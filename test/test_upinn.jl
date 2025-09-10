using Flux
using Random
using Base.Threads
using TimeSeriesPowerFlow
using Distributions
using Plots
using LinearAlgebra
using SparseArrays
using Statistics
using Zygote
using TimeSeriesPowerFlow.PowerFlow: dSbus_dV, makeSbus
using DelimitedFiles

# 包含必要的文件
include("../src/generate_random_sample_for_upinn.jl")
include("../src/adjacent_matrix_calculation.jl")

# 定义Leaky ReLU激活函数
leaky_relu(x, α=0.01f0) = max(α * x, x)

# 定义SiLu激活函数（Sigmoid Linear Unit）
silu(x) = x * sigmoid(x)

# 1. 图卷积层实现
struct GraphConv
    weight::Matrix{Float32}
    bias::Vector{Float32}
    α::Float32  # Leaky ReLU参数
end

function GraphConv(feature_dimension::Int, α::Float32=0.01f0)
    weight = Flux.glorot_uniform(feature_dimension, feature_dimension)
    bias = zeros(Float32, feature_dimension)
    return GraphConv(weight, bias, α)
end

function (gc::GraphConv)(X, A)
    # 转换输入为Float32类型
    X_f32 = Float32.(X)
    A_f32 = Float32.(A)
    
    # 如果A是稀疏矩阵，转换为密集矩阵
    if typeof(A_f32) <: SparseMatrixCSC
        A_f32 = Matrix(A_f32)
    end

    # 计算度矩阵的逆平方根
    D = Diagonal(sum(A_f32, dims=2)[:] .+ 1f-10)  # 添加小值防止除零
    D_inv_sqrt = Diagonal(1f0 ./ sqrt.(diag(D)))

    # 归一化邻接矩阵: D^(-1/2) * A * D^(-1/2)
    A_norm = D_inv_sqrt * A_f32 * D_inv_sqrt

    # 图卷积操作: D^(-1/2) * A * D^(-1/2) * X * W
    X_conv = A_norm * X_f32

    # 线性变换
    linear_output = X_conv * gc.weight .+ gc.bias'

    # 应用Leaky ReLU激活函数
    return leaky_relu.(linear_output, gc.α)
end

Flux.@functor GraphConv

function create_gcn_network(α::Float32=0.01f0)
    feature_dimension = 2
    # 创建GCN网络（使用Leaky ReLU）
    gcn = GraphConv(feature_dimension, α)
    return gcn
end

# 2. 全连接层实现
struct FCN
    weight::Matrix{Float32}
    bias::Vector{Float32}
    σ::Function
end

function FCN(in_dim::Int, out_dim::Int; σ=silu)
    weight = Flux.glorot_uniform(in_dim, out_dim)
    bias = zeros(Float32, out_dim)
    return FCN(weight, bias, σ)
end

function (fc::FCN)(X)
    X_f32 = Float32.(X)
    linear_output = X_f32 * fc.weight .+ fc.bias'
    return fc.σ.(linear_output)
end

Flux.@functor FCN

# 3. 残差连接块实现
struct ResidualBlock
    main_path
    shortcut
    σ::Function
end

function ResidualBlock(in_dim::Int, out_dim::Int; σ=silu)
    main_path = Chain(
        FCN(in_dim, out_dim, σ=σ),
        FCN(out_dim, out_dim, σ=identity)  # 注意这里不应用激活函数
    )
    
    # 如果输入输出维度不同，需要投影
    shortcut = in_dim == out_dim ? identity : FCN(in_dim, out_dim, σ=identity)

    return ResidualBlock(main_path, shortcut, σ)
end

function (rb::ResidualBlock)(x)
    main_output = rb.main_path(x)
    shortcut_output = rb.shortcut(x)
    return rb.σ.(main_output + shortcut_output)
end

Flux.@functor ResidualBlock (main_path, shortcut,)

# 4. Beta缩放层实现
struct BetaScaling
    beta::Vector{Float32}
end

function BetaScaling(output_dim::Int; init_value::Float32=1.0f0)
    return BetaScaling(fill(init_value, output_dim))
end

function (bs::BetaScaling)(x)
    return 0.01.*x .* bs.beta'
end

Flux.@functor BetaScaling

# 5. 完整的网络结构
struct UPINNNetwork
    gcn1::GraphConv
    gcn2::GraphConv
    res_block1::ResidualBlock
    res_block2::ResidualBlock
    fc3::FCN
    projection::FCN
    beta_scaling::BetaScaling
end

function UPINNNetwork(input_dim::Int, α::Float32=0.01f0)
    gcn1 = create_gcn_network(α)
    gcn2 = create_gcn_network(α)
    
    # 计算增强特征维度: GCN输出(2) + 全局特征(2) + 原始特征(2) = 6
    enhanced_features_dim = 6

    res_block1 = ResidualBlock(enhanced_features_dim, 64)
    res_block2 = ResidualBlock(64, 32)
    fc3 = FCN(32, 2, σ=identity)
    projection = FCN(enhanced_features_dim, 2, σ=identity)
    beta_scaling = BetaScaling(2)

    return UPINNNetwork(gcn1, gcn2, res_block1, res_block2, fc3, projection, beta_scaling)
end

function (net::UPINNNetwork)(x, A)
    # GCN层
    output_1 = net.gcn1(x, A)
    output_2 = net.gcn2(output_1, A)
    
    # 全局池化
    pooled_output = mean(output_2, dims=1)
    num_nodes = size(output_2, 1)
    repeated_global_features = repeat(pooled_output, num_nodes, 1)

    # 特征增强
    enhanced_features = hcat(
        output_2,                 # GCN输出
        repeated_global_features, # 全局池化特征
        x                        # 原始特征
    )

    # 全连接层（带残差连接）
    result_1 = net.res_block1(enhanced_features)
    result_2 = net.res_block2(result_1)
    result_3 = net.fc3(result_2)

    # 残差连接（从输入到最后一层）
    residual = net.projection(enhanced_features)
    result_with_residual = result_3 + residual

    # Beta缩放
    final_output = net.beta_scaling(result_with_residual)

    return final_output
end

Flux.@functor UPINNNetwork

# 计算损失函数
function calculate_loss_fully_fixed(ΔP, ΔQ, pv_idx, ref_idx)
    # 创建权重向量
    n_p = length(ΔP)
    n_q = length(ΔQ)
    
    ω_p = map(i -> Float32(i in ref_idx ? 0.0 : 1.0), 1:n_p)
    ω_q = map(i -> Float32(i in ref_idx || i in pv_idx ? 0.0 : 1.0), 1:n_q)
    
    n = length(ΔP) + length(ΔQ)
    L = (sum(ω_p .* ΔP.^2) + sum(ω_q .* ΔQ.^2)) / n
    
    return L, ω_p, ω_q
end

# 前向传播函数
function forward_pass_fully_fixed(network::UPINNNetwork, x, A, pq_idx, pv_idx, ref_idx, Ybus)
    # 网络前向传播
    final_output = network(x, A)
    
    # 构建电力系统变量
    num_buses = size(final_output, 1)
    
    P = map(i -> Float32(
        if i in pq_idx
            x[i, 1]
        elseif i in pv_idx
            x[i, 1]
        else  # ref_idx
            final_output[i, 1]
        end
    ), 1:num_buses)
    
    Q = map(i -> Float32(
        if i in pq_idx
            x[i, 2]
        elseif i in pv_idx
            final_output[i, 2]
        else  # ref_idx
            final_output[i, 2]
        end
    ), 1:num_buses)
    
    Vm = map(i -> Float32(
        if i in pq_idx
            final_output[i, 2]
        elseif i in pv_idx
            x[i, 2]
        else  # ref_idx
            x[i, 1]
        end
    ), 1:num_buses)
    
    θ = map(i -> Float32(
        if i in pq_idx
            final_output[i, 1]
        elseif i in pv_idx
            final_output[i, 1]
        else  # ref_idx
            x[i, 2]
        end
    ), 1:num_buses)
    
    # 转换为向量
    P_vec = collect(P)
    Q_vec = collect(Q)
    Vm_vec = collect(Vm)
    θ_vec = collect(θ)
    
    # 构建复数电压
    V = Vm_vec .* (cos.(θ_vec * π / 180) + im * sin.(θ_vec * π / 180))
    
    # 计算功率不平衡
    ΔS = V .* conj.(Ybus * V) - (P_vec + im .* Q_vec)
    all_idx = vcat(pq_idx, pv_idx, ref_idx)
    ΔP = real(ΔS[all_idx])
    ΔQ = imag(ΔS[all_idx])
    
    # 计算损失
    loss, ω_p, ω_q = calculate_loss_fully_fixed(ΔP, ΔQ, pv_idx, ref_idx)
    
    return loss, final_output, ΔP, ΔQ, V
end

Base.length(gc::GraphConv) = length(gc.weight) + length(gc.bias)
Base.vec(gc::GraphConv) = vcat(vec(gc.weight), vec(gc.bias))

# 直接生成完整雅可比矩阵的函数
function compute_full_ordered_jacobian(network::UPINNNetwork, x, A, pq_idx, pv_idx, ref_idx)
    # 将网络参数向量化
    θ, re = Flux.destructure(network)
    param_size = length(θ)
    
    # 获取网络输出
    final_output = network(x, A)
    num_buses = size(final_output, 1)
    
    # 创建索引映射，记录每个节点的输出在最终雅可比矩阵中的位置
    ordered_indices = Dict()
    descriptions = String[]  # 用于存储每行代表的物理意义
    
    # 1. PQ节点的电压(V)
    current_idx = 1
    for i in pq_idx
        ordered_indices[(i, 2)] = current_idx  # V对应final_output的第2列
        push!(descriptions, "PQ节点 $i 的电压(V)")
        current_idx += 1
    end
    
    # 2. PQ节点的相角(δ)
    for i in pq_idx
        ordered_indices[(i, 1)] = current_idx  # δ对应final_output的第1列
        push!(descriptions, "PQ节点 $i 的相角(δ)")
        current_idx += 1
    end
    
    # 3. PV节点的相角(δ)
    for i in pv_idx
        ordered_indices[(i, 1)] = current_idx  # δ对应final_output的第1列
        push!(descriptions, "PV节点 $i 的相角(δ)")
        current_idx += 1
    end
    
    # 4. 平衡节点的有功功率(Pn)
    for i in ref_idx
        ordered_indices[(i, 1)] = current_idx  # Pn对应final_output的第1列
        push!(descriptions, "平衡节点 $i 的有功功率(Pn)")
        current_idx += 1
    end
    
    # 5. PV节点的无功功率(Q)
    for i in pv_idx
        ordered_indices[(i, 2)] = current_idx  # Q对应final_output的第2列
        push!(descriptions, "PV节点 $i 的无功功率(Q)")
        current_idx += 1
    end
    
    # 6. 平衡节点的无功功率(Q)
    for i in ref_idx
        ordered_indices[(i, 2)] = current_idx  # Q对应final_output的第2列
        push!(descriptions, "平衡节点 $i 的无功功率(Q)")
        current_idx += 1
    end
    
    total_outputs = current_idx - 1
    
    # 初始化完整的雅可比矩阵
    jacobian = zeros(Float32, total_outputs, param_size)
    
    # 计算每个输出变量关于所有参数的梯度
    for i in 1:num_buses
        for j in 1:2  # 每个节点有2个输出
            if haskey(ordered_indices, (i, j))
                row_idx = ordered_indices[(i, j)]
                
                # 创建一个函数，它接受参数向量并返回特定的输出元素
                function output_selector(params)
                    # 重构网络
                    reconstructed_network = re(params)
                    # 计算输出
                    output = reconstructed_network(x, A)
                    # 返回特定元素
                    return output[i, j]
                end
                
                # 计算梯度
                element_grad = Zygote.gradient(output_selector, θ)[1]
                
                if !isnothing(element_grad)
                    jacobian[row_idx, :] = element_grad
                end
            end
        end
    end
    
    return jacobian, descriptions, total_outputs
end

# 创建输出变量描述
function create_output_descriptions(pq_idx, pv_idx, ref_idx)
    descriptions = String[]
    
    # 1. PQ节点的电压(V)
    for i in pq_idx
        push!(descriptions, "PQ节点 $i 的电压(V)")
    end
    
    # 2. PQ节点的相角(δ)
    for i in pq_idx
        push!(descriptions, "PQ节点 $i 的相角(δ)")
    end
    
    # 3. PV节点的相角(δ)
    for i in pv_idx
        push!(descriptions, "PV节点 $i 的相角(δ)")
    end
    
    # 4. 平衡节点的有功功率(Pn)
    for i in ref_idx
        push!(descriptions, "平衡节点 $i 的有功功率(Pn)")
    end
    
    # 5. PV节点的无功功率(Q)
    for i in pv_idx
        push!(descriptions, "PV节点 $i 的无功功率(Q)")
    end
    
    # 6. 平衡节点的无功功率(Q)
    for i in ref_idx
        push!(descriptions, "平衡节点 $i 的无功功率(Q)")
    end
    
    return descriptions
end

function calculate_the_deviation(pq_idx, pv_idx, ref_idx, Ybus, V)
    # 计算雅可比矩阵
    dSbus_dVa, dSbus_dVm = PowerFlow.dSbus_dV(Ybus, V)
    dP_dV = real.(dSbus_dVm)
    dP_dδ = real.(dSbus_dVa)
    dQ_dV = imag.(dSbus_dVm)
    dQ_dδ = real.(dSbus_dVa)

    # dP_dV
    dP_dV_pq_pq =  dP_dV[pq_idx, :]
    dP_dV_pq_pq =  dP_dV_pq_pq[:, pq_idx]

    dP_dV_pv_pq = dP_dV[pv_idx, :]
    dP_dV_pv_pq = dP_dV_pv_pq[:, pq_idx]

    dP_dV_ref_pq = spzeros(1, length(pq_idx))

    # dP_dδ
    dP_dδ_pq_pq = dP_dδ[pq_idx, :]
    dP_dδ_pq_pq = dP_dδ_pq_pq[:, pq_idx]
    dP_dδ_pq_pv = dP_dδ[pq_idx, :]
    dP_dδ_pq_pv = dP_dδ_pq_pv[:, pv_idx]

    dP_dδ_pv_pq = dP_dδ[pv_idx, :]
    dP_dδ_pv_pq = dP_dδ_pv_pq[:, pq_idx]
    dP_dδ_pv_pv = dP_dδ[pv_idx, :]
    dP_dδ_pv_pv = dP_dδ_pv_pv[:, pv_idx]

    dP_dδ_ref_pq = spzeros(1, length(pq_idx))
    dP_dδ_ref_pv = spzeros(1, length(pv_idx))

    # dP_dPn
    dP_dPn_pq_ref = zeros(length(pq_idx),1)
    dP_dPn_pv_ref = zeros(length(pv_idx),1)
    dP_dPn_ref_ref = -ones(length(ref_idx),1)

    # dP_dQ
    dP_dQ_pq_pv = zeros(length(pq_idx),length(pv_idx))
    dP_dQ_pq_ref = zeros(length(pq_idx),length(ref_idx))
    dP_dQ_pv_pv = zeros(length(pv_idx),length(pv_idx))
    dP_dQ_pv_ref = zeros(length(pv_idx),length(ref_idx))
    dP_dQ_ref_pv = zeros(length(ref_idx),length(pv_idx))
    dP_dQ_ref_ref = zeros(length(ref_idx),1)

    # dQ_dV
    dQ_dV_pq_pq =  dQ_dV[pq_idx, :]
    dQ_dV_pq_pq =  dQ_dV_pq_pq[:, pq_idx]

    dQ_dV_pv_pq = spzeros(length(pv_idx), length(pq_idx))

    dQ_dV_ref_pq = spzeros(1, length(pq_idx))

    # dQ_dδ
    dQ_dδ_pq_pq = dQ_dδ[pq_idx, :]
    dQ_dδ_pq_pq = dQ_dδ_pq_pq[:, pq_idx]
    dQ_dδ_pq_pv = dQ_dδ[pq_idx, :]
    dQ_dδ_pq_pv = dQ_dδ_pq_pv[:, pv_idx]

    dQ_dδ_pv_pq = spzeros(length(pv_idx), length(pq_idx))
    dQ_dδ_pv_pv = spzeros(length(pv_idx), length(pv_idx))

    dQ_dδ_ref_pq = spzeros(1, length(pq_idx))
    dQ_dδ_ref_pv = spzeros(1, length(pv_idx))

    # dQ_dPn
    dQ_dPn_pq_ref = spzeros(length(pq_idx), 1)
    dQ_dPn_pv_ref = spzeros(length(pv_idx), 1)
    dQ_dPn_ref_ref = spzeros(length(ref_idx), 1)

    # dQ_dQ
    dQ_dQ_pq_pv = spzeros(length(pq_idx), length(pv_idx))
    dQ_dQ_pq_ref = spzeros(length(pq_idx), length(ref_idx))

    dQ_dQ_pv_pv = spdiagm(0 => -ones(length(pv_idx)))
    dQ_dQ_pv_ref = spzeros(length(pv_idx), length(ref_idx))

    dQ_dQ_ref_pv = spzeros(length(ref_idx), length(pv_idx))
    dQ_dQ_ref_ref = spdiagm(0 => -ones(length(ref_idx)))

    # 拼接矩阵
    J = [dP_dV_pq_pq dP_dδ_pq_pq dP_dδ_pq_pv dP_dPn_pq_ref dP_dQ_pq_pv dP_dQ_pq_ref;
     dP_dV_pv_pq dP_dδ_pv_pq dP_dδ_pv_pv dP_dPn_pv_ref dP_dQ_pv_pv dP_dQ_pv_ref;
     dP_dV_ref_pq dP_dδ_ref_pq dP_dδ_ref_pv dP_dPn_ref_ref dP_dQ_ref_pv dP_dQ_ref_ref;
     dQ_dV_pq_pq dQ_dδ_pq_pq dQ_dδ_pq_pv dQ_dPn_pq_ref dQ_dQ_pq_pv dQ_dQ_pq_ref;
     dQ_dV_pv_pq dQ_dδ_pv_pq dQ_dδ_pv_pv dQ_dPn_pv_ref dQ_dQ_pv_pv dQ_dQ_pv_ref;
     dQ_dV_ref_pq dQ_dδ_ref_pq dQ_dδ_ref_pv dQ_dPn_ref_ref dQ_dQ_ref_pv dQ_dQ_ref_ref
     ]

    return J
end

function calculate_chain_deviation(J, ordered_jacobians, pv_idx, pq_idx, ref_idx, ΔP, ΔQ)
    n = length(pq_idx) + length(pv_idx) + length(ref_idx)
    ω_p = ones(size(ΔP,1))
    ω_q = ones(size(ΔQ,1))

    ω_p[ref_idx] .= 0
    ω_q[ref_idx] .= 0
    ω_q[pv_idx] .= 0

    P_vector = (ω_p./n).* ΔP
    Q_vector = (ω_q./n).* ΔQ
    mismatch = vcat(P_vector, Q_vector)
    deviation = mismatch' * J
    dL_dθ = deviation * ordered_jacobians

    return dL_dθ
end

# 近端算子实现 - L1正则化的近端算子就是软阈值函数
function prox_l1(x, λ)
    return sign.(x) .* max.(abs.(x) .- λ, 0)
end

# 修正后的参数更新函数
function update_parameters(network, dL_dθ, η, β, t, v_prev=nothing)
    # 将网络参数向量化
    θ, re = Flux.destructure(network)
    
    # 确保dL_dθ是向量
    dL_dθ_vec = vec(dL_dθ)
    
    # 计算二阶矩
    if isnothing(v_prev)
        # 初始化v_t
        v_t = (dL_dθ_vec).^2
    else
        # 更新v_t
        v_t = β .* v_prev .+ (1 - β) .* (dL_dθ_vec).^2
    end
    
    # 防止除零
    v_t_safe = v_t .+ 1e-8
    
    # 计算自适应学习率
    adaptive_lr = η ./ sqrt.(v_t_safe)
    
    # 近端梯度下降更新
    θ_new = prox_l1.(θ .- adaptive_lr .* dL_dθ_vec, adaptive_lr)
    
    # 重构网络
    updated_network = re(θ_new)
    
    return updated_network, v_t
end

# 单次epoch的前向传播和梯度计算
# 训练一个epoch
function train_one_epoch(network, x_batch, A, pq_idx, pv_idx, ref_idx, Ybus, η, β, v_prev=nothing)
    total_loss = 0.0
    num_batches = size(x_batch, 3)
    
    for i in 1:num_batches
        x = x_batch[:, :, i]
        
        # 1. 前向传播
        loss, final_output, ΔP, ΔQ, V = forward_pass_fully_fixed(network, x, A, pq_idx, pv_idx, ref_idx, Ybus)
        
        # 2. 计算有序梯度雅可比矩阵
        jacobian, descriptions, total_outputs = compute_full_ordered_jacobian(network, x, A, pq_idx, pv_idx, ref_idx)
        
        # 3. 计算物理约束的雅可比矩阵
        J = calculate_the_deviation(pq_idx, pv_idx, ref_idx, Ybus, V)
        
        # 4. 根据链式法则计算dL/dθ
        dL_dθ = calculate_chain_deviation(J, jacobian, pv_idx, pq_idx, ref_idx, ΔP, ΔQ)
        
        # 5. 更新参数
        network, v_prev = update_parameters(network, dL_dθ, η, β, 1, v_prev)
        
        total_loss += loss
    end
    
    avg_loss = total_loss / num_batches
    return network, avg_loss, v_prev
end

# 完整的训练流程
function train_upinn_full(case_file, epochs=2000, batch_size=64, learning_rate=1e-4, β=0.9)
    # 1. 生成数据
    X, Ybus, pq_idx, pv_idx, ref_idx, baseMVA = generate_random_sample_for_upinn(case_file)
    A = admittance_to_adjacency(Ybus)
    
    # 2. 创建网络
    input_dim = size(X, 1)
    network = UPINNNetwork(input_dim, 0.01f0)
    
    # 3. 准备训练数据
    num_samples = size(X, 3)
    num_batches = ceil(Int, num_samples / batch_size)
    
    # 4. 初始化训练状态
    losses = Float32[]
    v_prev = nothing
    best_loss = Inf
    best_network = deepcopy(network)
    
    # 5. 训练循环
    for epoch in 1:epochs
        # 打乱数据
        indices = shuffle(1:num_samples)
        X_shuffled = X[:, :, indices]
        
        epoch_loss = 0.0
        
        # 按批次训练
        for batch in 1:num_batches
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, num_samples)
            X_batch = X_shuffled[:, :, start_idx:end_idx]
            
            # 训练一个批次
            network, batch_loss, v_prev = train_one_epoch(network, X_batch, A, pq_idx, pv_idx, ref_idx, Ybus, learning_rate, β, v_prev)
            
            epoch_loss += batch_loss * (end_idx - start_idx + 1)
        end
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / num_samples
        push!(losses, avg_epoch_loss)
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss
            best_loss = avg_epoch_loss
            best_network = deepcopy(network)
        end
        
        # 打印训练进度

        # 评估当前模型
        x_eval = X[:, :, 1]  # 使用第一个样本进行评估
        eval_loss, _, ΔP, ΔQ, _ = forward_pass_fully_fixed(network, x_eval, A, pq_idx, pv_idx, ref_idx, Ybus)
        
        println("Epoch $epoch/$epochs, Loss: $avg_epoch_loss, Eval Loss: $eval_loss")
        println("ΔP max: $(maximum(abs.(ΔP))), ΔQ max: $(maximum(abs.(ΔQ)))")

        
        # 早停条件（可选）
        if epoch > 100 && all(losses[end-9:end] .> losses[end-10])
            println("Early stopping at epoch $epoch")
            break
        end
        
        # 学习率衰减（可选）
        if epoch % 500 == 0
            learning_rate *= 0.5
            println("Reducing learning rate to $learning_rate")
        end
    end
    
    # 6. 绘制损失曲线
    p = plot(1:length(losses), losses, xlabel="Epoch", ylabel="Loss", title="UPINN Training Loss", legend=false)
    savefig(p, "upinn_training_loss.png")
    
    # 7. 评估最终模型
    x_eval = X[:, :, 1]
    final_loss, final_output, ΔP, ΔQ, V = forward_pass_fully_fixed(best_network, x_eval, A, pq_idx, pv_idx, ref_idx, Ybus)
    
    println("\n========== 训练结果 ==========")
    println("初始损失: $(losses[1])")
    println("最终损失: $final_loss")
    println("功率不平衡ΔP最大值: $(maximum(abs.(ΔP)))")
    println("功率不平衡ΔQ最大值: $(maximum(abs.(ΔQ)))")
    
    return best_network, losses, final_output
end

# 运行完整训练
function run_full_training()
    case_file = "D:/luosipeng/matpower8.1/data/case118.m"
    epochs = 2000
    batch_size = 64
    learning_rate = 1e-4
    
    println("开始训练 UPINN 模型...")
    println("案例文件: $case_file")
    println("训练参数: epochs=$epochs, batch_size=$batch_size, learning_rate=$learning_rate")
    
    trained_network, losses, final_output = train_upinn_full(case_file, epochs, batch_size, learning_rate)
    
    # 保存模型（可选）
    # BSON.@save "upinn_model.bson" trained_network
    
    return trained_network, losses
end

# 执行训练
trained_network, losses = run_full_training()
