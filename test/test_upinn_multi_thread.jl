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
using Dates

# 包含必要的文件
include("../src/generate_random_sample_for_upinn.jl")
include("../src/adjacent_matrix_calculation.jl")

# 定义Leaky ReLU激活函数
leaky_relu(x, α=0.01f0) = max(α * x, x)

# 定义SiLu激活函数（Sigmoid Linear Unit）
silu(x) = x * sigmoid(x)

# 原子操作用于线程安全的累加
mutable struct AtomicFloat
    value::Float32
    lock::ReentrantLock
    
    AtomicFloat(val::Float32) = new(val, ReentrantLock())
end

function atomic_add!(a::AtomicFloat, val::Float32)
    lock(a.lock) do
        a.value += val
    end
    return a.value
end

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

    # 检查X的维度并相应处理
    if ndims(X_f32) == 3
        # 处理批量输入 (节点数, 特征数, 批量大小)
        batch_size = size(X_f32, 3)
        result = similar(X_f32)
        
        for b in 1:batch_size
            # 对每个批次单独进行图卷积操作
            X_batch = X_f32[:, :, b]
            X_conv = A_norm * X_batch
            linear_output = X_conv * gc.weight .+ gc.bias'
            result[:, :, b] = leaky_relu.(linear_output, gc.α)
        end
        
        return result
    else
        # 处理单个输入 (节点数, 特征数)
        X_conv = A_norm * X_f32
        linear_output = X_conv * gc.weight .+ gc.bias'
        return leaky_relu.(linear_output, gc.α)
    end
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
    
    # 检查X的维度并相应处理
    if ndims(X_f32) == 3
        # 处理批量输入 (节点数, 特征数, 批量大小)
        batch_size = size(X_f32, 3)
        result = similar(X_f32, size(X_f32, 1), size(fc.weight, 2), batch_size)
        
        for b in 1:batch_size
            # 对每个批次单独进行全连接操作
            X_batch = X_f32[:, :, b]
            linear_output = X_batch * fc.weight .+ fc.bias'
            result[:, :, b] = fc.σ.(linear_output)
        end
        
        return result
    else
        # 处理单个输入 (节点数, 特征数)
        linear_output = X_f32 * fc.weight .+ fc.bias'
        return fc.σ.(linear_output)
    end
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
    
    # 检查维度并适当处理
    if ndims(main_output) == 3
        return rb.σ.(main_output .+ shortcut_output)
    else
        return rb.σ.(main_output + shortcut_output)
    end
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
    if ndims(x) == 3
        # 处理批量输入
        batch_size = size(x, 3)
        result = similar(x)
        
        for b in 1:batch_size
            result[:, :, b] = 0.01 .* x[:, :, b] .* bs.beta'
        end
        
        return result
    else
        # 处理单个输入
        return 0.01 .* x .* bs.beta'
    end
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
    
    # 检查是否为批量输入
    if ndims(x) == 3
        batch_size = size(x, 3)
        num_nodes = size(x, 1)
        result = similar(x)
        
        for b in 1:batch_size
            # 全局池化 (对每个批次单独处理)
            pooled_output = mean(output_2[:, :, b], dims=1)
            repeated_global_features = repeat(pooled_output, num_nodes, 1)
            
            # 特征增强
            enhanced_features = hcat(
                output_2[:, :, b],          # GCN输出
                repeated_global_features,   # 全局池化特征
                x[:, :, b]                  # 原始特征
            )
            
            # 全连接层（带残差连接）
            result_1 = net.res_block1(enhanced_features)
            result_2 = net.res_block2(result_1)
            result_3 = net.fc3(result_2)
            
            # 残差连接（从输入到最后一层）
            residual = net.projection(enhanced_features)
            result_with_residual = result_3 + residual
            
            # Beta缩放
            result[:, :, b] = net.beta_scaling(result_with_residual)
        end
        
        return result
    else
        # 原始的单个样本处理逻辑
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

# 优化的前向传播函数
function optimized_forward_pass(network::UPINNNetwork, x, A, pq_idx, pv_idx, ref_idx, Ybus)
    # 网络前向传播
    final_output = network(x, A)
    
    # 构建电力系统变量
    num_buses = size(final_output, 1)
    
    # 预分配数组
    P = zeros(Float32, num_buses)
    Q = zeros(Float32, num_buses)
    Vm = zeros(Float32, num_buses)
    θ = zeros(Float32, num_buses)
    
    # 填充数组（使用向量化操作）
    P[pq_idx] .= x[pq_idx, 1]
    P[pv_idx] .= x[pv_idx, 1]
    P[ref_idx] .= final_output[ref_idx, 1]
    
    Q[pq_idx] .= x[pq_idx, 2]
    Q[pv_idx] .= final_output[pv_idx, 2]
    Q[ref_idx] .= final_output[ref_idx, 2]
    
    Vm[pq_idx] .= final_output[pq_idx, 2]
    Vm[pv_idx] .= x[pv_idx, 2]
    Vm[ref_idx] .= x[ref_idx, 1]
    
    θ[pq_idx] .= final_output[pq_idx, 1]
    θ[pv_idx] .= final_output[pv_idx, 1]
    θ[ref_idx] .= x[ref_idx, 2]
    
    # 构建复数电压
    θ_rad = θ .* (π / 180)
    V = Vm .* (cos.(θ_rad) .+ im .* sin.(θ_rad))
    
    # 计算功率不平衡
    S_calc = V .* conj.(Ybus * V)
    S_spec = P .+ im .* Q
    ΔS = S_calc .- S_spec
    
    all_idx = vcat(pq_idx, pv_idx, ref_idx)
    ΔP = real(ΔS[all_idx])
    ΔQ = imag(ΔS[all_idx])
    
    # 计算损失
    loss, ω_p, ω_q = calculate_loss_fully_fixed(ΔP, ΔQ, pv_idx, ref_idx)
    
    return loss, final_output, ΔP, ΔQ, V
end

# 并行计算雅可比矩阵
function compute_full_ordered_jacobian_parallel(network::UPINNNetwork, x, A, pq_idx, pv_idx, ref_idx)
    # 将网络参数向量化
    θ, re = Flux.destructure(network)
    param_size = length(θ)
    
    # 获取网络输出
    final_output = network(x, A)
    num_buses = size(final_output, 1)
    
    # 创建索引映射
    ordered_indices = Dict()
    descriptions = String[]
    
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
    
    # 创建索引列表
    indices_list = collect(keys(ordered_indices))
    
    # 并行计算每个输出变量关于所有参数的梯度
    @threads for idx in 1:length(indices_list)
        i, j = indices_list[idx]
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
    
    return jacobian, descriptions, total_outputs
end

# 计算物理约束的雅可比矩阵
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

# 计算链式法则的梯度
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

# 并行处理多个批次
function train_batches_parallel(network, x_batches, A, pq_idx, pv_idx, ref_idx, Ybus, η, β, v_prev=nothing)
    num_batches = length(x_batches)
    batch_results = Vector{Tuple}(undef, num_batches)
    
    # 并行处理每个批次
    @threads for i in 1:num_batches
        x_batch = x_batches[i]
        batch_size = size(x_batch, 3)
        
        # 初始化批次结果
        batch_loss = 0.0
        batch_dL_dθ = nothing
        
        # 处理批次中的每个样本
        for j in 1:batch_size
            x = x_batch[:, :, j]
            
            # 1. 前向传播
            loss, final_output, ΔP, ΔQ, V = optimized_forward_pass(network, x, A, pq_idx, pv_idx, ref_idx, Ybus)
            
            # 2. 计算有序梯度雅可比矩阵
            jacobian, _, _ = compute_full_ordered_jacobian_parallel(network, x, A, pq_idx, pv_idx, ref_idx)
            
            # 3. 计算物理约束的雅可比矩阵
            J = calculate_the_deviation(pq_idx, pv_idx, ref_idx, Ybus, V)
            
            # 4. 根据链式法则计算dL/dθ
            dL_dθ = calculate_chain_deviation(J, jacobian, pv_idx, pq_idx, ref_idx, ΔP, ΔQ)
            
            # 累加损失和梯度
            batch_loss += loss
            
            if isnothing(batch_dL_dθ)
                batch_dL_dθ = dL_dθ
            else
                batch_dL_dθ += dL_dθ
            end
        end
        
        # 计算批次平均
        batch_loss /= batch_size
        batch_dL_dθ /= batch_size
        
        # 存储结果
        batch_results[i] = (batch_loss, batch_dL_dθ)
    end
    
    # 聚合结果
    total_loss = 0.0
    avg_dL_dθ = nothing
    
    for (loss, dL_dθ) in batch_results
        total_loss += loss
        
        if isnothing(avg_dL_dθ)
            avg_dL_dθ = dL_dθ
        else
            avg_dL_dθ += dL_dθ
        end
    end
    
    # 计算平均梯度
    avg_dL_dθ = avg_dL_dθ / num_batches
    
    # 更新参数
    updated_network, new_v_prev = update_parameters(network, avg_dL_dθ, η, β, 1, v_prev)
    
    avg_loss = total_loss / num_batches
    return updated_network, avg_loss, new_v_prev
end


# 并行生成样本
function parallel_generate_samples(case_file, num_samples)
    # 基本设置
    X, Ybus, pq_idx, pv_idx, ref_idx, baseMVA = generate_random_sample_for_upinn(case_file, 1)
    num_buses = size(X, 1)
    
    # 预分配结果数组
    all_samples = zeros(Float32, num_buses, 2, num_samples)
    
    # 并行生成样本
    @threads for i in 1:num_samples
        sample, _, _, _, _, _ = generate_random_sample_for_upinn(case_file, 1)
        all_samples[:, :, i] = sample[:, :, 1]
    end
    
    return all_samples, Ybus, pq_idx, pv_idx, ref_idx, baseMVA
end

# 完整的并行训练流程
# 完整的并行训练流程
function train_upinn_parallel(case_file, epochs=2000, batch_size=64, learning_rate=1e-4, β=0.9, num_samples=1000)
    # 显示线程信息
    println("使用 $(Threads.nthreads()) 个线程进行训练")
    
    # 1. 并行生成数据
    println("开始并行生成 $num_samples 个训练样本...")
    start_time = time()
    X, Ybus, pq_idx, pv_idx, ref_idx, baseMVA = parallel_generate_samples(case_file, num_samples)
    A = admittance_to_adjacency(Ybus)
    println("数据生成完成，耗时: $(round(time() - start_time, digits=2)) 秒")
    
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
    println("开始训练，总共 $epochs 个epochs...")
    total_start_time = time()
    
    # 创建日志文件
    log_filename = "upinn_training_log_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    open(log_filename, "w") do log_file
        write(log_file, "Epoch,Loss,Eval_Loss,Max_ΔP,Max_ΔQ,Epoch_Time,Elapsed_Time,Remaining_Time\n")
    end
    
    for epoch in 1:epochs
        epoch_start_time = time()
        
        # 打乱数据
        indices = shuffle(1:num_samples)
        X_shuffled = X[:, :, indices]
        
        # 将数据分成批次
        x_batches = []
        for i in 1:num_batches
            start_idx = (i - 1) * batch_size + 1
            end_idx = min(i * batch_size, num_samples)
            push!(x_batches, X_shuffled[:, :, start_idx:end_idx])
        end
        
        # 将批次分组以便并行处理
        batch_groups = []
        group_size = max(1, ceil(Int, num_batches / Threads.nthreads()))
        for i in 1:group_size:num_batches
            end_idx = min(i + group_size - 1, num_batches)
            push!(batch_groups, x_batches[i:end_idx])
        end
        
        # 并行处理批次组
        epoch_loss = 0.0
        for group in batch_groups
            network, group_loss, v_prev = train_batches_parallel(network, group, A, pq_idx, pv_idx, ref_idx, Ybus, learning_rate, β, v_prev)
            epoch_loss += group_loss * length(group)
        end
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / length(batch_groups)
        push!(losses, avg_epoch_loss)
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss
            best_loss = avg_epoch_loss
            best_network = deepcopy(network)
        end
        
        # 评估当前模型 - 每个epoch都评估
        x_eval = X[:, :, 1]  # 使用第一个样本进行评估
        eval_loss, _, ΔP, ΔQ, _ = optimized_forward_pass(network, x_eval, A, pq_idx, pv_idx, ref_idx, Ybus)
        
        # 计算时间信息
        epoch_time = time() - epoch_start_time
        elapsed_time = time() - total_start_time
        remaining_time = elapsed_time / epoch * (epochs - epoch)
        
        # 打印训练进度 - 每个epoch都输出
        println("Epoch $epoch/$epochs ($(round(epoch_time, digits=2))s), Loss: $(round(avg_epoch_loss, digits=6)), " *
                "Eval Loss: $(round(eval_loss, digits=6))")
        println("ΔP max: $(round(maximum(abs.(ΔP)), digits=6)), ΔQ max: $(round(maximum(abs.(ΔQ)), digits=6))")
        println("已用时间: $(round(elapsed_time/60, digits=1))分钟, 预计剩余: $(round(remaining_time/60, digits=1))分钟")
        
        # 记录日志
        open(log_filename, "a") do log_file
            write(log_file, "$epoch,$(avg_epoch_loss),$(eval_loss),$(maximum(abs.(ΔP))),$(maximum(abs.(ΔQ))),$(epoch_time),$(elapsed_time),$(remaining_time)\n")
        end
        
        # 每100个epoch保存一次损失曲线
        if epoch % 100 == 0
            p = plot(1:length(losses), losses, xlabel="Epoch", ylabel="Loss", 
                    title="UPINN Training Loss (Epoch $epoch)", legend=false, 
                    lw=2, grid=true, color=:blue)
            savefig(p, "upinn_training_loss_epoch_$(epoch).png")
        end
        
        # 早停条件
        if epoch > 100 && all(losses[end-9:end] .> losses[end-10])
            println("Early stopping at epoch $epoch")
            break
        end
        
        # 学习率衰减
        if epoch % 500 == 0
            learning_rate *= 0.5
            println("Reducing learning rate to $learning_rate")
        end
    end
    
    total_training_time = time() - total_start_time
    println("\n训练完成！总用时: $(round(total_training_time/60, digits=1))分钟")
    
    # 6. 绘制最终损失曲线
    p = plot(1:length(losses), losses, xlabel="Epoch", ylabel="Loss", 
             title="UPINN Training Loss (Parallel)", legend=false, 
             lw=2, grid=true, color=:blue)
    savefig(p, "upinn_training_loss_final.png")
    
    # 7. 评估最终模型
    x_eval = X[:, :, 1]
    final_loss, final_output, ΔP, ΔQ, V = optimized_forward_pass(best_network, x_eval, A, pq_idx, pv_idx, ref_idx, Ybus)
    
    println("\n========== 训练结果 ==========")
    println("初始损失: $(losses[1])")
    println("最终损失: $final_loss")
    println("功率不平衡ΔP最大值: $(maximum(abs.(ΔP)))")
    println("功率不平衡ΔQ最大值: $(maximum(abs.(ΔQ)))")
    println("线程数: $(Threads.nthreads())")
    
    return best_network, losses, final_output
end


# 并行评估函数
function parallel_evaluate_model(network, X, A, pq_idx, pv_idx, ref_idx, Ybus, num_samples=100)
    losses = zeros(Float32, num_samples)
    max_dp = zeros(Float32, num_samples)
    max_dq = zeros(Float32, num_samples)
    
    # 并行评估多个样本
    @threads for i in 1:num_samples
        x = X[:, :, i]
        loss, _, ΔP, ΔQ, _ = optimized_forward_pass(network, x, A, pq_idx, pv_idx, ref_idx, Ybus)
        
        losses[i] = loss
        max_dp[i] = maximum(abs.(ΔP))
        max_dq[i] = maximum(abs.(ΔQ))
    end
    
    avg_loss = mean(losses)
    avg_max_dp = mean(max_dp)
    avg_max_dq = mean(max_dq)
    
    return avg_loss, avg_max_dp, avg_max_dq
end

# 运行完整的并行训练
function run_parallel_training()
    case_file = "D:/luosipeng/matpower8.1/data/case118.m"
    epochs = 2000
    batch_size = 64
    learning_rate = 1e-4
    num_samples = 1000  # 增加样本数以充分利用并行计算
    
    # 设置线程数（如果没有通过环境变量设置）
    if Threads.nthreads() == 1
        println("警告：当前只使用了1个线程。建议通过设置环境变量增加线程数：")
        println("在命令行中使用: export JULIA_NUM_THREADS=<核心数>")
        println("或在Julia启动时使用: julia --threads <核心数>")
    end
    
    println("\n========== UPINN 并行训练 ==========")
    println("案例文件: $case_file")
    println("训练参数: epochs=$epochs, batch_size=$batch_size, learning_rate=$learning_rate")
    println("样本数量: $num_samples")
    println("线程数: $(Threads.nthreads())")
    println("======================================\n")
    
    trained_network, losses, final_output = train_upinn_parallel(
        case_file, epochs, batch_size, learning_rate, 0.9, num_samples
    )
    
    # 保存模型（可选）
    # 可以使用BSON或JLD2包保存模型
    # using BSON
    # BSON.@save "upinn_model_parallel.bson" trained_network
    
    return trained_network, losses
end

# 性能比较函数
function compare_performance(case_file, epochs=100)
    # 准备数据
    X, Ybus, pq_idx, pv_idx, ref_idx, baseMVA = generate_random_sample_for_upinn(case_file, 100)
    A = admittance_to_adjacency(Ybus)
    
    # 创建网络
    input_dim = size(X, 1)
    network = UPINNNetwork(input_dim, 0.01f0)
    
    # 1. 测试串行版本
    println("测试串行版本性能...")
    serial_start = time()
    
    # 简化版串行训练循环
    v_prev = nothing
    for epoch in 1:epochs
        for i in 1:10  # 每个epoch处理10个样本
            x = X[:, :, i]
            
            # 前向传播
            loss, final_output, ΔP, ΔQ, V = forward_pass_fully_fixed(network, x, A, pq_idx, pv_idx, ref_idx, Ybus)
            
            # 计算雅可比矩阵
            jacobian, _, _ = compute_full_ordered_jacobian(network, x, A, pq_idx, pv_idx, ref_idx)
            
            # 计算物理约束的雅可比矩阵
            J = calculate_the_deviation(pq_idx, pv_idx, ref_idx, Ybus, V)
            
            # 计算梯度
            dL_dθ = calculate_chain_deviation(J, jacobian, pv_idx, pq_idx, ref_idx, ΔP, ΔQ)
            
            # 更新参数
            network, v_prev = update_parameters(network, dL_dθ, 1e-4, 0.9, 1, v_prev)
        end
    end
    
    serial_time = time() - serial_start
    
    # 2. 测试并行版本
    println("测试并行版本性能...")
    network = UPINNNetwork(input_dim, 0.01f0)  # 重置网络
    parallel_start = time()
    
    # 简化版并行训练循环
    v_prev = nothing
    for epoch in 1:epochs
        # 准备10个批次
        x_batches = [X[:, :, i:i] for i in 1:10]
        
        # 并行处理批次
        network, _, v_prev = train_batches_parallel(network, x_batches, A, pq_idx, pv_idx, ref_idx, Ybus, 1e-4, 0.9, v_prev)
    end
    
    parallel_time = time() - parallel_start
    
    # 输出比较结果
    speedup = serial_time / parallel_time
    println("\n========== 性能比较 ==========")
    println("串行版本用时: $(round(serial_time, digits=2))秒")
    println("并行版本用时: $(round(parallel_time, digits=2))秒")
    println("加速比: $(round(speedup, digits=2))倍")
    println("线程数: $(Threads.nthreads())")
    
    return serial_time, parallel_time, speedup
end

# 执行并行训练
function main()
    # 检查线程数并提示
    if Threads.nthreads() == 1
        println("警告: 当前只使用1个线程，无法发挥并行计算优势。")
        println("请使用以下方式启动Julia以启用多线程:")
        println("  命令行: JULIA_NUM_THREADS=<核心数> julia")
        println("  或: julia --threads <核心数>")
        println("\n是否仍要继续? (y/n)")
        response = readline()
        if lowercase(response) != "y"
            println("已取消训练。")
            return
        end
    end
    
    println("开始UPINN模型并行训练...")
    start_time = time()
    trained_network, losses = run_parallel_training()
    total_time = time() - start_time
    
    println("\n训练完成！")
    println("总训练时间: $(round(total_time/60, digits=2))分钟")
    println("最终损失: $(losses[end])")
    println("初始损失: $(losses[1])")
    println("损失下降率: $(round((1 - losses[end]/losses[1])*100, digits=2))%")
    
    return trained_network, losses
end


main()


