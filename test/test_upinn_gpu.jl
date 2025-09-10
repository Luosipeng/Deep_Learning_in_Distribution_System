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
using CUDA
using CUDA: @allowscalar

# 检查GPU是否可用
has_cuda = CUDA.functional()
if has_cuda
    println("CUDA is available! Using GPU for training.")
else
    println("CUDA is not available. Using CPU for training.")
end

# 包含必要的文件
include("../src/generate_random_sample_for_upinn.jl")
include("../src/adjacent_matrix_calculation.jl")

# 定义Leaky ReLU激活函数
leaky_relu(x, α=0.01f0) = max(α * x, x)

# 定义SiLu激活函数（Sigmoid Linear Unit）
silu(x) = x * sigmoid(x)

# 将数据移动到GPU或CPU的辅助函数
function to_device(x)
    if has_cuda
        return CuArray(x)
    else
        return x
    end
end

function from_device(x)
    if has_cuda && x isa CuArray
        return Array(x)
    else
        return x
    end
end

# 确保一致的数据类型
function ensure_consistent_device(x, y)
    if x isa CuArray && !(y isa CuArray)
        return x, to_device(y)
    elseif !(x isa CuArray) && y isa CuArray
        return to_device(x), y
    else
        return x, y
    end
end

# 1. 图卷积层实现
struct GraphConv
    weight::AbstractMatrix{Float32}
    bias::AbstractVector{Float32}
    α::Float32  # Leaky ReLU参数
end

function GraphConv(feature_dimension::Int, α::Float32=0.01f0)
    weight = to_device(Flux.glorot_uniform(feature_dimension, feature_dimension))
    bias = to_device(zeros(Float32, feature_dimension))
    return GraphConv(weight, bias, α)
end

# 修改 GraphConv 函数，确保输入在同一设备上
function (gc::GraphConv)(X, A)
    # 确保输入类型一致
    X_f32 = to_device(Float32.(X))
    A_f32 = to_device(Float32.(A))
    
    # 如果A是稀疏矩阵，转换为密集矩阵
    if typeof(A_f32) <: SparseMatrixCSC
        A_f32 = to_device(Matrix(A_f32))
    end

    # 计算度矩阵的逆平方根 - 避免标量索引
    D_diag = vec(sum(A_f32, dims=2))
    D_diag = D_diag .+ 1f-10  # 添加小值防止除零
    D_inv_sqrt_diag = 1f0 ./ sqrt.(D_diag)
    
    # 使用矩阵乘法而不是标量索引
    # D^(-1/2) * A * D^(-1/2)
    A_norm = Diagonal(D_inv_sqrt_diag) * A_f32 * Diagonal(D_inv_sqrt_diag)

    # 图卷积操作: D^(-1/2) * A * D^(-1/2) * X * W
    X_conv = A_norm * X_f32
    
    # 确保权重和偏置在与输入相同的设备上
    weight = X_f32 isa CuArray ? to_device(gc.weight) : from_device(gc.weight)
    bias = X_f32 isa CuArray ? to_device(gc.bias) : from_device(gc.bias)

    # 线性变换
    linear_output = X_conv * weight .+ bias'

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
    weight::AbstractMatrix{Float32}
    bias::AbstractVector{Float32}
    σ::Function
end

function FCN(in_dim::Int, out_dim::Int; σ=silu)
    weight = to_device(Flux.glorot_uniform(in_dim, out_dim))
    bias = to_device(zeros(Float32, out_dim))
    return FCN(weight, bias, σ)
end

# 修改 FCN 函数，确保设备一致性
function (fc::FCN)(X)
    # 确保输入在正确的设备上
    X_device = X isa CuArray ? X : to_device(X)
    
    # 确保权重和偏置在与输入相同的设备上
    weight = X_device isa CuArray ? to_device(fc.weight) : from_device(fc.weight)
    bias = X_device isa CuArray ? to_device(fc.bias) : from_device(fc.bias)
    
    # 线性变换
    linear_output = X_device * weight .+ bias'
    
    # 应用激活函数
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
    beta::AbstractVector{Float32}
end

function BetaScaling(output_dim::Int; init_value::Float32=1.0f0)
    return BetaScaling(to_device(fill(init_value, output_dim)))
end

# 修改 BetaScaling 函数，确保设备一致性
function (bs::BetaScaling)(x)
    # 确保beta在与输入相同的设备上
    beta = x isa CuArray ? to_device(bs.beta) : from_device(bs.beta)
    return 0.01 .* x .* beta'
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

# 修改 UPINNNetwork 函数，确保设备一致性
function (net::UPINNNetwork)(x, A)
    # 确保输入在同一设备上
    x_device = to_device(x)
    A_device = to_device(A)
    
    # GCN层
    output_1 = net.gcn1(x_device, A_device)
    output_2 = net.gcn2(output_1, A_device)
    
    # 全局池化 - 避免标量索引
    pooled_output = mean(output_2, dims=1)
    num_nodes = size(output_2, 1)
    repeated_global_features = repeat(pooled_output, num_nodes, 1)

    # 特征增强
    enhanced_features = hcat(
        output_2,                 # GCN输出
        repeated_global_features, # 全局池化特征
        x_device                  # 原始特征
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

# 添加一个函数来创建网络的CPU副本
function cpu_copy(network::UPINNNetwork)
    # 创建一个新的UPINNNetwork实例，所有参数都在CPU上
    return UPINNNetwork(
        cpu_copy(network.gcn1),
        cpu_copy(network.gcn2),
        cpu_copy(network.res_block1),
        cpu_copy(network.res_block2),
        cpu_copy(network.fc3),
        cpu_copy(network.projection),
        cpu_copy(network.beta_scaling)
    )
end

# 为每个组件添加cpu_copy方法
function cpu_copy(gc::GraphConv)
    return GraphConv(
        Array(gc.weight),
        Array(gc.bias),
        gc.α
    )
end

function cpu_copy(fc::FCN)
    return FCN(
        Array(fc.weight),
        Array(fc.bias),
        fc.σ
    )
end

function cpu_copy(rb::ResidualBlock)
    return ResidualBlock(
        cpu_copy(rb.main_path),
        typeof(rb.shortcut) == typeof(identity) ? identity : cpu_copy(rb.shortcut),
        rb.σ
    )
end

function cpu_copy(bs::BetaScaling)
    return BetaScaling(Array(bs.beta))
end

function cpu_copy(chain::Chain)
    return Chain(map(cpu_copy, chain.layers)...)
end

# 计算损失函数
function calculate_loss_fully_fixed(ΔP, ΔQ, pv_idx, ref_idx)
    # 创建权重向量
    n_p = length(ΔP)
    n_q = length(ΔQ)
    
    ω_p = Float32[i in ref_idx ? 0.0f0 : 1.0f0 for i in 1:n_p]
    ω_q = Float32[i in ref_idx || i in pv_idx ? 0.0f0 : 1.0f0 for i in 1:n_q]
    
    n = length(ΔP) + length(ΔQ)
    L = (sum(ω_p .* ΔP.^2) + sum(ω_q .* ΔQ.^2)) / n
    
    return L, ω_p, ω_q
end

# 修改 forward_pass_fully_fixed 函数，确保设备一致性
function forward_pass_fully_fixed(network::UPINNNetwork, x, A, pq_idx, pv_idx, ref_idx, Ybus)
    # 确保输入在同一设备上
    x_device = to_device(x)
    A_device = to_device(A)
    
    # 网络前向传播
    final_output = network(x_device, A_device)
    
    # 为了计算电力系统变量，需要将输出移回CPU
    final_output_cpu = from_device(final_output)
    x_cpu = from_device(x_device)
    
    # 构建电力系统变量
    num_buses = size(final_output_cpu, 1)
    
    # 使用数组操作代替标量索引
    P = zeros(Float32, num_buses)
    Q = zeros(Float32, num_buses)
    Vm = zeros(Float32, num_buses)
    θ = zeros(Float32, num_buses)
    
    # 为PQ节点设置值
    for i in pq_idx
        P[i] = x_cpu[i, 1]
        Q[i] = x_cpu[i, 2]
        Vm[i] = final_output_cpu[i, 2]
        θ[i] = final_output_cpu[i, 1]
    end
    
    # 为PV节点设置值
    for i in pv_idx
        P[i] = x_cpu[i, 1]
        Q[i] = final_output_cpu[i, 2]
        Vm[i] = x_cpu[i, 2]
        θ[i] = final_output_cpu[i, 1]
    end
    
    # 为参考节点设置值
    for i in ref_idx
        P[i] = final_output_cpu[i, 1]
        Q[i] = final_output_cpu[i, 2]
        Vm[i] = x_cpu[i, 1]
        θ[i] = x_cpu[i, 2]
    end
    
    # 构建复数电压
    V = Vm .* (cos.(θ * π / 180) + im * sin.(θ * π / 180))
    
    # 计算功率不平衡
    ΔS = V .* conj.(Ybus * V) - (P + im .* Q)
    all_idx = vcat(pq_idx, pv_idx, ref_idx)
    ΔP = real(ΔS[all_idx])
    ΔQ = imag(ΔS[all_idx])
    
    # 计算损失
    loss, ω_p, ω_q = calculate_loss_fully_fixed(ΔP, ΔQ, pv_idx, ref_idx)
    
    return loss, final_output, ΔP, ΔQ, V
end

Base.length(gc::GraphConv) = length(gc.weight) + length(gc.bias)
Base.vec(gc::GraphConv) = vcat(vec(from_device(gc.weight)), vec(from_device(gc.bias)))

# 完全重构的 compute_full_ordered_jacobian 函数，避免任何形式的标量索引
function compute_full_ordered_jacobian(network::UPINNNetwork, x, A, pq_idx, pv_idx, ref_idx)
    # 确保在CPU上计算梯度 - 将所有数据移至CPU
    x_cpu = from_device(x)
    A_cpu = from_device(A)
    network_cpu = cpu_copy(network)  # 创建网络的CPU副本
    
    # 将网络参数向量化
    θ, re = Flux.destructure(network_cpu)
    param_size = length(θ)
    
    # 获取网络输出 - 确保在CPU上计算
    final_output = network_cpu(x_cpu, A_cpu)  # 直接使用CPU版本
    num_buses = size(final_output, 1)
    
    # 创建索引映射
    # 不使用字典，而是直接创建映射数组
    num_pq = length(pq_idx)
    num_pv = length(pv_idx)
    num_ref = length(ref_idx)
    
    # 计算总输出数量
    total_outputs = 2*num_pq + num_pv + num_ref + num_pv + num_ref
    
    # 初始化完整的雅可比矩阵
    jacobian = zeros(Float32, total_outputs, param_size)
    descriptions = String[]
    
    # 创建一个函数，计算整个网络输出的梯度
    function full_output_gradient(params)
        # 重构网络
        reconstructed_network = re(params)
        # 计算输出 - 使用CPU数据
        return reconstructed_network(x_cpu, A_cpu)
    end
    
    # 计算整个输出矩阵相对于参数的梯度
    # 这将返回一个包含所有元素梯度的结构
    full_jac = Zygote.jacobian(full_output_gradient, θ)[1]
    
    # 现在我们需要按照特定顺序提取这些梯度
    current_row = 1
    
    # 1. PQ节点的电压(V)
    for i in pq_idx
        push!(descriptions, "PQ节点 $i 的电压(V)")
        jacobian[current_row, :] = full_jac[i, 2, :]
        current_row += 1
    end
    
    # 2. PQ节点的相角(δ)
    for i in pq_idx
        push!(descriptions, "PQ节点 $i 的相角(δ)")
        jacobian[current_row, :] = full_jac[i, 1, :]
        current_row += 1
    end
    
    # 3. PV节点的相角(δ)
    for i in pv_idx
        push!(descriptions, "PV节点 $i 的相角(δ)")
        jacobian[current_row, :] = full_jac[i, 1, :]
        current_row += 1
    end
    
    # 4. 平衡节点的有功功率(Pn)
    for i in ref_idx
        push!(descriptions, "平衡节点 $i 的有功功率(Pn)")
        jacobian[current_row, :] = full_jac[i, 1, :]
        current_row += 1
    end
    
    # 5. PV节点的无功功率(Q)
    for i in pv_idx
        push!(descriptions, "PV节点 $i 的无功功率(Q)")
        jacobian[current_row, :] = full_jac[i, 2, :]
        current_row += 1
    end
    
    # 6. 平衡节点的无功功率(Q)
    for i in ref_idx
        push!(descriptions, "平衡节点 $i 的无功功率(Q)")
        jacobian[current_row, :] = full_jac[i, 2, :]
        current_row += 1
    end
    
    # 如果原始网络在GPU上，将结果移回GPU
    if network isa UPINNNetwork && any(p isa CuArray for p in Flux.params(network))
        jacobian = to_device(jacobian)
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

# 修改后的calculate_chain_deviation函数，确保设备一致性
function calculate_chain_deviation(J, ordered_jacobians, pv_idx, pq_idx, ref_idx, ΔP, ΔQ)
    # 确保所有输入在同一设备上
    if ordered_jacobians isa CuArray
        J = to_device(J)
        ΔP = to_device(ΔP)
        ΔQ = to_device(ΔQ)
    else
        J = from_device(J)
        ΔP = from_device(ΔP)
        ΔQ = from_device(ΔQ)
    end
    
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

# 修改后的参数更新函数，确保设备一致性
function update_parameters(network, dL_dθ, η, β, t, v_prev=nothing)
    # 将网络参数向量化
    θ, re = Flux.destructure(network)
    
    # 确保dL_dθ和θ在同一设备上
    if θ isa CuArray && !(dL_dθ isa CuArray)
        dL_dθ_vec = to_device(vec(dL_dθ))
    elseif !(θ isa CuArray) && dL_dθ isa CuArray
        dL_dθ_vec = from_device(vec(dL_dθ))
    else
        dL_dθ_vec = vec(dL_dθ)
    end
    
        # 计算二阶矩
    if isnothing(v_prev)
        # 初始化v_t
        v_t = (dL_dθ_vec).^2
    else
        # 确保v_prev和dL_dθ_vec在同一设备上
        if v_prev isa CuArray && !(dL_dθ_vec isa CuArray)
            v_prev = from_device(v_prev)
        elseif !(v_prev isa CuArray) && dL_dθ_vec isa CuArray
            v_prev = to_device(v_prev)
        end
        
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

# 修改 train_one_epoch 函数，确保设备一致性
function train_one_epoch(network, x_batch, A, pq_idx, pv_idx, ref_idx, Ybus, η, β, v_prev=nothing)
    total_loss = 0.0
    num_batches = size(x_batch, 3)
    
    # 预处理邻接矩阵
    A_device = to_device(A)
    
    for i in 1:num_batches
        x = to_device(x_batch[:, :, i])
        
        # 1. 前向传播
        loss, final_output, ΔP, ΔQ, V = forward_pass_fully_fixed(network, x, A_device, pq_idx, pv_idx, ref_idx, Ybus)
        
        # 2. 计算有序梯度雅可比矩阵 - 完全在CPU上计算
        x_cpu = from_device(x)
        A_cpu = from_device(A_device)
        jacobian, descriptions, total_outputs = compute_full_ordered_jacobian(network, x_cpu, A_cpu, pq_idx, pv_idx, ref_idx)
        
        # 3. 计算物理约束的雅可比矩阵
        J = calculate_the_deviation(pq_idx, pv_idx, ref_idx, Ybus, V)
        
        # 确保J和jacobian在同一设备上 - 都在CPU上
        jacobian = from_device(jacobian)
        
        # 4. 根据链式法则计算dL/dθ
        dL_dθ = calculate_chain_deviation(J, jacobian, pv_idx, pq_idx, ref_idx, ΔP, ΔQ)
        
        # 5. 更新参数
        network, v_prev = update_parameters(network, dL_dθ, η, β, 1, v_prev)
        
        total_loss += loss
    end
    
    avg_loss = total_loss / num_batches
    return network, avg_loss, v_prev
end


# 批量并行处理函数 - 修改为避免GPU标量索引
function process_batch_parallel(batch_data, network, A_device, pq_idx, pv_idx, ref_idx, Ybus)
    results = Vector{Any}(undef, length(batch_data))
    
    # 创建CPU版本的邻接矩阵，用于梯度计算
    A_cpu = from_device(A_device)
    
    Threads.@threads for i in 1:length(batch_data)
        x = to_device(batch_data[i])
        x_cpu = from_device(x)  # CPU版本用于梯度计算
        
        # 前向传播
        loss, final_output, ΔP, ΔQ, V = forward_pass_fully_fixed(network, x, A_device, pq_idx, pv_idx, ref_idx, Ybus)
        
        # 计算有序梯度雅可比矩阵 - 使用CPU版本避免GPU标量索引
        jacobian, _, _ = compute_full_ordered_jacobian(network, x_cpu, A_cpu, pq_idx, pv_idx, ref_idx)
        
        # 计算物理约束的雅可比矩阵
        J = calculate_the_deviation(pq_idx, pv_idx, ref_idx, Ybus, V)
        
        # 根据链式法则计算dL/dθ
        dL_dθ = calculate_chain_deviation(J, jacobian, pv_idx, pq_idx, ref_idx, ΔP, ΔQ)
        
        results[i] = (loss, dL_dθ)
    end
    
    return results
end

# 完整的训练流程
function train_upinn_full(case_file, epochs=2000, batch_size=64, learning_rate=1e-4, β=0.9)
    # 1. 生成数据
    X, Ybus, pq_idx, pv_idx, ref_idx, baseMVA = generate_random_sample_for_upinn(case_file)
    A = admittance_to_adjacency(Ybus)
    
    # 将A移至GPU
    A_device = to_device(A)
    
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
            network, batch_loss, v_prev = train_one_epoch(network, X_batch, A_device, pq_idx, pv_idx, ref_idx, Ybus, learning_rate, β, v_prev)
            
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
        if epoch % 50 == 0 || epoch == 1
            # 评估当前模型
            x_eval = to_device(X[:, :, 1])  # 使用第一个样本进行评估
            eval_loss, _, ΔP, ΔQ, _ = forward_pass_fully_fixed(network, x_eval, A_device, pq_idx, pv_idx, ref_idx, Ybus)
            
            println("Epoch $epoch/$epochs, Loss: $avg_epoch_loss, Eval Loss: $eval_loss")
            println("ΔP max: $(maximum(abs.(ΔP))), ΔQ max: $(maximum(abs.(ΔQ)))")
            
            # 显示GPU内存使用情况(如果可用)
            if has_cuda
                println("GPU Memory: $(CUDA.memory_status())")
            end
        end
        
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
        
        # 定期清理GPU内存
        if has_cuda && epoch % 10 == 0
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    # 6. 绘制损失曲线
    p = plot(1:length(losses), losses, xlabel="Epoch", ylabel="Loss", title="UPINN Training Loss", legend=false)
    savefig(p, "upinn_training_loss.png")
    
    # 7. 评估最终模型
    x_eval = to_device(X[:, :, 1])
    final_loss, final_output, ΔP, ΔQ, V = forward_pass_fully_fixed(best_network, x_eval, A_device, pq_idx, pv_idx, ref_idx, Ybus)
    
    println("\n========== 训练结果 ==========")
    println("初始损失: $(losses[1])")
    println("最终损失: $final_loss")
    println("功率不平衡ΔP最大值: $(maximum(abs.(ΔP)))")
    println("功率不平衡ΔQ最大值: $(maximum(abs.(ΔQ)))")
    
    # 清理GPU内存
    if has_cuda
        GC.gc()
        CUDA.reclaim()
    end
    
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
    println("使用GPU: $(has_cuda ? "是" : "否")")
    
    # 设置线程数
    num_threads = Threads.nthreads()
    println("使用 $num_threads 个CPU线程进行并行计算")
    
    trained_network, losses, final_output = train_upinn_full(case_file, epochs, batch_size, learning_rate)
    
    # 保存模型（可选）
    # BSON.@save "upinn_model.bson" trained_network
    
    return trained_network, losses, final_output
end

# 执行训练
trained_network, losses, final_output = run_full_training()

