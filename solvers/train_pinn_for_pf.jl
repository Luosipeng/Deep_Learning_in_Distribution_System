# 1. 图卷积层实现
struct GraphConv
    weight::Matrix{Float32}
    bias::Vector{Float32}
    σ::Function
end

function GraphConv(feature_dimension::Int, σ=relu)
    weight = Flux.glorot_uniform(feature_dimension, feature_dimension)
    bias = zeros(Float32, feature_dimension)
    return GraphConv(weight, bias, σ)
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
    
    # 线性变换和激活
    return gc.σ.(X_conv * gc.weight .+ gc.bias')
end

Flux.@functor GraphConv

function create_gcn_network()

    feature_dimension = 2
    # 创建GCN网络
    gcn = GraphConv(feature_dimension)

    return gcn
end

# 定义SiLu激活函数（Sigmoid Linear Unit）
silu(x) = x * sigmoid(x)

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
    return fc.σ.(X_f32 * fc.weight .+ fc.bias')
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
    return rb.σ.(rb.main_path(x) + rb.shortcut(x))
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
    return x .* bs.beta'
end

Flux.@functor BetaScaling

# 创建全连接网络（带残差连接和Beta缩放）
function create_fc_network_with_residual(enhanced_features_y)
    # 创建残差块
    res_block1 = ResidualBlock(enhanced_features_y, 64)
    res_block2 = ResidualBlock(64, 32)
    
    # 最后一个全连接层
    fc3 = FCN(32, 2, σ=identity)  # 输出层通常不使用激活函数
    
    # 用于残差连接的投影（从输入直接到输出）
    projection = FCN(enhanced_features_y, 2, σ=identity)
    
    # Beta缩放
    beta_scaling = BetaScaling(2)
    
    return res_block1, res_block2, fc3, projection, beta_scaling
end


# 构建损失函数
function calculate_loss(ΔP, ΔQ, pv_idx, ref_idx)
    # 初始化
    ω_p = ones(size(ΔP,1))
    ω_q = ones(size(ΔQ,1))

    ω_p[ref_idx] .= 0
    ω_q[ref_idx] .= 0
    ω_q[pv_idx] .= 0

    n = length(ΔP) + length(ΔQ)

    L = (sum(ω_p.*ΔP.^2) + sum(ω_q.*ΔQ.^2))/n

    return L, ω_p, ω_q
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

function calculate_chain_deviation(J, pv_idx, pq_idx, ref_idx, ΔP, ΔQ, ω_p, ω_q)
    n = length(pq_idx) + length(pv_idx) + length(ref_idx)
    P_vector = (ω_p./n).* ΔP
    Q_vector = (ω_q./n).* ΔQ
    mismatch = vcat(P_vector, Q_vector)
    deviation = mismatch' * J

    return deviation
end