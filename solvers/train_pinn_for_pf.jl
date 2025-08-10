# 1. 图卷积层实现 - 保持不变
struct GraphConv
    weight::Matrix{Float32}
    bias::Vector{Float32}
    σ::Function
end

function GraphConv(in_channels::Int, out_channels::Int, σ=relu)
    weight = Flux.glorot_uniform(out_channels, in_channels)
    bias = zeros(Float32, out_channels)
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
    return gc.σ.(X_conv * gc.weight' .+ gc.bias')
end

Flux.@functor GraphConv

# 定义SiLU/Swish激活函数
swish(x) = x * sigmoid(x)

# 2. 设计新的网络架构: 2GCN + 池化 + 2FC
struct GCN2FC2
    # 两层图卷积
    gcn1::GraphConv
    gcn2::GraphConv
    
    # 两层全连接
    fc1::Dense
    fc2::Dense
    
    # Dropout层
    dropout::Dropout
end

# 创建2GCN+池化+2FC网络
function create_gcn2fc2(input_dim::Int, gcn1_dim::Int, gcn2_dim::Int, 
                       fc1_dim::Int, output_dim::Int, dropout_rate::Float32=0.2f0)
    # 创建两个图卷积层
    gcn1 = GraphConv(input_dim, gcn1_dim)
    gcn2 = GraphConv(gcn1_dim, gcn2_dim)
    
    # 创建两个全连接层，使用SiLU激活函数
    fc1 = Dense(gcn2_dim, fc1_dim, swish)  # 修改为swish
    fc2 = Dense(fc1_dim, output_dim, swish)  # 输出层也使用swish
    
    # 创建Dropout层
    dropout = Dropout(dropout_rate)
    
    return GCN2FC2(gcn1, gcn2, fc1, fc2, dropout)
end

# 网络的前向传播
function (model::GCN2FC2)(X, A)
    # 第一个图卷积层
    h = model.gcn1(X, A)
    h = model.dropout(h)
    
    # 第二个图卷积层
    h = model.gcn2(h, A)
    
    # 池化层 - 使用平均池化将节点特征聚合为图特征
    h_pooled = mean(h, dims=1)  # 结果维度: [1, gcn2_dim]
    h_pooled = vec(h_pooled)    # 转换为向量 [gcn2_dim]
    
    # 第一个全连接层
    h = model.fc1(h_pooled)
    h = model.dropout(h)
    
    # 第二个全连接层 (输出层)
    output_flat = model.fc2(h)
    
    # 重塑输出为矩阵形式 (size(X, 1)-1, 2)
    n_nodes = Int(length(output_flat) / 2)
    output_matrix = reshape(output_flat, 2, n_nodes)'
    
    return output_matrix
end


# 使Flux能够识别并训练GCN2FC2的参数
Flux.@functor GCN2FC2

# 3. 提供一个便捷的创建函数，支持hidden_dims数组
function create_gcn_network(input_dim::Int, hidden_dims::Vector{Int}, output_dim::Int, dropout_rate::Float32=0.2f0)
    # 确保hidden_dims至少有3个元素 (2个GCN层 + 1个FC层)
    if length(hidden_dims) < 3
        error("hidden_dims应至少包含3个元素: [gcn1_dim, gcn2_dim, fc1_dim]")
    end
    
    # 提取维度
    gcn1_dim = hidden_dims[1]
    gcn2_dim = hidden_dims[2]
    fc1_dim = hidden_dims[3]
    
    # 创建网络
    return create_gcn2fc2(input_dim, gcn1_dim, gcn2_dim, fc1_dim, output_dim, dropout_rate)
end
