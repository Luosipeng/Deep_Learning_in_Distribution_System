using SparseArrays, LinearAlgebra

"""
    admittance_to_adjacency(Y::AbstractMatrix{<:Number}; threshold::Real=1e-6)

从导纳矩阵计算邻接矩阵。支持稠密矩阵和稀疏矩阵(SparseMatrixCSC)。

参数:
- `Y`: 导纳矩阵，通常是复数矩阵，可以是稠密矩阵或SparseMatrixCSC
- `threshold`: 判断两节点是否相连的阈值，默认为1e-6

返回:
- 邻接矩阵(稀疏二值矩阵)
"""
function admittance_to_adjacency(Y::AbstractMatrix{<:Number}; threshold::Real=1e-6)
    # 检查Y是否为方阵
    n = size(Y, 1)
    if size(Y, 2) != n
        error("导纳矩阵必须是方阵")
    end
    
    # 对于稀疏矩阵，直接使用其非零元素结构
    if Y isa SparseMatrixCSC
        # 初始化行、列和值数组
        rows = Int[]
        cols = Int[]
        
        # 遍历非零元素
        for j in 1:n
            for i in nzrange(Y, j)
                row = Y.rowval[i]
                if row != j && abs(Y.nzval[i]) > threshold
                    push!(rows, row)
                    push!(cols, j)
                end
            end
        end
        
        # 创建稀疏邻接矩阵
        return sparse(rows, cols, ones(Int, length(rows)), n, n)
    else
        # 对于稠密矩阵，使用原来的方法
        rows = Int[]
        cols = Int[]
        
        for i in 1:n
            for j in 1:n
                if i != j && abs(Y[i, j]) > threshold
                    push!(rows, i)
                    push!(cols, j)
                end
            end
        end
        
        return sparse(rows, cols, ones(Int, length(rows)), n, n)
    end
end

"""
    admittance_to_weighted_adjacency(Y::AbstractMatrix{<:Number}; threshold::Real=1e-6)

从导纳矩阵计算加权邻接矩阵。支持稠密矩阵和稀疏矩阵(SparseMatrixCSC)。

参数:
- `Y`: 导纳矩阵，通常是复数矩阵，可以是稠密矩阵或SparseMatrixCSC
- `threshold`: 判断两节点是否相连的阈值，默认为1e-6

返回:
- 加权邻接矩阵(稀疏矩阵)，权重为导纳值的绝对值
"""
function admittance_to_weighted_adjacency(Y::AbstractMatrix{<:Number}; threshold::Real=1e-6)
    # 检查Y是否为方阵
    n = size(Y, 1)
    if size(Y, 2) != n
        error("导纳矩阵必须是方阵")
    end
    
    # 对于稀疏矩阵，直接使用其非零元素结构
    if Y isa SparseMatrixCSC
        # 初始化行、列和值数组
        rows = Int[]
        cols = Int[]
        vals = Float64[]
        
        # 遍历非零元素
        for j in 1:n
            for i in nzrange(Y, j)
                row = Y.rowval[i]
                val = abs(Y.nzval[i])
                if row != j && val > threshold
                    push!(rows, row)
                    push!(cols, j)
                    push!(vals, val)
                end
            end
        end
        
        # 创建稀疏加权邻接矩阵
        return sparse(rows, cols, vals, n, n)
    else
        # 对于稠密矩阵，使用原来的方法
        rows = Int[]
        cols = Int[]
        vals = Float64[]
        
        for i in 1:n
            for j in 1:n
                if i != j && abs(Y[i, j]) > threshold
                    push!(rows, i)
                    push!(cols, j)
                    push!(vals, abs(Y[i, j]))
                end
            end
        end
        
        return sparse(rows, cols, vals, n, n)
    end
end

"""
    normalize_adjacency(A::AbstractMatrix{<:Real})

归一化邻接矩阵，用于图卷积网络。支持稠密矩阵和稀疏矩阵(SparseMatrixCSC)。
计算 Â = D^(-1/2) * (A + I) * D^(-1/2)，其中D是度矩阵，I是单位矩阵。

参数:
- `A`: 邻接矩阵，可以是稠密矩阵或SparseMatrixCSC

返回:
- 归一化后的邻接矩阵(稀疏矩阵)
"""
function normalize_adjacency(A::AbstractMatrix{<:Real})
    n = size(A, 1)
    
    # 添加自环 (A + I)
    A_tilde = A + sparse(I, n, n)
    
    # 计算度矩阵
    D_tilde = vec(sum(A_tilde, dims=2))
    
    # 计算D^(-1/2)
    D_tilde_inv_sqrt = 1.0 ./ sqrt.(D_tilde)
    
    # 创建对角矩阵D^(-1/2)
    D_inv_sqrt = spdiagm(0 => D_tilde_inv_sqrt)
    
    # 计算归一化邻接矩阵 D^(-1/2) * A_tilde * D^(-1/2)
    A_hat = D_inv_sqrt * A_tilde * D_inv_sqrt
    
    return A_hat
end
