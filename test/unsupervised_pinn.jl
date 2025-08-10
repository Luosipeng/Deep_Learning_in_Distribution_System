using Flux
using Random
# 采样潮流数据和结果
# 使用matpower中的算例
using Base.Threads
using TimeSeriesPowerFlow
using Distributions
using Plots
using LinearAlgebra
using SparseArrays
using Statistics
using Zygote


include("../src/generate_random_sample_for_upinn.jl")
include("../src/adjacent_matrix_calculation.jl")
include("../solvers/train_pinn_for_pf.jl")


case_file ="C:/Users/13733/Desktop/matpower-8.0/data/case57.m"
X, Ybus, pq_idx, pv_idx, ref_idx = generate_random_sample_for_upinn(case_file)

A = admittance_to_adjacency(Ybus)

# 创建GCN网络
input_dim = 2
hidden_dims = [128, 96, 64]
output_dim =  size(X, 1)-1  # 输入特征维度
gcn_network = create_gcn_network(input_dim, hidden_dims, output_dim)

x = X[:, :, 1]  # 取第一列作为输入特征
output = gcn_network(x, A)
output = Matrix(output)

function physical_loss(output,pq_idx, pv_idx, ref_idx, Ybus, x)
    # 提取输出的功率注入
    V_pq = output[pq_idx, 1]
    θ_pq = output[pq_idx, 2]
    θ_pv = output[pv_idx, 1]
    Q_pv = output[pv_idx, 2]
    V_ref = x[ref_idx, 1]
    θ_ref = x[ref_idx, 2]

    # 计算物理损失
    loss = 0.0
    for i in pq_idx
        loss += (P_pq[i] - Ybus[i, i])^2 + (Q_pq[i] - Ybus[i, i])^2
    end

    for i in pv_idx
        loss += (P_pv[i] - Ybus[i, i])^2 + (V_pv[i] - Ybus[i, i])^2
    end

    for i in ref_idx
        loss += (V_ref[i] - Ybus[i, i])^2 + (θ_ref[i] - Ybus[i, i])^2
    end

    return loss

end