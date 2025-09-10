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
using TimeSeriesPowerFlow.PowerFlow:dSbus_dV, makeSbus

include("../src/generate_random_sample_for_upinn.jl")
include("../src/adjacent_matrix_calculation.jl")
include("../solvers/train_pinn_for_pf.jl")

# Input Parameters: the active and reactive power of PQ buses, the active power and voltage magnitudes of PV buses, and the voltage magnitude and phase angle of the balance bus
# Output Parameters: reactive power values of the PV buses, the voltage magnitudes of the PQ buses, and the phase angles of all buses except the balance bus

case_file ="D:/luosipeng/matpower8.1/data//case118.m"
X, Ybus, pq_idx, pv_idx, ref_idx, baseMVA = generate_random_sample_for_upinn(case_file)


A = admittance_to_adjacency(Ybus)

x = X[:,:,1]

# 创建GCN网络
input_dim = size(X,1)

gcn_network_1 = create_gcn_network()
gcn_network_2 = create_gcn_network()

output_1 = gcn_network_1(x, A)
output_2 = gcn_network_2(output_1, A)
pooled_output = mean(output_2, dims=1)
num_nodes = size(output_2, 1)
repeated_global_features = repeat(pooled_output, num_nodes, 1) 
enhanced_features = hcat(
    output_2,                 # GCN输出
    repeated_global_features, # 全局池化特征
    x
)

# 全连接层
enhanced_features_y = size(enhanced_features, 2)
res_block1, res_block2, fc3, projection, beta_scaling = create_fc_network_with_residual(enhanced_features_y)

# 前向传播（带残差连接）
result_1 = res_block1(enhanced_features)
result_2 = res_block2(result_1)
result_3 = fc3(result_2)

# 应用残差连接（从输入到最后一层）
residual = projection(enhanced_features)
result_with_residual = result_3 + residual

# 应用Beta缩放
final_output = beta_scaling(result_with_residual)

P = zeros(size(final_output,1))
Q = zeros(size(final_output,1))
Vm = zeros(size(final_output,1))
θ = zeros(size(final_output,1))

P[pq_idx] = x[pq_idx, 1]
P[pv_idx] = x[pv_idx, 1]
P[ref_idx] = final_output[ref_idx, 1]

Q[pq_idx] = x[pq_idx, 2]
Q[pv_idx] = final_output[pv_idx, 2]
Q[ref_idx] = final_output[ref_idx, 2]

Vm[pq_idx] = final_output[pq_idx, 2]
Vm[pv_idx] = x[pv_idx, 2]
Vm[ref_idx] = x[ref_idx, 1]

θ[pq_idx] = final_output[pq_idx, 1]
θ[pv_idx] = final_output[pv_idx, 1]
θ[ref_idx] = x[ref_idx, 2]

V = Vm.*(cos.(θ*pi/180)+im*sin.(θ*pi/180))

ΔS = V .* conj.(Ybus * V)- (P + im.*Q)
ΔP = real(ΔS[vcat(pq_idx, pv_idx, ref_idx)])
ΔQ = imag(ΔS[vcat(pq_idx, pv_idx, ref_idx)])

loss, ω_p, ω_q = calculate_loss(ΔP, ΔQ, pv_idx, ref_idx)

J = calculate_the_deviation(pq_idx, pv_idx, ref_idx, Ybus, V)

deviation =  calculate_chain_deviation(J, pv_idx, pq_idx, ref_idx, ΔP, ΔQ, ω_p, ω_q)