using Random, LinearAlgebra, SparseArrays, Statistics
using Flux
import Flux: sigmoid
using Optim
Random.seed!(42)
include("../solvers/train_pinn_for_pf.jl")
include("../ios/data_generation_upinn.jl")
case_file = "D:/luosipeng/matpower8.1/data/case30.m"
model, metrics, aux = run_training(case_file=case_file, total_samples=800, use_lbfgs=true)
println("完成。")

# 测试model有效性
case_file1 = "D:/luosipeng/matpower8.1/data/case30.m"
samples, Ybus, A_f32, idx, baseMVA = generator_matpower_case(case_file1)
s = samples[1]

# 将特征矩阵转成 Float64（与你训练脚本里 const T 一致）
feat = Float64.(s.feat)
# A_f32 -> Float64
A = SparseMatrixCSC{Float64,Int}(A_f32)

# 前向
out = model(feat, A; training=false)

# 构建 mask
n_bus = size(feat,1)
pq_mask    = zeros(Float64,n_bus); pq_mask[idx.pq]    .= 1
pv_mask    = zeros(Float64,n_bus); pv_mask[idx.pv]    .= 1
slack_mask = zeros(Float64,n_bus); slack_mask[idx.ref].=1
masks = Dict(:pq_mask=>pq_mask, :pv_mask=>pv_mask, :slack_mask=>slack_mask)

# 利用你已有的 map_outputs
V0, delta, Vmag = map_outputs(out, feat, idx, masks)

