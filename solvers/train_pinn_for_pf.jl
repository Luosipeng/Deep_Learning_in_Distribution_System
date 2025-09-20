# ############################################################
# # UPINN 极简复现脚本 修正版7
# # 修复：LBFGS 行搜索缺少 LineSearches 导入的问题
# # 增加：自动回退机制，如果未安装 LineSearches 则使用默认 LBFGS()
# ############################################################
# using Random, LinearAlgebra, SparseArrays, Statistics
# using Flux
# import Flux: sigmoid
# using Optim
# Random.seed!(42)
include("../ios/data_generation_upinn.jl")
# #---------------- 行搜索可选加载 ----------------#
const USE_BACKTRACKING = true  # 想禁用直接设为 false
const HAS_LINESEARCHES = let flag=false
    if USE_BACKTRACKING
        try
            @eval using LineSearches
            flag = true
        catch e
            @warn "未找到 LineSearches，将使用默认 LBFGS() 行搜索 (无 BackTracking)。安装：] add LineSearches 以启用。"
            flag = false
        end
    end
    flag
end

#---------------- 超参数 ----------------#
const USE_FLOAT64   = true
const T = USE_FLOAT64 ? Float64 : Float32

const EPOCHS_ADAM   = 300
const EPOCHS_LBFGS  = 200
const BATCH_SIZE    = 256
const LR_ADAM       = 1e-3
const WP            = 1.0
const WQ            = 1.0
const ANGLE_TANH    = true
const VMIN          = 0.90
const VMAX          = 1.10
const PRINT_EVERY   = 20
const CLIP_NORM     = 5.0
const L2_REG        = 0.0        # 可设 1e-8 微调
const G_HIDDEN      = 64
const MLP_HIDDEN    = 128
const USE_LBFGS_DEF = true
const ANGLE_SCALE   = π


#---------------- 基础函数 ----------------#
leaky_relu(x,α=0.01)= x>0 ? x : α*x
silu(x)= x*sigmoid(x)

#---------------- 图卷积 ----------------#
struct GraphConv
    W::Matrix{T}
    b::Vector{T}
    act::Function
end
Flux.@functor GraphConv
function GraphConv(in_dim::Int,out_dim::Int; act=leaky_relu)
    W=randn(T,in_dim,out_dim) * sqrt(T(2)/in_dim)
    b=zeros(T,out_dim)
    GraphConv(W,b,act)
end
function (g::GraphConv)(X::AbstractMatrix{<:Real}, A::SparseMatrixCSC{T,Int})
    Xc = T.(X)
    d  = vec(sum(A,dims=2))
    D_inv = Diagonal( 1.0 ./ sqrt.(d .+ 1e-12) )
    A_norm = D_inv * A * D_inv
    Z = A_norm * Xc
    g.act.(Z*g.W .+ g.b')
end

#---------------- 全连接 ----------------#
struct FCN
    W::Matrix{T}
    b::Vector{T}
    σ::Function
end
Flux.@functor FCN
function FCN(in_dim::Int,out_dim::Int; σ=silu)
    W=randn(T,in_dim,out_dim) * sqrt(T(2)/in_dim)
    b=zeros(T,out_dim)
    FCN(W,b,σ)
end
(f::FCN)(X)= f.σ.(T.(X)*f.W .+ f.b')

#---------------- 模型 ----------------#
mutable struct UPINNModel
    g1::GraphConv
    g2::GraphConv
    fc1::FCN
    fc2::FCN
    head::FCN
end
Flux.@functor UPINNModel
function UPINNModel(in_dim; g_hidden=G_HIDDEN, mlp_hidden=MLP_HIDDEN)
    g1=GraphConv(in_dim,g_hidden)
    g2=GraphConv(g_hidden,g_hidden)
    fc1=FCN(g_hidden+in_dim, mlp_hidden)
    fc2=FCN(mlp_hidden, mlp_hidden)
    head=FCN(mlp_hidden,2,σ=identity)
    UPINNModel(g1,g2,fc1,fc2,head)
end

function (m::UPINNModel)(X,A; training=true)
    H1 = m.g1(X,A)
    H2 = m.g2(H1,A)
    enh = hcat(H2, X)
    Z = m.fc1(enh)
    Z = m.fc2(Z)
    m.head(Z)
end

#---------------- 输出映射 ----------------#
function map_outputs(out, feat, idx::BusIndex, masks)
    pq_mask    = masks[:pq_mask]
    pv_mask    = masks[:pv_mask]
    slack_mask = masks[:slack_mask]

    a_code = out[:,1]
    v_code = out[:,2]

    pred_delta = ANGLE_TANH ? T.(ANGLE_SCALE .* tanh.(a_code)) : T.(a_code)
    δ = pred_delta .* (1 .- slack_mask)

    v_col6 = T.(feat[:,6])
    v_col5 = T.(feat[:,5])

    pred_v_pq = 0.5 .* (tanh.(v_code) .+ 1) .* (VMAX-VMIN) .+ VMIN
    v_pv_given   = pv_mask    .* ( (v_col6 .> 0) .* v_col6 .+ (v_col6 .<= 0) .* v_col5 )
    v_slack_given= slack_mask .* v_col5
    Vmag = pred_v_pq .* pq_mask .+ v_pv_given .+ v_slack_given

    V = Vmag .* cis.(δ)
    return V, δ, Vmag
end

#---------------- 物理损失 ----------------#
function physics_loss(s::DatasetSample, model::UPINNModel,
                      A, Ybus, idx::BusIndex,
                      masks, non_ref_idx, pq_idx; training=true)
    out = model(s.feat, A; training=training)
    V, δ, Vmag = map_outputs(out, s.feat, idx, masks)

    S = V .* conj.(Ybus * V)
    P_calc = real.(S)
    Q_calc = imag.(S)

    pv_mask    = masks[:pv_mask]
    slack_mask = masks[:slack_mask]

    Qpv_calc_vec = Q_calc .* pv_mask
    Pslack_calc  = sum(P_calc .* slack_mask)

    ΔP = s.P_spec[non_ref_idx] .- P_calc[non_ref_idx]
    ΔQ = s.Q_spec[pq_idx]      .- Q_calc[pq_idx]

    nP = length(ΔP); nQ = length(ΔQ)
    lossP = WP * sum(abs2,ΔP)/max(1,nP)
    lossQ = WQ * sum(abs2,ΔQ)/max(1,nQ)
    reg   = L2_REG > 0 ? L2_REG * sum(abs2,Flux.params(model)) : 0
    loss  = 0.5*(lossP + lossQ) + reg

    return loss, ΔP, ΔQ, Dict(
        :Vmag=>Vmag,
        :δ=>δ,
        :Qpv_calc=>Qpv_calc_vec,
        :Pslack_calc=>Pslack_calc
    )
end
function physics_loss(s::DatasetSample, model::UPINNModel,
                      A, idx::BusIndex,
                      masks, non_ref_idx, pq_idx; training=true)
    error("physics_loss 缺少 Ybus 参数，请使用 physics_loss(s, model, A, Ybus, idx, masks, non_ref_idx, pq_idx)")
end

#---------------- 批采样 ----------------#
sample_batch(samples, b) = samples[rand(1:length(samples), min(b,length(samples)))]

#---------------- 简易 Adam ----------------#
mutable struct SimpleAdam
    η::T; β1::T; β2::T; eps::T
    m::IdDict{Any,Any}; v::IdDict{Any,Any}; t::Int
end
function SimpleAdam(η=LR_ADAM, β1=0.9, β2=0.999, eps=1e-8)
    SimpleAdam(T(η),T(β1),T(β2),T(eps),IdDict{Any,Any}(),IdDict{Any,Any}(),0)
end
function adam_update!(opt::SimpleAdam, ps, gs)
    opt.t += 1
    for p in ps
        g = gs[p]; g === nothing && continue
        m = get!(opt.m,p, zeros(T,size(p)))
        v = get!(opt.v,p, zeros(T,size(p)))
        @. m = opt.β1*m + (1 - opt.β1)*g
        @. v = opt.β2*v + (1 - opt.β2)*g^2
        mhat = m / (1 - opt.β1^opt.t)
        vhat = v / (1 - opt.β2^opt.t)
        p .-= opt.η * mhat ./ (sqrt.(vhat) .+ opt.eps)
    end
end

#---------------- Adam 阶段 ----------------#
function train_adam!(model, samples, Ybus, A, idx;
                     epochs=EPOCHS_ADAM, batch=BATCH_SIZE)
    n_bus = size(samples[1].feat,1)
    ref_set = Set(idx.ref)
    non_ref_idx = [i for i in 1:n_bus if !(i in ref_set)]
    pq_idx = idx.pq

    pq_mask    = zeros(T,n_bus); pq_mask[idx.pq]    .= one(T)
    pv_mask    = zeros(T,n_bus); pv_mask[idx.pv]    .= one(T)
    slack_mask = zeros(T,n_bus); slack_mask[idx.ref].= one(T)
    masks = Dict(:pq_mask=>pq_mask, :pv_mask=>pv_mask, :slack_mask=>slack_mask)

    ps = Flux.params(model)
    opt = SimpleAdam()

    for ep in 1:epochs
        bt = sample_batch(samples, batch)
        gs = gradient(ps) do
            total = zero(T)
            for s in bt
                L,_,_,_ = physics_loss(s, model, A, Ybus, idx, masks, non_ref_idx, pq_idx; training=true)
                total += L
            end
            total / length(bt)
        end
        for p in ps
            g = gs[p]; g===nothing && continue
            gn = norm(g)
            if gn > CLIP_NORM
                g .= (CLIP_NORM/gn) .* g
            end
        end
        adam_update!(opt, ps, gs)

        if ep % PRINT_EVERY == 0 || ep == 1
            accL=0.0; mP=0.0; mQ=0.0
            for s in bt
                L,ΔP,ΔQ,_ = physics_loss(s, model, A, Ybus, idx, masks, non_ref_idx, pq_idx; training=false)
                accL += L
                !isempty(ΔP) && (mP=max(mP, maximum(abs.(ΔP))))
                !isempty(ΔQ) && (mQ=max(mQ, maximum(abs.(ΔQ))))
            end
            println("Epoch $(lpad(ep,4)) | Loss=$(round(accL/length(bt),digits=8))  ΔP_max=$(round(mP,digits=8))  ΔQ_max=$(round(mQ,digits=8))")
        end
    end
    return (non_ref_idx=non_ref_idx, pq_idx=pq_idx, masks=masks)
end

#---------------- LBFGS 精调 ----------------#
function lbfgs_refine!(model, samples, Ybus, A, idx, aux; max_iter=EPOCHS_LBFGS)
    println("[LBFGS] 开始精调 ...")
    ps = Flux.params(model)
    arrays = [p for p in ps]
    sizes  = map(size, arrays)
    lengths= map(length, arrays)
    offsets= cumsum(vcat(0, lengths[1:end-1]))
    total_len = sum(lengths)

    function pack()
        v = zeros(T,total_len)
        for (i,arr) in enumerate(arrays)
            rng = offsets[i]+1 : offsets[i]+lengths[i]
            v[rng] .= vec(arr)
        end
        v
    end
    function unpack!(v)
        for (i,arr) in enumerate(arrays)
            rng = offsets[i]+1 : offsets[i]+lengths[i]
            arr .= reshape(view(v,rng), sizes[i])
        end
    end

    non_ref_idx = aux.non_ref_idx
    pq_idx      = aux.pq_idx
    masks       = aux.masks

    function total_loss_and_grad(v)
        unpack!(v)
        gs = gradient(ps) do
            acc = zero(T)
            for s in samples
                l,_,_,_ = physics_loss(s, model, A, Ybus, idx, masks, non_ref_idx, pq_idx; training=true)
                acc += l
            end
            acc / length(samples)
        end
        L = zero(T)
        for s in samples
            l,_,_,_ = physics_loss(s, model, A, Ybus, idx, masks, non_ref_idx, pq_idx; training=false)
            L += l
        end
        L /= length(samples)
        gvec = zeros(T,total_len)
        for (i,arr) in enumerate(arrays)
            g = gs[arr]; g === nothing && continue
            rng = offsets[i]+1 : offsets[i]+lengths[i]
            gvec[rng] .= vec(g)
        end
        return L, gvec
    end

    iter_cb = 0
    function fg!(F,G,v)
        f, g = total_loss_and_grad(v)
        F[] = f
        G[:] = g
        iter_cb += 1
        if iter_cb % 10 == 0
            mP=0.0; mQ=0.0
            for s in samples
                _,ΔP,ΔQ,_ = physics_loss(s, model, A, Ybus, idx, masks, non_ref_idx, pq_idx; training=false)
                !isempty(ΔP) && (mP=max(mP, maximum(abs.(ΔP))))
                !isempty(ΔQ) && (mQ=max(mQ, maximum(abs.(ΔQ))))
            end
            println("[LBFGS iter $(iter_cb)] loss=$(round(f,digits=10))  ΔP_max=$(round(mP,digits=8))  ΔQ_max=$(round(mQ,digits=8))")
            if mP < 1e-3 && mQ < 1e-3
                println("[LBFGS] 提前停止：达到阈值")
                return true
            end
        end
        return false
    end

    v0 = pack()
    obj = OnceDifferentiable(v -> begin F=Ref(zero(T)); G=zeros(T,length(v)); fg!(F,G,v); F[]
             end,
             (G,v)->begin F=Ref(zero(T)); fg!(F,G,v); end,
             v0)
    options = Optim.Options(iterations = max_iter, show_trace=false)
    solver = HAS_LINESEARCHES ? LBFGS(linesearch=LineSearches.BackTracking()) : LBFGS()
    res = optimize(obj, v0, solver, options)
    unpack!(Optim.minimizer(res))
    println("[LBFGS] 完成：final loss = $(Optim.minimum(res))")
end

#---------------- 评估 ----------------#
function evaluate(model, samples, Ybus, A, idx, aux)
    non_ref_idx = aux.non_ref_idx
    pq_idx      = aux.pq_idx
    masks       = aux.masks
    ΔP_all=T[]; ΔQ_all=T[]
    v_err=T[]; qpv_err=T[]; ang_err=T[]; pslack_err=T[]
    for s in samples
        _,ΔP,ΔQ,extra = physics_loss(s, model, A, Ybus, idx, masks, non_ref_idx, pq_idx; training=false)
        append!(ΔP_all, T.(ΔP))
        append!(ΔQ_all, T.(ΔQ))
        for b in idx.pq
            push!(v_err, extra[:Vmag][b] - s.V_spec[b])
        end
        for b in idx.pv
            push!(qpv_err, extra[:Qpv_calc][b] - s.Q_spec[b])
        end
        for b in union(idx.pq, idx.pv)
            push!(ang_err, extra[:δ][b] - s.δ_true[b])
        end
        push!(pslack_err, extra[:Pslack_calc] - s.P_slack_true)
    end
    rmse(x)=sqrt(mean(abs2,x))
    Dict(
        :rmse_ΔP => rmse(ΔP_all),
        :rmse_ΔQ => rmse(ΔQ_all),
        :max_ΔP  => isempty(ΔP_all) ? 0 : maximum(abs.(ΔP_all)),
        :max_ΔQ  => isempty(ΔQ_all) ? 0 : maximum(abs.(ΔQ_all)),
        :rmse_Vpq => rmse(v_err),
        :rmse_Qpv => isempty(qpv_err) ? 0 : rmse(qpv_err),
        :rmse_angle => rmse(ang_err),
        :rmse_Pslack => rmse(pslack_err)
    )
end

#---------------- 主流程 ----------------#
function run_training(; case_file::String,
                        total_samples::Int=600,
                        load_scale_range=(0.9,1.1),
                        voltage_filter=(0.85,1.2),
                        seed=42,
                        use_lbfgs::Bool=USE_LBFGS_DEF)
    @info "生成样本..."
    # raw_samples, Ybus, A_f32, idx, baseMVA, _ =
        # generate_upinn_samples_tspf_for_training(case_file;
        #     total_samples=total_samples,
        #     load_scale_range=load_scale_range,
        #     voltage_filter=voltage_filter,
        #     seed=seed,
        #     keep_legacy_tensor=false)
    raw_samples, Ybus, A_f32, idx, baseMVA, _ =
        generate_upinn_samples_presolve(case_file;
            total_samples=total_samples,
            load_scale_range=load_scale_range,
            seed=seed,
            keep_legacy_tensor=false,
            perturb_genP=false   # 如需随机发电设定改 true
        )
    samples = DatasetSample[]
    for s in raw_samples
        push!(samples, DatasetSample(
            T.(s.feat),
            T.(s.P_spec),
            T.(s.Q_spec),
            T.(s.V_spec),
            T.(s.δ_true),
            T.(s.Q_PV_true),
            T(s.P_slack_true),
            s.bus_types,
            s.V_true
        ))
    end
    A = SparseMatrixCSC{T,Int}(A_f32)
    println("样本数=$(length(samples))  PQ=$(length(idx.pq)) PV=$(length(idx.pv)) Slack=$(length(idx.ref)) baseMVA=$baseMVA")

    model = UPINNModel(size(samples[1].feat,2))

    @info "阶段1 Adam 训练..."
    aux = train_adam!(model, samples, Ybus, A, idx)

    if use_lbfgs
        @info "阶段2 LBFGS 精调..."
        lbfgs_refine!(model, samples, Ybus, A, idx, aux)
    else
        println("跳过 LBFGS 精调。")
    end

    @info "评估..."
    metrics = evaluate(model, samples, Ybus, A, idx, aux)
    println("==== 评估指标 ====")
    for (k,v) in metrics
        println(rpad(String(k),14), " : ", v)
    end
    return model, metrics, aux
end


    
