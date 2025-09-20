############################################################
# 基于 TimeSeriesPowerFlow 的 UPINN 数据集生成（与训练脚本兼容）
# - 不再混叠通道，直接输出 7 维特征 + 真实注入
# - 可选择保留原 2 通道张量 (legacy_X) 以兼容旧代码
############################################################
using TimeSeriesPowerFlow
using Random, SparseArrays, LinearAlgebra
include("../ios/matlab2julia.jl")
# ============ MATPOWER 常量（内部编号格式） ============
const BUS_TYPE = 2
const PD = 3
const QD = 4
const VM = 8
const VA = 9

const GEN_BUS = 1
const PG = 2
const QG = 3

# ============ 数据结构（与 UPINN 脚本一致） ============
struct DatasetSample
    feat::Matrix{Float32}          # (nbus, 7)
    P_spec::Vector{Float32}        # 注入有功 (含 pv / pq / slack)
    Q_spec::Vector{Float32}        # 注入无功
    V_spec::Vector{Float32}        # 电压幅值
    δ_true::Vector{Float32}        # 角度 (rad)
    Q_PV_true::Vector{Float32}     # PV 母线的 Q 注入真值
    P_slack_true::Float32          # Slack 母线 P 注入
    bus_types::Vector{Int}         # 1/2/3
    V_true::Vector{ComplexF64}     # 复电压
end
struct BusIndex
    pq::Vector{Int}
    pv::Vector{Int}
    ref::Vector{Int}
end

# ============ 特征构造（7 维） ============
function build_input_features(P_spec,Q_spec,Vmag,δ,idx::BusIndex)
    n=length(P_spec)
    feat=zeros(Float32,n,7)
    for i in 1:n
        is_pq = in(i,idx.pq)
        is_pv = in(i,idx.pv)
        is_rf = in(i,idx.ref)
        feat[i,1] = is_pq ? 1f0 : 0f0
        feat[i,2] = is_pv ? 1f0 : 0f0
        feat[i,3] = is_rf ? 1f0 : 0f0
        feat[i,4] = is_rf ? 0f0 : Float32(P_spec[i])               # Slack 有功由模型预测
        feat[i,5] = is_pq ? Float32(Q_spec[i]) : (is_pv||is_rf ? Float32(Vmag[i]) : 0f0)
        feat[i,6] = (is_pv||is_rf) ? Float32(Vmag[i]) : 0f0
        feat[i,7] = 1f0
    end
    feat
end

# ============ 邻接矩阵 ============
function build_adjacency_from_Y(Ybus::SparseMatrixCSC{ComplexF64,Int})
    I,J,V=findnz(Ybus); n=size(Ybus,1)
    rows=Int[]; cols=Int[]
    for (i,j,v) in zip(I,J,V)
        if i!=j && abs(v)>1e-9
            push!(rows,i); push!(cols,j)
            push!(rows,j); push!(cols,i)
        end
    end
    append!(rows,1:n); append!(cols,1:n)
    sparse(rows,cols,ones(Float32,length(rows)),n,n)
end

# ============ 主函数 ============
"""
  generate_upinn_samples_tspf_for_training(case_file;
      total_samples=500,
      load_scale_range=(0.9,1.1),
      max_attempt_factor=30,
      seed=42,
      keep_legacy_tensor=true,
      voltage_filter=(0.85,1.2)
  )

  返回:
    samples::Vector{DatasetSample},
    Ybus::SparseMatrixCSC,
    A::SparseMatrixCSC,
    idx::BusIndex,
    baseMVA::Float64,
    legacy_X::Union{Array{Float32,3},Nothing}  # (nbus,2,N) 若 keep_legacy_tensor=true
"""
function generate_upinn_samples_tspf_for_training(case_file;
      total_samples::Int=500,
      load_scale_range::Tuple{Float64,Float64}=(0.9,1.1),
      max_attempt_factor::Int=30,
      seed::Int=42,
      keep_legacy_tensor::Bool=true,
      voltage_filter::Tuple{Float64,Float64}=(0.85,1.2)
    )

    Random.seed!(seed)
    jpc = convert_matpower_case_dp(case_file, "tmp_case_conv_upinn.jl")
    busAC = jpc.busAC; genAC = jpc.genAC; branchAC = jpc.branchAC; loadAC = jpc.loadAC; pvarray = jpc.pv
    (busAC,genAC,branchAC,loadAC,pvarray,_) =
        TimeSeriesPowerFlow.PowerFlow.ext2int(busAC,genAC,branchAC,loadAC,pvarray)
    jpc.busAC=busAC; jpc.genAC=genAC; jpc.branchAC=branchAC; jpc.loadAC=loadAC; jpc.pv=pvarray
    baseMVA = jpc.baseMVA

    pq_idx = findall(busAC[:,BUS_TYPE].==1)
    pv_idx = findall(busAC[:,BUS_TYPE].==2)
    ref_idx= findall(busAC[:,BUS_TYPE].==3)
    isempty(ref_idx) && error("无 Slack (BUS_TYPE=3)")
    idx = BusIndex(pq_idx,pv_idx,ref_idx)

    Ybus,Yf,Yt = TimeSeriesPowerFlow.PowerFlow.makeYbus(baseMVA, busAC, branchAC)
    A = build_adjacency_from_Y(Ybus)
    nbus = size(busAC,1)

    samples = DatasetSample[]
    legacy_X = keep_legacy_tensor ? zeros(Float32, nbus, 2, total_samples) : nothing
    collected = 0
    attempts = 0
    max_attempts = total_samples * max_attempt_factor

    println("TSPF 数据生成: 目标=$total_samples  最大尝试=$max_attempts  缩放范围=$(load_scale_range)")

    while collected < total_samples && attempts < max_attempts
        attempts += 1
        case = deepcopy(jpc)
        scale = rand()*(load_scale_range[2]-load_scale_range[1]) + load_scale_range[1]
        # 缩放负荷
        for b in 1:nbus
            case.busAC[b,PD] *= scale
            case.busAC[b,QD] *= scale
        end

        opt = TimeSeriesPowerFlow.PowerFlow.options()
        # opt["PF"]["NR_ALG"] = "bicgstab"  # 可选
        res = TimeSeriesPowerFlow.runpf(case,opt)
        res.success || continue

        solved = res
        # 计算注入
        P_inj = zeros(Float64,nbus)
        Q_inj = zeros(Float64,nbus)
        for g in 1:size(solved.genAC,1)
            b = Int(solved.genAC[g,GEN_BUS])
            P_inj[b] += solved.genAC[g,PG]
            Q_inj[b] += solved.genAC[g,QG]
        end
        P_inj .-= solved.busAC[:,PD]
        Q_inj .-= solved.busAC[:,QD]
        P_spec = P_inj ./ baseMVA
        Q_spec = Q_inj ./ baseMVA

        Vmag = solved.busAC[:,VM]
        Vdeg = solved.busAC[:,VA]
        δrad = Vdeg .* (π/180)
        V = Vmag .* cis.(δrad)

        # 电压过滤 (可关闭)
        if any(v->(v<voltage_filter[1] || v>voltage_filter[2]), Vmag)
            continue
        end

        # PV Q 真实值 / Slack P 真实值
        Q_PV_true = zeros(Float32,nbus)
        for b in pv_idx
            Q_PV_true[b] = Float32(Q_spec[b])
        end
        P_slack_true = Float32(P_spec[ref_idx[1]])

        feat = build_input_features(P_spec, Q_spec, Vmag, δrad, idx)
        bus_types_vec = Int.(solved.busAC[:,BUS_TYPE])

        push!(samples, DatasetSample(
            feat ,
            Float32.(P_spec),
            Float32.(Q_spec),
            Float32.(Vmag),
            Float32.(δrad),
            Q_PV_true,
            P_slack_true,
            bus_types_vec,
            V
        ))
        collected += 1

        if keep_legacy_tensor
            # 仅为了兼容旧 X( nb, 2, N ) 逻辑（不建议训练再用它）
            # 通道1 放 P_spec (Slack 位置也放真实注入, 旧方式可后处理成0)
            # 通道2 放: PQ => Q_spec; PV/Slack => Vmag
            Xslice = zeros(Float32, nbus,2)
            for i in 1:nbus
                Xslice[i,1] = Float32(P_spec[i])
                if bus_types_vec[i]==1
                    Xslice[i,2] = Float32(Q_spec[i])
                else
                    Xslice[i,2] = Float32(Vmag[i])
                end
            end
            legacy_X[:,:,collected] = Xslice
        end

        (collected % 50 == 0) && println("进度: $collected / $total_samples (尝试=$attempts)")
    end

    if collected < total_samples
        println("[WARN] 仅生成 $collected / $total_samples (尝试=$attempts)")
        if keep_legacy_tensor
            legacy_X = legacy_X[:,:,1:collected]
        end
    else
        println("[OK] 样本生成完成：$collected 条 (尝试=$attempts)")
    end

    return samples, Ybus, A, idx, baseMVA, legacy_X
end

function generator_matpower_case(case_file)
     jpc = convert_matpower_case_dp(case_file, "tmp_case_conv_upinn.jl")
    busAC = jpc.busAC; genAC = jpc.genAC; branchAC = jpc.branchAC; loadAC = jpc.loadAC; pvarray = jpc.pv
    (busAC,genAC,branchAC,loadAC,pvarray,_) =
        TimeSeriesPowerFlow.PowerFlow.ext2int(busAC,genAC,branchAC,loadAC,pvarray)
    jpc.busAC=busAC; jpc.genAC=genAC; jpc.branchAC=branchAC; jpc.loadAC=loadAC; jpc.pv=pvarray
    baseMVA = jpc.baseMVA

    pq_idx = findall(busAC[:,BUS_TYPE].==1)
    pv_idx = findall(busAC[:,BUS_TYPE].==2)
    ref_idx= findall(busAC[:,BUS_TYPE].==3)
    isempty(ref_idx) && error("无 Slack (BUS_TYPE=3)")
    idx = BusIndex(pq_idx,pv_idx,ref_idx)
    nbus = size(busAC,1)

    Ybus,Yf,Yt = TimeSeriesPowerFlow.PowerFlow.makeYbus(baseMVA, busAC, branchAC)
    A = build_adjacency_from_Y(Ybus)

    samples = DatasetSample[]
    P_inj = zeros(Float64,nbus)
    Q_inj = zeros(Float64,nbus)
    for g in 1:size(jpc.genAC,1)
        b = Int(jpc.genAC[g,GEN_BUS])
        P_inj[b] += jpc.genAC[g,PG]
        Q_inj[b] += jpc.genAC[g,QG]
    end
    P_inj .-= jpc.busAC[:,PD]
    Q_inj .-= jpc.busAC[:,QD]
    P_spec = P_inj ./ baseMVA
    Q_spec = Q_inj ./ baseMVA

    Q_spec = Q_inj ./ baseMVA

    Vmag = jpc.busAC[:,VM]
    Vdeg = jpc.busAC[:,VA]
    δrad = Vdeg .* (π/180)
    V = Vmag .* cis.(δrad)

    # PV Q 真实值 / Slack P 真实值
    Q_PV_true = zeros(Float32,nbus)
    for b in pv_idx
        Q_PV_true[b] = Float32(Q_spec[b])
    end
    P_slack_true = Float32(P_spec[ref_idx[1]])

    feat = build_input_features(P_spec, Q_spec, Vmag, δrad, idx)
    bus_types_vec = Int.(jpc.busAC[:,BUS_TYPE])

    push!(samples, DatasetSample(
        feat ,
        Float32.(P_spec),
        Float32.(Q_spec),
        Float32.(Vmag),
        Float32.(δrad),
        Q_PV_true,
        P_slack_true,
        bus_types_vec,
        V
    ))

    return samples, Ybus, A, idx, baseMVA
end

function build_input_features_presolve(P_spec_sched, Q_spec_sched, V_set, idx::BusIndex)
    n = length(P_spec_sched)
    feat = zeros(Float32, n, 7)
    for i in 1:n
        is_pq = i in idx.pq
        is_pv = i in idx.pv
        is_rf = i in idx.ref
        feat[i,1] = is_pq ? 1f0 : 0f0          # PQ
        feat[i,2] = is_pv ? 1f0 : 0f0          # PV
        feat[i,3] = is_rf ? 1f0 : 0f0          # Slack
        # 列4：P_spec（Slack 位置用 0，占位）
          feat[i,4] = is_rf ? 0f0 : Float32(P_spec_sched[i])
        # 列5：PQ -> Q_spec; PV/Slack -> V_set
        feat[i,5] = is_pq ? Float32(Q_spec_sched[i]) : Float32(V_set[i])
        # 列6：再次放 V_set (PV/Slack)，其余 0（保持与你 map_outputs 逻辑兼容）
        feat[i,6] = (is_pv || is_rf) ? Float32(V_set[i]) : 0f0
        # 列7：bias
        feat[i,7] = 1f0
    end
    feat
end

# ------------- 预求解数据集生成 -------------
"""
  generate_upinn_samples_presolve(case_file;
      total_samples=500,
      load_scale_range=(0.9,1.1),
      max_attempt_factor=30,
      seed=42,
      keep_legacy_tensor=false,
      perturb_genP=false,
      genP_scale=(0.95,1.05),
      voltage_filter=(0.85,1.2)
  )

  返回:
    samples::Vector{DatasetSample},
    Ybus, A, idx, baseMVA, legacy_X(可选)

  与旧 generate_upinn_samples_tspf_for_training 区别：
    * 特征用的是 pre-solve 规格 (不含已求解未知量)
    * runpf 只为得到真实解
"""
function generate_upinn_samples_presolve(case_file;
      total_samples::Int=500,
      load_scale_range::Tuple{Float64,Float64}=(0.9,1.1),
      max_attempt_factor::Int=30,
      seed::Int=42,
      keep_legacy_tensor::Bool=false,
      perturb_genP::Bool=false,
      genP_scale::Tuple{Float64,Float64}=(0.95,1.05),
      voltage_filter::Tuple{Float64,Float64}=(0.85,1.2)
    )

    Random.seed!(seed)
    jpc = convert_matpower_case_dp(case_file, "tmp_case_conv_upinn_presolve.jl")
    busAC = jpc.busAC; genAC = jpc.genAC; branchAC = jpc.branchAC; loadAC = jpc.loadAC; pvarray = jpc.pv
    (busAC,genAC,branchAC,loadAC,pvarray,_) =
        TimeSeriesPowerFlow.PowerFlow.ext2int(busAC,genAC,branchAC,loadAC,pvarray)
    jpc.busAC=busAC; jpc.genAC=genAC; jpc.branchAC=branchAC; jpc.loadAC=loadAC; jpc.pv=pvarray
    baseMVA = jpc.baseMVA

    pq_idx = findall(busAC[:,BUS_TYPE].==1)
    pv_idx = findall(busAC[:,BUS_TYPE].==2)
    ref_idx= findall(busAC[:,BUS_TYPE].==3)
    isempty(ref_idx) && error("无 Slack (BUS_TYPE=3)")
    idx = BusIndex(pq_idx,pv_idx,ref_idx)

    Ybus,Yf,Yt = TimeSeriesPowerFlow.PowerFlow.makeYbus(baseMVA, busAC, branchAC)
    A = let
        I,J,V=findnz(Ybus); n=size(Ybus,1)
        rows=Int[]; cols=Int[]
        for (i,j,v) in zip(I,J,V)
            if i!=j && abs(v)>1e-12
                push!(rows,i); push!(cols,j)
                push!(rows,j); push!(cols,i)
            end
        end
        append!(rows,1:n); append!(cols,1:n)
        sparse(rows,cols,ones(Float32,length(rows)),n,n)
    end

    nbus = size(busAC,1)
    samples = DatasetSample[]
    legacy_X = keep_legacy_tensor ? zeros(Float32, nbus, 2, total_samples) : nothing

    collected = 0; attempts = 0
    max_attempts = total_samples * max_attempt_factor
    println("Pre-solve 数据生成: 目标=$total_samples 最大尝试=$max_attempts")

    while collected < total_samples && attempts < max_attempts
        attempts += 1
        case = deepcopy(jpc)

        # 1) 随机缩放负荷
        load_scale = rand()*(load_scale_range[2]-load_scale_range[1]) + load_scale_range[1]
        for b in 1:nbus
            case.busAC[b,PD] *= load_scale
            case.busAC[b,QD] *= load_scale
        end

        # 2) 可选扰动非 Slack 发电机 P 设定 (不改变 Slack)
        if perturb_genP
            slack_bus = ref_idx[1]
            for g in 1:size(case.genAC,1)
                bus = Int(case.genAC[g,GEN_BUS])
                if bus != slack_bus
                    scaleP = rand()*(genP_scale[2]-genP_scale[1]) + genP_scale[1]
                    case.genAC[g,PG] *= scaleP
                end
            end
        end

        # 3) 构造 pre-solve 规格 P_spec_sched / Q_spec_sched
        P_spec_sched = zeros(Float64, nbus)
        Q_spec_sched = zeros(Float64, nbus)
        # 发电机 P 注入(除 Slack)，Slack 保留 0 作为未知
        slack_bus = ref_idx[1]
        for g in 1:size(case.genAC,1)
            b = Int(case.genAC[g,GEN_BUS])
            Pg = case.genAC[g,PG] / baseMVA
            if b == slack_bus
                # Slack P 不写入
            else
                P_spec_sched[b] += Pg
            end
        end
        # 负荷 -> 负注入
        for b in 1:nbus
            Pd = case.busAC[b,PD] / baseMVA
            Qd = case.busAC[b,QD] / baseMVA
            P_spec_sched[b] -= Pd     # PQ / PV / Slack 都减 (Slack 位置之前是 0-Pd)
            if b in idx.pq
                Q_spec_sched[b] -= Qd
            end
        end
        # 说明：Slack 行现在包含 -Pd (若有负荷)，这是“其它注入 + slack 未知 Pslack”求解后平衡
        # 我们训练时在特征列4把 Slack 置 0，是为了让模型不看到 slack 行总注入差额（严格去除未知）
        # 因此再做：把 Slack 行 P_spec_sched 置 0（占位）
        for b in idx.ref
            P_spec_sched[b] = 0.0
        end

        # 4) V_set: PV & Slack 的电压设定值（来自当前 case）
        V_set = case.busAC[:,VM]

        # 5) 构造特征（不含任何求解后未知）
        feat = build_input_features_presolve(P_spec_sched, Q_spec_sched, V_set, idx)

        # 6) 潮流求解得到真解
        opt = TimeSeriesPowerFlow.PowerFlow.options()
        res = TimeSeriesPowerFlow.runpf(case,opt)
        res.success || continue

        solved = res
        Vmag_true = solved.busAC[:,VM]
        Vdeg_true = solved.busAC[:,VA]
        δrad_true = Vdeg_true .* (π/180)
        V_true = Vmag_true .* cis.(δrad_true)

        # 过滤 (可选)
        if any(v->(v<voltage_filter[1] || v>voltage_filter[2]), Vmag_true)
            continue
        end

        # 真值：计算实际注入 (用于标签/残差)
        P_inj = zeros(Float64, nbus)
        Q_inj = zeros(Float64, nbus)
        for g in 1:size(solved.genAC,1)
            b = Int(solved.genAC[g,GEN_BUS])
            P_inj[b] += solved.genAC[g,PG]
            Q_inj[b] += solved.genAC[g,QG]
        end
        P_inj .-= solved.busAC[:,PD]
        Q_inj .-= solved.busAC[:,QD]
        P_spec_true = P_inj ./ baseMVA
        Q_spec_true = Q_inj ./ baseMVA

        # 真实 PV Q / Slack P
        Q_PV_true = zeros(Float32, nbus)
        for b in idx.pv
            Q_PV_true[b] = Float32(Q_spec_true[b])
        end
        P_slack_true = Float32(P_spec_true[idx.ref[1]])

        bus_types_vec = Int.(solved.busAC[:,BUS_TYPE])

        push!(samples, DatasetSample(
            feat,
            Float32.(P_spec_sched),       # 预求解 P 规格 (Slack=0 占位)
            Float32.(Q_spec_sched),       # 预求解 Q 规格 (只有 PQ 填值)
            Float32.(Vmag_true),          # 真值
            Float32.(δrad_true),
            Q_PV_true,
            P_slack_true,
            bus_types_vec,
            V_true
        ))
        collected += 1

        if keep_legacy_tensor
            Xslice = zeros(Float32, nbus,2)
            # 通道1：预求解 P_spec_sched
            # 通道2：PQ -> Q_spec_sched; PV/Slack -> V_set
            for i in 1:nbus
                Xslice[i,1] = Float32(P_spec_sched[i])
                if bus_types_vec[i] == 1
                    Xslice[i,2] = Float32(Q_spec_sched[i])
                else
                    Xslice[i,2] = Float32(V_set[i])
                end
            end
            legacy_X[:,:,collected] = Xslice
        end

        (collected % 50 == 0) && println("进度: $collected / $total_samples (尝试=$attempts)")
    end

    if collected < total_samples
        println("[WARN] 仅生成 $collected / $total_samples (尝试=$attempts)")
        if keep_legacy_tensor
            legacy_X = legacy_X[:,:,1:collected]
        end
    else
        println("[OK] 预求解样本生成完成：$collected 条 (尝试=$attempts)")
    end

    return samples, Ybus, A, idx, baseMVA, legacy_X
end
