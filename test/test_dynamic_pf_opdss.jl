# ====================================================================================
# IEEE 37 Feeder 24小时(0.1s步长)时间序列仿真脚本 (修正版)
#
# 说明:
# - 修复 Uniform 未定义问题：加入 using Distributions
# - 去掉重复的 build_load_profile 定义，只保留一个
# - 若不想依赖 Distributions，可将 Uniform 抽样改成线性变换 (已在注释中给出)
#
# 命令行用法:
#   julia this_script.jl "D:/path/to/ieee37" 0.1 24 100
#
# REPL 用法:
#   include("this_script.jl")
#   res = time_series_ieee37("D:/path/to/ieee37"; dt_s=0.1, hours=1.0, sample_every=50)
# ====================================================================================

using OpenDSSDirect
using DataFrames
using Statistics
using Random
using Printf
using Dates
using LinearAlgebra
using Logging
using Distributions   # <<< 关键修复：提供 Uniform, Normal 等分布

const dsscmd = OpenDSSDirect.dss

# ================= 工具函数 =================
function try_default(f::Function, default)
    try
        return f()
    catch
        return default
    end
end

if !(@isdefined _warn_flags)
    const _warn_flags = Dict{Symbol,Bool}()
end
function warn_once(key::Symbol, msg::String)
    if get(_warn_flags, key, false) == false
        @warn msg
        _warn_flags[key] = true
    end
end

function silent_text_query(cmd::String)
    Logging.with_logger(Logging.NullLogger()) do
        try
            return dsscmd(cmd)
        catch
            return ""
        end
    end
end

function parse_numeric_list(str::AbstractString)
    s = replace(str, '[' => ' ', ']' => ' ', ',' => ' ')
    toks = filter(!isempty, split(s))
    vals = Float64[]
    for t in toks
        v = try
            parse(Float64, t)
        catch
            continue
        end
        push!(vals, v)
    end
    return vals
end

# ================= 功率解析 =================
function parse_total_power(tp_raw)
    if tp_raw isa AbstractVector
        if length(tp_raw) >= 2 && !(tp_raw[1] isa Complex)
            return (tp_raw[1]/1000, tp_raw[2]/1000)
        elseif length(tp_raw) >= 1 && tp_raw[1] isa Complex
            c = tp_raw[1]; return (real(c)/1000, imag(c)/1000)
        end
    elseif tp_raw isa Complex
        return (real(tp_raw)/1000, imag(tp_raw)/1000)
    end
    return (NaN, NaN)
end

# ================= 相量提取 =================
function extract_phasors(vt_raw)
    if vt_raw isa AbstractVector{<:Real}
        len = length(vt_raw)
        if len < 2; return ComplexF64[] end
        if isodd(len)
            warn_once(:odd_volt_len, "Bus.Voltages() 返回奇数长度 $len, 末元素忽略")
            len -= 1
        end
        n = len ÷ 2
        return [complex(vt_raw[2i-1], vt_raw[2i]) for i in 1:n]
    elseif vt_raw isa AbstractVector{<:Complex}
        return ComplexF64[ComplexF64(v) for v in vt_raw]
    else
        warn_once(:unk_vtype, "未知电压类型 $(typeof(vt_raw))")
        return ComplexF64[]
    end
end

# ================= DataFrame 打印(调试) =================
function print_head(df::DataFrame; n::Int=12)
    nshow = min(n, nrow(df))
    if nshow == 0
        println("(空 DataFrame)"); return
    end
    try
        show(first(df, nshow); allcols=true)
        println()
    catch e
        warn_once(:show_fail, "DataFrame show 失败: $e")
        cols = names(df)
        println(join(cols," | "))
        for r in 1:nshow
            println(join([string(df[r,c]) for c in cols]," | "))
        end
    end
end

# ================= 线路功率解析内部 =================
function _parse_line_powers(pdata, nph::Int, nterm::Int)
    term_phase = Tuple{Int,Int,Float64,Float64}[]
    Pterm = zeros(nterm); Qterm = zeros(nterm)
    if pdata isa AbstractVector{<:Complex}
        L = length(pdata)
        if L == nterm*nph
            k=1
            for t in 1:nterm, ph in 1:nph
                c = pdata[k]; k+=1
                p=real(c); q=imag(c)
                push!(term_phase,(t,ph,p,q))
                Pterm[t]+=p; Qterm[t]+=q
            end
        elseif L == 2*nterm*nph
            warn_once(:line_complex_double,"线路功率长度为 2*nterm*nph, 拆分尝试")
            half = L ÷ 2
            if half == nterm*nph
                k=1
                for t in 1:nterm, ph in 1:nph
                    c=pdata[k]; k+=1
                    p=real(c); q=imag(c)
                    push!(term_phase,(t,ph,p,q)); Pterm[t]+=p; Qterm[t]+=q
                end
                for t in 1:nterm, ph in 1:nph
                    c=pdata[k]; k+=1
                    p=real(c); q=imag(c)
                    push!(term_phase,(t,ph,p,q)); Pterm[t]+=p; Qterm[t]+=q
                end
            end
        else
            warn_once(:line_c_len,"Complex 线路功率长度 $L 不匹配 nterm*nph=$(nterm*nph)")
        end
    elseif pdata isa AbstractVector{<:Real}
        L = length(pdata); expL = 2*nterm*nph
        if L != expL
            warn_once(:line_r_len,"Real 线路功率长度 $L != $expL")
        else
            k=1
            for t in 1:nterm, ph in 1:nph
                p=pdata[k]; q=pdata[k+1]; k+=2
                push!(term_phase,(t,ph,p,q))
                Pterm[t]+=p; Qterm[t]+=q
            end
        end
    else
        warn_once(:line_ptype,"未知线路功率类型 $(typeof(pdata))")
    end
    return term_phase, Pterm, Qterm
end

# ================= 主案例载入 =================
function load_ieee37(feeder_dir::String; main_file::String="ieee37.dss",
                     solve_mode::Union{Nothing,String}=nothing,
                     verbose::Bool=true)
    main_path = joinpath(feeder_dir, main_file)
    isfile(main_path) || error("主文件不存在: $main_path")
    old = pwd(); cd(feeder_dir)
    try
        dsscmd("Clear")
        dsscmd("Compile $main_file")
        if solve_mode !== nothing
            dsscmd("Set Mode=$solve_mode")
        end
        dsscmd("CalcVoltageBases")
        dsscmd("Solve")
        converged = OpenDSSDirect.Solution.Converged()
        iters = OpenDSSDirect.Solution.Iterations()
        if !converged
            dsscmd("Set Mode=direct"); dsscmd("Solve")
            converged = OpenDSSDirect.Solution.Converged()
            verbose && @info "初次未收敛，direct 重算: converged=$converged iter=$(OpenDSSDirect.Solution.Iterations())"
        else
            verbose && @info "收敛 iterations=$iters"
        end
        tp = OpenDSSDirect.Circuit.TotalPower()
        (MW,Mvar) = parse_total_power(tp)
        return converged, Dict(
            "converged"=>converged,
            "iterations"=>OpenDSSDirect.Solution.Iterations(),
            "total_power_MW_Mvar"=>(MW,Mvar),
            "buses"=>length(OpenDSSDirect.Circuit.AllBusNames()),
            "lines"=>length(OpenDSSDirect.Lines.AllNames()),
            "loads"=>length(OpenDSSDirect.Loads.AllNames()),
            "timestamp"=>string(Dates.now())
        )
    finally
        cd(old)
    end
end

function get_system_summary()
    Dict(
        "buses"=>OpenDSSDirect.Circuit.AllBusNames(),
        "lines"=>OpenDSSDirect.Lines.AllNames(),
        "loads"=>OpenDSSDirect.Loads.AllNames(),
        "transformers"=>OpenDSSDirect.Transformers.AllNames(),
        "capacitors"=>OpenDSSDirect.Capacitors.AllNames()
    )
end

# ================= 母线电压 (调试) =================
function get_bus_voltages_ll(; include_lg::Bool=false)
    buses = OpenDSSDirect.Circuit.AllBusNames()
    if include_lg
        df = DataFrame(Bus=String[], Pair=String[], VLL_kV=Float64[], Angle_deg=Float64[],
                       VLL_pu=Float64[], Phase=Int[], VLG_kV=Float64[],
                       VLG_pu=Float64[], VLG_Ang_deg=Float64[])
    else
        df = DataFrame(Bus=String[], Pair=String[], VLL_kV=Float64[], Angle_deg=Float64[], VLL_pu=Float64[])
    end
    for b in buses
        OpenDSSDirect.Circuit.SetActiveBus(b)
        base = OpenDSSDirect.Bus.kVBase()
        phasors = extract_phasors(OpenDSSDirect.Bus.Voltages()); n = length(phasors)
        n==0 && continue
        if n >= 2
            for i in 1:n-1, j in i+1:n
                vll = phasors[i]-phasors[j]
                mag = abs(vll)/1000
                ang = rad2deg(angle(vll))
                pu = base==0 ? NaN : mag/base
                if include_lg
                    for ph in 1:n
                        vlg = phasors[ph]
                        mag_lg = abs(vlg)/1000
                        ang_lg = rad2deg(angle(vlg))
                        pu_lg = base==0 ? NaN : mag_lg/(base/sqrt(3))
                        push!(df,(b,"$(i)-$(j)",mag,ang,pu,ph,mag_lg,pu_lg,ang_lg))
                    end
                else
                    push!(df,(b,"$(i)-$(j)",mag,ang,pu))
                end
            end
        elseif include_lg
            vlg = phasors[1]
            mag_lg = abs(vlg)/1000
            ang_lg = rad2deg(angle(vlg))
            pu_lg = base==0 ? NaN : mag_lg/(base/sqrt(3))
            push!(df,(b,"—",NaN,NaN,NaN,1,mag_lg,pu_lg,ang_lg))
        end
    end
    return df
end

# ================= 线路功率 (调试) =================
function _get_line_powers(name::String)
    try
        OpenDSSDirect.Lines.Name(name)
    catch
        dsscmd("Select Line.$name")
    end
    if hasproperty(OpenDSSDirect.Lines, :Powers)
        return try_default(() -> OpenDSSDirect.Lines.Powers(), OpenDSSDirect.CktElement.Powers())
    else
        warn_once(:old_lines_powers,"旧版无 Lines.Powers -> 使用 CktElement.Powers()")
        return OpenDSSDirect.CktElement.Powers()
    end
end

function get_line_flows(; per_phase::Bool=true)
    df = DataFrame(Line=String[], Bus1=String[], Bus2=String[], Terminal=Int[],
                   Phase=Int[], P_kW=Float64[], Q_kvar=Float64[],
                   Psum_kW=Float64[], Qsum_kvar=Float64[],
                   PLoss_kW=Float64[], QLoss_kvar=Float64[])
    for ln in OpenDSSDirect.Lines.AllNames()
        try
            OpenDSSDirect.Lines.Name(ln)
        catch
            dsscmd("Select Line.$ln")
        end
        bus1 = try_default(() -> OpenDSSDirect.Lines.Bus1(), "")
        bus2 = try_default(() -> OpenDSSDirect.Lines.Bus2(), "")
        nph = try_default(() -> OpenDSSDirect.Lines.Phases(), 3)
        nterm = try_default(() -> OpenDSSDirect.CktElement.NumTerminals(), 2)
        pdata = _get_line_powers(ln)
        phase_data, Pterm, Qterm = _parse_line_powers(pdata, nph, nterm)
        if per_phase
            for (t,ph,p,q) in phase_data
                push!(df,(ln,bus1,bus2,t,ph,p,q,NaN,NaN,NaN,NaN))
            end
        end
        PLoss = sum(Pterm); QLoss = sum(Qterm)
        for t in 1:nterm
            push!(df,(ln,bus1,bus2,t,0,NaN,NaN,Pterm[t],Qterm[t],PLoss,QLoss))
        end
    end
    return df
end

# ================= 负荷信息 =================
function get_load_bus(ld::String)
    if hasproperty(OpenDSSDirect.Loads, :BusName)
        return OpenDSSDirect.Loads.BusName()
    else
        dsscmd("Select Load.$ld")
        bns = try_default(() -> OpenDSSDirect.CktElement.BusNames(), String[])
        return isempty(bns) ? "" : strip(bns[1])
    end
end

function get_loads()
    names = OpenDSSDirect.Loads.AllNames()
    df = DataFrame(Load=String[], Bus=String[], BaseBus=String[],
                   Phases=Int[], Conn=String[],
                   kV_setting=Float64[], kW_setting=Float64[], kvar_setting=Float64[],
                   Pcalc_kW=Float64[], Qcalc_kvar=Float64[], PF=Float64[])
    for ld in names
        OpenDSSDirect.Loads.Name(ld)
        rawbus = get_load_bus(ld)
        basebus = split(rawbus, '.')[1]
        phs = try_default(() -> OpenDSSDirect.Loads.Phases(), 3)
        conncode = try_default(() -> OpenDSSDirect.Loads.Conn(), 0)
        kv = try_default(() -> OpenDSSDirect.Loads.kV(), NaN)
        kWset = try_default(() -> OpenDSSDirect.Loads.kW(), NaN)
        kvarset = try_default(() -> OpenDSSDirect.Loads.kvar(), NaN)
        dsscmd("Select Load.$ld")
        ep = OpenDSSDirect.CktElement.Powers()
        P=0.0; Q=0.0
        if ep isa AbstractVector{<:Complex}
            for c in ep
                P += real(c); Q += imag(c)
            end
        else
            for i in 1:2:length(ep)-1
                P += ep[i]; Q += ep[i+1]
            end
        end
        S = sqrt(P^2 + Q^2)
        pf = S==0 ? NaN : P / S
        push!(df,(ld, rawbus, basebus, phs, (conncode==1 ? "Delta" : "Wye"),
                  kv, kWset, kvarset, P, Q, pf))
    end
    return df
end

# ================= 调压器 =================
function reg_prop_float(rg::String, sym::Symbol, q::String, default)
    if hasproperty(OpenDSSDirect.RegControls, sym)
        return try_default(() -> (getproperty(OpenDSSDirect.RegControls, sym))(), default)
    else
        v = silent_text_query("? Regcontrol.$rg.$q")
        if isempty(v); return default end
        try parse(Float64, v) catch; default end
    end
end
function reg_prop_int(rg,sym,q,default)
    v = reg_prop_float(rg,sym,q,default)
    if v === default || isnan(v); return default end
    return Int(round(v))
end
function reg_prop_str(rg,sym,q,default)
    if hasproperty(OpenDSSDirect.RegControls, sym)
        return try_default(() -> (getproperty(OpenDSSDirect.RegControls, sym))(), default)
    else
        v = silent_text_query("? Regcontrol.$rg.$q")
        return isempty(v) ? default : strip(v)
    end
end

function get_regulators()
    names = try_default(() -> OpenDSSDirect.RegControls.AllNames(), String[])
    df = DataFrame(RegControl=String[], Transformer=String[], Winding=Int[],
                   Vreg=Float64[], Band=Float64[], PTRatio=Float64[],
                   CTPrim=Float64[], R=Float64[], X=Float64[],
                   TapNumber=Int[], TapPosition=Float64[])
    isempty(names) && return df
    for rg in names
        OpenDSSDirect.RegControls.Name(rg)
        tr  = reg_prop_str(rg,:Transformer,"transformer","")
        wdg = reg_prop_int(rg,:Winding,"winding",1)
        vreg = reg_prop_float(rg,:Vreg,"vreg",NaN)
        band = reg_prop_float(rg,:Band,"band",NaN)
        ptr  = reg_prop_float(rg,:PTratio,"ptratio",NaN)
        ctp  = reg_prop_float(rg,:CTPrim,"ctprim",NaN)
        rv   = reg_prop_float(rg,:R,"r",NaN)
        xv   = reg_prop_float(rg,:X,"x",NaN)
        tapnum=-1; tappos=NaN
        if !isempty(tr)
            try
                OpenDSSDirect.Transformers.Name(tr)
                tapnum = try_default(() -> OpenDSSDirect.Transformers.NumTaps(), -1)
                tappos = try
                    OpenDSSDirect.Transformers.Tap(wdg)
                catch
                    try
                        OpenDSSDirect.Transformers.Wdg(Float64(wdg))
                        OpenDSSDirect.Transformers.Tap()
                    catch
                        NaN
                    end
                end
            catch
            end
        end
        push!(df,(rg,tr,wdg,vreg,band,ptr,ctp,rv,xv,tapnum,tappos))
    end
    return df
end

# ================= 变压器 =================
function _get_winding_kV(tr::String, w::Int)
    try
        OpenDSSDirect.Transformers.Wdg(Float64(w))
        return OpenDSSDirect.Transformers.kV()
    catch
        kvs = silent_text_query("? Transformer.$tr.kvs")
        vals = parse_numeric_list(kvs)
        return (1 <= w <= length(vals)) ? vals[w] : NaN
    end
end

function _get_winding_kVA(tr::String, w::Int)
    try
        OpenDSSDirect.Transformers.Wdg(Float64(w))
        return OpenDSSDirect.Transformers.kVA()
    catch
        s = silent_text_query("? Transformer.$tr.kvas")
        if isempty(s)
            s = silent_text_query("? Transformer.$tr.kVAs")
        end
        vals = parse_numeric_list(s)
        return (1 <= w <= length(vals)) ? vals[w] : NaN
    end
end

function _get_winding_R(tr::String, w::Int)
    v = try
        OpenDSSDirect.Transformers.Wdg(Float64(w))
        OpenDSSDirect.Transformers.R()
    catch
        rs = silent_text_query("? Transformer.$tr.r$(w)")
        if isempty(rs)
            NaN
        else
            try parse(Float64, rs) catch; NaN end
        end
    end
    return v isa Number ? float(v) : NaN
end

function _get_winding_tap(tr::String, w::Int)
    try
        OpenDSSDirect.Transformers.Wdg(Float64(w))
        return OpenDSSDirect.Transformers.Tap()
    catch
        NaN
    end
end

function get_transformers()
    names = OpenDSSDirect.Transformers.AllNames()
    df = DataFrame(Transformer=String[], Windings=Int[], kVs=String[], kVAs=String[],
                   Xhl=Float64[], Rws=String[], TapPositions=String[],
                   P_kW=Float64[], Q_kvar=Float64[], Loss_kW=Float64[])
    for tr in names
        OpenDSSDirect.Transformers.Name(tr)
        nw = try_default(() -> OpenDSSDirect.Transformers.NumWindings(), 0)
        kvs = Float64[]; kvas = Float64[]; rws = Float64[]; taps = Float64[]
        for w in 1:nw
            push!(kvs, _get_winding_kV(tr, w))
            push!(kvas, _get_winding_kVA(tr, w))
            push!(rws, _get_winding_R(tr, w))
            push!(taps, _get_winding_tap(tr, w))
        end
        xhl = try
            parse(Float64, silent_text_query("? Transformer.$tr.xhl"))
        catch
            NaN
        end
        dsscmd("Select Transformer.$tr")
        ep = OpenDSSDirect.CktElement.Powers()
        P=0.0; Q=0.0
        if ep isa AbstractVector{<:Complex}
            for c in ep
                P+=real(c); Q+=imag(c)
            end
        else
            for i in 1:2:length(ep)-1
                P += ep[i]; Q += ep[i+1]
            end
        end
        loss_vec = try_default(() -> OpenDSSDirect.CktElement.Losses(), nothing)
        loss_kW = 0.0
        if loss_vec !== nothing && length(loss_vec) >= 1
            if loss_vec[1] isa Complex
                loss_kW = real(loss_vec[1])/1000
            else
                loss_kW = loss_vec[1]/1000
            end
        else
            if ep isa AbstractVector{<:Real} && length(ep) >= 4
                loss_kW = abs(ep[1] + ep[3])
            elseif ep isa AbstractVector{<:Complex} && length(ep) >= 2
                loss_kW = abs(real(ep[1]) + real(ep[2]))
            else
                loss_kW = NaN
            end
        end
        if loss_kW isa Complex
            loss_kW = real(loss_kW)
        end
        push!(df,(tr,nw,
                  join(round.(kvs,digits=4), ","),
                  join(round.(kvas,digits=4), ","),
                  xhl,
                  join(round.(rws,digits=4), ","),
                  join(round.(taps,digits=5), ","),
                  P,Q,loss_kW))
    end
    return df
end

# ================= 示例运行 (可选) =================
function example_run(feeder_dir::String)
    println("=== IEEE 37 Feeder 示例 ===")
    conv, info = load_ieee37(feeder_dir)
    println("收敛: $(info["converged"]) 迭代: $(info["iterations"]) 总功率(MW,Mvar)=$(info["total_power_MW_Mvar"])")
    summary = get_system_summary()
    println("总线: $(length(summary["buses"])) 线路: $(length(summary["lines"])) 负荷: $(length(summary["loads"])) 变压器: $(length(summary["transformers"]))")
    println("\n线—线电压(前若干行):")
    vll = get_bus_voltages_ll()
    print_head(vll; n=12)
    println("\n线路功率端汇总(前若干行):")
    lineflows = get_line_flows()
    agg = lineflows[lineflows.:Phase .== 0, :]
    print_head(agg; n=12)
    println("\n负荷功率(前若干行):")
    loads = get_loads()
    print_head(loads; n=12)
    println("\n调压器：")
    regs = get_regulators()
    print_head(regs; n=12)
    println("\n变压器：")
    txs = get_transformers()
    print_head(txs; n=12)
    return Dict(
        "info"=>info,
        "voltages_ll"=>vll,
        "line_flows"=>lineflows,
        "loads"=>loads,
        "regulators"=>regs,
        "transformers"=>txs
    )
end

# ====================================================================================
# 负荷曲线 + 时间序列仿真
# ====================================================================================

# -------- 基础日负荷曲线 --------
function build_base_profile(npts::Int; dt_s::Float64=0.1)
    prof = Vector{Float32}(undef, npts)
    for i in 1:npts
        t = (i-1)*dt_s
        h = t / 3600.0
        base =
            h < 5  ? 0.60 :
            h < 9  ? 0.60 + (h-5)/4 * (0.90-0.60) :
            h < 12 ? 0.95 :
            h < 14 ? 0.95 - (h-12)/2 * (0.95-0.85) :
            h < 19 ? 0.85 + (h-14)/5 * (1.05-0.85) :
            h < 21 ? 1.05 + (h-19)/2 * (1.10-1.05) :
                     1.10 - (h-21)/3 * (1.10-0.70)
        base += 0.03 * sin(2π*h/24)
        prof[i] = max(base, 0f0)
    end
    return prof
end

# -------- 单负荷曲线生成 (使用 Distributions.Uniform) --------
function build_load_profile(base_prof::Vector{Float32};
                            seed::Int=0,
                            pf_mode::Symbol=:constant,
                            pf_range::Tuple{Float64,Float64}=(0.92,0.99),
                            dt_s::Float64=0.1)
    rng = MersenneTwister(seed)
    npts = length(base_prof)
    α = rand(rng, Uniform(0.9,1.1))
    φ1 = rand(rng)*2π
    φ2 = rand(rng)*2π
    pf = rand(rng, Uniform(pf_range[1], pf_range[2]))
    β = 0.999
    σ = 0.02
    noise_prev = 0.0f0
    Pmult = Vector{Float32}(undef, npts)
    Qmult = Vector{Float32}(undef, npts)
    tanφ = tan(acos(pf))
    for i in 1:npts
        t = (i-1)*dt_s
        hf = 0.01*sin(2π*(1/600)*t + φ1) + 0.005*sin(2π*(1/60)*t + φ2)
        noise_prev = β*noise_prev + (1-β)*(σ*randn(rng))
        raw = base_prof[i]*α + hf + noise_prev
        pm = max(raw, 0.0f0)
        Pmult[i] = pm
        if pf_mode == :constant
            Qmult[i] = pm
        else
            pf_inst = clamp(pf + 0.01*randn(rng), 0.85, 0.995)
            # 若要真实 PF 波动: 可用 pm * tan(acos(pf_inst)) / tan(acos(pf)) 等比例缩放
            Qmult[i] = pm
        end
    end
    return Pmult, Qmult, pf
end

# -------- 注册 LoadShape --------
function register_loadshape(loadname::String,
                            Pmult::Vector{Float32},
                            Qmult::Vector{Float32};
                            dirname::String="loadshapes",
                            dt_s::Float64=0.1)
    isdir(dirname) || mkpath(dirname)
    pfile = joinpath(dirname, "$(loadname)_P.txt")
    qfile = joinpath(dirname, "$(loadname)_Q.txt")
    open(pfile, "w") do io
        for v in Pmult
            @printf(io, "%.6f\n", v)
        end
    end
    open(qfile, "w") do io
        for v in Qmult
            @printf(io, "%.6f\n", v)
        end
    end
    npts = length(Pmult)
    cmd = """
    New Loadshape.LD_$loadname npts=$npts sinterval=$dt_s Pmult=(file="$pfile") Qmult=(file="$qfile")
    """
    dsscmd(cmd)
    dsscmd("Edit Load.$loadname Yearly=LD_$loadname")
end

# -------- 全部负荷生成与绑定 --------
function build_load_shapes(; dt_s::Float64=0.1, hours::Float64=24.0,
                            shape_dir::String="loadshapes",
                            pf_mode::Symbol=:constant,
                            seed::Int=2024)
    loads = OpenDSSDirect.Loads.AllNames()
    npts = Int(round(hours*3600/dt_s))
    base_prof = build_base_profile(npts; dt_s=dt_s)
    df = DataFrame(Load=String[], kW_base=Float64[], kvar_base=Float64[],
                   pf_assumed=Float64[], shape=String[])
    rng_master = MersenneTwister(seed)
    for ld in loads
        OpenDSSDirect.Loads.Name(ld)
        kWb = OpenDSSDirect.Loads.kW()
        kvarb = OpenDSSDirect.Loads.kvar()
        seed_i = rand(rng_master, 1:10_000_000)
        Pmult, Qmult, pf = build_load_profile(base_prof; seed=seed_i,
                                              pf_mode=pf_mode, dt_s=dt_s)
        register_loadshape(ld, Pmult, Qmult; dirname=shape_dir, dt_s=dt_s)
        push!(df, (ld, kWb, kvarb, pf, "LD_$ld"))
    end
    return df
end

# ---------- 修改版 run_time_series ----------
function run_time_series(npts::Int; dt_s::Float64=0.1,
                         sample_every::Int=10,
                         collect::Vector{Symbol}=[:voltage_bus, :total_power],
                         save_phase::Bool=true,
                         max_phases::Int=3)
    # 进入 duty 或 daily 模式之前，可根据需要切换:
    # dsscmd("Set mode=daily stepsize=$(dt_s)s number=1")
    dsscmd("Set mode=Yearly stepsize=$(dt_s)s number=1")

    buses = OpenDSSDirect.Circuit.AllBusNames()
    nb = length(buses)
    n_samples = Int(ceil(npts / sample_every))

    time_s = Vector{Float64}(undef, n_samples)

    vol_pu_avg = nothing
    vol_pu_phase = nothing
    if :voltage_bus in collect
        vol_pu_avg = Matrix{Float32}(undef, nb, n_samples)
        if save_phase
            vol_pu_phase = Array{Float32}(undef, nb, max_phases, n_samples)
        end
    end

    totalP = Vector{Float64}()
    totalQ = Vector{Float64}()
    if :total_power in collect
        resize!(totalP, n_samples)
        resize!(totalQ, n_samples)
    end

    sample_idx = 0
    for step in 1:npts
        OpenDSSDirect.Solution.Solve()

        if step % sample_every == 0
            sample_idx += 1
            tcur = (step-1)*dt_s
            time_s[sample_idx] = tcur

            if vol_pu_avg !== nothing
                for (bi, b) in enumerate(buses)
                    OpenDSSDirect.Circuit.SetActiveBus(b)
                    base = OpenDSSDirect.Bus.kVBase()
                    phasors = extract_phasors(OpenDSSDirect.Bus.Voltages())
                    np = length(phasors)
                    if np == 0 || base <= 0
                        vol_pu_avg[bi, sample_idx] = NaN
                        if vol_pu_phase !== nothing
                            @inbounds for ph in 1:max_phases
                                vol_pu_phase[bi, ph, sample_idx] = NaN
                            end
                        end
                    else
                        mags = abs.(phasors) ./ 1000.0          # kV (相对地)
                        vbase_ph = base / sqrt(3)               # kV base (相)
                        pu_ph = mags ./ vbase_ph
                        vol_pu_avg[bi, sample_idx] = mean(pu_ph)
                        if vol_pu_phase !== nothing
                            @inbounds for ph in 1:max_phases
                                vol_pu_phase[bi, ph, sample_idx] = ph <= np ? pu_ph[ph] : NaN
                            end
                        end
                    end
                end
            end

            if :total_power in collect
                tp = OpenDSSDirect.Circuit.TotalPower()
                (MW,Mvar) = parse_total_power(tp)
                totalP[sample_idx] = MW
                totalQ[sample_idx] = Mvar
            end
        end
    end

    result = Dict{String,Any}()
    result["time_s"] = time_s
    if vol_pu_avg !== nothing
        result["voltage_pu_matrix"] = (buses=buses, data=vol_pu_avg)
        if vol_pu_phase !== nothing
            result["voltage_pu_phase"] = (
                buses=buses,
                data=vol_pu_phase,     # size: (nbus, max_phases, nsamples)
                phase_order="A,B,C"
            )
        end
    end
    if :total_power in collect
        result["total_power_MW_Mvar"] = (P=totalP, Q=totalQ)
    end
    return result
end

# ---------- 查询：整个时间序列某母线某相 ----------
# phase = 1,2,3 分别 A/B/C；若母线缺该相返回全 NaN
function get_bus_phase_voltage_series(res::Dict, bus::String, phase::Int)
    phase in 1:3 || error("phase 必须是 1~3 (A/B/C)")
    vtp = get(res["result"], "voltage_pu_phase", nothing)
    vtp === nothing && error("结果中没有 'voltage_pu_phase'，请运行 run_time_series 时 save_phase=true")
    buses = vtp.buses
    idx = findfirst(==(bus), buses)
    idx === nothing && error("母线 $bus 不存在")
    data = vtp.data
    # data 维度: (nbus, max_phases, nsamples)
    vseries = @view data[idx, phase, :]
    t = res["result"]["time_s"]
    return (time_s=t, v_pu=vseries)
end

# ---------- 查询：指定时间点（取最近采样） ----------
function get_bus_phase_voltage_at_time(res::Dict, bus::String, phase::Int, t_query::Float64)
    info = get_bus_phase_voltage_series(res, bus, phase)
    tvec = info.time_s
    i = searchsortedfirst(tvec, t_query)
    if i > length(tvec)
        i = length(tvec)
    elseif i > 1
        if abs(tvec[i] - t_query) > abs(tvec[i-1] - t_query)
            i -= 1
        end
    end
    return (bus=bus, phase=phase, time=tvec[i], v_pu=info.v_pu[i], index=i)
end


# -------- 综合执行入口 --------
function time_series_ieee37(feeder_dir::String;
                            dt_s::Float64=0.1,
                            hours::Float64=24.0,
                            sample_every::Int=100,
                            pf_mode::Symbol=:constant,
                            collect::Vector{Symbol}=[:voltage_bus, :total_power],
                            shape_dir::String="loadshapes")
    println("加载 IEEE 37 初始工况...")
    conv, info = load_ieee37(feeder_dir)
    println("初始收敛 = $(info["converged"]), 总功率(MW,Mvar) = $(info["total_power_MW_Mvar"])")

    println("生成并注册各负荷 LoadShape ...")
    df_shapes = build_load_shapes(dt_s=dt_s, hours=hours, pf_mode=pf_mode, shape_dir=shape_dir)
    println("已创建 $(nrow(df_shapes)) 个负荷曲线。")

    npts = Int(round(hours*3600/dt_s))
    println("启动时间序列仿真: 总步数=$npts, 步长=$(dt_s)s, 采样每 $sample_every 步 (≈保存 $(Int(ceil(npts/sample_every))) 条记录)")
    t0 = time()
    res = run_time_series(npts; dt_s=dt_s, sample_every=sample_every, collect=collect)
    tel = time() - t0
    println("仿真完成，耗时 $(round(tel,digits=2)) 秒 (≈ $(round(tel/60,digits=2)) 分).")

    return Dict(
        "shapes"=>df_shapes,
        "result"=>res,
        "meta"=>Dict("dt_s"=>dt_s,
                     "hours"=>hours,
                     "sample_every"=>sample_every,
                     "npts"=>npts,
                     "collected"=>collect,
                     "timestamp"=>string(Dates.now()))
    )
end




# ====================================================================================
# 命令行入口
# ====================================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1
        feeder_dir = ARGS[1]
        dt_s = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.1
        hours = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 24.0
        sample_every = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 100
        println("参数: feeder_dir=$feeder_dir dt_s=$dt_s hours=$hours sample_every=$sample_every")
        results = time_series_ieee37(feeder_dir; dt_s=dt_s, hours=hours, sample_every=sample_every)
        println("完成。结果已返回。")
    else
        println("未提供 feeder_dir，脚本未执行主流程。用法示例：")
        println("  julia $(PROGRAM_FILE) \"D:/path/to/ieee37\" 0.1 24 100")
    end
end

# ====================================================================================
# 可在 REPL 测试:
feeder_dir = "D:/luosipeng/Deep_Learning_in_Distribution_System/data"
# 假设你之前完整执行:
res = time_series_ieee37(feeder_dir; dt_s=0.1, hours=24.0, sample_every=1,
                         collect=[:voltage_bus, :total_power])
# 然后:
s = get_bus_phase_voltage_series(res, "704", 2)
@show s.v_pu[1:100]
pt = get_bus_phase_voltage_at_time(res, "704", 2, 123.4)
@show pt
# ====================================================================================