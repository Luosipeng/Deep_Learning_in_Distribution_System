using OpenDSSDirect
using DataFrames
using Statistics
using Random
using Printf
using Dates
using LinearAlgebra
using Logging
using Distributions

const dsscmd = OpenDSSDirect.dss
const NaN32 = Float32(NaN)

# ================= 通用工具 =================
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

# ====================================================================================
# 负荷曲线 + 时间序列仿真
# ====================================================================================

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
            Qmult[i] = pm
        end
    end
    return Pmult, Qmult, pf
end

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

# ====================================================================================
# 功率注入支持
# ====================================================================================

const TYPE_MAP = Dict(
    :load => "Load",
    :pv => "PVSystem",
    :gen => "Generator",
    :storage => "Storage",
    :capacitor => "Capacitor",
    :vsource => "Vsource"
)

_basebus(bus::String) = split(bus, '.')[1]

function _select_and_get_basebus(dss_type::String, name::String)
    try
        dsscmd("Select $dss_type.$name")
        bns = OpenDSSDirect.CktElement.BusNames()
        if isempty(bns)
            return ""
        else
            return _basebus(bns[1])
        end
    catch
        return ""
    end
end

function _element_total_PQ()
    ep = OpenDSSDirect.CktElement.Powers()
    P = 0.0; Q = 0.0
    if ep isa AbstractVector{<:Complex}
        @inbounds for c in ep
            P += real(c); Q += imag(c)
        end
    else
        @inbounds for i in 1:2:length(ep)-1
            P += ep[i]; Q += ep[i+1]
        end
    end
    return (P, Q)
end

function _allnames_or_empty(modsym::Symbol)
    if hasproperty(OpenDSSDirect, modsym)
        mod = getproperty(OpenDSSDirect, modsym)
        return try
            mod.AllNames()
        catch
            String[]
        end
    else
        return String[]
    end
end

function _gather_shunt_elements!(entries, bus_index::Dict{String,Int}, etype::Symbol)
    dss_type = TYPE_MAP[etype]
    names = _allnames_or_empty(Symbol(dss_type* "s"))
    if etype == :pv && isempty(names) && hasproperty(OpenDSSDirect, :PVSystems)
        names = try OpenDSSDirect.PVSystems.AllNames() catch; String[] end
    elseif etype == :vsource
        names = try
            hasproperty(OpenDSSDirect, :Vsources) ? OpenDSSDirect.Vsources.AllNames() : String[]
        catch
            String[]
        end
        if isempty(names)
            names = ["source"]
        end
    end
    for nm in names
        b = _select_and_get_basebus(dss_type, nm)
        isempty(b) && continue
        if haskey(bus_index, b)
            push!(entries, (etype, nm, bus_index[b]))
        end
    end
    return entries
end

function prepare_injection_elements(buses::Vector{String})
    bus_index = Dict{String,Int}()
    for (i,b) in enumerate(buses)
        bus_index[b] = i
    end
    entries = Vector{Tuple{Symbol,String,Int}}()
    for et in (:load, :pv, :gen, :storage, :capacitor, :vsource)
        _gather_shunt_elements!(entries, bus_index, et)
    end
    return entries
end

function _activate_element(etype::Symbol, name::String)
    dss_type = TYPE_MAP[etype]
    try
        dsscmd("Select $dss_type.$name")
    catch
    end
end

# ---- 取得相数和接线方式（0 Wye / 1 Delta；若失败视为 Wye） ----
function _element_phases_conn(etype::Symbol)
    nph = try
        OpenDSSDirect.CktElement.NumPhases()
    catch
        3
    end
    conn_code = try
        if etype == :load && hasproperty(OpenDSSDirect, :Loads)
            OpenDSSDirect.Loads.Conn()
        elseif etype == :gen && hasproperty(OpenDSSDirect, :Generators)
            OpenDSSDirect.Generators.Conn()
        elseif etype == :pv && hasproperty(OpenDSSDirect, :PVSystems)
            OpenDSSDirect.PVSystems.Conn()
        elseif etype == :storage && hasproperty(OpenDSSDirect, :Storage)
            OpenDSSDirect.Storage.Conn()
        elseif etype == :capacitor && hasproperty(OpenDSSDirect, :Capacitors)
            0
        elseif etype == :vsource
            0
        else
            0
        end
    catch
        0
    end
    return (nph, conn_code)
end

# ---- 修复版: 从 CktElement.Powers() 得到每相 P/Q（第一终端） ----
function _element_phase_powers(etype::Symbol; delta_method::Symbol=:equal, max_phases::Int=3)
    ep = OpenDSSDirect.CktElement.Powers()
    (nph, conn) = _element_phases_conn(etype)
    Pph = fill(NaN, max_phases)
    Qph = fill(NaN, max_phases)
    if ep === nothing || length(ep) == 0
        return Pph, Qph
    end

    rawP = Float64[]
    rawQ = Float64[]

    if ep isa AbstractVector{<:Complex}
        # 直接取前 nph 个复功率（默认第一终端相）
        m = min(nph, length(ep))
        for k in 1:m
            c = ep[k]
            push!(rawP, real(c))
            push!(rawQ, imag(c))
        end
    else
        # 典型格式: [P1,Q1,P2,Q2,...] (kW, kvar)
        lim = min(2nph, length(ep))
        i = 1
        while i <= lim-1
            p = ep[i]
            q = ep[i+1]
            if p isa Complex
                push!(rawP, real(p))
                push!(rawQ, imag(p))
            else
                # q 也可能是 Complex(少见)
                push!(rawP, float(p))
                push!(rawQ, q isa Complex ? imag(q) : float(q))
            end
            i += 2
        end
    end

    if isempty(rawP)
        return Pph, Qph
    end

    if conn == 0  # Wye
        for k in 1:min(length(rawP), max_phases)
            Pph[k] = rawP[k]
            Qph[k] = rawQ[k]
        end
    else  # Delta
        if nph == 3
            if delta_method == :pair_avg && length(rawP) == 3
                Pab, Pbc, Pca = rawP
                Qab, Qbc, Qca = rawQ
                Pph[1] = (Pab + Pca)/2
                Pph[2] = (Pab + Pbc)/2
                Pph[3] = (Pbc + Pca)/2
                Qph[1] = (Qab + Qca)/2
                Qph[2] = (Qab + Qbc)/2
                Qph[3] = (Qbc + Qca)/2
            else
                Ptot = sum(rawP)
                Qtot = sum(rawQ)
                eachP = Ptot/3
                eachQ = Qtot/3
                Pph[1] = eachP; Pph[2] = eachP; Pph[3] = eachP
                Qph[1] = eachQ; Qph[2] = eachQ; Qph[3] = eachQ
            end
        else
            m = min(length(rawP), max_phases)
            for k in 1:m
                Pph[k] = rawP[k]
                Qph[k] = rawQ[k]
            end
        end
    end
    return Pph, Qph
end

# ====================================================================================
# 主时间序列 (含电压/角度/注入 + 分相注入)
# ====================================================================================
function run_time_series_with_injection(npts::Int; dt_s::Float64=0.1,
                         sample_every::Int=10,
                         collect::Vector{Symbol}=[:voltage_bus, :total_power, :bus_injection],
                         save_phase::Bool=true,
                         max_phases::Int=3,
                         store_angle::Bool=true,
                         per_phase_injection::Bool=true,
                         delta_method::Symbol=:equal)

    dsscmd("Set mode=Yearly stepsize=$(dt_s)s number=1")

    buses = OpenDSSDirect.Circuit.AllBusNames()
    nb = length(buses)
    n_samples = Int(ceil(npts / sample_every))

    time_s = Vector{Float64}(undef, n_samples)

    vol_pu_avg = nothing
    vol_pu_phase = nothing
    vol_ang_phase = nothing
    if :voltage_bus in collect
        vol_pu_avg = Matrix{Float32}(undef, nb, n_samples)
        if save_phase
            vol_pu_phase = Array{Float32}(undef, nb, max_phases, n_samples)
            if store_angle
                vol_ang_phase = Array{Float32}(undef, nb, max_phases, n_samples)
            end
        end
    end

    totalP = Vector{Float64}()
    totalQ = Vector{Float64}()
    if :total_power in collect
        resize!(totalP, n_samples)
        resize!(totalQ, n_samples)
    end

    P_inj = nothing
    Q_inj = nothing
    P_inj_phase = nothing
    Q_inj_phase = nothing
    inj_elements = nothing
    phase_collect = per_phase_injection && (:bus_injection in collect)

    if :bus_injection in collect
        P_inj = Matrix{Float32}(undef, nb, n_samples)
        Q_inj = Matrix{Float32}(undef, nb, n_samples)
        if phase_collect
            P_inj_phase = Array{Float32}(undef, nb, max_phases, n_samples)
            Q_inj_phase = Array{Float32}(undef, nb, max_phases, n_samples)
        end
        inj_elements = prepare_injection_elements(buses)
        @info "注入元素数: $(length(inj_elements)), 分相记录: $(phase_collect)"
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
                                vol_pu_phase[bi, ph, sample_idx] = NaN32
                                if vol_ang_phase !== nothing
                                    vol_ang_phase[bi, ph, sample_idx] = NaN32
                                end
                            end
                        end
                    else
                        mags = abs.(phasors) ./ 1000.0
                        vbase_ph = base / sqrt(3)
                        pu_ph = mags ./ vbase_ph
                        vol_pu_avg[bi, sample_idx] = mean(pu_ph)
                        if vol_pu_phase !== nothing
                            @inbounds for ph in 1:max_phases
                                if ph <= np
                                    vol_pu_phase[bi, ph, sample_idx] = Float32(pu_ph[ph])
                                    if vol_ang_phase !== nothing
                                        vol_ang_phase[bi, ph, sample_idx] = Float32(angle(phasors[ph]))
                                    end
                                else
                                    vol_pu_phase[bi, ph, sample_idx] = NaN32
                                    if vol_ang_phase !== nothing
                                        vol_ang_phase[bi, ph, sample_idx] = NaN32
                                    end
                                end
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

            if :bus_injection in collect
                @inbounds for bi in 1:nb
                    P_inj[bi, sample_idx] = 0f0
                    Q_inj[bi, sample_idx] = 0f0
                    if phase_collect
                        for ph in 1:max_phases
                            P_inj_phase[bi, ph, sample_idx] = 0f0
                            Q_inj_phase[bi, ph, sample_idx] = 0f0
                        end
                    end
                end
                for (etype, nm, bi) in inj_elements
                    _activate_element(etype, nm)
                    (P_elem, Q_elem) = _element_total_PQ()
                    P_inj[bi, sample_idx] += Float32(-P_elem)
                    Q_inj[bi, sample_idx] += Float32(-Q_elem)
                    if phase_collect
                        Pph, Qph = _element_phase_powers(etype; delta_method=delta_method, max_phases=max_phases)
                        @inbounds for ph in 1:max_phases
                            if !isnan(Pph[ph])
                                P_inj_phase[bi, ph, sample_idx] += Float32(-Pph[ph])
                            end
                            if !isnan(Qph[ph])
                                Q_inj_phase[bi, ph, sample_idx] += Float32(-Qph[ph])
                            end
                        end
                    end
                end
            end
        end
    end

    result = Dict{String,Any}()
    result["time_s"] = time_s
    if vol_pu_avg !== nothing
        result["voltage_pu_matrix"] = (buses=buses, data=vol_pu_avg)
        if vol_pu_phase !== nothing
            result["voltage_pu_phase"] = (buses=buses, data=vol_pu_phase, phase_order="A,B,C")
        end
        if vol_ang_phase !== nothing
            result["voltage_phase_angle_rad"] = (buses=buses, data=vol_ang_phase, phase_order="A,B,C")
        end
    end
    if :total_power in collect
        result["total_power_MW_Mvar"] = (P=totalP, Q=totalQ)
    end
    if :bus_injection in collect
        inj_dict = Dict(
            "buses" => buses,
            "P_inj_kW" => P_inj,
            "Q_inj_kvar" => Q_inj,
            "definition" => "P_inj = generation - load (通过 -Σ 元件功率; 正=净发电)"
        )
        if phase_collect
            inj_dict_phase = Dict(
                "buses" => buses,
                "P_inj_phase_kW" => P_inj_phase,
                "Q_inj_phase_kvar" => Q_inj_phase,
                "definition" => "每相注入: 对每元件分配后求和; 符号=发电-负荷",
                "delta_method" => String(delta_method),
                "phase_order" => "A,B,C"
            )
            result["bus_injection_phase"] = inj_dict_phase
        end
        result["bus_injection"] = inj_dict
    end
    return result
end

# ====================================================================================
# 综合执行入口
# ====================================================================================
function time_series_ieee37(feeder_dir::String;
                            dt_s::Float64=0.1,
                            hours::Float64=24.0,
                            sample_every::Int=100,
                            pf_mode::Symbol=:constant,
                            collect::Vector{Symbol}=[:voltage_bus, :total_power, :bus_injection],
                            shape_dir::String="loadshapes",
                            save_phase::Bool=true,
                            store_angle::Bool=true,
                            per_phase_injection::Bool=true,
                            delta_method::Symbol=:equal)
    println("加载 IEEE 37 初始工况...")
    conv, info = load_ieee37(feeder_dir)
    println("初始收敛 = $(info["converged"]), 总功率(MW,Mvar) = $(info["total_power_MW_Mvar"])")

    println("生成并注册各负荷 LoadShape ...")
    df_shapes = build_load_shapes(dt_s=dt_s, hours=hours, pf_mode=pf_mode, shape_dir=shape_dir)
    println("已创建 $(nrow(df_shapes)) 个负荷曲线。")

    npts = Int(round(hours*3600/dt_s))
    println("启动时间序列仿真: 总步数=$npts, 步长=$(dt_s)s, 采样每 $sample_every 步 (≈保存 $(Int(ceil(npts/sample_every))) 条记录)")
    println("分相注入: $(per_phase_injection), delta_method=$(delta_method)")

    t0 = time()
    res_core = run_time_series_with_injection(npts;
        dt_s=dt_s,
        sample_every=sample_every,
        collect=collect,
        save_phase=save_phase,
        store_angle=store_angle,
        per_phase_injection=per_phase_injection,
        delta_method=delta_method)
    tel = time() - t0
    println("仿真完成，耗时 $(round(tel,digits=2)) 秒 (≈ $(round(tel/60,digits=2)) 分).")

    return Dict(
        "shapes"=>df_shapes,
        "result"=>res_core,
        "meta"=>Dict("dt_s"=>dt_s,
                     "hours"=>hours,
                     "sample_every"=>sample_every,
                     "npts"=>npts,
                     "collected"=>collect,
                     "per_phase_injection"=>per_phase_injection,
                     "delta_method"=>String(delta_method),
                     "timestamp"=>string(Dates.now()))
    )
end

# ====================================================================================
# 统一总线测量提取
# ====================================================================================
function get_bus_series(res::Dict, bus::String)
    r = res["result"]
    haskey(r, "voltage_pu_phase") || error("结果中没有 'voltage_pu_phase'")
    vtp = r["voltage_pu_phase"]
    buses = vtp.buses
    idx = findfirst(==(bus), buses)
    idx === nothing && error("母线 $bus 不存在")

    t = r["time_s"]
    vpu_ph = vtp.data
    ang_ph = haskey(r, "voltage_phase_angle_rad") ? r["voltage_phase_angle_rad"].data : nothing

    P_inj = nothing
    Q_inj = nothing
    if haskey(r, "bus_injection")
        inj = r["bus_injection"]
        buses_inj = inj["buses"]
        idx2 = findfirst(==(bus), buses_inj)
        idx2 === nothing && error("注入结构中缺失母线 $bus")
        P_inj = inj["P_inj_kW"][idx2, :]
        Q_inj = inj["Q_inj_kvar"][idx2, :]
    end

    ns = length(t)
    vpu_view = view(vpu_ph, idx, :, :)
    angle_mat = ang_ph === nothing ? fill(NaN32, 3, ns) : view(ang_ph, idx, :, :)

    mask = Array{Int8}(undef, 3, ns)
    for j in 1:ns
        for ph in 1:3
            mask[ph,j] = isnan(vpu_view[ph,j]) ? 0 : 1
        end
    end

    return Dict(
        "bus" => bus,
        "time_s" => t,
        "voltage_pu_phase" => vpu_view,
        "voltage_angle_rad" => angle_mat,
        "P_inj_kW" => (P_inj === nothing ? fill(NaN32, ns) : P_inj),
        "Q_inj_kvar" => (Q_inj === nothing ? fill(NaN32, ns) : Q_inj),
        "mask_phase" => mask,
        "angle_unit" => "radian",
        "definition_P_inj" => "P_inj = generation - load"
    )
end

function get_bus_phase_voltage_series(res::Dict, bus::String, phase::Int)
    phase in 1:3 || error("phase 必须 1~3")
    s = get_bus_series(res, bus)
    return Dict(
        "bus"=>bus,
        "phase"=>phase,
        "time_s"=>s["time_s"],
        "v_pu"=>s["voltage_pu_phase"][phase, :],
        "angle_rad"=>s["voltage_angle_rad"][phase, :],
        "P_inj_kW"=>s["P_inj_kW"],
        "Q_inj_kvar"=>s["Q_inj_kvar"]
    )
end

function get_bus_phase_injection(res::Dict, bus::String)
    r = res["result"]
    haskey(r, "bus_injection_phase") || error("结果中没有 'bus_injection_phase'")
    bip = r["bus_injection_phase"]
    buses = bip["buses"]
    idx = findfirst(==(bus), buses)
    idx === nothing && error("母线 $bus 不存在分相注入数据")
    t = r["time_s"]
    Pph = bip["P_inj_phase_kW"]
    Qph = bip["Q_inj_phase_kvar"]
    return Dict(
        "bus"=>bus,
        "time_s"=>t,
        "P_inj_phase_kW"=>view(Pph, idx, :, :),
        "Q_inj_phase_kvar"=>view(Qph, idx, :, :),
        "definition"=>bip["definition"],
        "delta_method"=>bip["delta_method"],
        "phase_order"=>bip["phase_order"]
    )
end

# ====================================================================================
# 长格式导出
# ====================================================================================
function bus_injection_long(res)
    r = res["result"]
    haskey(r,"bus_injection") || error("未收集 bus_injection")
    inj = r["bus_injection"]
    buses = inj["buses"]
    P = inj["P_inj_kW"]; Q = inj["Q_inj_kvar"]
    t = r["time_s"]
    nb, ns = size(P)
    rows = nb*ns
    df = DataFrame(Bus=Vector{String}(undef, rows),
                   Time_s=Vector{Float64}(undef, rows),
                   P_inj_kW=Vector{Float32}(undef, rows),
                   Q_inj_kvar=Vector{Float32}(undef, rows))
    k=0
    for j in 1:ns
        for i in 1:nb
            k+=1
            df.Bus[k] = buses[i]
            df.Time_s[k] = t[j]
            df.P_inj_kW[k] = P[i,j]
            df.Q_inj_kvar[k] = Q[i,j]
        end
    end
    return df
end

function bus_injection_phase_long(res)
    r = res["result"]
    haskey(r,"bus_injection_phase") || error("未收集 bus_injection_phase")
    bip = r["bus_injection_phase"]
    buses = bip["buses"]
    Pph = bip["P_inj_phase_kW"]; Qph = bip["Q_inj_phase_kvar"]
    t = r["time_s"]
    nb, np, ns = size(Pph)
    rows = nb*np*ns
    df = DataFrame(Bus=Vector{String}(undef, rows),
                   Phase=Vector{Int8}(undef, rows),
                   Time_s=Vector{Float64}(undef, rows),
                   P_inj_kW=Vector{Float32}(undef, rows),
                   Q_inj_kvar=Vector{Float32}(undef, rows))
    k=0
    for j in 1:ns
        for i in 1:nb
            for ph in 1:np
                k+=1
                df.Bus[k] = buses[i]
                df.Phase[k] = ph
                df.Time_s[k] = t[j]
                df.P_inj_kW[k] = Pph[i,ph,j]
                df.Q_inj_kvar[k] = Qph[i,ph,j]
            end
        end
    end
    return df
end

function bus_voltage_long(res)
    r = res["result"]
    haskey(r,"voltage_pu_phase") || error("未收集 voltage_pu_phase")
    vpu = r["voltage_pu_phase"]
    buses = vpu.buses
    data = vpu.data
    ang = haskey(r,"voltage_phase_angle_rad") ? r["voltage_phase_angle_rad"].data : nothing
    t = r["time_s"]
    nb, _, ns = size(data)
    rows = nb*3*ns
    df = DataFrame(Bus=Vector{String}(undef, rows),
                   Phase=Vector{Int8}(undef, rows),
                   Time_s=Vector{Float64}(undef, rows),
                   V_pu=Vector{Float32}(undef, rows),
                   Angle_rad=Vector{Float32}(undef, rows))
    k=0
    for j in 1:ns
        for i in 1:nb
            for ph in 1:3
                k+=1
                df.Bus[k] = buses[i]
                df.Phase[k] = ph
                df.Time_s[k] = t[j]
                df.V_pu[k] = data[i,ph,j]
                df.Angle_rad[k] = ang === nothing ? NaN32 : ang[i,ph,j]
            end
        end
    end
    return df
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
        results = time_series_ieee37(feeder_dir; dt_s=dt_s, hours=hours, sample_every=sample_every,
            per_phase_injection=true, delta_method=:equal)
        println("完成。示例: get_bus_phase_injection(results, \"701\")")
    else
        println("未提供 feeder_dir，脚本未执行主流程。用法示例：")
        println("  julia $(PROGRAM_FILE) \"D:/path/to/ieee37\" 0.1 24 100")
    end
end