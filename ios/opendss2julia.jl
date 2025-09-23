using OpenDSSDirect
using DataFrames
using Statistics
using LinearAlgebra
using Dates
using Logging

const dsscmd = OpenDSSDirect.dss

# ========== 通用 ==========
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

# 静默文本查询（避免 OpenDSSDirect.Text 内部 @warn 噪声）
function silent_text_query(cmd::String)
    Logging.with_logger(Logging.NullLogger()) do
        try
            return dsscmd(cmd)
        catch
            return ""
        end
    end
end

# 解析以空格 / 逗号 / 方括号分隔的数字列表
function parse_numeric_list(str::AbstractString)
    # 修复: 使用多个 Pair 参数而不是数组
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

# -------- 总功率解析 --------
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

# -------- 相量提取 --------
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

# -------- 简易显示 --------
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

# ========== 线路功率解析 ==========
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
                # 端1
                for t in 1:nterm, ph in 1:nph
                    c=pdata[k]; k+=1
                    p=real(c); q=imag(c)
                    push!(term_phase,(t,ph,p,q)); Pterm[t]+=p; Qterm[t]+=q
                end
                # 端2
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

# ========== 主求解 ==========
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

# ========== 母线电压 ==========
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

# ========== 线路功率 ==========
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

# ========== 负荷 ==========
# 全新版本：不使用 "? Load.xxx.bus1"
function get_load_bus(ld::String)
    # 很旧的 OpenDSSDirect 可能没有 Loads.BusName；采用选择后取 CktElement.BusNames()
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
        # base bus = 去掉相点编号
        basebus = split(rawbus, '.')[1]
        phs = try_default(() -> OpenDSSDirect.Loads.Phases(), 3)
        conncode = try_default(() -> OpenDSSDirect.Loads.Conn(), 0) # 0 Wye, 1 Delta
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

# ========== 调压器 ==========
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
                # 旧/新版本兼容
                tappos = try
                    # 可能支持 Transformers.Tap(wdg)
                    OpenDSSDirect.Transformers.Tap(wdg)
                catch
                    try
                        OpenDSSDirect.Transformers.Wdg(Float64(wdg))  # 修复: Float64(wdg)
                        OpenDSSDirect.Transformers.Tap()
                    catch
                        NaN
                    end
                end
            catch
                # ignore
            end
        end
        push!(df,(rg,tr,wdg,vreg,band,ptr,ctp,rv,xv,tapnum,tappos))
    end
    return df
end

# ========== 变压器 ==========
function _get_winding_kV(tr::String, w::Int)
    try
        # 修复: 将整数转换为浮点数
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
        # 修复: 将整数转换为浮点数
        OpenDSSDirect.Transformers.Wdg(Float64(w))
        return OpenDSSDirect.Transformers.kVA()
    catch
        s = silent_text_query("? Transformer.$tr.kvas")
        if isempty(s)
            s = silent_text_query("? Transformer.$tr.kVAs") # 大小写变体
        end
        vals = parse_numeric_list(s)
        return (1 <= w <= length(vals)) ? vals[w] : NaN
    end
end

function _get_winding_R(tr::String, w::Int)
    # 试 R() 失败再文本 r1 r2...
    v = try
        # 修复: 将整数转换为浮点数
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
        # 修复: 将整数转换为浮点数
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
        loss_kW = 0.0  # 初始化为实数
        if loss_vec !== nothing && length(loss_vec) >= 1
            # 确保取实部
            if loss_vec[1] isa Complex
                loss_kW = real(loss_vec[1])/1000
            else
                loss_kW = loss_vec[1]/1000
            end
        else
            # 备选：两端有功代数和绝对值
            if ep isa AbstractVector{<:Real} && length(ep) >= 4
                loss_kW = abs(ep[1] + ep[3])
            elseif ep isa AbstractVector{<:Complex} && length(ep) >= 2
                loss_kW = abs(real(ep[1]) + real(ep[2]))
            else
                loss_kW = NaN
            end
        end
        
        # 确保 loss_kW 是实数
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


# ========== 示例运行 ==========
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

# 使用示例:
feeder_dir = "D:/luosipeng/Deep_Learning_in_Distribution_System/data"
results = example_run(feeder_dir)
