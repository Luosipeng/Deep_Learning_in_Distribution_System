#############################
# IEEE 37 数据提取 - 简化版
# 需求：
#  - PMU (705) 只返回三相电压幅值 (0.1s, 24h)
#  - SCADA (702,703,730) 三相电压幅值 (1min, 24h)
#  - AMI 指定 (bus, phase) 的有功/无功 (15min, 24h)
#
# 使用：
#   res = time_series_ieee37(...)
#   ds = extract_requested_dataset(res)
#   访问:
#     ds[:PMU705][:times]        -> 0.1s 时间戳
#     ds[:PMU705][:Vmag][:A]     -> 705 A 相幅值
#     ds[:SCADA]["702"][:Vmag][:C] -> 702 C 相 1min 电压
#     ds[:AMI] 为 Vector{NamedTuple}
#############################
include("../ios/time_series_data_generation.jl")
const PHASES = (:A, :B, :C)
######### 工具函数 #########

"""
判断向量是否“平凡” (接近全 0 或方差很小)
返回 Bool
"""
function is_trivial_vector(v::AbstractVector;
                           zero_tol::Float32=1e-6f0,
                           var_tol::Float32=1e-8f0)
    if maximum(abs.(v)) < zero_tol
        return true
    end
    return var(v) < var_tol
end

######### PMU 提取（仅三相幅值） #########

"""
extract_pmu705_vmag(res) -> Dict
  返回:
    :times  => Vector{Float32} (0.1s)
    :Vmag   => Dict(:A=>Vector, :B=>Vector, :C=>Vector)
    :Vreal  => Dict(:A=>Vector, :B=>Vector, :C=>Vector)  # 新增：实部
    :Vimag  => Dict(:A=>Vector, :B=>Vector, :C=>Vector)  # 新增：虚部
"""
function extract_pmu705_vmag(res)
    s = get_bus_series(res, "705")
    Vm = Array(s["voltage_pu_phase"])   # (3,N)
    Va = Array(s["voltage_angle_rad"])  # (3,N)
    N = size(Vm, 2)
    times = Float32.(0:0.1:(N-1)*0.1)
    Vmag = Dict(PHASES[i] => Float32.(Vm[i, :]) for i in 1:3)
    Vreal = Dict(PHASES[i] => Float32.(Vm[i, :] .* cos.(Va[i, :])) for i in 1:3)
    Vimag = Dict(PHASES[i] => Float32.(Vm[i, :] .* sin.(Va[i, :])) for i in 1:3)
    return Dict(
        :times => times,
        :Vmag  => Vmag,
        :Vreal => Vreal,
        :Vimag => Vimag
    )
end

######### SCADA 提取（单个站点，聚合到 1 分钟） #########

"""
extract_scada_bus(res, bus::String)
  返回:
    :bus
    :times (60s)
    :Vmag  Dict(:A,:B,:C)
"""
function extract_scada_bus(res, bus::String)
    sb = get_bus_series(res, bus)
    vm_raw = Array(sb["voltage_pu_phase"])   # (3, raw_len) 0.1s
    raw_len = size(vm_raw, 2)
    @assert raw_len % 600 == 0 "SCADA 原始长度不是 600 的整数倍 (bus=$bus)"
    Ns = raw_len ÷ 600
    times = Float32.(0:60:(Ns-1)*60)
    vm_1min = vm_raw[:, 1:600:end]           # (3, Ns)
    Vmag = Dict(PHASES[i] => Float32.(vm_1min[i, :]) for i in 1:3)
    return Dict(
        :bus => bus,
        :times => times,
        :Vmag => Vmag
    )
end

######### AMI 提取（单个站点，聚合到 15 分钟） #########

"""
extract_ami_bus(res, bus::String)
  返回:
    :bus
    :times   (900s)
    :P_kW    Dict(:A,:B,:C)
    :Q_kvar  Dict(:A,:B,:C)

  若真实为单/双相，通常另两相可能全 0，可后续用 drop_trivial_ami_phases! 剔除。
"""
function extract_ami_bus(res, bus::String)
    sb = get_bus_phase_injection(res, bus)
    P_raw = Array(sb["P_inj_phase_kW"])       # (3, raw_len) 0.1s
    Q_raw = Array(sb["Q_inj_phase_kvar"])
    raw_len = size(P_raw, 2)
    @assert raw_len % 9000 == 0 "AMI 原始长度不是 9000 的整数倍 (bus=$bus)"
    Ns = raw_len ÷ 9000
    times = Float32.(0:900:(Ns-1)*900)
    P_15 = P_raw[:, 1:9000:end]
    Q_15 = Q_raw[:, 1:9000:end]
    Pdict = Dict(PHASES[i] => Float32.(P_15[i, :]) for i in 1:3)
    Qdict = Dict(PHASES[i] => Float32.(Q_15[i, :]) for i in 1:3)
    return Dict(
        :bus => bus,
        :times => times,
        :P_kW => Pdict,
        :Q_kvar => Qdict
    )
end

"""
批量提取多个 AMI 站点
ami_buses::Vector{String}
返回 Dict{String,Dict}
"""
function extract_ami_group(res, ami_buses::Vector{String})
    out = Dict{String,Dict}()
    for b in ami_buses
        out[b] = extract_ami_bus(res, b)
    end
    return out
end

######### 主统一提取入口 #########

"""
extract_requested_dataset(res;
    scada_buses = ["702","703","730"],
    ami_buses = ["701","744","728","729","736","727"])

返回:
  Dict(
    :PMU705 => Dict(:times,:Vmag)
    :SCADA  => Dict(bus=>Dict(:times,:Vmag), ...)
    :AMI    => Dict(bus=>Dict(:times,:P_kW,:Q_kvar), ...)
  )
"""
function extract_requested_dataset(res;
        scada_buses = ["702","703","730"],
        ami_buses = ["701","744","728","729","736","727"])

    pmu705 = extract_pmu705_vmag(res)

    scada_dict = Dict{String,Dict}()
    for b in scada_buses
        scada_dict[b] = extract_scada_bus(res, b)
    end

    ami_dict = extract_ami_group(res, ami_buses)

    return Dict(
        :PMU705 => pmu705,
        :SCADA  => scada_dict,
        :AMI    => ami_dict
    )
end

######### 辅助访问函数 #########

"""
get_pmu_vmag(ds, phase::Symbol)
  phase ∈ {:A,:B,:C}
"""
get_pmu_vmag(ds, phase::Symbol) = ds[:PMU705][:Vmag][phase]

"""
get_scada_vmag(ds, bus::String, phase::Symbol)
"""
function get_scada_vmag(ds, bus::String, phase::Symbol)
    return ds[:SCADA][bus][:Vmag][phase]
end

"""
get_ami_power(ds, bus::String, phase::Symbol) -> (P::Vector{Float32}, Q::Vector{Float32})
"""
function get_ami_power(ds, bus::String, phase::Symbol)
    bdict = ds[:AMI][bus]
    return bdict[:P_kW][phase], bdict[:Q_kvar][phase]
end

"""
subset_ami(ds; buses=nothing, phases=(:A,:B,:C)) -> Vector{NamedTuple}
  将 AMI 字典按条件展开为列表
"""
function subset_ami(ds; buses=nothing, phases=(:A,:B,:C))
    ami = ds[:AMI]
    sel_buses = buses === nothing ? collect(keys(ami)) : collect(buses)
    out = Vector{NamedTuple}(undef, 0)
    for b in sel_buses
        d = ami[b]
        times = d[:times]
        for ph in phases
            push!(out, (bus=b, phase=ph,
                        times=times,
                        P_kW=d[:P_kW][ph],
                        Q_kvar=d[:Q_kvar][ph]))
        end
    end
    return out
end

######### 可选：剔除 AMI 中“平凡”相 #########

"""
drop_trivial_ami_phases!(ds; zero_tol=1e-6f0, var_tol=1e-8f0)
  对 ds[:AMI][bus][:P_kW] / :Q_kvar 中检测平凡相 (P 和 Q 都 trivial) 则删除该相键
  返回被删除的 (bus, phase) 列表
"""
function drop_trivial_ami_phases!(ds; zero_tol=1e-6f0, var_tol=1e-8f0)
    removed = Vector{Tuple{String,Symbol}}()
    for (bus, d) in ds[:AMI]
        phases_to_delete = Symbol[]
        for ph in PHASES
            haskey(d[:P_kW], ph) || continue
            p = d[:P_kW][ph]
            q = d[:Q_kvar][ph]
            if is_trivial_vector(p; zero_tol=zero_tol, var_tol=var_tol) &&
               is_trivial_vector(q; zero_tol=zero_tol, var_tol=var_tol)
                push!(phases_to_delete, ph)
            end
        end
        for ph in phases_to_delete
            delete!(d[:P_kW], ph)
            delete!(d[:Q_kvar], ph)
            push!(removed, (bus, ph))
        end
    end
    return removed
end

