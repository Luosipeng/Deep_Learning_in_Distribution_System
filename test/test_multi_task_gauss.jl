# ============================================
# Includes & Packages
# ============================================

include("../src/get_sample_from_ieee37.jl")
include("../src/get_ieee37_multitask_data.jl")
include("../src/implement_data.jl")

using Flux
using LinearAlgebra
using Plots
using Statistics
using Random
using ProgressMeter
using DataFrames
using JLD2
using CSV

# Plot defaults
gr(fontfamily="Arial", legendfontsize=7, guidefontsize=9, titlefontsize=9)

# ============================================
# Data Structures
# ============================================

struct MultiSensorData
    S::Int
    times::Vector{Vector{Float32}}
    values::Vector{Vector{Float32}}
    sensor_names::Vector{String}
    sensor_types::Vector{Symbol}
end

struct NormalizationParams
    x_mean::Float32
    x_std::Float32
    y_means::Vector{Float32}
    y_stds::Vector{Float32}
end

# ============================================
# Helpers
# ============================================

# 纯函数：返回在对角线上加 d 的新矩阵副本（避免就地修改与 UniformScaling）
function add_diag_copy(A::AbstractMatrix{T}, d::T) where {T<:Real}
    Matrix{T}(A) + d * I
end

# Smart downsample
function smart_downsample(times::Vector, values::Vector, max_points::Int=500)
    n = length(times)
    if n <= max_points
        return Float32.(times), Float32.(values)
    end
    step = max(1, n ÷ max_points)
    idx = 1:step:n
    return Float32.(times[idx]), Float32.(values[idx])
end

# ============================================
# Build Dataset
# ============================================

function build_complete_multisensor_data(ds; max_points_per_sensor::Int=300)
    times_list = Vector{Vector{Float32}}()
    values_list = Vector{Vector{Float32}}()
    names_list = String[]
    types_list = Symbol[]
    
    println("="^70)
    println("Building Complete Multi-Sensor Dataset")
    println("="^70)
    
    # Sensors (可按需调整)
    scada_sensors = [
        ("702", [:A, :B, :C], :Vmag),
        ("703", [:A, :B, :C], :Vmag),
        ("730", [:A, :B, :C], :Vmag),
    ]
    ami_sensors = [
        ("701", [:A, :B, :C], :P_kW),
        ("701", [:A, :B, :C], :Q_kvar),
        ("744", [:A], :P_kW),
        ("744", [:A], :Q_kvar),
        ("728", [:A, :B, :C], :P_kW),
        ("728", [:A, :B, :C], :Q_kvar),
        ("729", [:A], :P_kW),
        ("729", [:A], :Q_kvar),
        ("736", [:B], :P_kW),
        ("736", [:B], :Q_kvar),
        ("727", [:C], :P_kW),
        ("727", [:C], :Q_kvar),
    ]
    
    println("\n[SCADA Sensors]")
    scada_count = 0
    for (bus, phases, measurement) in scada_sensors
        if !haskey(ds[:SCADA], bus)
            println("  ⚠️  Skip SCADA-$bus (not found)")
            continue
        end
        for ph in phases
            if !haskey(ds[:SCADA][bus][measurement], ph); continue; end
            t_raw = ds[:SCADA][bus][:times]
            v_raw = ds[:SCADA][bus][measurement][ph]
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip SCADA-$bus-$ph-$measurement (invalid data)")
                continue
            end
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, "SCADA-$bus-$ph-$(measurement == :Vmag ? "Vmag" : string(measurement))")
            push!(types_list, :SCADA)
            scada_count += 1
            println("  ✓ SCADA-$bus-$ph-$(measurement): $(length(v)) pts (range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total SCADA sensors: $scada_count")
    
    println("\n[AMI Sensors]")
    ami_count = 0
    for (bus, phases, measurement) in ami_sensors
        if !haskey(ds[:AMI], bus)
            println("  ⚠️  Skip AMI-$bus (not found)")
            continue
        end
        for ph in phases
            if !haskey(ds[:AMI][bus][measurement], ph); continue; end
            t_raw = ds[:AMI][bus][:times]
            v_raw = ds[:AMI][bus][measurement][ph]
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip AMI-$bus-$ph-$measurement (invalid data)")
                continue
            end
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, "AMI-$bus-$ph-$(measurement == :P_kW ? "P" : "Q")")
            push!(types_list, :AMI)
            ami_count += 1
            println("  ✓ AMI-$bus-$ph-$(measurement): $(length(v)) pts (range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total AMI sensors: $ami_count")
    
    println("\n[PMU Sensors]")
    pmu_count = 0
    if haskey(ds, :PMU705)
        for ph in [:A, :B, :C]
            if !haskey(ds[:PMU705][:Vmag], ph); continue; end
            t_raw = ds[:PMU705][:times]
            v_raw = ds[:PMU705][:Vmag][ph]
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip PMU-705-$ph (invalid data)")
                continue
            end
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, "PMU-705-$ph-Vmag")
            push!(types_list, :PMU)
            pmu_count += 1
            println("  ✓ PMU-705-$ph-Vmag: $(length(v)) pts (range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total PMU sensors: $pmu_count")
    
    S = length(times_list)
    total_points = sum(length.(values_list))
    
    println("\n" * "="^70)
    println("Dataset Built Successfully")
    println("="^70)
    println("  Total sensors: $S")
    println("    - SCADA: $scada_count")
    println("    - AMI: $ami_count")
    println("    - PMU: $pmu_count")
    println("  Total data points: $total_points")
    println("  Estimated memory: ~$(round(total_points * 8 / 1024^2, digits=2)) MB")
    println("="^70)
    
    return MultiSensorData(S, times_list, values_list, names_list, types_list)
end

function normalize_multisensor_data(data::MultiSensorData)
    all_times = vcat(data.times...)
    x_mean = Float32(mean(all_times))
    x_std = Float32(std(all_times))
    y_means = Float32[]
    y_stds = Float32[]
    times_norm = Vector{Vector{Float32}}()
    values_norm = Vector{Vector{Float32}}()
    for s in 1:data.S
        t_norm = (data.times[s] .- x_mean) ./ x_std
        push!(times_norm, t_norm)
        y_mean = Float32(mean(data.values[s]))
        y_std = Float32(std(data.values[s]))
        v_norm = (data.values[s] .- y_mean) ./ y_std
        push!(values_norm, v_norm)
        push!(y_means, y_mean)
        push!(y_stds, y_std)
    end
    norm_params = NormalizationParams(x_mean, x_std, y_means, y_stds)
    norm_data = MultiSensorData(data.S, times_norm, values_norm, data.sensor_names, data.sensor_types)
    return norm_data, norm_params
end

# ============================================
# ICM/LMC MTGP
# ============================================

# Shared mean network
function create_shared_mean_network(input_dim=1, hidden_dim=32)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 1)
    )
end

# Task correlation matrix B = L * L'
function build_task_correlation_matrix(L_params::Matrix{Float32})
    L = LowerTriangular(L_params)
    Matrix(L * L')
end

# RBF kernel
function rbf_kernel(t1::Vector{Float32}, t2::Vector{Float32}, σ::Float32, ℓ::Float32)
    Δ = t1 .- t2'
    @. (σ^2) * exp( - (Δ^2) / (2f0 * ℓ^2) )
end

# Construct joint covariance K_all and add per-task noise inside construction (non-inplace)
# K((s,x),(t,x′)) = B_{st} k_time(x,x′) + δ_{st} k_local_s(x,x′) + δ_{st} σ_noise[s]^2 I
function construct_joint_K_all(times::Vector{Vector{Float32}},
                               B::Matrix{Float32},
                               σ_time::Float32, ℓ_time::Float32,
                               σ_locals::Vector{Float32}, ℓ_locals::Vector{Float32},
                               σ_noise::Vector{Float32}, jitter::Float32)
    S = length(times)
    row_blocks = map(1:S) do s
        reduce(hcat, map(1:S) do t
            K_block = B[s, t] .* rbf_kernel(times[s], times[t], σ_time, ℓ_time)
            if s == t
                K_block = K_block .+ rbf_kernel(times[s], times[t], σ_locals[s], ℓ_locals[s])
                K_block = add_diag_copy(K_block, σ_noise[s]^2 + jitter)
            end
            K_block
        end)
    end
    reduce(vcat, row_blocks)
end

# Build joint dataset in normalized space
function build_joint_dataset(data::MultiSensorData)
    norm_data, norm_params = normalize_multisensor_data(data)
    x_all = vcat(norm_data.times...)
    y_all = vcat(norm_data.values...)
    n_per_task = length.(norm_data.times)
    offsets = cumsum(vcat(0, n_per_task[1:end-1]))
    return (; norm_data, norm_params, x_all, y_all, n_per_task, offsets)
end

# Mean forward
mean_forward(mean_func, x_all::Vector{Float32}) = vec(mean_func(reshape(x_all, 1, :)))

# Joint NLL
function joint_nll_icm(params, joint_pack; jitter::Float32=1f-5)
    mean_func    = params.mean_func
    L_params     = params.L_params
    log_σ_time   = params.log_σ_time
    log_ℓ_time   = params.log_ℓ_time
    log_σ_locals = params.log_σ_locals
    log_ℓ_locals = params.log_ℓ_locals
    log_σ_noise  = params.log_σ_noise

    norm_data   = joint_pack.norm_data
    x_all       = joint_pack.x_all
    y_all       = joint_pack.y_all

    S = norm_data.S

    B = build_task_correlation_matrix(L_params)
    σ_time = exp(log_σ_time[])
    ℓ_time = exp(log_ℓ_time[])
    σ_locals = exp.(log_σ_locals)
    ℓ_locals = exp.(log_ℓ_locals)
    σ_noise  = exp.(log_σ_noise)

    # Build K_all with noise inside (no in-place modifications)
    K_all = construct_joint_K_all(norm_data.times, B, σ_time, ℓ_time, σ_locals, ℓ_locals, σ_noise, jitter)

    m_all = mean_forward(mean_func, x_all)
    r = y_all .- m_all

    K_sym = Hermitian(0.5f0 .* (K_all .+ K_all'))
    L = cholesky(K_sym).L
    α = L' \ (L \ r)
    N = length(y_all)

    0.5f0 * dot(r, α) + sum(log.(diag(L))) + 0.5f0 * N * log(2f0 * Float32(pi))
end

# Train ICM MTGP
function train_icm_mtgp(data::MultiSensorData; num_epochs::Int=200, lr::Float64=0.01, verbose::Bool=true)
    joint_pack = build_joint_dataset(data)
    S = data.S

    mean_func = create_shared_mean_network(1, 32)

    # Initialize params
    L_params = Matrix{Float32}(I, S, S)          # B = I initially
    log_σ_time = Float32[log(1.0)]
    log_ℓ_time = Float32[log(0.5)]
    log_σ_locals = fill(Float32(log(0.3)), S)
    log_ℓ_locals = fill(Float32(log(0.3)), S)
    log_σ_noise  = fill(Float32(log(0.1)), S)

    params = (
        mean_func = mean_func,
        L_params = L_params,
        log_σ_time = log_σ_time,
        log_ℓ_time = log_ℓ_time,
        log_σ_locals = log_σ_locals,
        log_ℓ_locals = log_ℓ_locals,
        log_σ_noise = log_σ_noise
    )

    ps = Flux.params(mean_func, L_params, log_σ_time, log_ℓ_time, log_σ_locals, log_ℓ_locals, log_σ_noise)
    opt = Flux.Adam(lr)

    losses = Float32[]
    best = Inf32
    patience = 30
    stall = 0

    println("\n" * "="^70)
    println("Training ICM/LMC Multi-task GP (Joint LML)")
    println("="^70)

    @showprogress for epoch in 1:num_epochs
        gs = Flux.gradient(ps) do
            joint_nll_icm(params, joint_pack; jitter=1f-5)
        end
        Flux.update!(opt, ps, gs)
        loss = joint_nll_icm(params, joint_pack; jitter=1f-5)
        push!(losses, loss)

        if verbose && (epoch % 10 == 0 || epoch == 1)
            println("Epoch $epoch, NLL = $(round(loss, digits=4))")
        end

        if loss + 1e-5 < best
            best = loss
            stall = 0
        else
            stall += 1
            if stall >= patience && epoch > 50
                verbose && println("Early stopping at epoch $epoch")
                break
            end
        end
    end

    # For reporting in original scale (optional)
    σ_time_final = exp(log_σ_time[]) * mean(joint_pack.norm_params.y_stds)
    ℓ_time_final = exp(log_ℓ_time[]) * joint_pack.norm_params.x_std
    σ_locals_final = exp.(log_σ_locals) .* joint_pack.norm_params.y_stds
    ℓ_locals_final = exp.(log_ℓ_locals) .* joint_pack.norm_params.x_std
    σ_noise_final  = exp.(log_σ_noise)  .* joint_pack.norm_params.y_stds

    return (
        mean_func = mean_func,
        L_params = L_params,
        log_σ_time = log_σ_time,
        log_ℓ_time = log_ℓ_time,
        log_σ_locals = log_σ_locals,
        log_ℓ_locals = log_ℓ_locals,
        log_σ_noise  = log_σ_noise,
        σ_time = σ_time_final,
        ℓ_time = ℓ_time_final,
        σ_locals = σ_locals_final,
        ℓ_locals = ℓ_locals_final,
        σ_noise = σ_noise_final,
        losses = losses,
        norm_params = joint_pack.norm_params,
        joint_pack = joint_pack,
        data = data
    )
end

# Predict jointly for a given task s
function icm_predict(result, s::Int, x_test::Vector{Float32})
    data = result.data
    S = data.S
    normp = result.norm_params

    x_test_norm = (x_test .- normp.x_mean) ./ normp.x_std

    joint_pack = result.joint_pack
    x_all = joint_pack.x_all
    y_all = joint_pack.y_all

    B = build_task_correlation_matrix(result.L_params)

    # Predict in normalized space for stability
    σ_time = exp(result.log_σ_time[])
    ℓ_time = exp(result.log_ℓ_time[])
    σ_locals = exp.(result.log_σ_locals)
    ℓ_locals = exp.(result.log_ℓ_locals)
    σ_noise  = exp.(result.log_σ_noise)

    # K_all with noise inside construction (same as training)
    K_all = construct_joint_K_all(joint_pack.norm_data.times, B, σ_time, ℓ_time, σ_locals, ℓ_locals, σ_noise, 1f-6)
    K_sym = Hermitian(0.5f0 .* (K_all .+ K_all'))
    L = cholesky(K_sym).L

    # K_x_star: N x n*
    n_star = length(x_test_norm)
    K_x_star = zeros(Float32, length(x_all), n_star)

    offsets = joint_pack.offsets
    n_per_task = joint_pack.n_per_task

    for i in 1:S
        xi = joint_pack.norm_data.times[i]
        K_time_is = rbf_kernel(xi, x_test_norm, σ_time, ℓ_time)
        K_local = (i == s) ? rbf_kernel(xi, x_test_norm, σ_locals[s], ℓ_locals[s]) :
                             zeros(Float32, size(K_time_is))
        block = B[i,s] .* K_time_is .+ K_local
        os = offsets[i]; ns = n_per_task[i]
        @views K_x_star[os+1:os+ns, :] .= block
    end

    m_all = mean_forward(result.mean_func, x_all)
    r = y_all .- m_all
    α = L' \ (L \ r)

    m_star = vec(result.mean_func(reshape(x_test_norm, 1, :)))
    μ_star_norm = m_star .+ K_x_star' * α

    v = L \ K_x_star
    Kss = B[s,s] .* rbf_kernel(x_test_norm, x_test_norm, σ_time, ℓ_time) .+
          rbf_kernel(x_test_norm, x_test_norm, σ_locals[s], ℓ_locals[s])
    cov_star = Kss .- v' * v
    σ_star_norm = sqrt.(max.(diag(cov_star), 1f-8))

    μ_star = μ_star_norm .* joint_pack.norm_params.y_stds[s] .+ joint_pack.norm_params.y_means[s]
    σ_star = σ_star_norm .* joint_pack.norm_params.y_stds[s]
    return μ_star, σ_star
end

# ============================================
# Main
# ============================================

function main_multitask()
    Random.seed!(42)
    
    println("="^70)
    println("Multi-task Gaussian Process (MTGP) - ICM/LMC Joint Version (No in-place)")
    println("="^70)
    
    println("\n[1] Loading data...")
    feeder_dir = "D:/luosipeng/Deep_Learning_in_Distribution_System/data"  # adjust to your path
    res = time_series_ieee37(
        feeder_dir;
        dt_s=0.1,
        hours=24.0,
        sample_every=1,
        collect=[:voltage_bus, :total_power, :bus_injection],
        extract_ymatrix=true
    )
    ds = extract_requested_dataset(res)
    
    println("\n[2] Building complete multi-sensor dataset...")
    data = build_complete_multisensor_data(ds; max_points_per_sensor = 300)
    
    println("\n[3] Training multi-task GP (ICM/LMC)...")
    result = train_icm_mtgp(data; num_epochs = 200, lr = 0.01, verbose = true)
    
    println("\n[4] Final hyperparameters (reported in original scale):")
    println("="^70)
    println("Global time kernel:")
    println("  σ_time = $(round(result.σ_time, digits=4))")
    println("  ℓ_time = $(round(result.ℓ_time, digits=4)) hours")
    println("\nLocal per-sensor kernel (top 10 by σ_locals/σ_noise):")
    
    snrs = (result.σ_locals) ./ (result.σ_noise .+ 1e-12)
    sorted_indices = sortperm(snrs, rev=true)
    for i in 1:min(10, data.S)
        s = sorted_indices[i]
        println("  $(data.sensor_names[s]):")
        println("    σ_local = $(round(result.σ_locals[s], digits=4))")
        println("    ℓ_local = $(round(result.ℓ_locals[s], digits=4)) hours")
        println("    σ_noise = $(round(result.σ_noise[s], digits=4))")
        println("    SNR = $(round(snrs[s], digits=2))")
    end
    
    println("\n[5] Visualizations...")
    scada_indices = findall(x -> x == :SCADA, data.sensor_types)
    ami_indices = findall(x -> x == :AMI, data.sensor_types)
    pmu_indices = findall(x -> x == :PMU, data.sensor_types)
    
    selected_indices = Int[]
    if length(scada_indices) >= 2
        append!(selected_indices, scada_indices[1:2])
    end
    if length(ami_indices) >= 2
        append!(selected_indices, ami_indices[1:2])
    end
    if length(pmu_indices) >= 1
        push!(selected_indices, pmu_indices[1])
    end
    selected_indices = selected_indices[1:min(9, length(selected_indices))]
    
    plots_list = []
    for s in selected_indices
        x_test = range(minimum(data.times[s]), maximum(data.times[s]), length=200)
        μ_pred, σ_pred = icm_predict(result, s, collect(Float32, x_test))
        p = plot(x_test, μ_pred,
                 ribbon = 1.96 .* σ_pred,
                 label = "Pred (95% CI)",
                 xlabel = "Time (hours)",
                 ylabel = "Value",
                 title = data.sensor_names[s],
                 linewidth = 2,
                 fillalpha = 0.3,
                 legend = :topright,
                 size = (500, 350),
                 margin = 4Plots.mm,
                 titlefontsize = 8)
        scatter!(p, data.times[s], data.values[s],
                 label = "Data",
                 markersize = 1.5,
                 alpha = 0.6,
                 color = :red)
        push!(plots_list, p)
    end
    
    n_plots = length(plots_list)
    if n_plots > 0
        layout = (ceil(Int, n_plots/3), 3)
        combined = plot(plots_list..., 
                        layout = layout, 
                        size = (1400, 320 * layout[1]),
                        margin = 6Plots.mm)
        display(combined)
        savefig(combined, "mtgp_icm_predictions.png")
        println("  ✓ Saved: mtgp_icm_predictions.png")
    end
    
    valid_losses = filter(x -> isfinite(x) && x > 0, result.losses)
    if !isempty(valid_losses)
        p_loss = plot(valid_losses, 
                      xlabel = "Epoch", 
                      ylabel = "Total NLL",
                      title = "Training Loss (Joint LML)", 
                      linewidth = 2, 
                      legend = false,
                      yscale = :log10, 
                      size = (700, 450),
                      margin = 5Plots.mm)
        display(p_loss)
        savefig(p_loss, "mtgp_icm_loss.png")
        println("  ✓ Saved: mtgp_icm_loss.png")
    end
    
    if data.S > 0
        sorted_indices = sortperm(snrs, rev=true)
        colors = [data.sensor_types[i] == :SCADA ? :blue : 
                  (data.sensor_types[i] == :AMI ? :green : :red) 
                  for i in sorted_indices]
        p_snr = bar(1:data.S, snrs[sorted_indices],
                    xlabel = "Sensor Index (sorted by SNR)",
                    ylabel = "Signal-to-Noise Ratio",
                    title = "SNR by Sensor (Blue=SCADA, Green=AMI, Red=PMU)",
                    legend = false,
                    color = colors,
                    size = (1000, 500),
                    margin = 5Plots.mm,
                    xticks = (1:5:data.S, string.(1:5:data.S)))
        display(p_snr)
        savefig(p_snr, "mtgp_icm_snr.png")
        println("  ✓ Saved: mtgp_icm_snr.png")
    end
    
    println("\n[6] Summary:")
    println("="^70)
    println("  Mean SNR: $(round(mean(snrs), digits=2))")
    println("  Median SNR: $(round(median(snrs), digits=2))")
    println("  Max SNR: $(round(maximum(snrs), digits=2)) ($(data.sensor_names[argmax(snrs)]))")
    println("  Min SNR: $(round(minimum(snrs), digits=2)) ($(data.sensor_names[argmin(snrs)]))")
    println("  Global ℓ_time: $(round(result.ℓ_time, digits=4)) hours")
    println("  Mean ℓ_local: $(round(mean(result.ℓ_locals), digits=4)) hours")
    println("  Mean σ_noise: $(round(mean(result.σ_noise), digits=4))")
    
    println("\n" * "="^70)
    println("Complete!")
    println("="^70)
    
    return result
end

# ============================================
# Entry
# ============================================

println("\n" * "="^70)
println("Starting Multi-task GP (ICM/LMC) with Full Sensor Suite - No In-place")
println("="^70)

result = main_multitask()

if result !== nothing
unified_result = generate_1min_resolution_predictions(result)
end
