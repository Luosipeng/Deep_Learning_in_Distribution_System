#############################
# Multi-task Gaussian Process - Full Sensor Version
#############################

include("../src/get_sample_from_ieee37.jl")
include("../src/get_ieee37_multitask_data.jl")

using Flux
using LinearAlgebra
using Plots
using Statistics
using Random
using ProgressMeter

# Set plot defaults
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
# Data Preprocessing
# ============================================

function smart_downsample(times::Vector, values::Vector, max_points::Int=500)
    n = length(times)
    if n <= max_points
        return Float32.(times), Float32.(values)
    end
    step = n ÷ max_points
    indices = 1:step:n
    return Float32.(times[indices]), Float32.(values[indices])
end

"""
Build complete multi-sensor dataset with all available sensors
"""
function build_complete_multisensor_data(ds; max_points_per_sensor::Int=300)
    
    times_list = Vector{Vector{Float32}}()
    values_list = Vector{Vector{Float32}}()
    names_list = String[]
    types_list = Symbol[]
    
    println("="^70)
    println("Building Complete Multi-Sensor Dataset")
    println("="^70)
    
    # Define all sensors based on your specification
    scada_sensors = [
        # Bus 702: A, B, C phase voltage magnitude
        ("702", [:A, :B, :C], :Vmag),
        # Bus 703: A, B, C phase voltage magnitude
        ("703", [:A, :B, :C], :Vmag),
        # Bus 730: A, B, C phase voltage magnitude
        ("730", [:A, :B, :C], :Vmag),
    ]
    
    ami_sensors = [
        # Bus 701: A, B, C phase active and reactive power
        ("701", [:A, :B, :C], :P_kW),
        ("701", [:A, :B, :C], :Q_kvar),
        # Bus 744: A phase active and reactive power
        ("744", [:A], :P_kW),
        ("744", [:A], :Q_kvar),
        # Bus 728: A, B, C phase active and reactive power
        ("728", [:A, :B, :C], :P_kW),
        ("728", [:A, :B, :C], :Q_kvar),
        # Bus 729: A phase active and reactive power
        ("729", [:A], :P_kW),
        ("729", [:A], :Q_kvar),
        # Bus 736: B phase active and reactive power
        ("736", [:B], :P_kW),
        ("736", [:B], :Q_kvar),
        # Bus 727: C phase active and reactive power
        ("727", [:C], :P_kW),
        ("727", [:C], :Q_kvar),
    ]
    
    # Process SCADA sensors
    println("\n[SCADA Sensors]")
    scada_count = 0
    for (bus, phases, measurement) in scada_sensors
        if !haskey(ds[:SCADA], bus)
            println("  ⚠️  Skip SCADA-$bus (not found)")
            continue
        end
        
        for phase in phases
            if !haskey(ds[:SCADA][bus][measurement], phase)
                continue
            end
            
            t_raw = ds[:SCADA][bus][:times]
            v_raw = ds[:SCADA][bus][measurement][phase]
            
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip SCADA-$bus-$phase-$measurement (invalid data)")
                continue
            end
            
            push!(times_list, t)
            push!(values_list, v)
            
            # Create descriptive name
            meas_name = measurement == :Vmag ? "Vmag" : string(measurement)
            push!(names_list, "SCADA-$bus-$phase-$meas_name")
            push!(types_list, :SCADA)
            
            scada_count += 1
            println("  ✓ SCADA-$bus-$phase-$meas_name: $(length(v)) points " *
                    "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total SCADA sensors: $scada_count")
    
    # Process AMI sensors
    println("\n[AMI Sensors]")
    ami_count = 0
    for (bus, phases, measurement) in ami_sensors
        if !haskey(ds[:AMI], bus)
            println("  ⚠️  Skip AMI-$bus (not found)")
            continue
        end
        
        for phase in phases
            if !haskey(ds[:AMI][bus][measurement], phase)
                continue
            end
            
            t_raw = ds[:AMI][bus][:times]
            v_raw = ds[:AMI][bus][measurement][phase]
            
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip AMI-$bus-$phase-$measurement (invalid data)")
                continue
            end
            
            push!(times_list, t)
            push!(values_list, v)
            
            # Create descriptive name
            meas_name = measurement == :P_kW ? "P" : "Q"
            push!(names_list, "AMI-$bus-$phase-$meas_name")
            push!(types_list, :AMI)
            
            ami_count += 1
            println("  ✓ AMI-$bus-$phase-$meas_name: $(length(v)) points " *
                    "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
        end
    end
    println("  Total AMI sensors: $ami_count")
    
    # Process PMU sensor (if available)
    println("\n[PMU Sensors]")
    pmu_count = 0
    if haskey(ds, :PMU705)
        for phase in [:A, :B, :C]
            if !haskey(ds[:PMU705][:Vmag], phase)
                continue
            end
            
            t_raw = ds[:PMU705][:times]
            v_raw = ds[:PMU705][:Vmag][phase]
            
            t_hours = t_raw ./ 3600.0
            t, v = smart_downsample(t_hours, v_raw, max_points_per_sensor)
            
            if std(v) < 1e-6 || !all(isfinite.(v))
                println("  ⚠️  Skip PMU-705-$phase (invalid data)")
                continue
            end
            
            push!(times_list, t)
            push!(values_list, v)
            push!(names_list, "PMU-705-$phase-Vmag")
            push!(types_list, :PMU)
            
            pmu_count += 1
            println("  ✓ PMU-705-$phase-Vmag: $(length(v)) points (original: $(length(v_raw))) " *
                    "(range: $(round(minimum(v), digits=2)) ~ $(round(maximum(v), digits=2)))")
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
    norm_data = MultiSensorData(data.S, times_norm, values_norm, 
                                 data.sensor_names, data.sensor_types)
    
    return norm_data, norm_params
end

# ============================================
# Kernel Functions
# ============================================

function compute_multitask_kernel(times_s::Vector{Float32},
                                   times_t::Vector{Float32},
                                   s::Int, t::Int,
                                   σ_g::Float32, ℓ_g::Float32,
                                   σ_s::Vector{Float32}, ℓ_s::Vector{Float32})
    
    Δt = times_s .- times_t'
    K_global = σ_g^2 .* exp.(-Δt.^2 ./ (2 * ℓ_g^2))
    
    if s == t
        K_local = σ_s[s]^2 .* exp.(-Δt.^2 ./ (2 * ℓ_s[s]^2))
        return K_global .+ K_local
    else
        return K_global
    end
end

# ============================================
# Shared Mean Function
# ============================================

function create_shared_mean_network(input_dim=1, hidden_dim=32)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 1)
    )
end

# ============================================
# Training Functions
# ============================================

function compute_sensor_nll(times_s::Vector{Float32},
                            values_s::Vector{Float32},
                            s::Int,
                            mean_func,
                            σ_g::Float32, ℓ_g::Float32,
                            σ_s::Vector{Float32}, ℓ_s::Vector{Float32},
                            σ_noise::Vector{Float32};
                            jitter::Float32=1.0f-4)
    n = length(times_s)
    
    m = vec(mean_func(reshape(times_s, 1, :)))
    K = compute_multitask_kernel(times_s, times_s, s, s,
                                  σ_g, ℓ_g, σ_s, ℓ_s)
    K_noisy = K + (σ_noise[s]^2 + jitter) * I
    K_noisy = 0.5f0 .* (K_noisy .+ K_noisy')
    
    residual = values_s .- m
    L = cholesky(Hermitian(K_noisy)).L
    α = L' \ (L \ residual)
    
    nll = 0.5f0 * dot(residual, α) + 
          sum(log.(diag(L))) + 
          0.5f0 * n * log(2.0f0 * Float32(π))
    
    return nll
end

function train_multitask_gp(data::MultiSensorData;
                             num_epochs::Int=100,
                             lr::Float64=0.005,
                             verbose::Bool=true,
                             jitter::Float32=1.0f-4)
    
    S = data.S
    
    if verbose
        println("\nNormalizing data...")
    end
    norm_data, norm_params = normalize_multisensor_data(data)
    
    mean_func = create_shared_mean_network(1, 32)
    
    log_σ_g = Float32[log(1.0)]
    log_ℓ_g = Float32[log(0.5)]
    log_σ_s = Float32[log(0.3) for _ in 1:S]
    log_ℓ_s = Float32[log(0.3) for _ in 1:S]
    log_σ_noise = Float32[log(0.1) for _ in 1:S]
    
    if verbose
        println("\nInitial hyperparameters:")
        println("  Global: σ_g=$(round(exp(log_σ_g[1]), digits=4)), " *
                "ℓ_g=$(round(exp(log_ℓ_g[1]), digits=4))")
        println("  Jitter: $jitter")
        println("  Number of sensors: $S")
    end
    
    ps = Flux.params(mean_func, log_σ_g, log_ℓ_g, 
                     log_σ_s, log_ℓ_s, log_σ_noise)
    opt = Flux.Adam(lr)
    
    losses = Float32[]
    σ_g_history = Float32[]
    ℓ_g_history = Float32[]
    
    best_loss = Inf32
    patience = 20
    patience_counter = 0
    
    println("\n" * "="^70)
    println("Training Multi-task Gaussian Process")
    println("="^70)
    
    @showprogress for epoch in 1:num_epochs
        local total_loss
        
        try
            gs = Flux.gradient(ps) do
                σ_g = exp(log_σ_g[1])
                ℓ_g = exp(log_ℓ_g[1])
                σ_s = exp.(log_σ_s)
                ℓ_s = exp.(log_ℓ_s)
                σ_noise = exp.(log_σ_noise)
                
                total_nll = 0.0f0
                for s in 1:S
                    nll_s = compute_sensor_nll(
                        norm_data.times[s],
                        norm_data.values[s],
                        s, mean_func,
                        σ_g, ℓ_g, σ_s, ℓ_s, σ_noise;
                        jitter=jitter
                    )
                    total_nll += nll_s
                end
                
                total_loss = total_nll
                return total_loss
            end
            
            if !isnan(total_loss) && !isinf(total_loss) && total_loss > 0
                Flux.update!(opt, ps, gs)
                push!(losses, total_loss)
                push!(σ_g_history, exp(log_σ_g[1]))
                push!(ℓ_g_history, exp(log_ℓ_g[1]))
                
                if total_loss < best_loss
                    best_loss = total_loss
                    patience_counter = 0
                else
                    patience_counter += 1
                end
                
                if patience_counter >= patience && epoch > 30
                    if verbose
                        println("\nEarly stopping: no improvement for $patience epochs")
                    end
                    break
                end
            else
                if verbose
                    println("\nWarning: Invalid loss at epoch $epoch")
                end
                if !isempty(losses)
                    push!(losses, losses[end])
                    push!(σ_g_history, σ_g_history[end])
                    push!(ℓ_g_history, ℓ_g_history[end])
                end
            end
            
        catch e
            if verbose
                println("\nError at epoch $epoch: $e")
            end
            if !isempty(losses)
                push!(losses, losses[end])
                push!(σ_g_history, σ_g_history[end])
                push!(ℓ_g_history, ℓ_g_history[end])
            end
            continue
        end
        
        if verbose && (epoch % 10 == 0 || epoch == 1)
            println("\nEpoch $epoch/$num_epochs")
            println("  Loss: $(round(total_loss, digits=4))")
            println("  σ_g: $(round(exp(log_σ_g[1]), digits=4)), " *
                    "ℓ_g: $(round(exp(log_ℓ_g[1]), digits=4))")
        end
    end
    
    println("\n" * "="^70)
    println("Training Complete!")
    println("="^70)
    
    σ_g_final = exp(log_σ_g[1])
    ℓ_g_final = exp(log_ℓ_g[1]) * norm_params.x_std
    σ_s_final = exp.(log_σ_s) .* norm_params.y_stds
    ℓ_s_final = exp.(log_ℓ_s) .* norm_params.x_std
    σ_noise_final = exp.(log_σ_noise) .* norm_params.y_stds
    
    return (
        mean_func = mean_func,
        σ_g = σ_g_final,
        ℓ_g = ℓ_g_final,
        σ_s = σ_s_final,
        ℓ_s = ℓ_s_final,
        σ_noise = σ_noise_final,
        losses = losses,
        σ_g_history = σ_g_history,
        ℓ_g_history = ℓ_g_history,
        norm_params = norm_params,
        data = data,
        jitter = jitter
    )
end

# ============================================
# Prediction Functions
# ============================================

function robust_cholesky(K::Matrix{Float32}; max_tries::Int=5, initial_jitter::Float32=1.0f-6)
    jitter = initial_jitter
    
    for i in 1:max_tries
        try
            K_noisy = K + jitter * I
            K_noisy = Hermitian(0.5f0 .* (K_noisy .+ K_noisy'))
            L = cholesky(K_noisy).L
            return L, true, jitter
        catch e
            if i < max_tries
                jitter *= 10.0f0
            else
                return nothing, false, jitter
            end
        end
    end
end

function multitask_gp_predict(result, sensor_idx::Int, x_test::Vector{Float32})
    data = result.data
    norm_params = result.norm_params
    
    x_train = (data.times[sensor_idx] .- norm_params.x_mean) ./ norm_params.x_std
    y_train = (data.values[sensor_idx] .- norm_params.y_means[sensor_idx]) ./ 
              norm_params.y_stds[sensor_idx]
    x_test_norm = (x_test .- norm_params.x_mean) ./ norm_params.x_std
    
    σ_g_norm = result.σ_g / norm_params.y_stds[sensor_idx]
    ℓ_g_norm = result.ℓ_g / norm_params.x_std
    σ_s_norm = result.σ_s ./ norm_params.y_stds
    ℓ_s_norm = result.ℓ_s ./ norm_params.x_std
    σ_noise_norm = result.σ_noise[sensor_idx] / norm_params.y_stds[sensor_idx]
    
    m_train = vec(result.mean_func(reshape(x_train, 1, :)))
    m_test = vec(result.mean_func(reshape(x_test_norm, 1, :)))
    
    K_xx = compute_multitask_kernel(x_train, x_train, sensor_idx, sensor_idx,
                                     σ_g_norm, ℓ_g_norm, σ_s_norm, ℓ_s_norm)
    
    K_xx_base = K_xx + σ_noise_norm^2 * I
    L, success, used_jitter = robust_cholesky(K_xx_base, initial_jitter=result.jitter)
    
    if !success
        @warn "Cholesky failed during prediction for sensor $(data.sensor_names[sensor_idx])"
        μ_star = m_test .* norm_params.y_stds[sensor_idx] .+ norm_params.y_means[sensor_idx]
        σ_star = ones(Float32, length(m_test)) .* norm_params.y_stds[sensor_idx]
        return μ_star, σ_star
    end
    
    K_x_star = compute_multitask_kernel(x_train, x_test_norm, sensor_idx, sensor_idx,
                                         σ_g_norm, ℓ_g_norm, σ_s_norm, ℓ_s_norm)
    K_star_star = compute_multitask_kernel(x_test_norm, x_test_norm, 
                                            sensor_idx, sensor_idx,
                                            σ_g_norm, ℓ_g_norm, σ_s_norm, ℓ_s_norm)
    
    residual = y_train .- m_train
    α = L' \ (L \ residual)
    μ_star_norm = m_test .+ K_x_star' * α
    
    v = L \ K_x_star
    cov_star = K_star_star - v' * v
    σ_star_norm = sqrt.(max.(diag(cov_star), 1.0f-8))
    
    μ_star = μ_star_norm .* norm_params.y_stds[sensor_idx] .+ norm_params.y_means[sensor_idx]
    σ_star = σ_star_norm .* norm_params.y_stds[sensor_idx]
    
    return μ_star, σ_star
end

# ============================================
# Main Program
# ============================================

function main_multitask()
    Random.seed!(42)
    
    println("="^70)
    println("Multi-task Gaussian Process (MTGP) - Full Sensor Version")
    println("="^70)
    
    println("\n[1] Loading data...")
    feeder_dir = "D:/luosipeng/Deep_Learning_in_Distribution_System/data"
    res = time_series_ieee37(
        feeder_dir;
        dt_s=0.1,
        hours=24.0,
        sample_every=1,
        collect=[:voltage_bus, :total_power, :bus_injection]
    )
    ds = extract_requested_dataset(res)
    
    println("\n[2] Building complete multi-sensor dataset...")
    data = build_complete_multisensor_data(
        ds;
        max_points_per_sensor = 300
    )
    
    println("\n[3] Training multi-task GP...")
    result = train_multitask_gp(
        data;
        num_epochs = 100,
        lr = 0.005,
        verbose = true,
        jitter = 1.0f-4
    )
    
    println("\n[4] Final hyperparameters:")
    println("="^70)
    println("Global parameters:")
    println("  σ_g = $(round(result.σ_g, digits=4))")
    println("  ℓ_g = $(round(result.ℓ_g, digits=4)) hours")
    println("\nLocal parameters (top 10 sensors by SNR):")
    
    # Sort sensors by SNR
    snrs = result.σ_s ./ result.σ_noise
    sorted_indices = sortperm(snrs, rev=true)
    
    for i in 1:min(10, data.S)
        s = sorted_indices[i]
        println("  $(data.sensor_names[s]):")
        println("    σ_s = $(round(result.σ_s[s], digits=4))")
        println("    ℓ_s = $(round(result.ℓ_s[s], digits=4)) hours")
        println("    σ_noise = $(round(result.σ_noise[s], digits=4))")
        println("    SNR = $(round(snrs[s], digits=2))")
    end
    
    println("\n[5] Generating visualizations...")
    
    # Select diverse sensors for visualization
    # Pick 2 SCADA, 2 AMI, 1 PMU (if available)
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
    
    # Limit to 9 plots maximum
    selected_indices = selected_indices[1:min(9, length(selected_indices))]
    
    plots_list = []
    
    for s in selected_indices
        x_test = range(minimum(data.times[s]), maximum(data.times[s]), length=200)
        μ_pred, σ_pred = multitask_gp_predict(result, s, collect(Float32, x_test))
        
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
    
    # Combined plot
    n_plots = length(plots_list)
    layout = (ceil(Int, n_plots/3), 3)
    combined = plot(plots_list..., 
                   layout = layout, 
                   size = (1400, 320 * layout[1]),
                   margin = 6Plots.mm)
    display(combined)
    savefig(combined, "multitask_gp_predictions_full.png")
    println("  ✓ Saved: multitask_gp_predictions_full.png")
    
    # Loss curve
    valid_losses = filter(x -> isfinite(x) && x > 0, result.losses)
    if !isempty(valid_losses)
        p_loss = plot(valid_losses, 
                      xlabel = "Epoch", 
                      ylabel = "Total NLL",
                      title = "Training Loss", 
                      linewidth = 2, 
                      legend = false,
                      yscale = :log10, 
                      size = (700, 450),
                      margin = 5Plots.mm)
        display(p_loss)
        savefig(p_loss, "multitask_gp_loss_full.png")
        println("  ✓ Saved: multitask_gp_loss_full.png")
    end
    
    # Hyperparameter evolution
    if !isempty(result.ℓ_g_history) && !isempty(result.σ_g_history)
        p_hyper = plot(result.ℓ_g_history,
                       label = "Global length scale (ℓ_g)",
                       xlabel = "Epoch",
                       ylabel = "Value (normalized)",
                       title = "Hyperparameter Evolution",
                       linewidth = 2,
                       legend = :topright,
                       size = (700, 450),
                       margin = 5Plots.mm)
        plot!(p_hyper, result.σ_g_history,
              label = "Global signal variance (σ_g)",
                            linewidth = 2)
        display(p_hyper)
        savefig(p_hyper, "multitask_gp_hyperparams_full.png")
        println("  ✓ Saved: multitask_gp_hyperparams_full.png")
    end
    
    # SNR analysis plot
    println("\n[6] Generating SNR analysis...")
    snrs = result.σ_s ./ result.σ_noise
    sorted_indices = sortperm(snrs, rev=true)
    
    # Color by sensor type
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
    savefig(p_snr, "multitask_gp_snr_analysis.png")
    println("  ✓ Saved: multitask_gp_snr_analysis.png")
    
    # Length scale analysis
    println("\n[7] Generating length scale analysis...")
    sorted_ls_indices = sortperm(result.ℓ_s, rev=true)
    colors_ls = [data.sensor_types[i] == :SCADA ? :blue : 
                 (data.sensor_types[i] == :AMI ? :green : :red) 
                 for i in sorted_ls_indices]
    
    p_ls = bar(1:data.S, result.ℓ_s[sorted_ls_indices],
               xlabel = "Sensor Index (sorted by length scale)",
               ylabel = "Length Scale (hours)",
               title = "Local Length Scales (Blue=SCADA, Green=AMI, Red=PMU)",
               legend = false,
               color = colors_ls,
               size = (1000, 500),
               margin = 5Plots.mm,
               xticks = (1:5:data.S, string.(1:5:data.S)))
    hline!(p_ls, [result.ℓ_g], 
           label = "Global ℓ_g", 
           linewidth = 2, 
           linestyle = :dash, 
           color = :black,
           legend = :topright)
    display(p_ls)
    savefig(p_ls, "multitask_gp_lengthscale_analysis.png")
    println("  ✓ Saved: multitask_gp_lengthscale_analysis.png")
    
    # Summary statistics
    println("\n[8] Summary Statistics:")
    println("="^70)
    println("Signal-to-Noise Ratio:")
    println("  Mean SNR: $(round(mean(snrs), digits=2))")
    println("  Median SNR: $(round(median(snrs), digits=2))")
    println("  Max SNR: $(round(maximum(snrs), digits=2)) ($(data.sensor_names[argmax(snrs)]))")
    println("  Min SNR: $(round(minimum(snrs), digits=2)) ($(data.sensor_names[argmin(snrs)]))")
    
    println("\nLength Scales:")
    println("  Global ℓ_g: $(round(result.ℓ_g, digits=4)) hours")
    println("  Mean local ℓ_s: $(round(mean(result.ℓ_s), digits=4)) hours")
    println("  Median local ℓ_s: $(round(median(result.ℓ_s), digits=4)) hours")
    println("  Max local ℓ_s: $(round(maximum(result.ℓ_s), digits=4)) hours")
    println("  Min local ℓ_s: $(round(minimum(result.ℓ_s), digits=4)) hours")
    
    println("\nNoise Levels:")
    println("  Mean σ_noise: $(round(mean(result.σ_noise), digits=4))")
    println("  Median σ_noise: $(round(median(result.σ_noise), digits=4))")
    
    # Sensor type breakdown
    scada_snrs = snrs[data.sensor_types .== :SCADA]
    ami_snrs = snrs[data.sensor_types .== :AMI]
    pmu_snrs = snrs[data.sensor_types .== :PMU]
    
    println("\nSNR by Sensor Type:")
    if !isempty(scada_snrs)
        println("  SCADA - Mean: $(round(mean(scada_snrs), digits=2)), " *
                "Median: $(round(median(scada_snrs), digits=2))")
    end
    if !isempty(ami_snrs)
        println("  AMI   - Mean: $(round(mean(ami_snrs), digits=2)), " *
                "Median: $(round(median(ami_snrs), digits=2))")
    end
    if !isempty(pmu_snrs)
        println("  PMU   - Mean: $(round(mean(pmu_snrs), digits=2)), " *
                "Median: $(round(median(pmu_snrs), digits=2))")
    end
    
    println("\n" * "="^70)
    println("Complete!")
    println("="^70)
    
    return result
end

# ============================================
# Additional Analysis Functions
# ============================================

"""
Compute correlation matrix between sensors using learned kernel
"""
function compute_sensor_correlation_matrix(result)
    S = result.data.S
    corr_matrix = zeros(Float32, S, S)
    
    # Use a common time point (e.g., midpoint)
    t_ref = Float32[0.0]  # Normalized time
    
    for i in 1:S
        for j in 1:S
            K_ij = compute_multitask_kernel(
                t_ref, t_ref, i, j,
                result.σ_g / result.norm_params.y_stds[i],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            K_ii = compute_multitask_kernel(
                t_ref, t_ref, i, i,
                result.σ_g / result.norm_params.y_stds[i],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            K_jj = compute_multitask_kernel(
                t_ref, t_ref, j, j,
                result.σ_g / result.norm_params.y_stds[j],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            corr_matrix[i, j] = K_ij[1] / sqrt(K_ii[1] * K_jj[1])
        end
    end
    
    return corr_matrix
end

"""
Plot sensor correlation heatmap
"""
function plot_sensor_correlations(result; max_sensors::Int=20)
    S = min(result.data.S, max_sensors)
    
    println("\nComputing sensor correlations (first $S sensors)...")
    
    # Compute correlation for subset
    corr_subset = zeros(Float32, S, S)
    t_ref = Float32[0.0]
    
    for i in 1:S
        for j in 1:S
            K_ij = compute_multitask_kernel(
                t_ref, t_ref, i, j,
                result.σ_g / result.norm_params.y_stds[i],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            K_ii = compute_multitask_kernel(
                t_ref, t_ref, i, i,
                result.σ_g / result.norm_params.y_stds[i],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            K_jj = compute_multitask_kernel(
                t_ref, t_ref, j, j,
                result.σ_g / result.norm_params.y_stds[j],
                result.ℓ_g / result.norm_params.x_std,
                result.σ_s ./ result.norm_params.y_stds,
                result.ℓ_s ./ result.norm_params.x_std
            )
            
            corr_subset[i, j] = K_ij[1] / sqrt(K_ii[1] * K_jj[1])
        end
    end
    
    # Create short labels
    short_names = [length(name) > 15 ? name[1:12]*"..." : name 
                   for name in result.data.sensor_names[1:S]]
    
    p = heatmap(corr_subset,
                xlabel = "Sensor Index",
                ylabel = "Sensor Index",
                title = "Learned Sensor Correlations",
                color = :RdBu,
                clims = (-1, 1),
                size = (800, 700),
                margin = 8Plots.mm,
                xticks = (1:S, short_names),
                yticks = (1:S, short_names),
                xrotation = 45,
                aspect_ratio = :equal)
    
    display(p)
    savefig(p, "multitask_gp_correlations.png")
    println("  ✓ Saved: multitask_gp_correlations.png")
    
    return corr_subset
end

"""
Save results to file
"""
function save_results(result, filename::String="multitask_gp_results.txt")
    open(filename, "w") do io
        println(io, "="^70)
        println(io, "Multi-task Gaussian Process Results")
        println(io, "="^70)
        println(io, "\nGlobal Hyperparameters:")
        println(io, "  σ_g = $(result.σ_g)")
        println(io, "  ℓ_g = $(result.ℓ_g) hours")
        
        println(io, "\nPer-Sensor Parameters:")
        println(io, "="^70)
        println(io, "Sensor Name | σ_s | ℓ_s (hrs) | σ_noise | SNR")
        println(io, "-"^70)
        
        snrs = result.σ_s ./ result.σ_noise
        for s in 1:result.data.S
            println(io, "$(rpad(result.data.sensor_names[s], 35)) | " *
                        "$(round(result.σ_s[s], digits=4)) | " *
                        "$(round(result.ℓ_s[s], digits=4)) | " *
                        "$(round(result.σ_noise[s], digits=4)) | " *
                        "$(round(snrs[s], digits=2))")
        end
        
        println(io, "\n" * "="^70)
        println(io, "Summary Statistics:")
        println(io, "  Total sensors: $(result.data.S)")
        println(io, "  Mean SNR: $(round(mean(snrs), digits=2))")
        println(io, "  Median SNR: $(round(median(snrs), digits=2))")
        println(io, "  Mean local ℓ_s: $(round(mean(result.ℓ_s), digits=4)) hours")
        # println(io, "  Final training loss: $(round(result.losses[end], digits=4))")
    end
    
    println("  ✓ Saved: $filename")
end

# ============================================
# Run Main Program
# ============================================

println("\n" * "="^70)
println("Starting Multi-task GP with Full Sensor Suite")
println("="^70)

# Run main training
# result = nothing
result = main_multitask()


# Additional analyses (only if training succeeded)
if result !== nothing
    println("\n[9] Additional Analyses...")
    
    # Correlation plot
    try
        plot_sensor_correlations(result, max_sensors=20)
    catch e
        println("  ⚠️  Correlation plot failed: $e")
    end
    
    # Save detailed results
    try
        save_results(result)
    catch e
        println("  ⚠️  Failed to save results: $e")
        println("  Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
    
    println("\n" * "="^70)
    println("All analyses complete!")
    println("="^70)
    println("\nGenerated files:")
    println("  - multitask_gp_predictions_full.png")
    println("  - multitask_gp_loss_full.png")
    println("  - multitask_gp_hyperparams_full.png")
    println("  - multitask_gp_snr_analysis.png")
    println("  - multitask_gp_lengthscale_analysis.png")
    println("  - multitask_gp_correlations.png")
    println("  - multitask_gp_results.txt")
else
    println("\n" * "="^70)
    println("Training failed - no results to analyze")
    println("="^70)
end

