function generate_1min_resolution_predictions(result)
    println("\n[9] Generating 1-minute resolution predictions for 24 hours...")
    println("="^70)

    dt_unified = 1.0f0 / 60.0f0
    t_min = minimum([minimum(result.data.times[s]) for s in 1:result.data.S])
    t_max = maximum([maximum(result.data.times[s]) for s in 1:result.data.S])
    t_unified = collect(Float32, t_min:dt_unified:t_max)

    println("  Time grid configuration:")
    println("    Sampling interval: 1 minute ($(round(dt_unified*60, digits=2)) min)")
    println("    Time range: $(round(t_min, digits=2)) ~ $(round(t_max, digits=2)) hours")
    println("    Total duration: $(round(t_max - t_min, digits=2)) hours")
    println("    Number of time points: $(length(t_unified))")
    println("    Expected for 24h: 1440 points")

    println("\n  Generating GP predictions for all sensors...")

    unified_predictions = Dict{String, Any}()
    unified_predictions["time_hours"] = t_unified
    unified_predictions["time_minutes"] = t_unified .* 60.0f0
    unified_predictions["sampling_interval_minutes"] = 1.0
    unified_predictions["num_timepoints"] = length(t_unified)
    unified_predictions["sensors"] = Dict{String, Any}()

    @showprogress for s in 1:result.data.S
        sensor_name = result.data.sensor_names[s]
        sensor_type = result.data.sensor_types[s]

        try
            μ_pred, σ_pred = multitask_gp_predict(result, s, t_unified)
            t_meas = result.data.times[s]
            v_meas = result.data.values[s]

            if length(t_meas) > 1
                orig_dt = mean(diff(sort(t_meas)))
                if orig_dt < 0.01
                    orig_rate_str = "$(round(orig_dt * 3600, digits=1))s"
                elseif orig_dt < 1.0
                    orig_rate_str = "$(round(orig_dt * 60, digits=1))min"
                else
                    orig_rate_str = "$(round(orig_dt, digits=2))h"
                end
            else
                orig_rate_str = "single point"
            end

            unified_predictions["sensors"][sensor_name] = Dict(
                "type" => String(sensor_type),
                "prediction_mean" => μ_pred,
                "prediction_std" => σ_pred,
                "prediction_ci_lower" => μ_pred .- 1.96f0 .* σ_pred,
                "prediction_ci_upper" => μ_pred .+ 1.96f0 .* σ_pred,
                "measurement_times" => t_meas,
                "measurement_values" => v_meas,
                "original_sampling_rate" => orig_rate_str,
                "num_measurements" => length(t_meas),
                "num_predictions" => length(t_unified)
            )
        catch e
            @warn "Failed to generate predictions for sensor $sensor_name: $e"
            continue
        end
    end

    println("  ✓ Successfully generated predictions for $(length(unified_predictions["sensors"])) sensors")

    println("\n  Computing prediction quality metrics...")

    metrics = DataFrame(
        Sensor = String[],
        Type = String[],
        Original_Rate = String[],
        Num_Measurements = Int[],
        RMSE = Float32[],
        MAE = Float32[],
        MAPE = Float32[],
        R2 = Float32[],
        Coverage_95 = Float32[],
        Mean_Uncertainty = Float32[]
    )

    for (sensor_name, sensor_data) in unified_predictions["sensors"]
        t_meas = sensor_data["measurement_times"]
        v_meas = sensor_data["measurement_values"]
        s_idx = findfirst(==(sensor_name), result.data.sensor_names)
        μ_at_meas, σ_at_meas = multitask_gp_predict(result, s_idx, t_meas)
        residuals = v_meas .- μ_at_meas
        rmse = sqrt(mean(residuals.^2))
        mae = mean(abs.(residuals))
        mape = mean(abs.(residuals ./ (abs.(v_meas) .+ 1e-8))) * 100
        ss_res = sum(residuals.^2)
        ss_tot = sum((v_meas .- mean(v_meas)).^2)
        r2 = 1 - ss_res / ss_tot
        ci_lower = μ_at_meas .- 1.96f0 .* σ_at_meas
        ci_upper = μ_at_meas .+ 1.96f0 .* σ_at_meas
        coverage = mean((v_meas .>= ci_lower) .& (v_meas .<= ci_upper)) * 100
        mean_uncertainty = mean(σ_at_meas)

        push!(metrics, (
            sensor_name,
            sensor_data["type"],
            sensor_data["original_sampling_rate"],
            length(t_meas),
            rmse,
            mae,
            mape,
            r2,
            coverage,
            mean_uncertainty
        ))
    end

    sort!(metrics, [:Type, order(:R2, rev=true)])

    println("\n" * "="^70)
    println("Prediction Quality Metrics (evaluated at original measurement points)")
    println("="^70)
    println(metrics)

    println("\n  Saving unified predictions to files...")

    try
        println("    Creating prediction time series DataFrame...")
        df_predictions = DataFrame(
            Time_hours = Float32[],
            Time_minutes = Float32[],
            Sensor = String[],
            Type = String[],
            Prediction_Mean = Float32[],
            Prediction_Std = Float32[],
            CI_Lower = Float32[],
            CI_Upper = Float32[]
        )

        for (sensor_name, sensor_data) in unified_predictions["sensors"]
            n = length(t_unified)
            append!(df_predictions, DataFrame(
                Time_hours = t_unified,
                Time_minutes = t_unified .* 60.0f0,
                Sensor = fill(sensor_name, n),
                Type = fill(sensor_data["type"], n),
                Prediction_Mean = sensor_data["prediction_mean"],
                Prediction_Std = sensor_data["prediction_std"],
                CI_Lower = sensor_data["prediction_ci_lower"],
                CI_Upper = sensor_data["prediction_ci_upper"]
            ))
        end

        CSV.write("gp_predictions_1min_timeseries.csv", df_predictions)
        println("  ✓ Saved: gp_predictions_1min_timeseries.csv")
        println("    Size: $(nrow(df_predictions)) rows × $(ncol(df_predictions)) columns")
        println("    File size: ~$(round(nrow(df_predictions) * ncol(df_predictions) * 4 / 1024^2, digits=2)) MB")

        println("    Creating measurements DataFrame...")
        df_measurements = DataFrame(
            Time_hours = Float32[],
            Time_minutes = Float32[],
            Sensor = String[],
            Type = String[],
            Measurement = Float32[]
        )

        for (sensor_name, sensor_data) in unified_predictions["sensors"]
            n = length(sensor_data["measurement_times"])
            append!(df_measurements, DataFrame(
                Time_hours = sensor_data["measurement_times"],
                Time_minutes = sensor_data["measurement_times"] .* 60.0f0,
                Sensor = fill(sensor_name, n),
                Type = fill(sensor_data["type"], n),
                Measurement = sensor_data["measurement_values"]
            ))
        end

        CSV.write("original_measurements.csv", df_measurements)
        println("  ✓ Saved: original_measurements.csv")
        println("    Size: $(nrow(df_measurements)) rows × $(ncol(df_measurements)) columns")

        CSV.write("prediction_quality_metrics.csv", metrics)
        println("  ✓ Saved: prediction_quality_metrics.csv")

        println("    Creating wide-format predictions...")
        df_wide = DataFrame(Time_hours = t_unified, Time_minutes = t_unified .* 60.0f0)

        for (sensor_name, sensor_data) in unified_predictions["sensors"]
            df_wide[!, Symbol(sensor_name)] = sensor_data["prediction_mean"]
        end

        CSV.write("gp_predictions_1min_wide.csv", df_wide)
        println("  ✓ Saved: gp_predictions_1min_wide.csv")
        println("    Size: $(nrow(df_wide)) rows × $(ncol(df_wide)) columns")
    catch e
        println("  ⚠️  CSV save failed: $e")
        println("    Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end

    println("\n  Generating comparison visualizations...")

    sensor_types_available = unique(metrics.Type)
    selected_for_plot = String[]

    for stype in sensor_types_available
        type_metrics = metrics[metrics.Type .== stype, :]
        if nrow(type_metrics) > 0
            best_idx = argmax(type_metrics.R2)
            push!(selected_for_plot, type_metrics.Sensor[best_idx])
        end
    end

    if length(selected_for_plot) < 9
        for stype in sensor_types_available
            type_metrics = metrics[metrics.Type .== stype, :]
            sorted_type = sort(type_metrics, :R2, rev=true)
            for i in 2:min(3, nrow(sorted_type))
                if length(selected_for_plot) < 9
                    push!(selected_for_plot, sorted_type.Sensor[i])
                end
            end
        end
    end

    selected_for_plot = unique(selected_for_plot)[1:min(9, length(unique(selected_for_plot)))]

    plots_short = []
    time_window = 6.0

    for sensor_name in selected_for_plot
        sensor_data = unified_predictions["sensors"][sensor_name]
        sensor_metrics = metrics[metrics.Sensor .== sensor_name, :]
        window_indices = findall(t_unified .<= (t_min + time_window))
        t_plot = t_unified[window_indices]
        μ_plot = sensor_data["prediction_mean"][window_indices]
        σ_plot = sensor_data["prediction_std"][window_indices]
        meas_window = findall((sensor_data["measurement_times"] .>= t_min) .&
                              (sensor_data["measurement_times"] .<= t_min + time_window))

        p = plot(t_plot, μ_plot,
                 ribbon = 1.96f0 .* σ_plot,
                 label = "GP Prediction (95% CI)",
                 xlabel = "Time (hours)",
                 ylabel = "Value",
                 title = "$(sensor_name)\n" *
                         "R²=$(round(sensor_metrics.R2[1], digits=3)), " *
                         "Cov=$(round(sensor_metrics.Coverage_95[1], digits=1))%",
                 linewidth = 2,
                 fillalpha = 0.3,
                 legend = :topright,
                 size = (500, 350),
                 margin = 4Plots.mm,
                 titlefontsize = 7,
                 legendfontsize = 6)

        if !isempty(meas_window)
            scatter!(p, sensor_data["measurement_times"][meas_window],
                     sensor_data["measurement_values"][meas_window],
                     label = "Measurements ($(sensor_data["original_sampling_rate"]))",
                     markersize = 3,
                     alpha = 0.8,
                     color = :red,
                     markerstrokewidth = 0)
        end

        push!(plots_short, p)
    end

    if isempty(plots_short)
        println("  ⚠ No sensors available for short-window plots; skipping.")
    else
        n_plots = length(plots_short)
        layout = (ceil(Int, n_plots/3), 3)
        combined_short = plot(plots_short...,
                              layout = layout,
                              size = (1400, 320 * layout[1]),
                              margin = 6Plots.mm)
        display(combined_short)
        savefig(combined_short, "gp_predictions_1min_6hours.png")
        println("  ✓ Saved: gp_predictions_1min_6hours.png")
    end

    plots_full = []

    for sensor_name in selected_for_plot[1:min(6, length(selected_for_plot))]
        sensor_data = unified_predictions["sensors"][sensor_name]
        sensor_metrics = metrics[metrics.Sensor .== sensor_name, :]

        p = plot(t_unified, sensor_data["prediction_mean"],
                 ribbon = 1.96f0 .* sensor_data["prediction_std"],
                 label = "GP (95% CI)",
                 xlabel = "Time (hours)",
                 ylabel = "Value",
                 title = "$(sensor_name) - Full 24h\n" *
                         "R²=$(round(sensor_metrics.R2[1], digits=3)), " *
                         "RMSE=$(round(sensor_metrics.RMSE[1], digits=3))",
                 linewidth = 1.5,
                 fillalpha = 0.25,
                 legend = :topright,
                 size = (600, 350),
                 margin = 4Plots.mm,
                 titlefontsize = 7,
                 legendfontsize = 6)

        scatter!(p, sensor_data["measurement_times"],
                 sensor_data["measurement_values"],
                 label = "Measurements",
                 markersize = 2,
                 alpha = 0.6,
                 color = :red,
                 markerstrokewidth = 0)

        push!(plots_full, p)
    end

    if isempty(plots_full)
        println("  ⚠ No sensors available for full-range plots; skipping.")
    else
        n_plots_full = length(plots_full)
        layout_full = (ceil(Int, n_plots_full/2), 2)
        combined_full = plot(plots_full...,
                             layout = layout_full,
                             size = (1200, 350 * layout_full[1]),
                             margin = 6Plots.mm)
        display(combined_full)
        savefig(combined_full, "gp_predictions_1min_24hours.png")
        println("  ✓ Saved: gp_predictions_1min_24hours.png")
    end

    println("\n  Generating metrics comparison plots...")

    p_r2 = bar(1:nrow(metrics), metrics.R2,
               xlabel = "Sensor Index",
               ylabel = "R² Score",
               title = "Prediction Quality (R²) by Sensor",
               legend = false,
               color = [m == "SCADA" ? :blue : (m == "AMI" ? :green : :red)
                        for m in metrics.Type],
               size = (1000, 500),
               margin = 5Plots.mm,
               xticks = (1:nrow(metrics),
                         [length(s) > 10 ? s[1:8]*".." : s for s in metrics.Sensor]),
               xrotation = 45)
    hline!(p_r2, [0.9], label="R²=0.9", linestyle=:dash, linewidth=2, color=:black)
    display(p_r2)
    savefig(p_r2, "prediction_quality_r2.png")
    println("  ✓ Saved: prediction_quality_r2.png")

    p_cov = bar(1:nrow(metrics), metrics.Coverage_95,
                xlabel = "Sensor Index",
                ylabel = "95% CI Coverage (%)",
                title = "Prediction Uncertainty Coverage by Sensor",
                legend = false,
                color = [m == "SCADA" ? :blue : (m == "AMI" ? :green : :red)
                         for m in metrics.Type],
                size = (1000, 500),
                margin = 5Plots.mm,
                xticks = (1:nrow(metrics),
                          [length(s) > 10 ? s[1:8]*".." : s for s in metrics.Sensor]),
                xrotation = 45)
    hline!(p_cov, [95], label="Target 95%", linestyle=:dash, linewidth=2, color=:black)
    display(p_cov)
    savefig(p_cov, "prediction_quality_coverage.png")
    println("  ✓ Saved: prediction_quality_coverage.png")

    println("\n" * "="^70)
    println("Prediction Quality Summary")
    println("="^70)

    println("\nOverall Statistics:")
    println("  Mean R²: $(round(mean(metrics.R2), digits=3))")
    println("  Median R²: $(round(median(metrics.R2), digits=3))")
    println("  Min R²: $(round(minimum(metrics.R2), digits=3)) ($(metrics.Sensor[argmin(metrics.R2)]))")
    println("  Max R²: $(round(maximum(metrics.R2), digits=3)) ($(metrics.Sensor[argmax(metrics.R2)]))")

    println("\n  Mean Coverage: $(round(mean(metrics.Coverage_95), digits=1))%")
    println("  Median Coverage: $(round(median(metrics.Coverage_95), digits=1))%")

    println("\n  Mean RMSE: $(round(mean(metrics.RMSE), digits=4))")
    println("  Mean MAE: $(round(mean(metrics.MAE), digits=4))")
    println("  Mean MAPE: $(round(mean(metrics.MAPE), digits=2))%")

    for stype in sensor_types_available
        type_metrics = metrics[metrics.Type .== stype, :]
        if nrow(type_metrics) > 0
            println("\n$stype Sensors (n=$(nrow(type_metrics))):")
            println("  Mean R²: $(round(mean(type_metrics.R2), digits=3))")
            println("  Mean Coverage: $(round(mean(type_metrics.Coverage_95), digits=1))%")
            println("  Mean RMSE: $(round(mean(type_metrics.RMSE), digits=4))")
        end
    end

    println("\n" * "="^70)
    println("Sampling Rate Information")
    println("="^70)

    scada_sensors = [k for (k, v) in unified_predictions["sensors"] if v["type"] == "SCADA"]
    ami_sensors = [k for (k, v) in unified_predictions["sensors"] if v["type"] == "AMI"]
    pmu_sensors = [k for (k, v) in unified_predictions["sensors"] if v["type"] == "PMU"]

    if !isempty(scada_sensors)
        rates = unique([unified_predictions["sensors"][s]["original_sampling_rate"] for s in scada_sensors])
        println("  SCADA ($(length(scada_sensors)) sensors): $rates")
    end
    if !isempty(ami_sensors)
        rates = unique([unified_predictions["sensors"][s]["original_sampling_rate"] for s in ami_sensors])
        println("  AMI ($(length(ami_sensors)) sensors): $rates")
    end
    if !isempty(pmu_sensors)
        rates = unique([unified_predictions["sensors"][s]["original_sampling_rate"] for s in pmu_sensors])
        println("  PMU ($(length(pmu_sensors)) sensors): $rates")
    end

    println("\nUnified output:")
    println("  All sensors now at: 1 minute resolution")
    println("  Total time points: $(length(t_unified))")
    println("  Duration: $(round(t_max - t_min, digits=2)) hours")

    println("\n" * "="^70)
    println("1-Minute Resolution Prediction Complete!")
    println("="^70)
    println("\nGenerated files:")
    println("  Data files:")
    println("    - unified_predictions_1min.jld2 (full data structure)")
    println("    - gp_predictions_1min_timeseries.csv (long format)")
    println("    - gp_predictions_1min_wide.csv (wide format)")
    println("    - original_measurements.csv")
    println("    - prediction_quality_metrics.csv")
    println("  Visualization files:")
    println("    - gp_predictions_1min_6hours.png")
    println("    - gp_predictions_1min_24hours.png")
    println("    - prediction_quality_r2.png")
    println("    - prediction_quality_coverage.png")
    println("="^70)

    unified_result = (
        predictions = unified_predictions,
        metrics = metrics,
        time_grid = t_unified
    )
    return unified_result
end

function multitask_gp_predict(result, s::Int, x::AbstractVector)
    μ, σ = icm_predict(result, s, Float32.(x))
    return μ, σ
end