include("../src/get_sample_from_ieee37.jl")
include("../solvers/gauss_process.jl")
include("../src/get_ieee37_multitask_data.jl")

using KernelFunctions
using AbstractGPs
using Flux, Optim
using LinearAlgebra
using Plots
using Statistics
using Random

# ============================================
# 改进版本：更稳定的训练
# ============================================

"""
创建神经网络均值函数
"""
function create_mean_network(input_dim=1, hidden_dim=64)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 1)
    )
end

"""
使用 KernelFunctions.jl 计算核矩阵（完全兼容Zygote）
"""
function compute_kernel_matrix(x1::AbstractVector, x2::AbstractVector, σ_s::Real, ℓ::Real)
    # 使用 KernelFunctions.jl 的 SqExponentialKernel
    kernel = σ_s^2 * SqExponentialKernel() ∘ ScaleTransform(1.0 / ℓ)
    
    # 转换为矩阵格式
    X1 = reshape(x1, 1, :)
    X2 = reshape(x2, 1, :)
    
    # 计算核矩阵
    K = kernelmatrix(kernel, X1, X2)
    
    return K
end

"""
计算对数边际似然
"""
function log_marginal_likelihood(x::AbstractVector, y::AbstractVector, mean_func, 
                                  σ_s::Real, ℓ::Real, σ_noise::Real)
    n = length(x)
    
    # 计算均值
    x_reshaped = reshape(x, 1, :)
    m_x = vec(mean_func(x_reshaped))
    
    # 计算协方差矩阵
    K_xx = compute_kernel_matrix(x, x, σ_s, ℓ)
    K_xx_noisy = K_xx + (σ_noise^2 + 1e-6) * I
    
    # 残差
    residual = y .- m_x
    
    # Cholesky分解
    try
        L = cholesky(Hermitian(K_xx_noisy)).L
        α = L' \ (L \ residual)
        
        # 计算对数似然
        term1 = -0.5f0 * dot(residual, α)
        term2 = -sum(log.(diag(L)))
        term3 = -0.5f0 * n * log(2.0f0 * π)
        
        return term1 + term2 + term3
    catch e
        return -1.0f10
    end
end

"""
训练独立高斯过程（改进版）
"""
function train_independent_gp(x_train::AbstractVector, y_train::AbstractVector; 
                               num_epochs::Int=200, 
                               lr::Float64=0.001,
                               verbose::Bool=true)
    
    # 数据标准化
    x_mean = mean(x_train)
    x_std = std(x_train)
    y_mean = mean(y_train)
    y_std = std(y_train)
    
    x_normalized = Float32.((x_train .- x_mean) ./ x_std)
    y_normalized = Float32.((y_train .- y_mean) ./ y_std)
    
    if verbose
        println("数据标准化:")
        println("  X: μ=$(round(x_mean, digits=2)), σ=$(round(x_std, digits=2))")
        println("  Y: μ=$(round(y_mean, digits=2)), σ=$(round(y_std, digits=2))")
    end
    
    # 初始化模型参数
    mean_func = create_mean_network(1, 64)
    
    # 核函数超参数 - 更好的初始化
    # 基于数据范围估计合理的长度尺度
    data_range = maximum(x_normalized) - minimum(x_normalized)
    initial_ℓ = data_range / 10.0  # 初始长度尺度约为数据范围的1/10
    
    log_σ_s = Float32[log(1.0)]  # 信号标准差初始化为1
    log_ℓ = Float32[log(initial_ℓ)]  # 长度尺度
    log_σ_noise = Float32[log(0.1)]  # 噪声标准差初始化为0.1
    
    if verbose
        println("\n初始超参数:")
        println("  σ_s: $(round(exp(log_σ_s[1]), digits=4))")
        println("  ℓ: $(round(exp(log_ℓ[1]), digits=4))")
        println("  σ_noise: $(round(exp(log_σ_noise[1]), digits=4))")
    end
    
    # 收集所有参数
    ps = Flux.params(mean_func, log_σ_s, log_ℓ, log_σ_noise)
    
    # 使用学习率调度
    opt = Flux.Adam(lr)
    
    # 训练循环
    losses = Float32[]
    σ_s_history = Float32[]
    ℓ_history = Float32[]
    σ_noise_history = Float32[]
    
    best_loss = Inf32
    patience = 20
    patience_counter = 0
    
    for epoch in 1:num_epochs
        # 计算损失和梯度
        local loss
        gs = Flux.gradient(ps) do
            # 转换为正值
            σ_s = exp(log_σ_s[1])
            ℓ = exp(log_ℓ[1])
            σ_noise = exp(log_σ_noise[1])
            
            # 计算负对数边际似然
            log_ml = log_marginal_likelihood(x_normalized, y_normalized, mean_func, 
                                              σ_s, ℓ, σ_noise)
            loss = -log_ml
            return loss
        end
        
        # 检查梯度是否有效
        if !isnan(loss) && !isinf(loss)
            # 更新参数
            Flux.update!(opt, ps, gs)
            push!(losses, loss)
            
            # 记录超参数历史
            push!(σ_s_history, exp(log_σ_s[1]))
            push!(ℓ_history, exp(log_ℓ[1]))
            push!(σ_noise_history, exp(log_σ_noise[1]))
            
            # 早停检查
            if loss < best_loss
                best_loss = loss
                patience_counter = 0
            else
                patience_counter += 1
            end
            
            if patience_counter >= patience && epoch > 50
                if verbose
                    println("\n早停: 损失在 $patience 个epoch内没有改善")
                end
                break
            end
        else
            if verbose
                println("警告: Epoch $epoch 出现无效损失值")
            end
            if !isempty(losses)
                push!(losses, losses[end])
                push!(σ_s_history, σ_s_history[end])
                push!(ℓ_history, ℓ_history[end])
                push!(σ_noise_history, σ_noise_history[end])
            else
                push!(losses, 1.0f10)
                push!(σ_s_history, 1.0f0)
                push!(ℓ_history, 1.0f0)
                push!(σ_noise_history, 0.1f0)
            end
        end
        
        if verbose && (epoch % 20 == 0 || epoch == 1)
            σ_s_curr = exp(log_σ_s[1])
            ℓ_curr = exp(log_ℓ[1])
            σ_noise_curr = exp(log_σ_noise[1])
            println("Epoch $epoch/$num_epochs, Loss: $(round(loss, digits=4)), " *
                    "σ_s: $(round(σ_s_curr, digits=4)), " *
                    "ℓ: $(round(ℓ_curr, digits=4)), " *
                    "σ_noise: $(round(σ_noise_curr, digits=4))")
        end
    end
    
    # 返回训练好的参数（在原始尺度上）
    σ_s = exp(log_σ_s[1]) * y_std
    ℓ = exp(log_ℓ[1]) * x_std
    σ_noise = exp(log_σ_noise[1]) * y_std
    
    return (mean_func=mean_func, 
            σ_s=σ_s, 
            ℓ=ℓ, 
            σ_noise=σ_noise, 
            losses=losses,
            σ_s_history=σ_s_history,
            ℓ_history=ℓ_history,
            σ_noise_history=σ_noise_history,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std)
end

"""
GP预测（支持标准化）
"""
function gp_predict(x_train::AbstractVector, y_train::AbstractVector, 
                     x_test::AbstractVector,
                     mean_func, σ_s::Real, ℓ::Real, σ_noise::Real;
                     x_mean=0.0, x_std=1.0, y_mean=0.0, y_std=1.0)
    
    # 标准化
    x_train_norm = Float32.((x_train .- x_mean) ./ x_std)
    y_train_norm = Float32.((y_train .- y_mean) ./ y_std)
    x_test_norm = Float32.((x_test .- x_mean) ./ x_std)
    
    # 标准化超参数
    σ_s_norm = σ_s / y_std
    ℓ_norm = ℓ / x_std
    σ_noise_norm = σ_noise / y_std
    
    n = length(x_train_norm)
    m = length(x_test_norm)
    
    # 计算均值
    x_train_reshaped = reshape(x_train_norm, 1, :)
    x_test_reshaped = reshape(x_test_norm, 1, :)
    
    m_train = vec(mean_func(x_train_reshaped))
    m_test = vec(mean_func(x_test_reshaped))
    
    # 计算协方差矩阵
    K_xx = compute_kernel_matrix(x_train_norm, x_train_norm, σ_s_norm, ℓ_norm)
    K_xx_noisy = K_xx + (σ_noise_norm^2 + 1e-6) * I
    
    K_x_star = compute_kernel_matrix(x_train_norm, x_test_norm, σ_s_norm, ℓ_norm)
    K_star_star = compute_kernel_matrix(x_test_norm, x_test_norm, σ_s_norm, ℓ_norm)
    
    # 求解
    residual = y_train_norm .- m_train
    L = cholesky(Hermitian(K_xx_noisy)).L
    α = L' \ (L \ residual)
    
    # 预测均值（标准化空间）
    μ_star_norm = m_test .+ K_x_star' * α
    
    # 预测协方差
    v = L \ K_x_star
    cov_star = K_star_star - v' * v
    
    # 提取标准差（标准化空间）
    σ_star_norm = sqrt.(max.(diag(cov_star), 0.0))
    
    # 转换回原始尺度
    μ_star = μ_star_norm .* y_std .+ y_mean
    σ_star = σ_star_norm .* y_std
    
    return μ_star, σ_star
end

# ============================================
# 主程序
# ============================================

function main()
    Random.seed!(42)
        
    println("="^70)
    println("Independent Gaussian Process (GP) Demo - Improved Version")
    println("="^70)

    # 1) Load data
    feeder_dir = "D:/luosipeng/Deep_Learning_in_Distribution_System/data"
    res = time_series_ieee37(
        feeder_dir;
        dt_s=0.1,
        hours=24.0,
        sample_every=1,
        collect=[:voltage_bus, :total_power, :bus_injection]
    )
    ds = extract_requested_dataset(res)

    # Get AMI data
    ami_data = ds[:AMI]["736"]
    times_ami_seconds = ami_data[:times]  # This is in seconds
    
    # ✅ Convert to hours for better numerical stability
    times_ami = times_ami_seconds ./ 3600.0  # Convert seconds to hours
    P_ami = ami_data[:P_kW][:A]

    println("\n[1] Data Information")
    println("-"^70)
    println("  Number of points: $(length(P_ami))")
    println("  Time range: $(round(minimum(times_ami), digits=2)) - $(round(maximum(times_ami), digits=2)) hours")
    println("  Time span: $(round(maximum(times_ami) - minimum(times_ami), digits=2)) hours")
    println("  Power range: $(round(minimum(P_ami), digits=2)) - $(round(maximum(P_ami), digits=2)) kW")
    println("  Power mean: $(round(mean(P_ami), digits=2)) kW")
    println("  Power std: $(round(std(P_ami), digits=2)) kW")

    # Train GP with adjusted hyperparameters
    println("\n[2] Training Gaussian Process")
    println("-"^70)
    result = train_independent_gp(
        times_ami, P_ami, 
        num_epochs=300,  # ✅ More epochs
        lr=0.005,        # ✅ Slightly higher learning rate
        verbose=true
    )

    println("\n[3] Training Completed!")
    println("-"^70)
    println("Final Hyperparameters (Original Scale):")
    println("  Signal std (σ_s): $(round(result.σ_s, digits=4)) kW")
    println("  Signal variance (σ_s²): $(round(result.σ_s^2, digits=4)) kW²")
    println("  Length scale (ℓ): $(round(result.ℓ, digits=4)) hours")
    println("  Noise std (σ_noise): $(round(result.σ_noise, digits=4)) kW")
    println("  Noise variance (σ_noise²): $(round(result.σ_noise^2, digits=4)) kW²")
    
    # ✅ Interpret length scale
    data_span = maximum(times_ami) - minimum(times_ami)
    println("\n  Interpretation:")
    println("  - Length scale as % of data span: $(round(result.ℓ/data_span*100, digits=1))%")
    println("  - Signal-to-noise ratio: $(round(result.σ_s/result.σ_noise, digits=2))")

    # Prediction
    println("\n[4] Making Predictions")
    println("-"^70)
    x_pred = range(minimum(times_ami), maximum(times_ami), length=300)
    μ_pred, σ_pred = gp_predict(times_ami, P_ami, collect(x_pred), 
                                  result.mean_func, result.σ_s, result.ℓ, result.σ_noise,
                                  x_mean=result.x_mean, x_std=result.x_std,
                                  y_mean=result.y_mean, y_std=result.y_std)

    # Evaluation
    μ_train, σ_train = gp_predict(times_ami, P_ami, times_ami, 
                                    result.mean_func, result.σ_s, result.ℓ, result.σ_noise,
                                    x_mean=result.x_mean, x_std=result.x_std,
                                    y_mean=result.y_mean, y_std=result.y_std)
    
    mae = mean(abs.(μ_train .- P_ami))
    rmse = sqrt(mean((μ_train .- P_ami).^2))
    mape = mean(abs.((μ_train .- P_ami) ./ P_ami)) * 100
    
    # 95% confidence interval coverage
    upper = μ_train .+ 1.96 .* σ_train
    lower = μ_train .- 1.96 .* σ_train
    coverage = mean((P_ami .>= lower) .& (P_ami .<= upper)) * 100
    
    # ✅ Additional metrics
    mean_uncertainty = mean(σ_train)
    max_uncertainty = maximum(σ_train)

    println("\n[5] Performance Metrics")
    println("-"^70)
    println("  MAE: $(round(mae, digits=4)) kW")
    println("  RMSE: $(round(rmse, digits=4)) kW")
    println("  MAPE: $(round(mape, digits=2))%")
    println("  95% CI Coverage: $(round(coverage, digits=2))%")
    println("  Mean Uncertainty: $(round(mean_uncertainty, digits=4)) kW")
    println("  Max Uncertainty: $(round(max_uncertainty, digits=4)) kW")
    
    # ✅ Diagnostic warnings
    if coverage < 90
        println("\n  ⚠️  WARNING: Low CI coverage - model may be overconfident")
    end
    if result.ℓ > data_span * 0.5
        println("  ⚠️  WARNING: Length scale > 50% of data span - model may be too smooth")
    end

    # Visualization
    println("\n[6] Generating Visualizations")
    println("-"^70)
    
    upper_pred = μ_pred .+ 1.96 .* σ_pred
    lower_pred = μ_pred .- 1.96 .* σ_pred

    # Main prediction plot
    p1 = plot(x_pred, μ_pred,
              ribbon=(μ_pred .- lower_pred, upper_pred .- μ_pred),
              label="GP Prediction (95% CI)",
              xlabel="Time (hours)",
              ylabel="Active Power (kW)",
              title="AMI Sensor 736 - Phase A Active Power",
              linewidth=2,
              fillalpha=0.3,
              legend=:topright,
              color=:blue,
              size=(700, 500))

    scatter!(p1, times_ami, P_ami,
             label="Training Data",
             markersize=4,
             alpha=0.7,
             color=:red)

    # Loss curve - fix negative values issue
    valid_losses = filter(x -> x > 0, result.losses)
    if isempty(valid_losses)
        valid_losses = abs.(result.losses)
    end
    
    p2 = plot(valid_losses,
              label="",
              xlabel="Epoch",
              ylabel="Negative Log Likelihood",
              title="Training Loss Curve",
              linewidth=2,
              color=:orange,
              yscale=:log10,
              size=(700, 500))

    # Residual plot
    residuals = μ_train .- P_ami
    p3 = scatter(times_ami, residuals,
                 label="Residuals",
                 xlabel="Time (hours)",
                 ylabel="Residuals (kW)",
                 title="Prediction Residuals",
                 markersize=4,
                 alpha=0.7,
                 color=:purple,
                 size=(700, 500))
    hline!(p3, [0], linestyle=:dash, color=:black, linewidth=2, label="Zero")
    hline!(p3, [2*std(residuals), -2*std(residuals)], 
           linestyle=:dot, color=:gray, linewidth=1, label="±2σ")

    # Uncertainty plot
    p4 = plot(x_pred, σ_pred,
              label="Prediction Std Dev",
              xlabel="Time (hours)",
              ylabel="Standard Deviation (kW)",
              title="Prediction Uncertainty",
              linewidth=2,
              color=:red,
              size=(700, 500))
    hline!(p4, [mean(σ_pred)], 
           linestyle=:dash, color=:black, linewidth=1, 
           label="Mean: $(round(mean(σ_pred), digits=3)) kW")

    # Hyperparameter evolution (normalized scale)
    p5 = plot(result.ℓ_history,
              label="Length Scale ℓ",
              xlabel="Epoch",
              ylabel="Value (Normalized)",
              title="Hyperparameter Evolution",
              linewidth=2,
              color=:green,
              legend=:topright,
              size=(700, 500))
    plot!(p5, result.σ_s_history,
          label="Signal Std σ_s",
          linewidth=2,
          color=:blue)
    plot!(p5, result.σ_noise_history,
          label="Noise Std σ_noise",
          linewidth=2,
          color=:red)

    # Q-Q plot
    sorted_residuals = sort(residuals)
    theoretical_quantiles = quantile(Normal(0, std(residuals)), 
                                     range(0.01, 0.99, length=length(residuals)))
    
    p6 = scatter(sorted_residuals, theoretical_quantiles,
                 label="Data",
                 xlabel="Sample Quantiles",
                 ylabel="Theoretical Quantiles",
                 title="Q-Q Plot (Normality Test)",
                 markersize=3,
                 alpha=0.6,
                 color=:purple,
                 size=(700, 500))
    plot!(p6, sorted_residuals, sorted_residuals,
          label="Theoretical Line",
          linestyle=:dash,
          linewidth=2,
          color=:black)

    # Display and save plots individually to avoid layout issues
    println("\nDisplaying plots...")
    
    try
        display(p1)
        savefig(p1, "gp_prediction.png")
        println("  ✓ Prediction plot saved: gp_prediction.png")
    catch e
        println("  ✗ Error saving prediction plot: $e")
    end
    
    try
        display(p2)
        savefig(p2, "gp_loss.png")
        println("  ✓ Loss curve saved: gp_loss.png")
    catch e
        println("  ✗ Error saving loss plot: $e")
    end
    
    try
        display(p3)
        savefig(p3, "gp_residuals.png")
        println("  ✓ Residuals plot saved: gp_residuals.png")
    catch e
        println("  ✗ Error saving residuals plot: $e")
    end
    
    try
        display(p4)
        savefig(p4, "gp_uncertainty.png")
        println("  ✓ Uncertainty plot saved: gp_uncertainty.png")
    catch e
        println("  ✗ Error saving uncertainty plot: $e")
    end
    
    try
        display(p5)
        savefig(p5, "gp_hyperparameters.png")
        println("  ✓ Hyperparameter evolution saved: gp_hyperparameters.png")
    catch e
        println("  ✗ Error saving hyperparameter plot: $e")
    end
    
    try
        display(p6)
        savefig(p6, "gp_qq_plot.png")
        println("  ✓ Q-Q plot saved: gp_qq_plot.png")
    catch e
        println("  ✗ Error saving Q-Q plot: $e")
    end
    
    # Try to create combined plot with better spacing
    try
        final_plot = plot(p1, p2, p3, p4, p5, p6, 
                         layout=(3, 2), 
                         size=(1600, 1200),
                         margin=5Plots.mm,
                         left_margin=10Plots.mm,
                         bottom_margin=10Plots.mm)
        display(final_plot)
        savefig(final_plot, "gp_results_combined.png")
        println("  ✓ Combined plot saved: gp_results_combined.png")
    catch e
        println("  ✗ Could not create combined plot (individual plots saved): $e")
    end

    println("\nCompleted!")
    println("="^70)
    
    # Print summary
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("Model Performance:")
    println("  - RMSE: $(round(rmse, digits=4)) kW ($(round(rmse/std(P_ami)*100, digits=2))% of data std)")
    println("  - MAPE: $(round(mape, digits=2))%")
    println("  - 95% CI Coverage: $(round(coverage, digits=2))% (Target: 95%)")
    println("\nHyperparameters:")
    println("  - Length scale: $(round(result.ℓ, digits=2)) hours ($(round(result.ℓ/data_span*100, digits=1))% of span)")
    println("  - Signal-to-noise ratio: $(round(result.σ_s/result.σ_noise, digits=2))")
    println("\nIssues Detected:")
    if coverage < 90
        println("  ⚠️  Low CI coverage suggests model is overconfident")
        println("      → Consider increasing noise variance or using different kernel")
    end
    if mae < 0.5 && coverage < 80
        println("  ⚠️  Very low MAE but poor coverage indicates overfitting")
        println("      → Model fits training data too closely")
    end
    println("="^70)
    
    return result
end

result = main()