using Flux
using Random
# 采样潮流数据和结果
# 使用matpower中的算例
using Base.Threads
using TimeSeriesPowerFlow
using Distributions
using Plots

# include("../data/output_case.jl")
include("../src/generate_random_pf_samples_matpower.jl")
include("../src/get_batches.jl")
include("../solvers/train_neural_network_with_MLP.jl")
include("../solvers/train_neural_network_with_BNN.jl")

include("../Utils/plot_history.jl")
include("../Utils/calculate_mape.jl")
include("../Utils/plot_mape.jl")
include("../Utils/calculate_rmse.jl")
include("../Utils/plot_rmse.jl")


# 主程序入口
# Convert a MATPOWER case file to a Julia PowerCase
# Case 1 IEEE57
case_file_1 = "D:/luosipeng/matpower8.1/data/case57.m"
# case_file_1 ="C:/Users/13733/Desktop/matpower-8.0/data/case118.m"
μ1, ω1, Pg_inputs1, Qg_inputs1, Pd_inputs1, Qd_inputs1, Vg_inputs1, Vr_inputs1, θr_inputs1, G, B = generate_random_pf_samples_matpower(case_file_1)

# 1. 训练MLP+MLP模型
println("\n========== 训练MLP+MLP模型 ==========")

# 使用修改后的训练函数
mlp_encoder, mlp_decoder, mlp_predict_voltage, mlp_predict_power, mlp_predict_power_from_input,
(mlp_train_losses, mlp_val_losses, mlp_train_supervised_losses, mlp_train_unsupervised_losses),
(mlp_X_test, mlp_Y_voltage_test, mlp_Y_power_test),
(mlp_input_mean, mlp_input_std, mlp_voltage_output_mean, mlp_voltage_output_std,
mlp_power_output_mean, mlp_power_output_std) =
train_neural_network_with_MLP(μ1, ω1, Pg_inputs1, Qg_inputs1, Pd_inputs1, Qd_inputs1, Vg_inputs1, Vr_inputs1, θr_inputs1, G, B;
epochs=1000, batch_size=32, learning_rate=0.001, αsup=1.0, αunsup=0.5)

# 2. 训练MLP+BNN模型
println("\n========== 训练MLP+BNN模型 ==========")

bnn_encoder, bnn_decoder, bnn_predict_voltage, bnn_predict_power, bnn_predict_power_from_input,
(bnn_train_losses, bnn_val_losses, bnn_train_supervised_losses, bnn_train_unsupervised_losses),
(bnn_X_test, bnn_Y_voltage_test, bnn_Y_power_test),
(bnn_input_mean, bnn_input_std, bnn_voltage_output_mean, bnn_voltage_output_std,
bnn_power_output_mean, bnn_power_output_std) =
train_neural_network_with_BNN(μ1, ω1, Pg_inputs1, Qg_inputs1, Pd_inputs1, Qd_inputs1, Vg_inputs1, Vr_inputs1, θr_inputs1, G, B;
epochs=1000, batch_size=32, learning_rate=0.001, αsup=1.0, αunsup=0.5)

# 绘制训练历史对比
function plot_training_history_comparison(mlp_train_losses, mlp_val_losses, bnn_train_losses, bnn_val_losses)
    p = plot(1:length(mlp_train_losses), mlp_train_losses, label="MLP Train Loss", 
             linewidth=2, color=:blue, alpha=0.7)
    plot!(p, 1:length(mlp_val_losses), mlp_val_losses, label="MLP Validation Loss", 
          linewidth=2, color=:blue, linestyle=:dash, alpha=0.7)
    plot!(p, 1:length(bnn_train_losses), bnn_train_losses, label="BNN Train Loss", 
          linewidth=2, color=:red, alpha=0.7)
    plot!(p, 1:length(bnn_val_losses), bnn_val_losses, label="BNN Validation Loss", 
          linewidth=2, color=:red, linestyle=:dash, alpha=0.7)
    
    xlabel!(p, "Epoch")
    ylabel!(p, "Loss")
    title!(p, "Training and Validation Loss Comparison")
    
    return p
end

# 绘制训练历史对比
history_comparison = plot_training_history_comparison(
    mlp_train_losses, mlp_val_losses, 
    bnn_train_losses, bnn_val_losses
)
display(history_comparison)
# savefig(history_comparison, "history_comparison.png")

# 定义predict_voltage_norm函数
mlp_predict_voltage_norm(x) = mlp_encoder(x)
bnn_predict_voltage_norm(x) = bnn_encoder(x)

# 获取测试集预测结果
mlp_voltage_predictions_norm = mlp_predict_voltage_norm(mlp_X_test)
bnn_voltage_predictions_norm = bnn_predict_voltage_norm(bnn_X_test)

# 计算节点数量
n_nodes = Int(size(mlp_voltage_predictions_norm, 1) / 2)

# 计算μ和ω的RMSE
mlp_μ_rmse_norm, mlp_ω_rmse_norm = calculate_component_rmse(mlp_voltage_predictions_norm, mlp_Y_voltage_test, n_nodes)
bnn_μ_rmse_norm, bnn_ω_rmse_norm = calculate_component_rmse(bnn_voltage_predictions_norm, bnn_Y_voltage_test, n_nodes)

# 打印结果
println("\n========== MLP模型标准化空间RMSE ==========")
println("μ Standard RMSE: $(round(mlp_μ_rmse_norm, digits=6))")
println("ω Standard RMSE: $(round(mlp_ω_rmse_norm, digits=6))")

println("\n========== BNN模型标准化空间RMSE ==========")
println("μ Standard RMSE: $(round(bnn_μ_rmse_norm, digits=6))")
println("ω Standard RMSE: $(round(bnn_ω_rmse_norm, digits=6))")

# 计算原始空间的RMSE
mlp_μ_std = mlp_voltage_output_std[1:n_nodes]
mlp_ω_std = mlp_voltage_output_std[n_nodes+1:end]
bnn_μ_std = bnn_voltage_output_std[1:n_nodes]
bnn_ω_std = bnn_voltage_output_std[n_nodes+1:end]

mlp_μ_rmse_original = mlp_μ_rmse_norm * mean(mlp_μ_std)
mlp_ω_rmse_original = mlp_ω_rmse_norm * mean(mlp_ω_std)
bnn_μ_rmse_original = bnn_μ_rmse_norm * mean(bnn_μ_std)
bnn_ω_rmse_original = bnn_ω_rmse_norm * mean(bnn_ω_std)

println("\n========== MLP模型原始空间RMSE ==========")
println("μ Origin RMSE: $(round(mlp_μ_rmse_original, digits=6))")
println("ω Origin RMSE: $(round(mlp_ω_rmse_original, digits=6))")

println("\n========== BNN模型原始空间RMSE ==========")
println("μ Origin RMSE: $(round(bnn_μ_rmse_original, digits=6))")
println("ω Origin RMSE: $(round(bnn_ω_rmse_original, digits=6))")

# 将预测和真实值反标准化
mlp_voltage_predictions_original = mlp_voltage_predictions_norm .* mlp_voltage_output_std .+ mlp_voltage_output_mean
mlp_voltage_true_original = mlp_Y_voltage_test .* mlp_voltage_output_std .+ mlp_voltage_output_mean

bnn_voltage_predictions_original = bnn_voltage_predictions_norm .* bnn_voltage_output_std .+ bnn_voltage_output_mean
bnn_voltage_true_original = bnn_Y_voltage_test .* bnn_voltage_output_std .+ bnn_voltage_output_mean

# 修改plot_mape_cdf函数以支持多模型对比
function plot_mape_cdf_comparison(mlp_preds, mlp_true, bnn_preds, bnn_true, n_nodes; 
                                 title="MAPE Comparison", max_x=nothing)
    # 计算MLP的MAPE
    mlp_μ_preds = mlp_preds[1:n_nodes, :]
    mlp_ω_preds = mlp_preds[n_nodes+1:end, :]
    mlp_μ_true = mlp_true[1:n_nodes, :]
    mlp_ω_true = mlp_true[n_nodes+1:end, :]
    
    mlp_μ_mape = vec(mean(abs.((mlp_μ_preds .- mlp_μ_true) ./ mlp_μ_true) .* 100.0, dims=2))
    mlp_ω_mape = vec(mean(abs.((mlp_ω_preds .- mlp_ω_true) ./ mlp_ω_true) .* 100.0, dims=2))
    
    # 计算BNN的MAPE
    bnn_μ_preds = bnn_preds[1:n_nodes, :]
    bnn_ω_preds = bnn_preds[n_nodes+1:end, :]
    bnn_μ_true = bnn_true[1:n_nodes, :]
    bnn_ω_true = bnn_true[n_nodes+1:end, :]
    
    bnn_μ_mape = vec(mean(abs.((bnn_μ_preds .- bnn_μ_true) ./ bnn_μ_true) .* 100.0, dims=2))
    bnn_ω_mape = vec(mean(abs.((bnn_ω_preds .- bnn_ω_true) ./ bnn_ω_true) .* 100.0, dims=2))
    
    # 排序MAPE值以绘制CDF
    sorted_mlp_μ_mape = sort(mlp_μ_mape)
    sorted_mlp_ω_mape = sort(mlp_ω_mape)
    sorted_bnn_μ_mape = sort(bnn_μ_mape)
    sorted_bnn_ω_mape = sort(bnn_ω_mape)
    
    # 计算CDF
    cdf_y_μ = collect(range(0, 1, length=length(sorted_mlp_μ_mape)))
    cdf_y_ω = collect(range(0, 1, length=length(sorted_mlp_ω_mape)))
    
    # 创建子图布局
    p = plot(layout=(1,2), size=(900, 400), margin=5Plots.mm)
    
    # 左侧子图：μ的MAPE-CDF
    plot!(p[1], sorted_mlp_μ_mape, cdf_y_μ, label="MLP", linewidth=2, color=:blue)
    plot!(p[1], sorted_bnn_μ_mape, cdf_y_μ, label="BNN", linewidth=2, color=:red)
    
    # 右侧子图：ω的MAPE-CDF
    plot!(p[2], sorted_mlp_ω_mape, cdf_y_ω, label="MLP", linewidth=2, color=:blue)
    plot!(p[2], sorted_bnn_ω_mape, cdf_y_ω, label="BNN", linewidth=2, color=:red)
    
    # 设置横坐标最大值（如果提供）
    if max_x !== nothing
        xlims!(p[1], 0, max_x)
        xlims!(p[2], 0, max_x)
    end
    
    # 设置子图标题和轴标签
    title!(p[1], "μ MAPE-CDF")
    title!(p[2], "ω MAPE-CDF")
    
    xlabel!(p[1], "MAPE (%)")
    xlabel!(p[2], "MAPE (%)")
    ylabel!(p[1], "Cumulative Probability")
    ylabel!(p[2], "Cumulative Probability")
    
    # 设置整体标题
    plot!(p, title=title)
    
    return p, (mlp_μ_mape, mlp_ω_mape), (bnn_μ_mape, bnn_ω_mape)
end

# 绘制MAPE-CDF对比图，将μ和ω分别放在两个子图中
mape_comparison_plot, (mlp_μ_mape, mlp_ω_mape), (bnn_μ_mape, bnn_ω_mape) = 
    plot_mape_cdf_comparison(
        mlp_voltage_predictions_original, mlp_voltage_true_original,
        bnn_voltage_predictions_original, bnn_voltage_true_original,
        n_nodes, 
        title="MLP vs BNN Voltage Prediction Error (MAPE)",
        max_x=1000.0  # 设置横坐标最大值，根据实际数据调整
    )
display(mape_comparison_plot)
# savefig(mape_comparison_plot, "mape_comparison.png")
