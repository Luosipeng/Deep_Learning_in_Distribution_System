"""
计算标准化空间的RMSE
"""
function calculate_rmse_norm(predictions, targets)
  return sqrt(mean((predictions .- targets).^2))
end

"""
计算原始空间的RMSE，考虑标准差的影响
"""
function calculate_rmse_original(predictions_norm, targets_norm, std_values)
  # 将标准化空间的RMSE转换回原始空间
  return calculate_rmse_norm(predictions_norm, targets_norm) * mean(std_values)
end

# 计算μ和ω的RMSE值
function calculate_component_rmse(predictions, targets, n_nodes)
    # 分离μ和ω
    μ_pred = predictions[1:n_nodes, :]
    ω_pred = predictions[n_nodes+1:end, :]
    
    μ_true = targets[1:n_nodes, :]
    ω_true = targets[n_nodes+1:end, :]
    
    # 计算标准化空间的RMSE
    μ_rmse_norm = sqrt(mean((μ_pred .- μ_true).^2))
    ω_rmse_norm = sqrt(mean((ω_pred .- ω_true).^2))
    
    return μ_rmse_norm, ω_rmse_norm
end