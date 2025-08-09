"""
计算平均绝对百分比误差(MAPE)
"""
function calculate_mape(predictions, targets)
    # 避免除以零
    mask = abs.(targets) .> 1e-5
    
    # 计算每个点的绝对百分比误差
    ape = abs.((predictions[mask] .- targets[mask]) ./ targets[mask]) .* 100.0
    
    return ape
end

