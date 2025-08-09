"""
创建批次数据用于训练
"""
function get_batches(X, Y_voltage, Y_power, batch_size, shuffle_data=true)
  n = size(X, 2)
  idx = shuffle_data ? shuffle(1:n) : 1:n
  
  num_batches = div(n, batch_size)
  
  batches = []
  for i in 1:num_batches
      batch_idx = idx[(i-1)*batch_size+1:i*batch_size]
      push!(batches, (X[:, batch_idx], Y_voltage[:, batch_idx], Y_power[:, batch_idx]))
  end
  
  if n % batch_size != 0
      batch_idx = idx[num_batches*batch_size+1:end]
      push!(batches, (X[:, batch_idx], Y_voltage[:, batch_idx], Y_power[:, batch_idx]))
  end
  
  return batches
end

function physics_informed_loss(voltage_pred, G, B)
# 从预测的电压计算功率注入
nnodes = Int(size(voltage_pred,1)/2)
μ = voltage_pred[1:nnodes, :]
ω = voltage_pred[nnodes+1:end, :]

# 使用正确的功率计算公式
function compute_power_for_sample(i)
    μi = μ[:,i]
    ωi = ω[:,i]
    
    # 正确的有功功率和无功功率计算
    P = μi.*(G*μi - B*ωi) + ωi.*(B*μi + G*ωi)
    Q = ωi.*(G*μi - B*ωi) - μi.*(B*μi + G*ωi)
    
    return P, Q
end

# 计算所有样本的功率
power_results = [compute_power_for_sample(i) for i in 1:size(μ,2)]

# 转换为矩阵形式
P_cal = hcat([result[1] for result in power_results]...)
Q_cal = hcat([result[2] for result in power_results]...)

return P_cal, Q_cal
end