"""
创建预测函数
"""
function create_prediction_functions(encoder, decoder, input_mean, input_std, 
                                  voltage_output_mean, voltage_output_std, 
                                  power_output_mean, power_output_std)
  # 预测函数 - 在标准化空间工作
  function predict_voltage_norm(x_norm)
      # 使用encoder预测标准化电压
      return encoder(x_norm)
  end
  
  function predict_power_norm(voltage_norm)
      # 使用decoder预测标准化功率
      return decoder(voltage_norm)
  end
  
  # 预测函数 - 处理原始数据并返回原始尺度结果
  function predict_voltage(x)
      # 标准化输入
      x_norm = Float32.((x .- input_mean) ./ input_std)
      # 使用encoder预测电压
      y_voltage_norm = encoder(x_norm)
      # 反标准化输出
      y_voltage = Array{eltype(x)}(y_voltage_norm .* voltage_output_std .+ voltage_output_mean)
      return y_voltage
  end
  
  function predict_power(voltage)
      # 标准化电压输入
      voltage_norm = Float32.((voltage .- voltage_output_mean) ./ voltage_output_std)
      # 使用decoder预测功率
      y_power_norm = decoder(voltage_norm)
      # 反标准化输出
      y_power = Array{eltype(voltage)}(y_power_norm .* power_output_std .+ power_output_mean)
      return y_power
  end
  
  function predict_power_from_input(x)
      # 先预测电压，再预测功率
      voltage = predict_voltage(x)
      power = predict_power(voltage)
      return power
  end
  
  return predict_voltage_norm, predict_power_norm, predict_voltage, predict_power, predict_power_from_input
end