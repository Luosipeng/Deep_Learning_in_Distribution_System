include("../src/prediction_function.jl")

"""
神经网络训练 MLP+MLP
"""
function train_neural_network_with_MLP(μ, ω, Pg_inputs, Qg_inputs, Pd_inputs, Qd_inputs, Vg_inputs, Vr_inputs, θr_inputs, G, B; 
                             epochs=1000, batch_size=64, learning_rate=0.005,
                             αsup=1.0, αunsup=0.5)
    # 计算净功率注入
    P_net = Pg_inputs - Pd_inputs  # 有功功率净注入
    Q_net = Qg_inputs - Qd_inputs  # 无功功率净注入
    
    # 定义输入和输出
    inputs = vcat(Pd_inputs, Pg_inputs, Qd_inputs, Vg_inputs, Vr_inputs, θr_inputs)
    voltage_outputs = vcat(μ, ω)  # 电压幅值和相角
    power_outputs = vcat(P_net, Q_net)  # 有功和无功功率净注入
    
    # 获取数据维度
    input_size = size(inputs, 1)
    voltage_output_size = size(voltage_outputs, 1)
    power_output_size = size(power_outputs, 1)
    num_samples = size(inputs, 2)
    
    # 数据预处理：标准化
    input_mean = mean(inputs, dims=2)
    input_std = std(inputs, dims=2)
    input_std[input_std .< 1e-5] .= 1.0  # 避免除以接近零的值
    
    voltage_output_mean = mean(voltage_outputs, dims=2)
    voltage_output_std = std(voltage_outputs, dims=2)
    voltage_output_std[voltage_output_std .< 1e-5] .= 1.0
    
    power_output_mean = mean(power_outputs, dims=2)
    power_output_std = std(power_outputs, dims=2)
    power_output_std[power_output_std .< 1e-5] .= 1.0
    
    # 标准化数据
    inputs_norm = (inputs .- input_mean) ./ input_std
    voltage_outputs_norm = (voltage_outputs .- voltage_output_mean) ./ voltage_output_std
    power_outputs_norm = (power_outputs .- power_output_mean) ./ power_output_std
    
    # 将数据分为训练集(70%)、验证集(15%)和测试集(15%)
    indices = shuffle(1:num_samples)
    
    train_size = floor(Int, 0.7 * num_samples)
    val_size = floor(Int, 0.15 * num_samples)
    
        train_indices = indices[1:train_size]
    val_indices = indices[train_size+1:train_size+val_size]
    test_indices = indices[train_size+val_size+1:end]
    
    X_train = Float32.(inputs_norm[:, train_indices])
    Y_voltage_train = Float32.(voltage_outputs_norm[:, train_indices])
    Y_power_train = Float32.(power_outputs_norm[:, train_indices])
    
    X_val = Float32.(inputs_norm[:, val_indices])
    Y_voltage_val = Float32.(voltage_outputs_norm[:, val_indices])
    Y_power_val = Float32.(power_outputs_norm[:, val_indices])
    
    X_test = Float32.(inputs_norm[:, test_indices])
    Y_voltage_test = Float32.(voltage_outputs_norm[:, test_indices])
    Y_power_test = Float32.(power_outputs_norm[:, test_indices])
    
    # 定义Encoder-Decoder架构
    hidden_size1 = max(128, min(256, input_size * 3))
    hidden_size2 = max(64, min(128, input_size * 2))
    
    # Encoder网络 - 预测电压幅值和相角 (μ, ω)
    encoder = Chain(
        Dense(input_size => hidden_size1, relu; init=Flux.glorot_uniform),
        Dropout(0.1),
        Dense(hidden_size1 => hidden_size2, relu; init=Flux.glorot_uniform),
        Dropout(0.05),
        Dense(hidden_size2 => voltage_output_size; init=Flux.glorot_uniform)
    )
    
    # Decoder网络 - 从电压幅值和相角重建功率注入 (p, q)
    decoder = Chain(
        Dense(voltage_output_size => hidden_size2, relu; init=Flux.glorot_uniform),
        Dropout(0.05),
        Dense(hidden_size2 => hidden_size1, relu; init=Flux.glorot_uniform),
        Dropout(0.1),
        Dense(hidden_size1 => power_output_size; init=Flux.glorot_uniform)
    )
    
    # 创建联合模型用于优化器设置
    combined_model = (encoder=encoder, decoder=decoder)
    
    # 存储训练历史
    train_losses = Float64[]
    val_losses = Float64[]
    train_supervised_losses = Float64[]
    train_unsupervised_losses = Float64[]  # 修复变量名
    learning_rates = Float64[]
    
    # 训练循环
    best_val_loss = Inf
    best_encoder = deepcopy(encoder)
    best_decoder = deepcopy(decoder)
    patience = 30
    patience_counter = 0
    
    @info "开始训练Encoder-Decoder架构 (αsup=$αsup, αunsup=$αunsup)..."
    @info "损失函数: 监督损失(电压预测误差) + 无监督损失(功率重建)"
    @info "使用节点净功率注入 (P_net = Pg - Pd, Q_net = Qg - Qd) 作为Decoder的目标"
    
    for epoch in 1:epochs
        # 学习率调度
        if epoch <= 50
            current_lr = learning_rate
        elseif epoch <= 150
            current_lr = learning_rate * 0.5
        elseif epoch <= 300
            current_lr = learning_rate * 0.2
        else
            current_lr = learning_rate * 0.1
        end
        
        # 记录当前学习率
        push!(learning_rates, current_lr)
        
        # 创建优化器和状态
        opt = ADAM(current_lr)
        opt_state = Flux.setup(opt, combined_model)
        
        # 训练阶段
        epoch_loss = 0.0
        epoch_supervised_loss = 0.0
        epoch_unsupervised_loss = 0.0
        num_batches = 0
        
        # 批次训练
        for i in 1:div(size(X_train, 2), batch_size)
            start_idx = (i-1)*batch_size + 1
            end_idx = min(i*batch_size, size(X_train, 2))
            batch_indices = start_idx:end_idx
            
            x_batch = X_train[:, batch_indices]
            y_voltage_batch = Y_voltage_train[:, batch_indices]
            y_power_batch = Y_power_train[:, batch_indices]
            
            # 计算梯度和更新参数
            loss_val, grads = Flux.withgradient(combined_model) do m
                compute_loss_simplified(m.encoder, m.decoder, x_batch, y_voltage_batch, y_power_batch, 
                                      G, B, αsup, αunsup)[1]
            end
            
            # 更新参数
            Flux.update!(opt_state, combined_model, grads[1])
            
            # 记录损失
            batch_loss, batch_supervised, batch_unsupervised = 
                compute_loss_simplified(combined_model.encoder, combined_model.decoder, 
                                      x_batch, y_voltage_batch, y_power_batch, 
                                      G, B, αsup, αunsup)
            
            epoch_loss += batch_loss
            epoch_supervised_loss += batch_supervised
            epoch_unsupervised_loss += batch_unsupervised
            num_batches += 1
        end
        
        # 平均损失
        avg_train_loss = epoch_loss / num_batches
        avg_train_supervised = epoch_supervised_loss / num_batches
        avg_train_unsupervised = epoch_unsupervised_loss / num_batches  # 修复变量名
        
                # 验证损失
        val_loss, val_supervised, val_unsupervised = 
            compute_loss_simplified(combined_model.encoder, combined_model.decoder, 
                                  X_val, Y_voltage_val, Y_power_val, 
                                  G, B, αsup, αunsup)
        
        # 记录历史
        push!(train_losses, avg_train_loss)
        push!(val_losses, val_loss)
        push!(train_supervised_losses, avg_train_supervised)
        push!(train_unsupervised_losses, avg_train_unsupervised)  # 修复变量名
        
        # 打印进度
        if epoch % 10 == 0 || epoch == 1
            @info "Epoch $epoch/$epochs - Train: $(round(avg_train_loss, digits=4)) (Sup: $(round(avg_train_supervised, digits=4)), Unsup: $(round(avg_train_unsupervised, digits=4))), Val: $(round(val_loss, digits=4)), LR: $current_lr"
        end
        
        # 早停检查
        if val_loss < best_val_loss * 0.995
            best_val_loss = val_loss
            best_encoder = deepcopy(combined_model.encoder)
            best_decoder = deepcopy(combined_model.decoder)
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                @info "早停在 epoch $epoch"
                break
            end
        end
    end
    
    # 使用最佳模型
    encoder = best_encoder
    decoder = best_decoder
    
    # 测试评估
    test_loss, test_supervised, test_unsupervised = 
        compute_loss_simplified(encoder, decoder, X_test, Y_voltage_test, Y_power_test, 
                              G, B, αsup, αunsup)
    
    @info "训练完成 - 测试损失: $(round(test_loss, digits=4))"
    @info "  监督损失(电压): $(round(test_supervised, digits=4))"
    @info "  无监督损失(功率重建): $(round(test_unsupervised, digits=4))"
    
    # 创建预测函数
    predict_voltage_norm, predict_power_norm, predict_voltage, predict_power, predict_power_from_input = 
        create_prediction_functions(encoder, decoder, input_mean, input_std, 
                                   voltage_output_mean, voltage_output_std, 
                                   power_output_mean, power_output_std)
    
    # 计算详细的RMSE
    voltage_predictions_norm = predict_voltage_norm(X_test)
    power_predictions_norm = predict_power_norm(voltage_predictions_norm)
    
    voltage_rmse_norm = sqrt(mean((voltage_predictions_norm - Y_voltage_test).^2))
    power_rmse_norm = sqrt(mean((power_predictions_norm - Y_power_test).^2))
    
    @info "最终RMSE (标准化空间):"
    @info "电压RMSE: $(round(voltage_rmse_norm, digits=6))"
    @info "功率RMSE: $(round(power_rmse_norm, digits=6))"
    
    return encoder, decoder, predict_voltage, predict_power, predict_power_from_input,
           (train_losses, val_losses, train_supervised_losses, train_unsupervised_losses),
           (X_test, Y_voltage_test, Y_power_test),
           (input_mean, input_std, voltage_output_mean, voltage_output_std, power_output_mean, power_output_std)
end

"""
计算损失函数 - 添加物理约束
"""
function compute_loss(encoder_model, decoder_model, x, y_voltage, y_power, G, B, 
                    αsup=1.0, αunsup=0.5)
  return compute_loss_simplified(encoder_model, decoder_model, x, y_voltage, y_power, G, B, αsup, αunsup)
end

"""
简化的损失函数 - 只包含监督损失和重建损失
"""
function compute_loss_simplified(encoder_model, decoder_model, x, y_voltage, y_power, G, B, 
                              αsup=1.0, αunsup=0.5)
  # Encoder预测电压
  y_voltage_pred = encoder_model(x)
  
  # Decoder预测功率
  y_power_pred = decoder_model(y_voltage_pred)
  
  # 1. 监督损失 - 电压预测误差
  supervised_loss = Flux.mse(y_voltage_pred, y_voltage)
  
  # 2. 无监督损失(重建损失) - 预测功率 vs 从预测电压计算出的功率
  P_cal, Q_cal = physics_informed_loss(y_voltage_pred, G, B)
  power_calculated = vcat(P_cal, Q_cal)
  reconstruction_loss = Flux.mse(y_power_pred, power_calculated)
  
  # 总损失
  total_loss = αsup * supervised_loss + αunsup * reconstruction_loss
  
  return total_loss, supervised_loss, reconstruction_loss
end