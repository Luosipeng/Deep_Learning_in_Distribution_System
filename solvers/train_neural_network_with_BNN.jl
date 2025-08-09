# 将BNN参数包装为一个结构体
struct BNNDecoder
    W_G::Matrix{Float32}
    W_B::Matrix{Float32}
    b_p::Vector{Float32}
    b_q::Vector{Float32}
    n_nodes::Int  # 添加节点数作为结构体字段
end

# 定义BNN解码器的前向传播函数 - 避免数组原地修改
function (decoder::BNNDecoder)(voltage)
    # 从voltage中提取μ和ω
    μ = voltage[1:decoder.n_nodes, :]
    ω = voltage[decoder.n_nodes+1:end, :]
    
    batch_size = size(voltage, 2)
    
    # 使用map或列表推导式而不是原地修改数组
    results = map(1:batch_size) do b
        μ_b = μ[:, b]
        ω_b = ω[:, b]
        
        # 计算S₁和S₂
        μμᵀ = μ_b * μ_b'
        ωωᵀ = ω_b * ω_b'
        ωμᵀ = ω_b * μ_b'
        μωᵀ = μ_b * ω_b'
        
        S₁ = (μμᵀ + ωωᵀ) * decoder.W_G + (ωμᵀ - μωᵀ) * decoder.W_B
        S₂ = (ωμᵀ - μωᵀ) * decoder.W_G - (μμᵀ + ωωᵀ) * decoder.W_B
        
        # 计算功率预测
        P_pred_b = vec(sum(S₁, dims=2)) + decoder.b_p
        Q_pred_b = vec(sum(S₂, dims=2)) + decoder.b_q
        
        return [P_pred_b; Q_pred_b]  # 连接单个样本的P和Q预测
    end
    
    # 将结果转换为矩阵
    return hcat(results...)
end

"""
神经网络训练 MLP+BNN
"""
function train_neural_network_with_BNN(μ, ω, Pg_inputs, Qg_inputs, Pd_inputs, Qd_inputs, Vg_inputs, Vr_inputs, θr_inputs, G, B; 
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
    
    # 定义Encoder架构
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
    
    # 获取系统节点数
    n_nodes = div(voltage_output_size, 2)
    
    # 初始化BNN解码器的参数
    W_G = Float32.(Flux.glorot_uniform(n_nodes, n_nodes))
    W_B = Float32.(Flux.glorot_uniform(n_nodes, n_nodes))
    b_p = zeros(Float32, n_nodes)
    b_q = zeros(Float32, n_nodes)
    
    # 创建BNN解码器实例
    bnn_decoder = BNNDecoder(W_G, W_B, b_p, b_q, n_nodes)
    
    # 创建联合模型
    model = (encoder=encoder, decoder=bnn_decoder)
    
    # 存储训练历史
    train_losses = Float64[]
    val_losses = Float64[]
    train_supervised_losses = Float64[]
    train_unsupervised_losses = Float64[]
    learning_rates = Float64[]
    
    # 训练循环
    best_val_loss = Inf
    best_model = deepcopy(model)
    patience = 30
    patience_counter = 0
    
    @info "开始训练Encoder-BNN架构 (αsup=$αsup, αunsup=$αunsup)..."
    @info "损失函数: 监督损失(电压预测误差) + 无监督损失(功率重建)"
    @info "使用节点净功率注入 (P_net = Pg - Pd, Q_net = Qg - Qd) 作为BNN的目标"
    @info "BNN解码器使用物理模型: S₁ = (μμᵀ + ωωᵀ)W_G + (ωμᵀ - μωᵀ)W_B, S₂ = (ωμᵀ - μωᵀ)W_G - (μμᵀ + ωωᵀ)W_B"
    
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
        opt_state = Flux.setup(opt, model)
        
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
            loss_val, grads = Flux.withgradient(model) do m
                # 前向传播
                y_voltage_pred = m.encoder(x_batch)
                y_power_pred = m.decoder(y_voltage_pred)
                
                # 1. 监督损失 - 电压预测误差
                supervised_loss = Flux.mse(y_voltage_pred, y_voltage_batch)
                
                # 2. 无监督损失 - 功率预测误差
                unsupervised_loss = Flux.mse(y_power_pred, y_power_batch)
                
                # 总损失
                αsup * supervised_loss + αunsup * unsupervised_loss
            end
            
            # 更新参数
            Flux.update!(opt_state, model, grads[1])
            
            # 计算当前批次的损失
            y_voltage_pred = model.encoder(x_batch)
            y_power_pred = model.decoder(y_voltage_pred)
            
            supervised_loss = Flux.mse(y_voltage_pred, y_voltage_batch)
            unsupervised_loss = Flux.mse(y_power_pred, y_power_batch)
            batch_loss = αsup * supervised_loss + αunsup * unsupervised_loss
            
            epoch_loss += batch_loss
            epoch_supervised_loss += supervised_loss
            epoch_unsupervised_loss += unsupervised_loss
            num_batches += 1
        end
        
        # 平均损失
        avg_train_loss = epoch_loss / num_batches
        avg_train_supervised = epoch_supervised_loss / num_batches
        avg_train_unsupervised = epoch_unsupervised_loss / num_batches
        
        # 验证损失
        y_voltage_val_pred = model.encoder(X_val)
        y_power_val_pred = model.decoder(y_voltage_val_pred)
        
        val_supervised_loss = Flux.mse(y_voltage_val_pred, Y_voltage_val)
        val_unsupervised_loss = Flux.mse(y_power_val_pred, Y_power_val)
        val_loss = αsup * val_supervised_loss + αunsup * val_unsupervised_loss
        
        # 记录历史
        push!(train_losses, avg_train_loss)
        push!(val_losses, val_loss)
        push!(train_supervised_losses, avg_train_supervised)
        push!(train_unsupervised_losses, avg_train_unsupervised)
        
        # 打印进度
        if epoch % 10 == 0 || epoch == 1
            @info "Epoch $epoch/$epochs - Train: $(round(avg_train_loss, digits=4)) (Sup: $(round(avg_train_supervised, digits=4)), Unsup: $(round(avg_train_unsupervised, digits=4))), Val: $(round(val_loss, digits=4)), LR: $current_lr"
        end
        
        # 早停检查
        if val_loss < best_val_loss * 0.995
            best_val_loss = val_loss
            best_model = deepcopy(model)
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
    model = best_model
    
    # 测试评估
    y_voltage_test_pred = model.encoder(X_test)
    y_power_test_pred = model.decoder(y_voltage_test_pred)
    
    test_supervised_loss = Flux.mse(y_voltage_test_pred, Y_voltage_test)
    test_unsupervised_loss = Flux.mse(y_power_test_pred, Y_power_test)
    test_loss = αsup * test_supervised_loss + αunsup * test_unsupervised_loss
    
    @info "训练完成 - 测试损失: $(round(test_loss, digits=4))"
    @info "  监督损失(电压): $(round(test_supervised_loss, digits=4))"
    @info "  无监督损失(功率重建): $(round(test_unsupervised_loss, digits=4))"
    
    # 创建预测函数
    function predict_voltage_norm(x_norm)
        return model.encoder(x_norm)
    end
    
    function predict_power_norm(voltage_norm)
        return model.decoder(voltage_norm)
    end
    
    function predict_voltage(x)
        # 标准化输入
        x_norm = (x .- input_mean) ./ input_std
        # 预测标准化电压
        voltage_norm = predict_voltage_norm(x_norm)
        # 反标准化电压
        return voltage_norm .* voltage_output_std .+ voltage_output_mean
    end
    
    function predict_power(voltage)
        # 标准化电压
        voltage_norm = (voltage .- voltage_output_mean) ./ voltage_output_std
        # 预测标准化功率
        power_norm = predict_power_norm(voltage_norm)
        # 反标准化功率
        return power_norm .* power_output_std .+ power_output_mean
    end
    
    function predict_power_from_input(x)
        # 预测电压
        voltage = predict_voltage(x)
        # 从电压预测功率
        return predict_power(voltage)
    end
    
    # 计算详细的RMSE
    voltage_predictions_norm = predict_voltage_norm(X_test)
    power_predictions_norm = predict_power_norm(voltage_predictions_norm)
    
    voltage_rmse_norm = sqrt(mean((voltage_predictions_norm - Y_voltage_test).^2))
    power_rmse_norm = sqrt(mean((power_predictions_norm - Y_power_test).^2))
    
    @info "最终RMSE (标准化空间):"
    @info "电压RMSE: $(round(voltage_rmse_norm, digits=6))"
    @info "功率RMSE: $(round(power_rmse_norm, digits=6))"
    
    # 提取BNN参数
    bnn_params = (
        W_G = model.decoder.W_G,
        W_B = model.decoder.W_B,
        b_p = model.decoder.b_p,
        b_q = model.decoder.b_q
    )
    
    return model.encoder, bnn_params, predict_voltage, predict_power, predict_power_from_input,
           (train_losses, val_losses, train_supervised_losses, train_unsupervised_losses),
           (X_test, Y_voltage_test, Y_power_test),
           (input_mean, input_std, voltage_output_mean, voltage_output_std, power_output_mean, power_output_std)
end

"""
使用BNN计算功率 - 无原地修改版本
"""
function bnn_power_calculation(voltage, W_G, W_B, b_p, b_q)
    # 获取节点数
    n_nodes = div(size(voltage, 1), 2)
    
    # 从voltage中提取μ和ω
    μ = voltage[1:n_nodes, :]
    ω = voltage[n_nodes+1:end, :]
    
    batch_size = size(voltage, 2)
    
    # 使用函数式编程方法
    results = map(1:batch_size) do b
        μ_b = μ[:, b]
        ω_b = ω[:, b]
        
        # 计算S₁和S₂
        μμᵀ = μ_b * μ_b'
        ωωᵀ = ω_b * ω_b'
        ωμᵀ = ω_b * μ_b'
        μωᵀ = μ_b * ω_b'
        
        S₁ = (μμᵀ + ωωᵀ) * W_G + (ωμᵀ - μωᵀ) * W_B
        S₂ = (ωμᵀ - μωᵀ) * W_G - (μμᵀ + ωωᵀ) * W_B
        
        # 计算功率预测
        P_pred_b = vec(sum(S₁, dims=2)) + b_p
        Q_pred_b = vec(sum(S₂, dims=2)) + b_q
        
        return P_pred_b, Q_pred_b
    end
    
    # 解构结果
    P_preds = [p for (p, _) in results]
    Q_preds = [q for (_, q) in results]
    
    # 将结果转换为矩阵
    P_pred = hcat(P_preds...)
    Q_pred = hcat(Q_preds...)
    
    return P_pred, Q_pred
end

"""
计算损失函数 - 使用BNN物理模型
"""
function compute_loss(encoder_model, decoder_model, x, y_voltage, y_power, G, B, 
                    αsup=1.0, αunsup=0.5)
    # Encoder预测电压
    y_voltage_pred = encoder_model(x)
    
    # Decoder预测功率
    y_power_pred = decoder_model(y_voltage_pred)
    
    # 1. 监督损失 - 电压预测误差
    supervised_loss = Flux.mse(y_voltage_pred, y_voltage)
    
    # 2. 无监督损失 - 功率预测误差
    unsupervised_loss = Flux.mse(y_power_pred, y_power)
    
    # 总损失
    total_loss = αsup * supervised_loss + αunsup * unsupervised_loss
    
    return total_loss, supervised_loss, unsupervised_loss
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
    
    # 2. 无监督损失 - 功率预测误差
    unsupervised_loss = Flux.mse(y_power_pred, y_power)
    
    # 总损失
    total_loss = αsup * supervised_loss + αunsup * unsupervised_loss
    
    return total_loss, supervised_loss, unsupervised_loss
end
