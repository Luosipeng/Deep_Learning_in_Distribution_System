# 分析每个节点的预测精度
function analyze_node_accuracy(predictions, targets, n_nodes)
    # 分离μ和ω
    μ_pred = predictions[1:n_nodes, :]
    ω_pred = predictions[n_nodes+1:end, :]
    
    μ_true = targets[1:n_nodes, :]
    ω_true = targets[n_nodes+1:end, :]
    
    # 计算每个节点的RMSE
    μ_node_rmse = [sqrt(mean((μ_pred[i,:] .- μ_true[i,:]).^2)) for i in 1:n_nodes]
    ω_node_rmse = [sqrt(mean((ω_pred[i,:] .- ω_true[i,:]).^2)) for i in 1:n_nodes]
    
    # 绘制每个节点的RMSE
    p = plot(1:n_nodes, μ_node_rmse, 
             label="μ RMSE", 
             linewidth=2, 
             title="The prediction error of each node", 
             xlabel="node", 
             ylabel="RMSE")
    plot!(p, 1:n_nodes, ω_node_rmse, 
          label="ω RMSE", 
          linewidth=2)
    
    return p, μ_node_rmse, ω_node_rmse
end