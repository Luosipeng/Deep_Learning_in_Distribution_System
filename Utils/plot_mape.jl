"""
绘制平均绝对百分比误差(MAPE)的累积分布函数(CDF)图
"""
function plot_mape_cdf(predictions, targets, n_nodes; title="Cumulative distribution function of forecast error", max_mape=1000.0)
    # 分离μ和ω
    μ_pred = predictions[1:n_nodes, :]
    ω_pred = predictions[n_nodes+1:end, :]
    
    μ_true = targets[1:n_nodes, :]
    ω_true = targets[n_nodes+1:end, :]
    
    # 计算MAPE
    μ_mape = calculate_mape(μ_pred, μ_true)
    ω_mape = calculate_mape(ω_pred, ω_true)
    
    # 过滤掉极端异常值
    μ_mape_filtered = filter(x -> x <= max_mape, μ_mape)
    ω_mape_filtered = filter(x -> x <= max_mape, ω_mape)
    
    # 排序MAPE值用于CDF
    μ_mape_sorted = sort(vec(μ_mape_filtered))
    ω_mape_sorted = sort(vec(ω_mape_filtered))
    
    # 计算CDF
    n_μ = length(μ_mape_sorted)
    n_ω = length(ω_mape_sorted)
    
    μ_cdf = collect(1:n_μ) ./ n_μ
    ω_cdf = collect(1:n_ω) ./ n_ω
    
    # 绘制CDF
    p = plot(μ_mape_sorted, μ_cdf, 
             label="μ (Reality)", 
             linewidth=2, 
             title=title, 
             xlabel="Mean absolute percentage error (MAPE, %)", 
             ylabel="Cumulative probability",
             legend=:bottomright,
             xlims=(0, min(max_mape, maximum(μ_mape_sorted), maximum(ω_mape_sorted))))
    
    plot!(p, ω_mape_sorted, ω_cdf, 
          label="ω (imagnary)", 
          linewidth=2)
    
    # 添加垂直线标记中位数MAPE
    μ_median_mape = μ_mape_sorted[Int(round(n_μ/2))]
    ω_median_mape = ω_mape_sorted[Int(round(n_ω/2))]
    
    vline!(p, [μ_median_mape], label="μMedian: $(round(μ_median_mape, digits=2))%", 
           linestyle=:dash, linecolor=:blue)
    vline!(p, [ω_median_mape], label="ωMedian: $(round(ω_median_mape, digits=2))%", 
           linestyle=:dash, linecolor=:red)
    
    # 添加95%分位点
    μ_95_mape = μ_mape_sorted[Int(round(n_μ*0.95))]
    ω_95_mape = ω_mape_sorted[Int(round(n_ω*0.95))]
    
    annotate!(p, [(μ_95_mape, 0.95, text("95%: $(round(μ_95_mape, digits=2))%", 8, :right, :blue)),
                  (ω_95_mape, 0.95, text("95%: $(round(ω_95_mape, digits=2))%", 8, :right, :red))])
    
    # 返回过滤前的完整MAPE数据，以便进行其他统计分析
    return p, μ_mape, ω_mape
end


"""
分析每个节点的MAPE
"""
function analyze_node_mape(predictions, targets, n_nodes)
    # 分离μ和ω
    μ_pred = predictions[1:n_nodes, :]
    ω_pred = predictions[n_nodes+1:end, :]
    
    μ_true = targets[1:n_nodes, :]
    ω_true = targets[n_nodes+1:end, :]
    
    # 计算每个节点的平均MAPE
    μ_node_mape = zeros(n_nodes)
    ω_node_mape = zeros(n_nodes)
    
    for i in 1:n_nodes
        μ_mape_i = calculate_mape(μ_pred[i,:], μ_true[i,:])
        ω_mape_i = calculate_mape(ω_pred[i,:], ω_true[i,:])
        
        μ_node_mape[i] = mean(μ_mape_i)
        ω_node_mape[i] = mean(ω_mape_i)
    end
    
    # 绘制每个节点的平均MAPE
    p = plot(1:n_nodes, μ_node_mape, 
             label="μ平均MAPE", 
             linewidth=2, 
             title="每个节点的平均绝对百分比误差", 
             xlabel="节点", 
             ylabel="MAPE (%)")
    
    plot!(p, 1:n_nodes, ω_node_mape, 
          label="ω平均MAPE", 
          linewidth=2)
    
    # 添加平均值标记
    hline!(p, [mean(μ_node_mape)], label="μ平均: $(round(mean(μ_node_mape), digits=2))%", 
           linestyle=:dash, linecolor=:blue)
    hline!(p, [mean(ω_node_mape)], label="ω平均: $(round(mean(ω_node_mape), digits=2))%", 
           linestyle=:dash, linecolor=:red)
    
    return p, μ_node_mape, ω_node_mape
end

