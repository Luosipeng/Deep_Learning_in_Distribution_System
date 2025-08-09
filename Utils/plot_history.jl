"""
绘制训练历史
"""
function plot_training_history(train_losses, val_losses)
  plt = plot(1:length(train_losses), train_losses, 
             label="Training Loss", 
             linewidth=2, 
             xlabel="Epoch", 
             ylabel="Loss (MSE)",
             title="Training and Validation Loss History",
             legend=:topright)
  
  plot!(plt, 1:length(val_losses), val_losses, 
        label="Validation Loss", 
        linewidth=2)
  
  # 添加最终损失值标注
  annotate!(plt, [(length(train_losses), train_losses[end] + 0.05, 
                  text("Final Training Loss: $(round(train_losses[end], digits=4))", 8, :left)),
                  (length(val_losses), val_losses[end] + 0.05, 
                  text("Final Validation Loss: $(round(val_losses[end], digits=4))", 8, :left))])
  
  return plt
end

"""
简化版训练历史可视化
"""
function plot_simplified_training_history(train_losses, val_losses, train_supervised_losses, 
                                       train_unsupervised_losses)  # 修复参数名
  
  # 总损失图
  p1 = plot(1:length(train_losses), train_losses, 
            label="Training Total", linewidth=2, color=:blue,
            xlabel="Epoch", ylabel="Loss", title="Total Loss")
  plot!(p1, 1:length(val_losses), val_losses, 
        label="Validation Total", linewidth=2, color=:red)
  
  # 组件损失图
  p2 = plot(1:length(train_supervised_losses), train_supervised_losses, 
            label="Supervised (Voltage)", linewidth=2, color=:green,
            xlabel="Epoch", ylabel="Loss", title="Component Losses")
  plot!(p2, 1:length(train_unsupervised_losses), train_unsupervised_losses, 
        label="Unsupervised (Power Reconstruction)", linewidth=2, color=:orange)
  
  # 组合图
  combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
  
  return combined_plot
end