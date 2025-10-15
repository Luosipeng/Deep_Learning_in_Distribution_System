# MLP求解器 - 使用Flux.jl，包含Gurobi对比测试（优化版）
# 需要安装: Pkg.add(["Flux", "Statistics", "Random", "Plots", "JuMP", "Gurobi", "LinearAlgebra"])

using Flux
using Flux: train!, params
using Statistics
using Random
using Plots
using JuMP
using Gurobi
using LinearAlgebra

# 关闭Gurobi的所有输出
ENV["GRB_LICENSE_FILE"] = ""

"""
MLP求解器结构体
"""
mutable struct MLPSolver
    model::Chain
    loss_history::Vector{Float64}
    val_loss_history::Vector{Float64}
    optimizer
    
    function MLPSolver(input_dim::Int, hidden_dims::Vector{Int}, output_dim::Int; 
                       activation=relu, output_activation=identity, learning_rate=0.001)
        # 构建网络层（增加Dropout和BatchNorm）
        layers = []
        
        # 输入层到第一个隐藏层
        push!(layers, Dense(input_dim, hidden_dims[1], activation))
        push!(layers, Dropout(0.2))
        
        # 隐藏层之间
        for i in 1:(length(hidden_dims)-1)
            push!(layers, Dense(hidden_dims[i], hidden_dims[i+1], activation))
            push!(layers, Dropout(0.2))
        end
        
        # 最后一个隐藏层到输出层
        push!(layers, Dense(hidden_dims[end], output_dim, output_activation))
        
        # 创建模型
        model = Chain(layers...)
        
        # 优化器
        opt = Adam(learning_rate)
        
        new(model, Float64[], Float64[], opt)
    end
end

"""
训练模型
"""
function train_model!(solver::MLPSolver, X_train, y_train, X_val, y_val;
                     epochs=100, batch_size=32, loss_fn=Flux.mse, verbose=true)
    
    n_samples = size(X_train, 2)
    n_batches = ceil(Int, n_samples / batch_size)
    
    best_val_loss = Inf
    patience = 20
    patience_counter = 0
    
    for epoch in 1:epochs
        # 打乱数据
        indices = shuffle(1:n_samples)
        epoch_loss = 0.0
        
        for batch in 1:n_batches
            # 获取批次数据
            batch_start = (batch - 1) * batch_size + 1
            batch_end = min(batch * batch_size, n_samples)
            batch_indices = indices[batch_start:batch_end]
            
            X_batch = X_train[:, batch_indices]
            y_batch = y_train[:, batch_indices]
            
            # 计算梯度并更新参数
            loss, grads = Flux.withgradient(params(solver.model)) do
                ŷ = solver.model(X_batch)
                loss_fn(ŷ, y_batch)
            end
            
            Flux.update!(solver.optimizer, params(solver.model), grads)
            epoch_loss += loss
        end
        
        # 记录训练损失
        avg_train_loss = epoch_loss / n_batches
        push!(solver.loss_history, avg_train_loss)
        
        # 计算验证损失（关闭Dropout）
        Flux.testmode!(solver.model)
        ŷ_val = solver.model(X_val)
        val_loss = loss_fn(ŷ_val, y_val)
        push!(solver.val_loss_history, val_loss)
        Flux.trainmode!(solver.model)
        
        # 早停
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        if patience_counter >= patience
            println("早停于 epoch $epoch")
            break
        end
        
        # 打印进度
        if verbose && (epoch % 10 == 0 || epoch == 1)
            println("Epoch $epoch/$epochs - Train Loss: $(round(avg_train_loss, digits=6)) - Val Loss: $(round(val_loss, digits=6))")
        end
    end
end

"""
预测
"""
function predict(solver::MLPSolver, X)
    Flux.testmode!(solver.model)
    result = solver.model(X)
    Flux.trainmode!(solver.model)
    return result
end

"""
评估模型
"""
function evaluate(solver::MLPSolver, X_test, y_test; loss_fn=Flux.mse)
    ŷ = predict(solver, X_test)
    test_loss = loss_fn(ŷ, y_test)
    
    # 计算R²分数（用于回归问题）
    ss_res = sum((y_test .- ŷ).^2)
    ss_tot = sum((y_test .- mean(y_test)).^2)
    r2_score = 1 - ss_res / ss_tot
    
    # 计算MAE
    mae = mean(abs.(y_test .- ŷ))
    
    return Dict("test_loss" => test_loss, "r2_score" => r2_score, "mae" => mae)
end

"""
绘制训练历史
"""
function plot_history(solver::MLPSolver)
    p = plot(1:length(solver.loss_history), solver.loss_history, 
             label="Training Loss", xlabel="Epoch", ylabel="Loss", 
             linewidth=2, legend=:topright, yscale=:log10)
    plot!(p, 1:length(solver.val_loss_history), solver.val_loss_history, 
          label="Validation Loss", linewidth=2)
    return p
end

# ============ Gurobi优化求解器 ============

"""
使用Gurobi求解二次规划问题
"""
function solve_quadratic_program_gurobi(Q, c, A_eq, b_eq, A_ineq, b_ineq, lb, ub; silent=true)
    n = length(c)
    
    # 创建Gurobi环境（完全静默）
    env = Gurobi.Env()
    if silent
        # 关闭所有输出
        Gurobi.GRBsetintparam(env, "OutputFlag", 0)
        Gurobi.GRBsetintparam(env, "LogToConsole", 0)
    end
    
    # 创建模型
    model = Model(() -> Gurobi.Optimizer(env))
    set_silent(model)
    
    # 定义变量
    @variable(model, lb[i] <= x[i=1:n] <= ub[i])
    
    # 二次目标函数
    @objective(model, Min, 0.5 * sum(Q[i,j] * x[i] * x[j] for i in 1:n, j in 1:n) + 
                           sum(c[i] * x[i] for i in 1:n))
    
    # 等式约束
    if !isnothing(A_eq) && size(A_eq, 1) > 0
        for i in 1:size(A_eq, 1)
            @constraint(model, sum(A_eq[i, j] * x[j] for j in 1:n) == b_eq[i])
        end
    end
    
    # 不等式约束
    if !isnothing(A_ineq) && size(A_ineq, 1) > 0
        for i in 1:size(A_ineq, 1)
            @constraint(model, sum(A_ineq[i, j] * x[j] for j in 1:n) <= b_ineq[i])
        end
    end
    
    # 求解
    optimize!(model)
    
    # 返回结果
    return Dict(
        "status" => termination_status(model),
        "objective" => objective_value(model),
        "solution" => value.(x),
        "solve_time" => solve_time(model)
    )
end

"""
计算二次规划的目标函数值
"""
function compute_qp_objective(x, Q, c)
    return 0.5 * dot(x, Q * x) + dot(c, x)
end

"""
训练MLP来学习优化问题的求解
"""
function train_mlp_for_optimization(problem_generator, n_samples; 
                                    input_dim, output_dim, 
                                    hidden_dims=[256, 128, 64, 32],
                                    epochs=200, batch_size=64)
    
    println("生成训练数据...")
    X_data = []
    y_data = []
    
    for i in 1:n_samples
        if i % 500 == 0
            println("  生成第 $i/$n_samples 个样本...")
        end
        problem, solution, _, _ = problem_generator()
        push!(X_data, problem)
        push!(y_data, solution)
    end
    
    # 转换为矩阵
    X = hcat(X_data...)
    y = hcat(y_data...)
    
    # 数据标准化
    X_mean = mean(X, dims=2)
    X_std = std(X, dims=2) .+ 1e-8
    X = (X .- X_mean) ./ X_std
    
    y_mean = mean(y, dims=2)
    y_std = std(y, dims=2) .+ 1e-8
    y = (y .- y_mean) ./ y_std
    
    # 划分数据集
    train_size = Int(floor(0.8 * n_samples))
    val_size = Int(floor(0.1 * n_samples))
    
    X_train = X[:, 1:train_size]
    y_train = y[:, 1:train_size]
    
    X_val = X[:, train_size+1:train_size+val_size]
    y_val = y[:, train_size+1:train_size+val_size]
    
    X_test = X[:, train_size+val_size+1:end]
    y_test = y[:, train_size+val_size+1:end]
    
    println("训练集: $(size(X_train, 2)), 验证集: $(size(X_val, 2)), 测试集: $(size(X_test, 2))")
    
    # 创建并训练MLP
    solver = MLPSolver(input_dim, hidden_dims, output_dim, learning_rate=0.001)
    
    println("\n训练MLP模型...")
    train_model!(solver, X_train, y_train, X_val, y_val,
                 epochs=epochs, batch_size=batch_size, verbose=true)
    
    return solver, X_test, y_test, (X_mean, X_std, y_mean, y_std)
end

"""
对比MLP和Gurobi的性能
"""
function compare_mlp_gurobi(mlp_solver, test_problems, normalization_params)
    X_mean, X_std, y_mean, y_std = normalization_params
    
    mlp_times = Float64[]
    gurobi_times = Float64[]
    mlp_objectives = Float64[]
    gurobi_objectives = Float64[]
    relative_errors = Float64[]
    solution_errors = Float64[]
    
    println("\n" * "="^80)
    println("MLP vs Gurobi 对比测试")
    println("="^80)
    
    for (i, (problem_params, Q, c, gurobi_solver_fn)) in enumerate(test_problems)
        # MLP预测
        problem_vec = (problem_params .- X_mean) ./ X_std
        
        mlp_time = @elapsed begin
            mlp_solution = predict(mlp_solver, problem_vec)
            mlp_solution = mlp_solution .* y_std .+ y_mean
        end
        
        # Gurobi求解
        gurobi_time = @elapsed begin
            gurobi_result = gurobi_solver_fn()
        end
        
        gurobi_solution = gurobi_result["solution"]
        
        # 使用正确的目标函数计算
        mlp_obj = compute_qp_objective(mlp_solution, Q, c)
        gurobi_obj = gurobi_result["objective"]
        
        # 目标函数值的相对误差
        obj_rel_error = abs(mlp_obj - gurobi_obj) / (abs(gurobi_obj) + 1e-8) * 100
        
        # 解的相对误差（L2范数）
        sol_error = norm(mlp_solution - gurobi_solution) / (norm(gurobi_solution) + 1e-8) * 100
        
        push!(mlp_times, mlp_time * 1000)
        push!(gurobi_times, gurobi_time * 1000)
        push!(mlp_objectives, mlp_obj)
        push!(gurobi_objectives, gurobi_obj)
        push!(relative_errors, obj_rel_error)
        push!(solution_errors, sol_error)
        
        if i <= 5
            println("\n测试问题 $i:")
            println("  MLP时间: $(round(mlp_times[i], digits=4)) ms")
            println("  Gurobi时间: $(round(gurobi_times[i], digits=4)) ms")
            println("  加速比: $(round(gurobi_times[i]/mlp_times[i], digits=2))x")
            println("  MLP目标值: $(round(mlp_obj, digits=6))")
            println("  Gurobi目标值: $(round(gurobi_obj, digits=6))")
            println("  目标值相对误差: $(round(obj_rel_error, digits=4))%")
            println("  解的相对误差: $(round(sol_error, digits=4))%")
        end
    end
    
    # 统计结果
    println("\n" * "="^80)
    println("统计结果 (基于 $(length(test_problems)) 个测试问题)")
    println("="^80)
    println("MLP平均求解时间: $(round(mean(mlp_times), digits=4)) ms")
    println("Gurobi平均求解时间: $(round(mean(gurobi_times), digits=4)) ms")
    println("平均加速比: $(round(mean(gurobi_times ./ mlp_times), digits=2))x")
    println("\n目标函数值误差:")
    println("  平均相对误差: $(round(mean(relative_errors), digits=4))%")
    println("  中位数相对误差: $(round(median(relative_errors), digits=4))%")
    println("  最大相对误差: $(round(maximum(relative_errors), digits=4))%")
    println("  最小相对误差: $(round(minimum(relative_errors), digits=4))%")
    println("\n解的误差:")
    println("  平均相对误差: $(round(mean(solution_errors), digits=4))%")
    println("  中位数相对误差: $(round(median(solution_errors), digits=4))%")
    println("  最大相对误差: $(round(maximum(solution_errors), digits=4))%")
    println("  最小相对误差: $(round(minimum(solution_errors), digits=4))%")
    
    # 绘制对比图
    p1 = bar(["MLP", "Gurobi"], [mean(mlp_times), mean(gurobi_times)],
             ylabel="平均求解时间 (ms)", title="求解时间对比",
             legend=false, color=[:blue, :red])
    
    p2 = histogram(relative_errors, bins=30, xlabel="目标值相对误差 (%)", 
                   ylabel="频数", title="目标值误差分布", legend=false)
    
    p3 = scatter(gurobi_objectives, mlp_objectives, 
                 xlabel="Gurobi目标值", ylabel="MLP目标值",
                 title="目标值对比", legend=false, alpha=0.6, markersize=4)
    plot!(p3, [minimum(gurobi_objectives), maximum(gurobi_objectives)],
          [minimum(gurobi_objectives), maximum(gurobi_objectives)],
          line=:dash, color=:red, linewidth=2, label="理想线")
    
    p4 = histogram(solution_errors, bins=30, xlabel="解的相对误差 (%)", 
                   ylabel="频数", title="解的误差分布", legend=false, color=:green)
    
    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))
    
    return Dict(
        "mlp_times" => mlp_times,
        "gurobi_times" => gurobi_times,
        "relative_errors" => relative_errors,
        "solution_errors" => solution_errors,
        "plot" => p
    )
end

# ============ 使用示例：求解二次规划问题 ============

println("\n" * "="^80)
println("示例：使用MLP学习求解二次规划问题")
println("="^80)

Random.seed!(42)

# 问题生成器：生成随机二次规划问题并用Gurobi求解
function generate_qp_problem()
    n = 5  # 变量数量
    
    # 生成随机正定矩阵Q（条件数不要太大）
    A_rand = randn(n, n) * 0.5
    Q = A_rand' * A_rand + Matrix(2.0I, n, n)  # 增加对角占优
    
    # 生成随机线性项
    c = randn(n) * 0.5
    
    # 边界
    lb = -ones(n) * 2.0
    ub = ones(n) * 2.0
    
    # 用Gurobi求解
    result = solve_quadratic_program_gurobi(Q, c, nothing, nothing, nothing, nothing, lb, ub, silent=true)
    
    # 问题参数作为输入（Q的上三角和c）
    problem_params = vcat([Q[i,j] for i in 1:n for j in i:n], c)
    solution = result["solution"]
    
    return Float32.(problem_params), Float32.(solution), Q, c
end

# 训练MLP学习QP求解（增加数据量和模型容量）
n_train_samples = 5000  # 增加到5000
input_dim = 5 * 6 ÷ 2 + 5
output_dim = 5

mlp_solver, X_test, y_test, norm_params = train_mlp_for_optimization(
    generate_qp_problem, 
    n_train_samples,
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dims=[256, 128, 64, 32],  # 更深的网络
    epochs=200,  # 更多轮次
    batch_size=64
)

# 评估MLP
println("\n" * "="^80)
println("MLP模型评估")
println("="^80)
results = evaluate(mlp_solver, X_test, y_test)
println("测试集MSE: $(round(results["test_loss"], digits=6))")
println("R² 分数: $(round(results["r2_score"], digits=6))")
println("MAE: $(round(results["mae"], digits=6))")

# 生成测试问题进行对比
println("\n生成测试问题...")
n_test = 100
test_problems = []

for i in 1:n_test
    if i % 25 == 0
        println("  生成测试问题 $i/$n_test...")
    end
    
    problem_params, _, Q, c = generate_qp_problem()
    
    lb = -ones(5) * 2.0
    ub = ones(5) * 2.0
    
    gurobi_solver_fn = () -> solve_quadratic_program_gurobi(Q, c, nothing, nothing, nothing, nothing, lb, ub, silent=true)
    
    push!(test_problems, (problem_params, Q, c, gurobi_solver_fn))
end

# 执行对比测试
comparison_results = compare_mlp_gurobi(mlp_solver, test_problems, norm_params)

# 显示对比图
display(comparison_results["plot"])

# 显示训练历史
println("\n显示训练历史...")
display(plot_history(mlp_solver))

println("\n" * "="^80)
println("完成！")
println("="^80)