using TimeSeriesPowerFlow
include("../ios/matlab2julia.jl")
function generate_random_sample_for_upinn(case_file, num_samples=1000, load_factor_range=(0.8, 1.2))
  # 设置随机数种子，确保可重复性
  Random.seed!(42)
  
  # 加载MATPOWER格式的电力系统案例
  jpc = convert_matpower_case_dp(case_file, "data/output_case.jl")
  busAC = jpc.busAC
  genAC = jpc.genAC
  branchAC = jpc.branchAC
  loadAC = jpc.loadAC
  pvarray = jpc.pv
  (busAC, genAC, branchAC, loadAC, pvarray, i2e) = PowerFlow.ext2int(busAC, genAC, branchAC, loadAC, pvarray)
  jpc.busAC = busAC
  jpc.genAC = genAC   
  jpc.branchAC = branchAC
  jpc.loadAC = loadAC
  jpc.pv = pvarray

  bus_data = jpc.busAC
  branch_data = jpc.branchAC
  baseMVA = jpc.baseMVA
  
  # 获取系统的导纳矩阵
  Ybus, Yf, Yt = PowerFlow.makeYbus(baseMVA, bus_data, branch_data)

  pq_idx = findall(bus_data[:, BUS_TYPE] .== 1)  # PQ节点索引
  pv_idx = findall(bus_data[:, BUS_TYPE] .== 2)  # PV节点索引
  ref_idx = findall(bus_data[:, BUS_TYPE] .== 3)  # 参考节点索引
  # 存储结果的数组
  mpc_inputs = []  # 存储修改后的MATPOWER案例（输入）
  
  # 收集样本直到达到目标数量
  collected_samples = 0
  attempts = 0
  max_attempts = num_samples * 10  # 设置最大尝试次数
  # 定义样本矩阵
  X= zeros(size(jpc.busAC,1), 2, num_samples)
  
  println("开始生成随机潮流样本...")
  
  while collected_samples < num_samples && attempts < max_attempts
      attempts += 1
      
      # 复制原始案例
      case = deepcopy(jpc)
      
      # 随机修改负荷
      for i in 1:size(case.busAC, 1)
          if case.busAC[i, PD] > 0 || case.busAC[i, QD] > 0  # 只修改有负荷的节点
              # 生成随机负荷因子
               load_factor = rand(Uniform(load_factor_range[1], load_factor_range[2]))
              
              # 更新负荷值
              case.busAC[i, PD] = case.busAC[i, PD] * load_factor
              case.busAC[i, QD] = case.busAC[i, QD] * load_factor
          end
      end
      
      # 可选：调整发电机出力以平衡总负荷
      adjust_generation!(case)
      
      # 运行潮流计算
      opt = PowerFlow.options() # The initial settings
      opt["PF"]["NR_ALG"] = "bicgstab";
      opt["PF"]["ENFORCE_Q_LIMS"] = 0;
      opt["PF"]["DC_PREPROCESS"] = 0;

      input_case = deepcopy(case)
      result = runpf(case, opt)
      
      # 检查是否收敛
      if result.success == true
          # 保存输入案例
          push!(mpc_inputs, input_case)
          
          collected_samples += 1

          # 提取功率注入
          P_pq, Q_pq, P_pv, V_pv, V_ref, θ_ref = extract_input_variables(input_case)
          X[:,1,collected_samples] += P_pq # PQ节点
          X[:,2,collected_samples] += Q_pq # PQ节点
          X[:,1,collected_samples] += P_pv # PV节点
          X[:,2,collected_samples] += V_pv # PV节点
          X[:,1,collected_samples] += V_ref # 参考节点电压
          X[:,2,collected_samples] += θ_ref # 参考节点电压角度
          
          # 打印进度
          if collected_samples % 10 == 0
              println("已收集 $collected_samples/$num_samples 组样本 (尝试次数: $attempts)")
          end
      end
  end
  

  if collected_samples < num_samples
      println("警告：在 $max_attempts 次尝试后仅收集到 $collected_samples 组有效样本")
  else
      println("成功收集到 $num_samples 组样本，总尝试次数：$attempts")
  end
  
  # 返回收集的样本
  return  X, Ybus, pq_idx, pv_idx, ref_idx
end

"""
调整发电机出力以平衡总负荷
确保系统总发电量与总负荷相匹配
"""
function adjust_generation!(case)
  # 计算总负荷
  total_load = sum(case.busAC[:, PD])
  
  # 获取所有非参考节点的发电机
  non_ref_gens = []
  ref_gens = []

  for (i, gen_row) in enumerate(eachrow(case.genAC))
      bus_id = Int(gen_row[GEN_BUS])
      if case.busAC[bus_id, BUS_TYPE] != 3  # 非参考节点
          push!(non_ref_gens, (i, gen_row))
      else
          push!(ref_gens, (i, gen_row))
      end
  end

  
  # 先调整非参考节点的发电机
  if !isempty(non_ref_gens)
      # 计算这些发电机的当前总出力
      current_gen = sum(case.genAC[:,PG])
      
      # 计算需要调整的出力
      total_adjustment = total_load - current_gen
      
      # 按比例分配调整量
      if current_gen > 0
          for (gen_id, gen) in non_ref_gens
              adjustment_factor = gen[PG] / current_gen
              new_pg = gen[PG] + total_adjustment * adjustment_factor
              
              # 确保发电机出力在限制范围内
              new_pg = min(max(new_pg, gen[PMIN]), gen[PMAX])
              
              # 更新发电机出力
              case.genAC[gen_id,PG] = new_pg
          end
      end
  end
  
  # 如果非参考节点的发电机无法完全平衡负荷，剩余部分由参考节点平衡
  if !isempty(ref_gens)
      # 重新计算总负荷和总发电量
      total_load = sum(case.busAC[:, PD])
      total_gen = sum(case.genAC[:, PG])
      
      # 计算参考节点需要提供的功率
      ref_power_needed = total_load - total_gen
      
      # 将这个功率分配给参考节点的发电机
      if !isempty(ref_gens)
          ref_gen_id, ref_gen = ref_gens[1]  # 使用第一个参考节点发电机
          
          # 确保在限制范围内
          new_pg = min(max(ref_power_needed, ref_gen[PMIN]), ref_gen[PMAX])
          
          # 更新参考节点发电机出力
          case.genAC[ref_gen_id,PG] = new_pg
      end
  end
end

"""
提取功率注入数据
从潮流计算结果中提取所有节点的功率注入
"""
function extract_input_variables(input_case)
    # 提取PQ节点
    pq_idx = findall(input_case.busAC[:, BUS_TYPE] .== 1)
    # 提取PV节点
    pv_idx = findall(input_case.busAC[:, BUS_TYPE] .== 2)
    # 提取参考节点
    ref_idx = findall(input_case.busAC[:, BUS_TYPE] .== 3)

    # 初始化输入矩阵
    P_pq = zeros(size(input_case.busAC,1))
    Q_pq = zeros(size(input_case.busAC,1))
    P_pv = zeros(size(input_case.busAC,1))
    V_pv = zeros(size(input_case.busAC,1))
    V_ref = zeros(size(input_case.busAC,1))
    θ_ref = zeros(size(input_case.busAC,1))

    P_pq[pq_idx] = input_case.busAC[pq_idx, PD]
    Q_pq[pq_idx] = input_case.busAC[pq_idx, QD]

    P_pv[pv_idx] = input_case.busAC[pv_idx, PD]
    V_pv[pv_idx] = input_case.busAC[pv_idx, VM]

    V_ref[ref_idx] = input_case.busAC[ref_idx, VM]
    θ_ref[ref_idx] = input_case.busAC[ref_idx, VA]
 
  return P_pq, Q_pq, P_pv, V_pv, V_ref, θ_ref
end