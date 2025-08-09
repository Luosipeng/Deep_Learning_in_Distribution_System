using TimeSeriesPowerFlow
include("../ios/matlab2julia.jl")
function generate_random_pf_samples_matpower(case_file, num_samples=1000, load_factor_range=(0.8, 1.2))
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
  G = real.(Ybus)
  B = imag.(Ybus)

  # 存储结果的数组
  mpc_inputs = []  # 存储修改后的MATPOWER案例（输入）
  voltage_magnititude_outputs = zeros(size(jpc.busAC,1),0)  # 存储电压结果（输出）
  voltage_angle_outputs = zeros(size(jpc.busAC,1),0)  # 存储电压角度（输出）
  Pg_inputs = zeros(size(jpc.busAC,1),0)  # 存储功率注入（辅助输出）
  Qg_inputs = zeros(size(jpc.busAC,1),0)  # 存储功率注入（输出）
  Vg_inputs = zeros(size(jpc.busAC,1),0)  # 存储电压（输出）
  Pd_inputs = zeros(size(jpc.busAC,1),0)  # 存储负荷（输出）
  Qd_inputs = zeros(size(jpc.busAC,1),0)  # 存储负荷（输出）

  Pg_outputs = zeros(size(jpc.busAC,1),0)  # 存储发电机有功功率（输出）
  Qg_outputs = zeros(size(jpc.busAC,1),0)  # 存储发电机无功功率（输出）
  Pd_outputs = zeros(size(jpc.busAC,1),0)  # 存储负荷有功功率（输出）
  Qd_outputs = zeros(size(jpc.busAC,1),0)  # 存储负荷无功功率（输出）

  Vr_inputs = zeros(1,0)  # 存储参考节点电压（输出）
  θr_inputs = zeros(1,0)  # 存储参考节点电压角度（输出）
  
  # 收集样本直到达到目标数量
  collected_samples = 0
  attempts = 0
  max_attempts = num_samples * 10  # 设置最大尝试次数
  
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
          
          # 提取电压结果
          voltage_magnititude_data = result.busAC[:,VM]
          voltage_angle_data = result.busAC[:,VA]
          voltage_magnititude_outputs = hcat(voltage_magnititude_outputs,voltage_magnititude_data)
          voltage_angle_outputs = hcat(voltage_angle_outputs,voltage_angle_data)
          
          # 提取功率注入
          Pg, Qg, Pd, Qd, Vg = extract_power_injections(input_case)
          Pg_inputs = hcat(Pg_inputs, Pg)
          Qg_inputs = hcat(Qg_inputs, Qg)
          Pd_inputs = hcat(Pd_inputs, Pd)
          Qd_inputs = hcat(Qd_inputs, Qd)
          Vg_inputs = hcat(Vg_inputs, Vg)
          
          # 提取参考节点的VM和VA
          ref_bus = findall(input_case.busAC[:, BUS_TYPE] .== 3)  # 找到参考节点
          Vr = input_case.busAC[ref_bus, VM]  # 参考节点电压幅值
          θr = input_case.busAC[ref_bus, VA]  # 参考节点电压角度
          Vr_inputs = hcat(Vr_inputs, Vr)
          θr_inputs = hcat(θr_inputs, θr)
          
          collected_samples += 1
          
          # 打印进度
          if collected_samples % 10 == 0
              println("已收集 $collected_samples/$num_samples 组样本 (尝试次数: $attempts)")
          end
      end
  end

  μ = voltage_magnititude_outputs.*cos.(voltage_angle_outputs)
  ω = voltage_magnititude_outputs.*sin.(voltage_angle_outputs)
  

  if collected_samples < num_samples
      println("警告：在 $max_attempts 次尝试后仅收集到 $collected_samples 组有效样本")
  else
      println("成功收集到 $num_samples 组样本，总尝试次数：$attempts")
  end
  
  # 返回收集的样本
  return μ, ω, Pg_inputs, Qg_inputs, Pd_inputs, Qd_inputs, Vg_inputs, Vr_inputs, θr_inputs, G, B
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
function extract_power_injections(result)
  Pd = result.busAC[:, PD]  # 有功负荷
  Qd = result.busAC[:, QD]  # 无功负荷

  pg = result.genAC[:, PG]  # 有功发电
  qg = result.genAC[:, QG]  # 无功发电

  vg = result.genAC[:, VG]  # 电压幅值
  # 计算功率注入
  Pg = zeros(size(result.busAC,1))  # 初始化功率注入数组
  Qg = zeros(size(result.busAC,1))  # 初始化无功功率注入数组
  Vg = zeros(size(result.busAC,1))  # 初始化电压数组
  for i in 1:size(result.genAC,1)
      bus_id = Int(result.genAC[i, GEN_BUS])
      Pg[bus_id] = pg[i]
      Qg[bus_id] = qg[i]
      Vg[bus_id] = vg[i]
  end
 
  return Pg, Qg, Pd, Qd, Vg
end