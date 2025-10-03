include("../ios/time_series_data_generation.jl")
function get_sample_from_ieee37()
    feeder_dir = "D:/luosipeng/Deep_Learning_in_Distribution_System/data"
    # 假设你之前完整执行:
    res = time_series_ieee37(feeder_dir; dt_s=0.1, hours=24.0, sample_every=1, collect=[:voltage_bus, :total_power, :bus_injection])
    # 提取 PMU 采样点数据： PMU 采样点为 “705”， 0.1s 采样间隔， 24 小时数据
    s_pmu = get_bus_series(res, "705")
    Vm_pmu = Array(s_pmu["voltage_pu_phase"])
    Va_pmu = Array(s_pmu["voltage_angle_rad"])
    Vreal_pmu = Vm_pmu .* cos.(Va_pmu)
    Vimag_pmu = Vm_pmu .* sin.(Va_pmu)

    # 提取SCADA采样点数据： SCADA 采样点为 “702”， 1 分钟采样间隔， 24 小时数据
    s_scada_1 = get_bus_series(res, "702")
    s_scada_1_voltage = Array(s_scada_1["voltage_pu_phase"])
    s_scada_1_voltage = s_scada_1_voltage[:, 1:600:end]  # 1分钟采样间隔

    # 提取SCADA采样点数据： SCADA 采样点为 “703”， 1 分钟采样间隔， 24 小时数据
    s_scada_2 = get_bus_series(res, "703")
    s_scada_2_voltage = Array(s_scada_2["voltage_pu_phase"])
    s_scada_2_voltage = s_scada_2_voltage[:, 1:600:end]  # 1分钟采样间隔

    # 提取SCADA采样点数据： SCADA 采样点为 “730”， 1 分钟采样间隔， 24 小时数据
    s_scada_3 = get_bus_series(res, "730")
    s_scada_3_voltage = Array(s_scada_3["voltage_pu_phase"])
    s_scada_3_voltage = s_scada_3_voltage[:, 1:600:end]  # 1分钟采样间隔

    # 提取 AMI 采样点数据： AMI 采样点为 “701”， 15 分钟采样间隔， 24 小时数据, A相有功无功功率
    s_ami_1 = get_bus_phase_injection(res, "701")
    s_ami_1_active_power = Array(s_ami_1["P_inj_phase_kW"])
    s_ami_1_active_power = s_ami_1_active_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_1_active_power = s_ami_1_active_power[1,:]  # 只取A相
    s_ami_1_reactive_power = Array(s_ami_1["Q_inj_phase_kvar"])
    s_ami_1_reactive_power = s_ami_1_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_1_reactive_power = s_ami_1_reactive_power[1,:]  # 只取A相

    # 提取 AMI 采样点数据： AMI 采样点为 “744”， 15 分钟采样间隔， 24 小时数据, A相有功无功功率
    s_ami_2 = get_bus_phase_injection(res, "744")
    s_ami_2_active_power = Array(s_ami_2["P_inj_phase_kW"])
    s_ami_2_active_power = s_ami_2_active_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_2_active_power = s_ami_2_active_power[1,:]  # 只取A相
    s_ami_2_reactive_power = Array(s_ami_2["Q_inj_phase_kvar"])
    s_ami_2_reactive_power = s_ami_2_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_2_reactive_power = s_ami_2_reactive_power[1,:]  # 只取A相

    # 提取 AMI 采样点数据： AMI 采样点为 “728”， 15 分钟采样间隔， 24 小时数据, A相有功无功功率
    s_ami_3 = get_bus_phase_injection(res, "728")
    s_ami_3_active_power = Array(s_ami_3["P_inj_phase_kW"])
    s_ami_3_active_power = s_ami_3_active_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_3_active_power = s_ami_3_active_power[1,:]  # 只取A相
    s_ami_3_reactive_power = Array(s_ami_3["Q_inj_phase_kvar"])
    s_ami_3_reactive_power = s_ami_3_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_3_reactive_power = s_ami_3_reactive_power[1,:]  # 只取A相

    # 提取 AMI 采样点数据： AMI 采样点为 “729”， 15 分钟采样间隔， 24 小时数据, A相有功无功功率
    s_ami_4 = get_bus_phase_injection(res, "729")
    s_ami_4_active_power = Array(s_ami_4["P_inj_phase_kW"])
    s_ami_4_active_power = s_ami_4_active_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_4_active_power = s_ami_4_active_power[1,:]  # 只取A相
    s_ami_4_reactive_power = Array(s_ami_4["Q_inj_phase_kvar"])
    s_ami_4_reactive_power = s_ami_4_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_4_reactive_power = s_ami_4_reactive_power[1,:]  # 只取A相

    # 提取 AMI 采样点数据： AMI 采样点为 “701”， 15 分钟采样间隔， 24 小时数据, B相有功无功功率
    s_ami_1_B = Array(s_ami_1["P_inj_phase_kW"])
    s_ami_1_B = s_ami_1_B[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_1_B = s_ami_1_B[2,:]  # 只取B相
    s_ami_1B_reactive_power = Array(s_ami_1["Q_inj_phase_kvar"])
    s_ami_1B_reactive_power = s_ami_1B_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_1B_reactive_power = s_ami_1B_reactive_power[2,:]  # 只取B相

    # 提取 AMI 采样点数据： AMI 采样点为 “728”， 15 分钟采样间隔， 24 小时数据, B相有功无功功率
    s_ami_3_B = Array(s_ami_3["P_inj_phase_kW"])
    s_ami_3_B = s_ami_3_B[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_3_B = s_ami_3_B[2,:]  # 只取B相
    s_ami_3B_reactive_power = Array(s_ami_3["Q_inj_phase_kvar"])
    s_ami_3B_reactive_power = s_ami_3B_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_3B_reactive_power = s_ami_3B_reactive_power[2,:]  # 只取B相

    # 提取 AMI 采样点数据： AMI 采样点为 “736”， 15 分钟采样间隔， 24 小时数据, B相有功无功功率
    s_ami_5 = get_bus_phase_injection(res, "736")
    s_ami_5_B = Array(s_ami_5["P_inj_phase_kW"])
    s_ami_5_B = s_ami_5_B[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_5_B = s_ami_5_B[2,:]  # 只取B相
    s_ami_5B_reactive_power = Array(s_ami_5["Q_inj_phase_kvar"])
    s_ami_5B_reactive_power = s_ami_5B_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_5B_reactive_power = s_ami_5B_reactive_power[2,:]  # 只取B相

    # 提取 AMI 采样点数据： AMI 采样点为 “701”， 15 分钟采样间隔， 24 小时数据, C相有功无功功率
    s_ami_1_C = Array(s_ami_1["P_inj_phase_kW"])
    s_ami_1_C = s_ami_1_C[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_1_C = s_ami_1_C[3,:]  # 只取C相
    s_ami_1C_reactive_power = Array(s_ami_1["Q_inj_phase_kvar"])
    s_ami_1C_reactive_power = s_ami_1C_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_1C_reactive_power = s_ami_1C_reactive_power[3,:]  # 只取C相

    # 提取 AMI 采样点数据： AMI 采样点为 “727”， 15 分钟采样间隔， 24 小时数据, C相有功无功功率
    s_ami_6 = get_bus_phase_injection(res, "727")
    s_ami_6_C = Array(s_ami_6["P_inj_phase_kW"])
    s_ami_6_C = s_ami_6_C[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_6_C = s_ami_6_C[3,:]  # 只取C相
    s_ami_6C_reactive_power = Array(s_ami_6["Q_inj_phase_kvar"])
    s_ami_6C_reactive_power = s_ami_6C_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_6C_reactive_power = s_ami_6C_reactive_power[3,:]  # 只取C相

    # 提取 AMI 采样点数据： AMI 采样点为 “728”， 15 分钟采样间隔， 24 小时数据, C相有功无功功率
    s_ami_3_C = Array(s_ami_3["P_inj_phase_kW"])
    s_ami_3_C = s_ami_3_C[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_3_C = s_ami_3_C[3,:]  # 只取C相
    s_ami_3C_reactive_power = Array(s_ami_3["Q_inj_phase_kvar"])
    s_ami_3C_reactive_power = s_ami_3C_reactive_power[:, 1:9000:end]  # 15分钟采样间隔
    s_ami_3C_reactive_power = s_ami_3C_reactive_power[3,:]  # 只取C相

    # 构建输入张量，尺寸 (3, 5, 10, 86400)
    # 3 个相位: A相, B相, C相
    # 5 个通道: 实部电压, 虚部电压, 电压幅值, 有功功率, 无功功率
    # 10 个测量点： PMU 705, SCADA 702, SCADA 703, SCADA 730, AMI 701, AMI 744, AMI 728, AMI 729, AMI 736, AMI 727
    # 86400 个时间步长 (24 小时，每 0.1 秒采样)
    # 注意：对于 SCADA 和 AMI 数据，我们需要将其插值到与 PMU 数据相同的时间步长
    # 未采样到的数据标记缺失值NaN
    # 初始化张量
    x = NaN .*zeros(3,5,10,864000)

    # 填充 PMU 数据 (测量点 1)
    x[1,1,1,:] = Vreal_pmu[1,:]  # A相实部电压
    x[2,1,1,:] = Vreal_pmu[2,:]  # B相实部电压
    x[3,1,1,:] = Vreal_pmu[3,:]  # C相实部电压
    x[1,2,1,:] = Vimag_pmu[1,:]  # A相虚部电压
    x[2,2,1,:] = Vimag_pmu[2,:]  # B相虚部电压
    x[3,2,1,:] = Vimag_pmu[3,:]  # C相虚部电压

    # 填充 SCADA 数据 (测量点 2, 3, 4)
    for t in 1:1440
        idx = (t-1)*600 + 1  # 对应 PMU 时间步长索引
        x[1,3,2,idx] = s_scada_1_voltage[1,t]  # A相电压幅值 SCADA 702
        x[2,3,2,idx] = s_scada_1_voltage[2,t]  # B相电压幅值 SCADA 702
        x[3,3,2,idx] = s_scada_1_voltage[3,t]  # C相电压幅值 SCADA 702

        x[1,3,3,idx] = s_scada_2_voltage[1,t]  # A相电压幅值 SCADA 703
        x[2,3,3,idx] = s_scada_2_voltage[2,t]  # B相电压幅值 SCADA 703
        x[3,3,3,idx] = s_scada_2_voltage[3,t]  # C相电压幅值 SCADA 703

        x[1,3,4,idx] = s_scada_3_voltage[1,t]  # A相电压幅值 SCADA 730
        x[2,3,4,idx] = s_scada_3_voltage[2,t]  # B相电压幅值 SCADA 730
        x[3,3,4,idx] = s_scada_3_voltage[3,t]  # C相电压幅值 SCADA 730

    end

    # 填充 AMI 数据 (测量点 5, 6, 7, 8, 9, 10)
    for t in 1:96
        idx = (t-1)*9000 + 1  # 对应 PMU 时间步长索引
        x[1,4,5,idx] = s_ami_1_active_power[t]  # A相有功功率 AMI 701
        x[2,4,5,idx] = s_ami_1_B[t]               # B相有功功率 AMI 701
        x[3,4,5,idx] = s_ami_1_C[t]               # C相有功功率 AMI 701
        x[1,5,5,idx] = s_ami_1_reactive_power[t]  # A相无功功率 AMI 701
        x[2,5,5,idx] = s_ami_1B_reactive_power[t] # B相无功功率 AMI 701
        x[3,5,5,idx] = s_ami_1C_reactive_power[t] # C相无功功率 AMI 701

        x[1,4,6,idx] = s_ami_2_active_power[t]  # A相有功功率 AMI 744
        x[1,5,6,idx] = s_ami_2_reactive_power[t]  # A相无功功率 AMI 744

        x[1,4,7,idx] = s_ami_3_active_power[t]  # A相有功功率 AMI 728
        x[2,4,7,idx] = s_ami_3_B[t]               # B相有功功率 AMI 728
        x[3,4,7,idx] = s_ami_3_C[t]               # C相有功功率 AMI 728
        x[1,5,7,idx] = s_ami_3_reactive_power[t]  # A相无功功率 AMI 728
        x[2,5,7,idx] = s_ami_3B_reactive_power[t] # B相无功功率 AMI 728
        x[3,5,7,idx] = s_ami_3C_reactive_power[t] # C相无功功率 AMI 728

        x[1,4,8,idx] = s_ami_4_active_power[t]  # A相有功功率 AMI 729
        x[1,5,8,idx] = s_ami_4_reactive_power[t]  # A相无功功率 AMI 729

        x[1,4,9,idx] = s_ami_5_B[t]               # B相有功功率 AMI 736
        x[1,5,9,idx] = s_ami_5B_reactive_power[t] # B相无功功率 AMI 736

        x[1,4,10,idx] = s_ami_6_C[t]               # C相有功功率 AMI 727
        x[1,5,10,idx] = s_ami_6C_reactive_power[t] # C相无功功率 AMI 727
       
    end

    return Float32.(x)
end