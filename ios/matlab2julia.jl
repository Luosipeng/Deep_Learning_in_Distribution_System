function parse_matlab_case_file(filepath)
    # 读取文件内容
    content = read(filepath, String)
    
    # 创建空字典存储中间结果
    mpc = Dict{String, Any}()
    
    # 解析 baseMVA
    if occursin(r"mpc\.baseMVA\s*=\s*(\d+)", content)
        basemva_match = match(r"mpc\.baseMVA\s*=\s*(\d+)", content)
        mpc["baseMVA"] = parse(Float64, basemva_match[1])
    end
    
    # 解析 version
    if occursin(r"mpc\.version\s*=\s*'(\d+)'", content)
        version_match = match(r"mpc\.version\s*=\s*'(\d+)'", content)
        mpc["version"] = version_match[1]
    end
    
    # 解析矩阵或字符串数据的函数
    function extract_data(content, key)
        # 分割内容为行
        lines = split(content, '\n')
        
        # 找到矩阵开始的行
        start_pattern = "mpc.$key = ["
        end_pattern = "];"
        
        start_idx = 0
        end_idx = 0
        
        # 查找矩阵的开始和结束位置
        for (i, line) in enumerate(lines)
            if occursin(start_pattern, line)
                start_idx = i
            elseif start_idx > 0 && occursin(end_pattern, line)
                end_idx = i
                break
            end
        end
        
        # 如果找到了矩阵
        if start_idx > 0 && end_idx > 0
            # 提取矩阵内容
            matrix_lines = lines[start_idx+1:end_idx-1]
            
            # 过滤掉空行和注释行
            matrix_lines = filter(line -> !isempty(strip(line)) && !startswith(strip(line), '%'), matrix_lines)
            
            # 检查是否包含字符串数据
            contains_strings = any(line -> occursin("'", line) || occursin("\"", line), matrix_lines)
            
            if contains_strings
                # 处理字符串数据
                matrix = []
                for line in matrix_lines
                    # 移除行尾的分号和注释
                    line = replace(line, r";.*$" => "")
                    # 提取引号中的内容和数字
                    parts = String[]
                    current_str = ""
                    in_quotes = false
                    quote_char = nothing
                    
                    for char in line
                        if char in ['\'', '"']
                            if !in_quotes
                                in_quotes = true
                                quote_char = char
                            elseif char == quote_char
                                in_quotes = false
                                if !isempty(current_str)
                                    push!(parts, current_str)
                                    current_str = ""
                                end
                            else
                                current_str *= char
                            end
                        elseif in_quotes
                            current_str *= char
                        elseif !isspace(char) && char != ';'
                            current_str *= char
                        elseif !isempty(current_str)
                            push!(parts, current_str)
                            current_str = ""
                        end
                    end
                    
                    if !isempty(current_str)
                        push!(parts, current_str)
                    end
                    
                    # 过滤掉空字符串
                    parts = filter(!isempty, parts)
                    
                    if !isempty(parts)
                        push!(matrix, parts)
                    end
                end
                return length(matrix) > 0 ? reduce(vcat, transpose.(matrix)) : nothing
            else
                # 处理数值数据
                matrix = []
                for line in matrix_lines
                    # 移除行尾的分号和注释
                    line = replace(line, r";.*$" => "")
                    # 分割并转换为数值
                    try
                        row = parse.(Float64, split(strip(line)))
                        if !isempty(row)
                            push!(matrix, row)
                        end
                    catch
                        @warn "无法解析行: $line"
                        continue
                    end
                end
                return length(matrix) > 0 ? reduce(vcat, transpose.(matrix)) : nothing
            end
        end
        return nothing
    end
    
    # 查找所有可能的矩阵名称
    matrix_names = String[]
    for line in split(content, '\n')
        m = match(r"mpc\.(\w+)\s*=\s*\[", line)
        if m !== nothing
            push!(matrix_names, m[1])
        end
    end
    
    # 解析每个找到的矩阵
    for name in matrix_names
        if name ∉ ["version", "baseMVA"]  # 跳过已处理的特殊字段
            matrix = extract_data(content, name)
            if matrix !== nothing
                mpc[name] = matrix
            end
        end
    end
    
    return mpc
end

# 将字典格式转换为JPC结构体
function dict_to_jpc(mpc_dict)
    # 创建JPC实例
    jpc = JPC()
    
    # 设置基本属性
    if haskey(mpc_dict, "version")
        jpc.version = mpc_dict["version"]
    end
    
    if haskey(mpc_dict, "baseMVA")
        jpc.baseMVA = mpc_dict["baseMVA"]
    end
    
    # 映射MATPOWER字段到JPC字段
    field_mapping = Dict(
        "bus" => "busAC",
        "gen" => "genAC",
        "branch" => "branchAC",
        "load" => "loadAC"
        # 可以根据需要添加更多映射
    )
    
    # 将数据从字典转移到JPC结构体
    for (matpower_field, jpc_field) in field_mapping
        if haskey(mpc_dict, matpower_field)
            data = mpc_dict[matpower_field]
            # 确保数据是二维数组
            if ndims(data) == 1
                data = reshape(data, 1, length(data))
            end
            
            # 获取JPC结构体中对应字段的引用
            field_ref = getfield(jpc, Symbol(jpc_field))
            
            # 检查数据列数是否与目标字段匹配
            target_cols = size(field_ref, 2)
            data_cols = size(data, 2)
            
            if data_cols <= target_cols
                # 创建适当大小的新数组
                new_data = zeros(size(data, 1), target_cols)
                # 复制数据
                new_data[:, 1:data_cols] = data
                # 设置字段值
                setfield!(jpc, Symbol(jpc_field), new_data)
            else
                # 如果数据列数超过目标字段，截断数据
                setfield!(jpc, Symbol(jpc_field), data[:, 1:target_cols])
                @warn "数据 $matpower_field 的列数($data_cols)超过目标字段 $jpc_field 的列数($target_cols)，已截断。"
            end
        end
    end
    
    # 处理其他未映射的字段
    for field in keys(mpc_dict)
        if field ∉ ["version", "baseMVA"] && !haskey(field_mapping, field)
            # 尝试直接匹配字段名
            try
                if hasproperty(jpc, Symbol(field))
                    data = mpc_dict[field]
                    # 确保数据是二维数组
                    if ndims(data) == 1
                        data = reshape(data, 1, length(data))
                    end
                    
                    # 获取JPC结构体中对应字段的引用
                    field_ref = getfield(jpc, Symbol(field))
                    
                    # 检查数据列数是否与目标字段匹配
                    target_cols = size(field_ref, 2)
                    data_cols = size(data, 2)
                    
                    if data_cols <= target_cols
                        # 创建适当大小的新数组
                        new_data = zeros(size(data, 1), target_cols)
                        # 复制数据
                        new_data[:, 1:data_cols] = data
                        # 设置字段值
                        setfield!(jpc, Symbol(field), new_data)
                    else
                        # 如果数据列数超过目标字段，截断数据
                        setfield!(jpc, Symbol(field), data[:, 1:target_cols])
                        @warn "数据 $field 的列数($data_cols)超过目标字段的列数($target_cols)，已截断。"
                    end
                end
            catch e
                @warn "无法将字段 $field 映射到JPC结构体: $e"
            end
        end
    end
    
    return jpc
end

function save_to_julia_file(jpc, output_filepath)
    open(output_filepath, "w") do f
        write(f, """function case_data()
    # 创建JPC结构体实例
    jpc = JPC("$(jpc.version)", $(jpc.baseMVA), $(jpc.success), $(jpc.iterationsAC), $(jpc.iterationsDC))
    
""")
        
        # 获取JPC结构体的所有字段名
        field_names = fieldnames(JPC)
        
        # 遍历所有字段
        for field in field_names
            # 跳过基本属性
            if field ∉ [:version, :baseMVA, :success, :iterationsAC, :iterationsDC]
                data = getfield(jpc, field)
                
                # 只处理非空数组
                if !isempty(data)
                    write(f, "    # 设置 $field 数据\n")
                    write(f, "    jpc.$field = [\n")
                    
                    # 写入数组数据
                    for i in 1:size(data, 1)
                        write(f, "        ")
                        for j in 1:size(data, 2)
                            write(f, "$(data[i,j]) ")
                        end
                        write(f, ";\n")
                    end
                    
                    write(f, "    ]\n\n")
                end
            end
        end
        
        write(f, "    return jpc\nend")
    end
end

function convert_matpower_case_dp(input_filepath, output_filepath)
    try
        println("正在解析MATLAB文件...")
        mpc_dict = parse_matlab_case_file(input_filepath)
        
        println("正在转换为JPC结构体...")
        jpc = dict_to_jpc(mpc_dict)
        
        println("正在保存为Julia文件...")
        save_to_julia_file(jpc, output_filepath)
        
        println("转换完成！")
        println("输入文件：$input_filepath")
        println("输出文件：$output_filepath")
        
        # 打印数据统计
        println("\n数据统计：")
        for field in fieldnames(JPC)
            data = getfield(jpc, field)
            if isa(data, Array)
                if !isempty(data)
                    println("$field 矩阵大小: $(size(data))")
                end
            else
                println("$field: $(data)")
            end
        end
        
        return jpc
    catch e
        println("转换过程中出现错误：")
        println(e)
        return nothing
    end
end


