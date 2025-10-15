push!(LOAD_PATH, "/path/to/HyDistFlow")

# Import modules
using Dates
using XLSX
using DataFrames
using Base.Threads
using HyDistFlow

# Input Data
file_path = joinpath(pkgdir(HyDistFlow), "data", "test_case.xlsx")
load_path = joinpath(pkgdir(HyDistFlow), "data", "load.xlsx")  
price_path = joinpath(pkgdir(HyDistFlow), "data", "price.xlsx")  
irradiance_path = joinpath(pkgdir(HyDistFlow), "data", "irradiance.xlsx")  

# Process Data
case = load_julia_power_data(file_path)
time_column, time_str_column, load_names, data = read_load_data(load_path) 
time_column, time_str_column, price_profiles = read_price_data(price_path)  
time_column, time_str_column, irradiance_profiles = read_irradiance_data(irradiance_path) 

# Topology processing
results, new_case = topology_analysis(case, output_file="topology_results.xlsx")

# Clear existing storage data and add a new battery storage system
empty!(new_case.storageetap)
push!(new_case.storages, Storage(1, "Battery_ESS_1", 3, 0.75, 1.5, 0.3, 0.05, 0.95, 0.9, true, "lithium_ion", true))

# Set control mode for converters to Droop_Udc_Us (voltage droop control)
new_case.converters[3].control_mode = "Droop_Udc_Us"
new_case.converters[2].control_mode = "Droop_Udc_Us"
new_case.converters[1].control_mode = "Droop_Udc_Us"

opt = options() # The initial settings 
opt["PF"]["NR_ALG"] = "bicgstab";
opt["PF"]["ENFORCE_Q_LIMS"] = 0;
opt["PF"]["DC_PREPROCESS"] = 1;

# Run time-series power flow calculation and measure execution time
@time results = runtdpf(new_case, data, load_names, price_profiles, irradiance_profiles, opt)

# # Get voltage results for all nodes
plot_result = plot_voltage_time_series(results, "Bus_21", new_case, 366, "AC"; save_path="voltage_plot")