import subprocess

dpo_script_path = "run_dpo.py"
dataset_script_path = "generate_data.py"

data_command = ["python3", dataset_script_path, "0"]
process = subprocess.run(data_command, capture_output=True, text=True)

for i in range(1, 4):
    # Create the command to execute
    command = ['python3', dpo_script_path, str(i - 1)]
    
    # Run the subprocess and wait for it to finish
    process = subprocess.run(command, capture_output=True, text=True)

    data_command = ["python3", dataset_script_path, str(i)]
    process = subprocess.run(data_command, capture_output=True, text=True)
