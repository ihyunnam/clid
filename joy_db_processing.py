# used chatGPT for assistance

import json
# Uncomment to enable displaying download link for processed file
# import IPython.display as display

def transform_os_info(data):
    transformed_data = []
    for os_info, value in data.items():
        transformed_data.append(f"{os_info}:{value}")
    return transformed_data

# input_file_path has to be csv or JSONL
def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as json_input_file, open(output_file_path, 'w') as json_output_file:
        for line in json_input_file:
            data = json.loads(line)
            for process_info in data['process_info']:
                process_info['os_info'] = transform_os_info(process_info['os_info'])
            json.dump(data, json_output_file)
            json_output_file.write('\n')

    # Uncomment to enable: Display a download link for the processed file
    # display.display(display.FileLink(output_file_path))

# Call the function with your input and output file paths
process_file('input_file_path', 'output_file_path')
