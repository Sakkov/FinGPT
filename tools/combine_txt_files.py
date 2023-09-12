import os
    
def process_file(source, output_file):
    """
    Process the file or directory and write its contents to the output file.

    Keyword arguments:
    source -- source file or directory
    output_file -- output file
    """
    # If the source is a file, read and write it line-by-line to the output
    if os.path.isfile(source):
        print(f"Reading {source} of size {round(os.path.getsize(source)/1048576)}MiB") # noqa: E501
        
        with open(source, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                output_file.write(line)
        
        print("Done")
    # If the source is a directory, process each item in the directory
    else:
        for filename in os.listdir(source):
            process_file(os.path.join(source, filename), output_file)

def combine_txt_files(source_dir, output_filename):
    """
    Combine all the text files in the source directory recursively and save them in the output file.
    
    Keyword arguments:
    source_dir -- source directory
    output_filename -- output filename
    """
    
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        raise Exception('Source directory does not exist')
    
    # Process the files and directories and write the content to the output file
    with open(output_filename, "w", encoding="utf-8", errors="ignore") as output_file:
        process_file(source_dir, output_file)
    
    print("Content exported to", output_filename)


if __name__ == "__main__":
    # Combine the text files
    source_dir = "filtered_training_data\Finnish\custom_finGPT-1.2_21_8_2023"
    output_filename = "filtered_training_data\Finnish\custom_finGPT-1.2_21_8_2023.txt"

    combine_txt_files(source_dir, output_filename)