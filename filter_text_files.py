import os
import re
from tqdm import tqdm

def filter_text_files(source, destination):
    '''
    Filter the text files in the source directory recursively and save them in the destination directory

    source: path to the source directory
    destination: path to the destination directory
    '''
    count = 0
    
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    for filename in tqdm(os.listdir(source), desc='Filtering', unit='file'):
        if os.path.isdir(source + filename):
            filter_text_files(source + filename + '/', destination + filename + '/')
        else:
            
            with open(source + filename, 'r', encoding='utf-8', errors= 'ignore') as file:
                while True:
                    # Read the file few lines at a time
                    content = file.readlines(100000)

                    # Break if the file is empty
                    if not content:
                        break
            
                    # # Count the number of characters in the file
                    # num_chars_before = len(content)

                    # Join the lines into a single string
                    content = '\n'.join(content)

                    # Filter out all instances of multiple consecutive whitespace characters
                    content = re.sub(r' +', ' ', content)
                    content = re.sub(r'[ +][\n+]', '\n', content)
                    content = re.sub(r'\n{3,}', '\n\n', content)

                    # Filter out characters that are not in the Finnish alphabet or punctuation
                    content = re.sub(r'[^a-zA-ZåäöÅÄÖ,.-_;:_!?()\n\s\[\]\^\'\"@£$€\{\}]', '', content)

                    # # Count the number of characters in the file
                    # num_chars_after = len(content)
                    
                    # count += num_chars_before - num_chars_after
                    
                    # Add the filtered text to the destination file
                    with open(destination + filename, 'a', encoding='utf-8', errors='ignore') as destination_file:
                        destination_file.write(content)

    print(f'Filtered {count} characters')


if __name__ == '__main__':
    filter_text_files('training_data/Finnish/suomi24-2001-2017/', 'filtered_training_data/Finnish/suomi24-2001-2017/')