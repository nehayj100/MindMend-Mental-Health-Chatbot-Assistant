import shutil
import os

# copy all data into a single folder: get a folder with text data files from podcasts, blogs and videos.

# copy pre-processed data
# ONE TIME RUN : RUN FROM ROOT DIR


destination_dirs = {
    'Podcasts': 'Pipeline/Data/Audio',
    'Youtube': 'Pipeline/Data/Video',
    'Blogs': 'Pipeline/Data/Text'
}

source_dirs = {
    'Podcasts': 'Data-PreProcessing/Final-Dataset/Podcasts',
    'Youtube': 'Data-PreProcessing/Final-Dataset/Videos',
    'Blogs': 'Data-PreProcessing/Final-Dataset/Blogs'
}


for category in source_dirs:
    
    source_path = source_dirs[category]
    destination_path = destination_dirs[category]
    
    os.makedirs(destination_path, exist_ok=True)
    
    for file_name in os.listdir(source_path):
        source_file = os.path.join(source_path, file_name)
        if os.path.isfile(source_file):
            destination_file = os.path.join(destination_path, file_name)
            shutil.copy(source_file, destination_file)



data_folder = 'Data'  
subfolders = ['Audio', 'Video', 'Text']
all_folder = os.path.join(data_folder, 'All')

os.makedirs(all_folder, exist_ok=True)

for subfolder in subfolders:
    subfolder_path = os.path.join(data_folder, subfolder)
    
    if os.path.exists(subfolder_path):
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(file_path): 
                shutil.copy(file_path, all_folder)


