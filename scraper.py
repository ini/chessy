import re
import tqdm
import urllib.request
import zipfile


out_dir = './players/'
top_level_url = 'https://www.pgnmentor.com'


# Get links to downloads
links = []
data = urllib.request.urlopen(f'{top_level_url}/files.html')
for line in data:
    x = re.findall(r'players\/[a-zA-z]+\.zip', str(line))
    if len(x) > 0:
        links += x

links = [f'{top_level_url}/{link}' for link in links]


# Download zip files
for url in tqdm.tqdm(links):
    zip_path, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall(out_dir)
    

