from mega import Mega
import zipfile

mega_url = 'https://mega.nz/file/dEJVRYRT#zc_eWxp2GIKxMz8tHxsMcKLt5gnwHCh9o0_6UOXOTtk'


zip_file_path = './dependencies.zip'


extract_path = '.'


mega = Mega()


m = mega.login()


print("Downloading...")
try:
    file = m.download_url(mega_url)
except PermissionError as e:
    pass


print("Extracting...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
