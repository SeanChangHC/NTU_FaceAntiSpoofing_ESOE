import os
folder = '/home/fas3/example-img/SiW60/Test/spoof'
files = os.listdir(folder)

for f in files:
    oldname = os.path.join(folder, f)
    newname = os.path.join(folder, f).replace('.live.', '.spoof.')
    os.rename(oldname, newname)