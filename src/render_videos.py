import os, glob
# This script render all recorded attempts saved in records folder as mp4 and delete the bk2 files
os.chdir('../records')
records = [f for f in glob.glob('*.bk2')]

for record in records:
    _ = os.system('python3 -m retro.scripts.playback_movie {}'.format(record))
    _ = os.system('rm {}'.format(record))