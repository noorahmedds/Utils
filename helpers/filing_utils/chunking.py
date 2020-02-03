
import moviepy.editor
import os
import random
import fnmatch 
​
directory = 'original_videos'
ext = "*mp4"
length = 120
out_dir = "2_minute_chunks_"
​
# compile list of videos
inputs = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and fnmatch.fnmatch(f, ext)]
print(inputs)
for i in inputs:
    video_id = i.split(".")[0].split("/")[-1]
​
    # import to moviepy
    clip = moviepy.editor.VideoFileClip(i)
​
    # select a random time point
    start = round(random.uniform(0,clip.duration - length), 2) 
​
    # cut a subclip
    out_clip = clip.subclip(start,start+length)
​
    out_clip.write_videofile(out_dir + video_id + ".mp4")
​
