from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd

''' File to get clips at start, stop intervals from full length videos from dataset '''

WORKSPACE_PATH = "/mount/data/"
downloads_df = pd.read_csv(WORKSPACE_PATH + '/downloaded_videos.csv')
for i, row in downloads_df.iterrows():
    
    full_video = WORKSPACE_PATH + 'videos/' + row['renamed_title']
    video_clip_name = row['renamed_title'].replace('_', '_clip_')

    #download clip at start, stop interval
    ffmpeg_extract_subclip(full_video, row['clip_start'], row['clip_end'], targetname=WORKSPACE_PATH + 'video_clips/' + video_clip_name)
    
print('Done downloading video clips.')