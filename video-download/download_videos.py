from pytube import YouTube
import pandas as pd
import time

#download the videos from our sub_hw_df into our directory
'''downloads videos into the directory specified from YouTube urls
    ---------------------------------------------------------------
    Input:
        - url_dataset: csv file consisting of YouTube urls to videos we want to download

    Output:
        .mp4 video files in path specified
'''
DOWNLOAD_PATH = "/Users/micheller/Documents/nvidia-intern-project/data/videos/"
url_dataset = pd.read_csv('/Users/micheller/Documents/nvidia-intern-project/data/creative_commons_youtube_videos.csv')
num_unavailable = 0

print('Beginning download...')
for index, row in url_dataset[0:2].iterrows():
    try:
        #get video item and check availability
        video = YouTube(row['url'])
        video.check_availability() 

        #download the actual video mp4 with highest res available
        video_name = row['renamed_title']
        print(f'downloading {video_name} ...')
        
        highest_res_video = video.streams.filter(file_extension="mp4").order_by('resolution').last() 
        highest_res_video.download(output_path = DOWNLOAD_PATH, filename = video_name)

        #pause slightly to avoid raising error
        time.sleep(0.5)

    except:
        print('Video {} at url {} unavailable...skipping'.format(video_name, row['url']))
        num_unavailable += 1
    
print(f'Done Downloading videos. Downloaded videos can be found in: {DOWNLOAD_PATH}')

if num_unavailable > 0:
    print(f'Unable to download {num_unavailable} videos.')
