import googleapiclient.discovery
from IPython.display import YouTubeVideo, Image, display, Video
import time


class YouTube():
    '''Connect to YouTube API. Holds methods to retrieve videos and video attributes'''

    def __init__(self, KEY):
        self._YOUTUBE_API_SERVICE_NAME = 'youtube'
        self._YOUTUBE_API_VERSION = 'v3'
        self._KEY = KEY
        self.youtube = googleapiclient.discovery.build(self._YOUTUBE_API_SERVICE_NAME, 
                                                        self._YOUTUBE_API_VERSION, 
                                                        developerKey = self._KEY)

    def search(self, query, license_type, nextPage = '', num_videos=1):
        '''
        YOUTUBE API METHOD
        connects to the youtube search API resource
        
        Args
            query (string): search term query to retrieve results for
            license_type(string): type of license to filter search results by.
                                  Can be one of 'any', 'creativeCommon', or 'youtube' 

        Returns
            search_result['nextPageToken']: the token used to get the next page of results (useful for pagination)
            search_result['items']: search API results for YouTube videos that meet the specified parameters             
        '''

        search_result = self.youtube.search().list(
            q = query, 
            part="snippet",
            type="video",
            videoLicense = license_type,
            maxResults=num_videos if num_videos < 50 else 50,
            pageToken = nextPage,
            ).execute()

        return search_result['nextPageToken'], search_result['items']

    def videos(self, video_id):
        '''
        YOUTUBE API METHOD
        more on the returned fields found here: 
           https://developers.google.com/youtube/v3/docs/videos#contentDetails.licensedContent
           
        Args
            video_id (string): the id used to identify each unique YouTube video

        Returns
            snippet, status, content, and recording details (aka 'video resource' per API terminology)
        '''

        return self.youtube.videos().list(
            id = video_id,
            part="snippet, status, contentDetails, recordingDetails").execute()['items']

    def get_search_results(self, query, license_type, total_num_videos):
        '''
        paginates through search results until limit is met

        Args
            query (string): search query term
            license_type (string): type of license a video must have. See search() for more details
            limit (int): number of videos to retrieve 
        
        '''
        nextPage = ''
        formatted_results = []
        while(total_num_videos > 0):
            
            #get results
            nextPage, page_result = self.search(query, license_type, nextPage, num_videos=total_num_videos)

            #format results 
            formatted_results += self.format_search_results(page_result)

            total_num_videos -= 50 #each page is set to have 50 videos
            time.sleep(0.5) 
        
        return formatted_results


    def format_search_results(self, page_result):
        '''
        go through page_result retrieved and format them to only keep videoID, url,
        title, and license type
        '''

        #go through current page results and extract info
        formatted_page_results = []
        for search_item in page_result:
            
            #store video title, url and license type
            video_info = {'video_id': search_item['id']['videoId'],
                            'url': 'https://youtu.be/' + search_item['id']['videoId'],
                            'title': search_item['snippet'].get('title', None),
                            'license': self.videos(search_item['id']['videoId'])[0]['status']['license']
                            }

            formatted_page_results.append(video_info)
        
        return formatted_page_results


    def label_video(self, video_id):
        
        ''' asks user to assign label to video displayed'''
        
        #show video
        self.display_video(video_id)

        #ask user for label
        print('Relevant (y/n/delete):', end = " ")
        
        user_input = input()
        
        if user_input == "y":
            relevant_label = True
        elif user_input == "delete":
            relevant_label = "delete"
        else:
            relevant_label = False

        return relevant_label

    def display_video(self, video_id):

        ''' Display YouTube video as embedded item for filtration'''
        
        video_url = 'https://youtu.be/' + video_id
        print(f'VideoID: {video_id}     URL: {video_url}')

        display(YouTubeVideo(video_id))
