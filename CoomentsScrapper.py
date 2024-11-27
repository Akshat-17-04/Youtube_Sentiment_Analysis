from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv
import sys
if(len(sys.argv)!=3):
    print("Add video id")
    sys.exit(1)
vid_id=sys.argv[1]
output_file=sys.argv[2]
yt_client=build("youtube","v3",developerKey="AIzaSyDAEcS8TEMilvrGwQU88E1JGS2K1hwg-DQ")
def get_comments(client,video_id,token=None):
    try:
        response=(client.commentThreads().list(part="snippet",videoId=video_id,
                                               textFormat="plainText",maxResults=100,pageToken=token).execute())
        return response
    except HttpError as e:
        print(e.resp.status)
        return None
    except Exception as e:
        print(e)
        return None
comments=[]
next=None
while True:
    response=get_comments(yt_client,vid_id,next)
    if not response:
        break
    else:
        comments+=response["items"]
        next=response.get("nextPageToken")
    if not next:
        break
print(f"totals comments fetched:{len(comments)}")
with open(output_file,"w",newline="",encoding="utf-8") as file:
    csv_writer=csv.writer(file)
    for i in comments:
        row=[i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]]
        csv_writer.writerow(row)
#4gulVzzh82g
