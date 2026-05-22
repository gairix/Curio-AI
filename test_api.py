import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

params = {
    "api_key": SERPAPI_API_KEY,
    "engine": "youtube_video_transcript",
    "v": "fQNIfFzNLaE",  # Replace with target video ID
    "type": "asr"
}

search = requests.get("https://serpapi.com/search", params=params)
response = search.json()

# Check if SerpApi explicitly flagged an error (e.g., transcript not found)
if "error" in response:
    print("Sorry, no transcript available for this video.")

else:
    transcript_list = response.get("transcript", [])
    
    if not transcript_list:
        print("Sorry, no transcript available for this video.")
    else:
        # Loop and safely print segments if data exists
        for result in transcript_list:
            snippet = result.get("snippet")
            start_time_text = result.get("start_time_text")
            print(f"[{start_time_text}] {snippet}")