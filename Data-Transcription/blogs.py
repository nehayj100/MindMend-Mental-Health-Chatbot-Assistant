import requests
import os
from bs4 import BeautifulSoup

folder_path = 'Data-Blogs'

# URL of the blog webpage
urls = [
'https://trystressmanagement.com/mental-health/how-to-deal-with-separation-anxiety-in-relationships/',
'https://trystressmanagement.com/featured/tips-for-dealing-with-anxiety-in-public/',
'https://trystressmanagement.com/mental-health/how-to-deal-with-someone-with-anxiety/',
'https://trystressmanagement.com/mental-health/stress-management-techniques-in-the-workplace/',
'https://trystressmanagement.com/featured/sleep-behavior-disorder-treatment/',
'https://trystressmanagement.com/mental-health/6-best-sleep-medication-for-bipolar-disorder/',
'https://trystressmanagement.com/mental-health/cadhd-and-sleep-disorders/',
'https://trystressmanagement.com/featured/stress-in-teens-and-children/',
'https://trystressmanagement.com/featured/how-does-bipolar-disorder-affect-a-persons-daily-life/',
'https://trystressmanagement.com/mental-health/post-traumatic-stress-disorder/'
]

for u in range(len(urls)):
    # Send a GET request to fetch the page content
    url = urls[u]
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all text from the webpage
        text_content = soup.get_text()

       # Define the path for the text file
        file_path = os.path.join(folder_path, f'blog_{u}.txt')

        # Save the text content into the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)

        print(f"Blog text has been saved to blog_{u}.txt.")
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
