"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

file_path = 'repo_names.txt'

# Read repo names from the txt file and store in a list
REPOS = []
with open(file_path, 'r') as file:
    for repo in file:
        REPOS.append(repo.strip())

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )

#######################################################################################

def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


# def scrape_github_data() -> List[Dict[str, str]]:
#     """
#     Loop through all of the repos and process them. Returns the processed data.
#     """
#     return [process_repo(repo) for repo in REPOS]

#######################################################################################


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    processed_data = []
    
    for repo in REPOS:
        try:
            processed_repo = process_repo(repo)
            processed_data.append(processed_repo)
        except Exception as e:
            print(f"Error processing repo: {repo}\nError: {e}")
            continue
    
    return processed_data



def read_website(url, webdriver_path = '~/chromedriver'):
    '''
        read website and returns a data structure representing a parsed HTML document
    '''
    # Path to chromedriver executable
    webdriver_path = webdriver_path

    # Set up the Selenium driver with options
    options = Options()
    options.add_argument('--headless')  # Run in headless mode
    driver = webdriver.Chrome(service=Service(webdriver_path), options=options)
    
    # Load the webpage
    driver.get(url)

    # Wait for the dynamic content to load (if necessary)
    # You can use driver.implicitly_wait() or other wait methods

    # Extract the page source after the dynamic content has loaded
    source = driver.page_source

    # Close the Selenium driver
    driver.quit()
    
    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(source, 'lxml')

    return soup

def scrape_repo_names(pages_to_scrape = 100):
    """pages_to_scrape must be 100 or less"""
    repos = []

    for page in range(1, pages_to_scrape + 1):
        if page % 10 == 0: print('Page', page) 
        url_start = "https://github.com/search?q=stars%3A%3E1000&type=repositories&s=forks&o=desc&p="
        url = url_start + str(page)

        soup = read_website(url, webdriver_path='~/chromedriver')

        for h3 in soup.find_all('h3'):
            try:
                repo_link = h3.div.find_all('div')[1].a['href']
                repos.append(repo_link[1:])
            except (IndexError, AttributeError) as e:
                print(f"Error: {e}. Skipping this entry.")
                continue
    return repos

def add_new_repos(repos):
    file_path = '../repo_names.txt'  # Change this to the desired file path

    # Read existing content into a set
    existing_repos = set()
    with open(file_path, 'r') as file:
        for line in file:
            existing_repos.add(line.strip())

    # Open the file in append mode
    with open(file_path, 'a') as file:
        # Add new strings that are not already in the file
        for repo in repos:
            if repo not in existing_repos:
                file.write(repo + '\n')
                existing_repos.add(repo)


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)