#Checked LDS robot.txt file. It appears we can legally scrape (or attempt to)
#With aide from chat GPT:

from urllib.request import urlopen
from bs4 import BeautifulSoup
import sys
import json

url = 'https://www.churchofjesuschrist.org/study/scriptures/tg'

# Send HTTP request and open the URL
with urlopen(url) as response:
    # Read the HTML content
    html_content = response.read()

# Parse HTML content with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# first_paragraph = soup.find('p').get_text()
# print(first_paragraph[:1000])  # Print the first 1000 characters
# Extract all links on the page
links = soup.find_all('a')
topics = dict()
foundB = False
num = 0
try:
    for i, link in enumerate(links):
        num += 1
        if i > 2:
            topic = link.get_text()
            print(topic)
            if topic.startswith("B"):
                foundB = True
            if foundB and topic.startswith("A"):
                break
            url = 'https://www.churchofjesuschrist.org' + link.get('href')
            with urlopen(url) as response:
                topic = url.split("/")[-1].split("?")[0]
                topics[topic] = []
                # Read the HTML content
                html_content2 = response.read()
                s = BeautifulSoup(html_content2, 'html.parser')
                # print(str(s)[:5000])
                paragraphs = s.find_all('p', class_='entry')
                for paragraph in paragraphs:
                    # Find all scripture references within the current paragraph
                    references = paragraph.find_all('a', class_='scripture-ref')

                    for reference in references:
                        # Extract and append the text of each scripture reference
                        info = reference.get_text().replace('\xa0', ' ').strip()
                        if str.isdigit(info[0]) and info[1] != " ":
                            pastBook = " ".join(topics[topic][-1].split(" ")[:-1])
                            info = pastBook + " " + info
                        topics[topic].append(info)
except:
    print("Error in", num)
# with open("topical_guide.json", "w") as jsonFile:
#     jsonFile.write(json.dumps(topics))

cleanerTG = dict()
for topic in topics:
    if topics[topic] != []:
        cleanerTG[topic] = topics[topic]
print(cleanerTG)
# with open("datasets/topical_guide_clean.json", "w") as topGFile:
#     topGFile.write(json.dumps(cleanerTG))
correctAbbreviations = dict()
with open("translatePlease.tsv", "r") as readFile:
    header = readFile.readline()
    for line in readFile:
        line = line.rstrip()
        line = line.split("\t")
        correctAbbreviations[line[0]] = line[1]
for topic, references in cleanerTG.items():
    for i, ref in enumerate(references):
        splitting = ref.split(".")
        if len(splitting) == 2:
            splitting[0] = correctAbbreviations[splitting[0]]
            print(splitting[0])
        cleanerTG[topic][i] = "".join(splitting)
with open("topicalGuide/topical_guide_grace.json", "w") as topGFile:
    topGFile.write(json.dumps(cleanerTG))