# Description: This script creates a dataframe with all the works from the scriptures-json repository
import pandas as pd
import json
import re
from urllib.request import urlopen 

#Found JSONs from https://github.com/bcbooks/scriptures-json
works = ['old-testament','new-testament', 'book-of-mormon','doctrine-and-covenants','pearl-of-great-price']
works_url = [f'https://raw.githubusercontent.com/bcbooks/scriptures-json/master/{work}.json' for work in works]

df_list = []

#Iterate through all the works, parses the JSON, and adds the verses to a dataframe
for index, url in enumerate(works_url):
    response = urlopen(url) 
    # storing the JSON response  
    # from url in data 
    data_json = json.loads(response.read()) 
    

    if works[index] == 'doctrine-and-covenants':
        for i in data_json["sections"]:
            section = i["section"]
            for j in i["verses"]:
                verseNum = j["verse"]
                verse = j["text"]
                reference = j['reference']
                row = {
                        'Work': 'd&c',
                        'Book':'D&C',
                        'Chapter':section,
                        'VerseNum':verseNum,
                        'Reference': reference,
                        'Verse': verse
                }
                df_list.append(row)
    else: 
        for i in data_json['books']:
            book = i["book"]
            for j in i["chapters"]:
                chapter = j["chapter"]
                for k in j["verses"]:
                    verseNum = k["verse"]
                    verse = k["text"]
                    reference = k['reference']
                    row = {
                            'Work': works[index],
                            'Book':book,
                            'Chapter':chapter,
                            'VerseNum':verseNum,
                            'Reference': reference,
                            'Verse': verse
                    }
                    df_list.append(row)

df = pd.DataFrame(df_list)

#Clean the verses
def clean_verse(row):

    verse = row['Verse'].lower()
    #remove punctuation
    verse = re.sub('[^a-zA-Z]', ' ', verse)
    #remove tags
    verse=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",verse)

    #remove digits and special chars
    verse=re.sub("(\\d|\\W)+"," ",verse)
    return verse


df['CleanedVerse'] = df.apply(clean_verse, axis=1)

df.to_csv("datasets/allWorks.csv")

# Convert to JSON

dict_data = df.set_index('Reference')['Verse'].to_dict()
# json_data = df[["Reference","Verse"]].to_json(orient='records')
json_data = json.dumps(dict_data, indent=4)

# Write to a JSON file
with open('datasets/all_references.json', 'w') as f:
    f.write(json_data)