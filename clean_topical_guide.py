import json
import re
import pandas as pd


def work_chapter_split(input_string):
    #Regular expression pattern to split the string
    pattern = r'(.+)\s(\d+.*)' #This pattern matches the name and the number
    result = re.match(pattern, input_string)

    if result:
        # Extracting the parts
        book = result.group(1)
        chapter = result.group(2)
        return book, chapter
    else:
        print(input_string, "Pattern not matched.")
        return None

def clean_commas(input, output):
    with open(input, "r") as readFile:
        topicalGuide = json.load(readFile)

    outputDict = {}
    for topicName, topicVerses in topicalGuide.items():
        verseList = []
        for verse in topicVerses:
            #Fix issues for references later.
            verse = verse.replace("JS\u2014M", "Joseph Smith—Matthew")
            verse = verse.replace("JS\u2014H", "Joseph Smith—History")
            verse = verse.replace('A of F', 'Articles of Faith')
            verse = verse.replace('W of M', 'Words of Mormon')
            verse = verse.replace('Song', "Solomon's Song")
            #Remove facimilies.
            if "Abr, fac " in verse:
                continue
            
            if "," in verse:
                if ":" not in verse: #Apparently doesn't happen.
                    print(verse)

                    book, chapters = work_chapter_split(verse)
                    for chapter in chapters.split(", "):
                        verseList.append(f"{book} {chapter}")
                        print('appended', f"{book} {chapter}")

                else:
                    # print(verse)
                    book_chapter = verse.split(":")[0]
                    book, chapter = work_chapter_split(book_chapter)
                    split_verse_commas = verse.split(", ")

                    verseList.append(split_verse_commas[0])
                    for chap_verse in split_verse_commas[1:]:
                        if ":" in chap_verse:
                            # print('appended', f"{book} {chap_verse}")
                            verseList.append(f"{book} {chap_verse}")
                            chapter= chap_verse.split(":")[0]
                        else:
                            # print('appended', f"{book} {chapter}:{chap_verse}")
                            verseList.append(f"{book} {chapter}:{chap_verse}")

            else:
                verseList.append(verse)
        outputDict[topicName] = verseList

    with open(output, 'w') as f:
        json.dump(outputDict, f)



def clean_json_hyphens(input, output):
    with open(input, "r") as readFile:
        topicalGuide = json.load(readFile)

    outputDict = {}
    for topicName, topicVerses in topicalGuide.items():
        verseList = []
        for verse in topicVerses:

            if re.search(r'\d+\u2013\d+', verse):
                if ":" in verse:
                    # print(verse)

                    book_chapter = verse.split(":")[0]
                    verseRange = verse.split(":")[1]
                    startingVerse = int(verseRange.split("\u2013")[0])
                    endingVerse = int(verseRange.split("\u2013")[1])
                    book, chapter = work_chapter_split(book_chapter)

                    if endingVerse<=startingVerse: #For three digits it can be "108-9" meaning 108-109
                        if verse == "Matthew 9:35–11:1":
                            continue #give up on this for now
                        if endingVerse>10:
                            endingVerse = 100*(startingVerse//100)+ endingVerse
                        else:
                            endingVerse = 10*(startingVerse//10)+endingVerse
                    # print(verse)
                    for i in range(startingVerse, endingVerse+1):
                        # print('appended', f"{book} {chapter}:{i}")
                        verseList.append(f"{book} {chapter}:{i}")

                else:
                    book, chapters = work_chapter_split(verse)
                    startingChapter = int(chapters.split("\u2013")[0])
                    endingChapter = int(chapters.split("\u2013")[1])
                    for i in range(startingChapter, endingChapter+1):
                        verseList.append(f"{book} {i}")

            else:
                verseList.append(verse)

        outputDict[topicName] = verseList

    with open(output, 'w') as f:
        json.dump(outputDict, f)

def clean_json_chapters_to_verses(input, output, all_Verses):
    with open(input, "r") as readFile:
        topicalGuide = json.load(readFile)
    
    df = pd.read_csv(all_Verses)
    outputDict = {}

    for topicName, topicVerses in topicalGuide.items():
        verseList = []
        for verse in topicVerses:         
            if ":" not in verse:
                # print(verse)
                book, chapter = work_chapter_split(verse)
                if book=="chapters": #Remove those 'chapters' entries, unclear what they come from.
                    continue
                
                associated_df = df[(df["Book"]==book) & (df["Chapter"]==int(chapter))]
                # print(len(associated_df))

                for i in range(1, len(associated_df)+1):
                    # print('appended',f"{book} {chapter}:{i}")
                    verseList.append(f"{book} {chapter}:{i}")
            else:
                verseList.append(verse)
        outputDict[topicName] = verseList
    
    with open(output, 'w') as f:
        json.dump(outputDict, f)

def modify_json(input, output, all_Verses):
    with open(input, "r") as readFile:
        topicalGuide = json.load(readFile)
    
    df = pd.read_csv(all_Verses)
    #Add index to topicalGuide

    outputDict = {}
    for topicName, topicVerses in topicalGuide.items():
        verseDic = {}
        for verse in topicVerses:
            
            if verse in df["Reference"].values:
                verse_index = df.index[df["Reference"] == verse].tolist()
                if len(verse_index)!=1:
                    print("ERROR!",verse, verse_index)
                else:
                    verseDic[verse] = verse_index[0]
            else:
                print("Not in Reference", verse, topicName)
        outputDict[topicName] = verseDic


    with open(output, 'w') as f:
        json.dump(outputDict, f)

    
def limit_topical_guide(topical_guide_path, training_topics, output_topical_guide_path, training=True):
    topics = []
    with open(training_topics, 'r') as f:
        for line in f:
            topics.append(line.strip())
    with open(topical_guide_path) as f:
        topical_guide = json.load(f)

    if training:
        output_topical_guide = {key: topical_guide[key] for key in topics}
    else:
        output_topical_guide = {key: topical_guide[key] for key in topical_guide if key not in topics}

    with open(output_topical_guide_path, 'w') as f:
        json.dump(output_topical_guide, f) 
    return


if __name__ == "__main__":
    all_Verses = "datasets/allWorks.csv"

    inputJson ="topicalGuide/topical_guide_grace.json"
    cleaned_commas_json = "topicalGuide/topical_guide_cleaned_commas.json"

    clean_hyphens_json = "topicalGuide/topical_guide_cleaned_hyphens.json"
    cleaned_json = "topicalGuide/topical_guide_clean.json"
    indexed_json = "topicalGuide/indexed_topical_guide_clean.json"

    clean_commas(inputJson, cleaned_commas_json)
    clean_json_hyphens(cleaned_commas_json, clean_hyphens_json) #Im giving up on "Matthew 9:35–11:1" for now
    clean_json_chapters_to_verses(clean_hyphens_json, cleaned_json, all_Verses)

    modify_json(cleaned_json, indexed_json, all_Verses)

    training_topics = "training_topics.tsv"
    training_topical_guide = "topicalGuide/training_topical_guide.json"
    limit_topical_guide(cleaned_json, training_topics, training_topical_guide)
    testing_topical_guide = "topicalGuide/testing_topical_guide.json"
    limit_topical_guide(cleaned_json, training_topics, testing_topical_guide, training=False)