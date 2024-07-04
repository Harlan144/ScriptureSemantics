"""For each year of general conference talks, analyze how often the talks contain primary references to 
differet works of scripture. Visualize these results."""
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

#Import the best performing model
model = SentenceTransformer('all-MiniLM-L6-v2')
with open("datasets/all_references.json", "r") as readFile:
    allRef = json.load(readFile)

#List of all verses
verses_references = list(allRef.keys())

#This function will take in a list of paragraphs and a list of verses and return the top 3 verses that are most similar to each paragraph
def matrix_cosine_sim(paragraph_embeddings, verse_embeddings, verse_references, threshold):
    num_paragraphs = len(paragraph_embeddings)
    # num_verses = len(verse_embeddings)

    cosine_scores = util.cos_sim(paragraph_embeddings, verse_embeddings)
    top_results = torch.topk(cosine_scores, k=3)
    # # Normalize embeddings
    # norm_paragraphs = paragraph_embeddings / np.linalg.norm(paragraph_embeddings, axis=1)[:, np.newaxis]
    # norm_verses = verse_embeddings / np.linalg.norm(verse_embeddings, axis=1)[:, np.newaxis]

    # # Calculate cosine similarity matrix
    # similarities = np.dot(norm_paragraphs, norm_verses.T)

    # Find top 3 matches with similarity above threshold for each paragraph
    matches = []
    paragraph_index= 0
    for score, idx in zip(top_results[0], top_results[1]):
        for j in range(3):
            if score[j] > threshold:
                matches.append({
                    "paragraph_index": paragraph_index,
                    "reference": verse_references[int(idx[j])],
                    "similarity_score": float(score[j])
                })
        paragraph_index+=1


    # for i in range(num_paragraphs):
    #     top_matches = []
    #     for j in range(3):
    #         print(similarities[i][j])
    #         if similarities[i][j] > threshold:
    #             top_matches.append({
    #                 "paragraph_index": i,
    #                 "reference": verse_references[j],
    #                 "similarity_score": similarities[i][j]
    #             })
    #     top_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    #     matches.extend(top_matches[:3])

    return matches

#Hard code in the different books of scripture
gospelsList = set(["Matthew", "Mark", "Luke", "John"])
oldTestamentList = set(["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Solomon's Song", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", 
"Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"])
bookOfMormonList = set(['Enos', 'Jarom', 'Omni', 'Mosiah', 'Moroni', 'Words of Mormon', '4 Nephi', '3 Nephi', '2 Nephi', 'Ether', 'Jacob', 'Alma', 'Helaman', 'Mormon', '1 Nephi'])
doctrineCovList = set(["D&C"])
pearlOfGreatPrice = set(["Moses", "Joseph Smith—History", "Abraham", "Joseph Smith\u2014Matthew", "Joseph Smith—Matthew", "Joseph Smith\u2014History", "Articles of Faith"])
secondHalfNewTestamentList = set(["Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"])
newTestamenSet = gospelsList.union(secondHalfNewTestamentList)

#Load in the embeddings across the scriptures using the best performing model
verseEmbeddings = torch.load("modelTensors/all-MiniLM-L6-v2_with_punc.pt")


#Iterate through all the years of general conference and analyze the references to scripture
#Determine which book each reference is from
for yearOfInterest in range(1880, 2024):
    yearOfInterest = str(yearOfInterest)
    paragraphsToSearch = []
    with open("datasets/Conference.tsv", "r") as readFile:
        header = readFile.readline().rstrip()
        header = header.split()
        for line in readFile.readlines():
            line = line.rstrip().split("\t")
            if line[0] == yearOfInterest:
                if "." not in line[-1] and "?" not in line[-1] and "!" not in line[-1]:
                    continue
                if len(line[-1].split(" ")) < 4:
                    continue
                if line[-1] == "See online Liahona, listen, download.":
                    continue
                paragraphsToSearch.append(line[-1])
    with open("datasets/conference2023.tsv", "r") as readFile:
        header = readFile.readline().rstrip()
        for line in readFile.readlines():
            line = line.rstrip().split("\t")
            try:
                if int(line[0]) == int(yearOfInterest):
                    if "." not in line[-1] and "?" not in line[-1] and "!" not in line[-1]:
                        continue
                    if len(line[-1].split(" ")) < 4:
                        continue
                    if line[-1] == "See online Liahona, listen, download.":
                        continue
                    paragraphsToSearch.append(line[-1])
            except:
                if int(line[0].split(" ")[0]) == int(yearOfInterest):
                    if "." not in line[-1] and "?" not in line[-1] and "!" not in line[-1]:
                        continue
                    if len(line[-1].split(" ")) < 4:
                        continue
                    if line[-1] == "See online Liahona, listen, download.":
                        continue
                    paragraphsToSearch.append(line[-1])

    print("Completed")
    numReferences = 0
    bookRefDict = dict()
    for book in secondHalfNewTestamentList:
        bookRefDict[book] = 0
    for book in gospelsList:
        bookRefDict[book] = 0
    for book in bookOfMormonList:
        bookRefDict[book] = 0
    for book in oldTestamentList:
        bookRefDict[book] = 0
    for i in range(1, 139):
        bookRefDict[f"D&C {i}"] = 0
    for book in pearlOfGreatPrice:
        bookRefDict[book] = 0
    print("Num paragraphs", len(paragraphsToSearch))

    yearEmbeddings = model.encode(paragraphsToSearch, convert_to_tensor=True, show_progress_bar=True)

    SIGNIFICANCELEVEL = 0.75
    #for year in general conference:
    verses = matrix_cosine_sim(yearEmbeddings, verseEmbeddings, verses_references, SIGNIFICANCELEVEL)
    print(verses)
    for verse in verses:
        bookOfOrigin = " ".join(verse["reference"].split(":")[0].split(" ")[:-1])
        print(bookOfOrigin)
        if bookOfOrigin == "D&C":
            bookRefDict[verse["reference"].split(":")[0]] += 1
        else:
            bookRefDict[bookOfOrigin] += 1

    #Get the sums for each category...
    numRefBom = 0
    numRefOld = 0
    numRefDC = 0
    numRefGosp = 0
    numRef2NT = 0
    numPGP = 0
    for book in secondHalfNewTestamentList:
        numRef2NT += bookRefDict[book]
    for book in gospelsList:
        numRefGosp += bookRefDict[book]
    for book in bookOfMormonList:
        numRefBom += bookRefDict[book]
    for book in oldTestamentList:
        numRefOld += bookRefDict[book]
    for i in range(1, 139):
        numRefDC += bookRefDict[f"D&C {i}"]
    for book in pearlOfGreatPrice:
        numPGP += bookRefDict[book]
    
    #Write the results to a file
    with open(f"conferenceAnalysis/conferenceResults/reviewReferences{yearOfInterest}.tsv", "w") as writeFile:
        for match in verses:
            writeFile.write(f"{paragraphsToSearch[match['paragraph_index']]}\t{match['reference']}\t{match['similarity_score']}\n")
    #Write the specific verse references to files
    with open(f"conferenceAnalysis/conferenceResults/conferenceResults{yearOfInterest}.tsv", "w") as recordFile:
        recordFile.write(f"New Testament\t{yearOfInterest}\t{numRefGosp + numRef2NT}\n")
        recordFile.write(f"Gospels\t{yearOfInterest}\t{numRefGosp}\n")
        recordFile.write(f"Old Testament\t{yearOfInterest}\t{numRefOld}\n")
        recordFile.write(f"D&C\t{yearOfInterest}\t{numRefDC}\n")
        recordFile.write(f"2nd Half of New Testament\t{yearOfInterest}\t{numRef2NT}\n")
        recordFile.write(f"Book of Mormon\t{yearOfInterest}\t{numRefBom}\n")
        recordFile.write(f"Pearl of Great Price\t{yearOfInterest}\t{numPGP}\n")
        recordFile.write(f"All Scripture\t{yearOfInterest}\t{numRefBom + numRef2NT + numRefDC + numRefOld + numRefGosp + numPGP}\n")

    with open(f"conferenceAnalysis/conferenceResults/mostQuotedBooksByYear{yearOfInterest}.tsv", "w") as writeFile:
        quotedBook = max(bookRefDict, key=bookRefDict.get)
        del bookRefDict[quotedBook]
        quotedBook2 = max(bookRefDict, key=bookRefDict.get)
        del bookRefDict[quotedBook2]
        quotedBook3 = max(bookRefDict, key=bookRefDict.get)
        del bookRefDict[quotedBook3]
        quotedBook4 = max(bookRefDict, key=bookRefDict.get)
        del bookRefDict[quotedBook4]
        quotedBook5 = max(bookRefDict, key=bookRefDict.get)
        writeFile.write(f"{yearOfInterest}\t{quotedBook}\t{quotedBook2}\t{quotedBook3}\t{quotedBook4}\t{quotedBook5}\n")