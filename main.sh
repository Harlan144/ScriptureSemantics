# Description: This script runs the entire pipeline for the project.

# Create the datasets
mkdir -p datasets
python3 create_all_works_df.py


# Make and clean topical guide
mkdir -p topicalGuide
python3 scrapingLdsTopicalGuide.py
python3 clean_topical_guide.py #Cleans, splits topical guide in testing and training set based on "training_topics.tsv"


# Creates output directories for the embeddings, cosine similarity files, and analysis
mkdir -p modelTensors
mkdir -p cosineSimilarity/mapped
mkdir -p cosineSimilarity/unmapped
mkdir -p analysis

# Create tensors and calculate similarity at a depth of 25
# Output the model's performance on the topicalGuide using Recall at K
for model in 'all-MiniLM-L6-v2' 'sentence-transformers/all-roberta-large-v1'
do (
    for context_window in "0" "1" "2"
    do (
        python3 create_tensor.py $model $context_window 25
    )
    done
)
done


#Visualizes the cosine similarity of the tensors


#From previous testing, modelTensors/all-MiniLM-L6-v2_with_punc performs best, so that is used from now on
#Sets a threshold of 0.85 as the cutoff
#Only colors books with 200 or more verses.
#Potentially includes more than 1 connection from a verse (two lines from one verse is ok)
mkdir -p syntenyPlots
python3 circle_synteny.py

#Uses UMAP and PCA to visualize clustering
mkdir -p dimReducPlots
#Cluster example between Books. We use # to represent a space in a book title
python3 clustering.py JohnMatthew3NephiPlot books 3#Nephi John Matthew
#Cluster example between Chapters.
python3 clustering.py JacobPlot chapters Jacob
#Cluster example between Verses. We use ? to represent a space in a chapter title
python3 clustering.py Alma36Plot verses Alma?36

#Conference Analysis
#The Conference.tsv and conference2023.tsv files are not included in the repository. 
#They were sourced from a representative of The Church of Jesus Christ of Latter-day Saints.
#They can also be found at https://www.churchofjesuschrist.org/general-conference?lang=eng
python3 conferenceAnalysis/fastConferenceAnalysis.py
python3 conferenceAnalysis/visualizeConferenceResults.py
python3 conferenceAnalysis/visualizeConference2.py

# rm -rf modelTensors
# rm -rf cosineSimilarity
# rm -rf analysis
# rm -rf datasets
# rm -rf dimReducPlots
