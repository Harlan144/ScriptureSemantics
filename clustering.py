"""
This file processes subsets of the data using UMAP and PCA.
This file should be used to visualize how the different verses
relate to each other in high dimensional space.
"""

import pandas as pd
import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer 
import torch
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def plot_vectors(books, category, name):
    """
    This function plots different books of scripture onto a 2D plane

    Parameters
    ----------
        books : list
            A list of book names to include in the plot, make sure they are capitalized
        category : String 
            The way we want to differentiate hue (chapters, references, books)
        name : String 
            The name of the plot to be saved
    """
    df = pd.read_csv("datasets/allWorks.csv",index_col=0)
    
    # Filter to only consider the books we passed in
    book_subset = [book.replace("#"," ") for book in books]

    if len(books)==1 and "?" in books[0]:
            book_subset = [book.split("?")[0] for book in book_subset]
            chapter_subset = int(books[0].split("?")[1])

    else:
        chapter_subset = -1

    assert category in ["books", "verses", "chapters"], f"Invalid hue grouping, must be one of ['books', 'verses', 'chapters'] not '{category}'"
    # assert context in ["verse","context"], f"Invalid vector option, must be one of ['verse', 'context'] not '{context}'"
    embeddings= torch.load(f'modelTensors/all-MiniLM-L6-v2_(1)_with_punc.pt')
    for item in book_subset:
        assert item in df["Book"].tolist(),f"'{item}' is not a valid book name, perhaps try capitalizing it"

    df = df[df["Book"].isin(book_subset)]

    if chapter_subset!=-1:
        df = df[df["Chapter"]==chapter_subset]
    # df = df.reset_index(drop=True)
    embeddings_limited = embeddings[df.index,:]
    # df_x = df["Embeddings"]
    X = embeddings_limited.numpy()
    sns.set_theme()

    # Set up basic dimensionality reduction
    scaler = StandardScaler()
    pca = PCA(n_components=.9)
    umap = UMAP(
        n_components=2,
        n_neighbors=15,

        min_dist=0.0001,
        spread=1,

        # metric='cosine',
        metric='cosine',
        init='spectral',
        # # init='random',
        # random_state=0
    )

    # Perfom the dimensionality reduction
    print("rescaling data")
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    print("Reducing Data")
    pca.fit(X_scaled)

    print(f"reducing compnents via PCA to {pca.n_components_} components")
    X_reduced = pca.transform(X_scaled)

    print("UMAP")
    clusters = umap.fit_transform(X_reduced)

    print("Successfully performed dimensionality reduction")

    # Put the clusters in a dataframe
    df_plot = pd.DataFrame(clusters,columns=["x","y"])
    df_reset = df.reset_index(drop=True)

    df_plot["books"] = df_reset["Book"]
    df_plot["verses"] = df_reset["VerseNum"]
    df_plot["chapters"] = df_reset["Chapter"]

    # Plot clusters
    sns.set_palette(sns.color_palette()[2:])
    sns.relplot(data=df_plot,x="x",y="y",hue=category,kind="scatter")
    plt.title(name)

    # Save figure
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(f"dimReducPlots/{name}.png")

if __name__ == "__main__":
    name = sys.argv[1]
    category = sys.argv[2]
    books = sys.argv[3:]
    plot_vectors(books = books, category=category,name=name)

    # Outline of how to use this
    #>>> python3 analysis/clustering.py <name_of_plot> <hue_grouping> <vector_type> <Book1> <Book2> ...

    # <name_of_plot> is the name of the file you are creating (to be saved in the plots folder)
    # <hue_grouping> is how we will divide up the hues on the plot (chapters, books, references)
    # <vector_type> is wether we are using individual verse vectors or the ones that are context aware (verse, context)
    # We probably should generalize this to use whole chapters as well!
    
    # Example
    #>>> python3 analysis/clustering.py GalEphPlot chapters context Galatians Ephesians
    #>>> python3 analysis/clustering.py JohnNephMosesPlot books verse John "1 Nephi" Moses
    # Any you can enter any number of verses that you would like

# fig.update_traces(textposition='top center')
# fig.show()
# fig.write_html( "./all_counts.html" )
