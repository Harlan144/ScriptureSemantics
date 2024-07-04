# Purpose: Visualize the change in scripture quotation over time using a scatter plot.
import os
import matplotlib.pyplot as plt
import numpy as np

years = []
specialPlot = dict()
for y in range(1880, 2023):
    years.append(y)
percentDict = dict()
for year in years:
    percentDict[year] = dict()

# Iterate through all the files in the conferenceResults directory
for file in os.listdir("conferenceAnalysis/conferenceResults"):
    if file.startswith("conferenceResults"):
        print(file)
        with open("conferenceAnalysis/conferenceResults/" + file, "r") as readFile:
            tmpDict = dict()
            year = 0
            for section in ["Old Testament", "D&C", "New Testament", "Book of Mormon", "Pearl of Great Price"]:
                tmpDict[section] = 0
            allCounts = 0
            for i, line in enumerate(readFile):
                line = line.rstrip("\n").split("\t")
                if i == 0:
                    year = int(line[1])
                if line[0] in tmpDict:
                    tmpDict[line[0]] = int(line[2])
                else:
                    try:
                        allCounts = int(line[2])
                    except:
                        continue
            if allCounts == 0:
                continue
            for section in tmpDict:
                tmpDict[section] = tmpDict[section] / allCounts
            percentDict[year] = tmpDict
            specialPlot[year] = tmpDict

# Create a dictionary of book percentages for each year
year2 = sorted(list(specialPlot.keys()))
book_percentages = {
    book: [specialPlot[year].get(book, 0) * 100 for year in year2] for book in specialPlot[1884]
}

# Create the plot
plt.figure(figsize=(12, 6))
for book, percentages in book_percentages.items():
    plt.scatter(year2, percentages, label=book, s=10, alpha=0.5)  # Use scatter plot

# Set labels and legend
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.title("Change in Scripture Quotation Over Time")
plt.legend(loc="upper right")

# Fit lines of best fit
for book, percentages in book_percentages.items():
    z = np.polyfit(year2, percentages, 1)
    p = np.poly1d(z)
    plt.plot(year2, p(year2), linewidth=2, label=f'Line of Best Fit for {book}')

# Save the plot to a file (e.g., PNG format)
plt.grid(True)
plt.savefig("conferenceAnalysis/scatter_scripture_quotation_over_time.png")

# Show the plot
plt.show()