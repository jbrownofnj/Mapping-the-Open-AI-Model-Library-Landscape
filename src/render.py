import os
import sqlite3
from dotenv import load_dotenv
from huggingface_hub import HfApi
import json
import matplotlib.pyplot as plt
import pandas as pd
from db import *
from analysis import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from analysis import *


dbPath = "data/huggingface_models.db"
def getDF(command):
    conn = sqlite3.connect(dbPath)
    df = pd.read_sql_query(command, conn)
    conn.close()
    return df
def graphTopLibraries(limit=15):
    df = getDF(f"SELECT libraryName, COUNT(*) AS modelCount FROM models WHERE libraryName IS NOT NULL GROUP BY libraryName ORDER BY modelCount DESC LIMIT {limit};")
    x = range(len(df))
    plt.figure(figsize=(10, 6))
    plt.bar(x, df["modelCount"])
    plt.xticks(x, df["libraryName"], rotation=45, ha="right")
    plt.title("Top Libraries by Number of Models")
    plt.xlabel("Library")
    plt.ylabel("Number of Models")
    plt.tight_layout()
    plt.savefig("topLibraries.png")

def graphLibraryDownloads(limit=15):
    df = getDF(f"SELECT libraryName, SUM(downloads) AS totalDownloads FROM models WHERE libraryName IS NOT NULL AND downloads IS NOT NULL GROUP BY libraryName ORDER BY totalDownloads DESC LIMIT {limit};")
    x = range(len(df))
    plt.figure(figsize=(10,6))
    plt.bar(x, df["totalDownloads"])
    plt.xticks(x, df["libraryName"], rotation=45, ha="right")
    plt.title("Top Libraries by Total Downloads")
    plt.xlabel("Library")
    plt.ylabel("Total Downloads")
    plt.tight_layout()
    plt.savefig("libraryDownloads.png")

def graphTopTags(limit=15):
    df = getDF(f"SELECT tag, COUNT(*) AS count FROM model_tags GROUP BY tag ORDER BY count DESC LIMIT {limit};")
    x = range(len(df))
    plt.figure(figsize=(10,6))
    plt.bar(x, df["count"])
    plt.xticks(x, df["tag"], rotation=45, ha="right")
    plt.title("Top Tags by Count")
    plt.xlabel("Tag")
    plt.ylabel("Number of Models")
    plt.tight_layout()
    plt.savefig("topTags.png")
def graphTagDownloads(limit=15):
    df = getDF(f"SELECT model_tags.tag, SUM(models.downloads) AS totalDownloads FROM model_tags JOIN models ON model_tags.modelID = models.id WHERE models.downloads IS NOT NULL GROUP BY model_tags.tag ORDER BY totalDownloads DESC LIMIT {limit};")
    x = range(len(df))
    plt.figure(figsize=(10,6))
    plt.bar(x, df["totalDownloads"])
    plt.xticks(x, df["tag"], rotation=45, ha="right")
    plt.title("Top Tags by Downloads")
    plt.xlabel("Tag")
    plt.ylabel("Total Downloads")
    plt.tight_layout()
    plt.savefig("tag_downloads.png")


def graphModelsOverTime():
    df = getDF("SELECT strftime('%Y', createdAt) AS year, COUNT(*) AS count FROM models WHERE createdAt IS NOT NULL GROUP BY year ORDER BY year;")
    plt.figure(figsize=(10,6))
    plt.plot(df["year"], df["count"], marker='o')
    plt.title("Model Creation Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Models")
    plt.tight_layout()
    plt.savefig("modelsOT.png")

def produceGraphs():
    graphTopLibraries(15)
    graphLibraryDownloads(15)
    graphTopTags(15)
    graphTagDownloads(15)
    graphModelsOverTime()
