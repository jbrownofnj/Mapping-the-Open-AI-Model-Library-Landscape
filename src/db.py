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

dbPath = "data/huggingface_models.db"
load_dotenv()
hfToken = os.getenv("HF_TOKEN")
api = HfApi(token=hfToken)

def insertModel(conn, model):
    model_id = model.id
    author = model.author if model.author else None
    createdAt = model.created_at.isoformat() if model.created_at else None
    lastModified = model.last_modified.isoformat() if hasattr(model, "last_modified") and model.last_modified else None
    downloads = model.downloads
    likes = model.likes
    pipelineTag = model.pipeline_tag if hasattr(model, "pipeline_tag") else None
    libraryName = model.library_name if hasattr(model, "library_name") else None
    
    fullJSON = json.dumps(vars(model), default=str)
    conn.execute("INSERT OR REPLACE INTO models(id, author, createdAt, lastModified, downloads, likes, pipelineTag, libraryName) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", (model_id, author, createdAt, lastModified, downloads, likes, pipelineTag, libraryName))

    
    conn.execute("INSERT OR REPLACE INTO fullJSON (id, fullJSON) VALUES (?, ?);", (model_id, fullJSON))
    tags = model.tags if hasattr(model, "tags") and model.tags else []
    
    for tag in tags:
        conn.execute("INSERT OR IGNORE INTO tags (tag)VALUES (?);", (tag,))
        conn.execute("INSERT OR IGNORE INTO model_tags (modelID, tag)VALUES (?, ?);", (model_id, tag))

def populateDataDBBySearch(searchString, returnLimit):
    conn = sqlite3.connect(dbPath)
    returnedModels = api.list_models(search=searchString, limit=returnLimit, full=True)
    for model in returnedModels:
        insertModel(conn, model)
    conn.commit()
    conn.close()

def populateDataDBGlobal(returnLimit):
    conn = sqlite3.connect(dbPath)
    sortMethods = ["downloads", "likes", "created_at", "last_modified"]
    for sortMethod in sortMethods:
        print(f"colecting sorted by {sortMethod}")
        returnedModels = api.list_models(sort=sortMethod,limit=returnLimit,full=True)
        for model in returnedModels:
            insertModel(conn, model)
    conn.commit()
    conn.close()

def populateDataBase(returnLimit):
    tasks = [
        "text-generation",
        "automatic-speech-recognition",
        "text-classification",
        "text-to-image",
        "image-classification",
        "token-classification",
        "image-text-to-text",
        "fill-mask",
        "feature-extraction",
        "question-answering",
        "sentence-similarity",
        "summarization",
        "translation",
        "text-to-speech",
        "image-to-image",
        "zero-shot-image-classification",
        "reinforcement-learning",
        "image-feature-extraction",
        "image-to-video",
        "robotics",
        "text-to-video",
        "image-to-text",
        "object-detection",
        "any-to-any",
        "audio-classification",
        "text-to-audio",
        "image-segmentation",
        "audio-to-audio",
        "text-ranking",
        "time-series-forecasting",
        "depth-estimation",
        "zero-shot-classification",
        "image-to-3d",
        "mask-generation",
        "audio-text-to-text",
        "multiple-choice",
        "visual-question-answering",
        "video-to-video",
        "visual-document-retrieval",
        "video-classification",
        "keypoint-detection",
        "video-text-to-text",
        "unconditional-image-generation",
        "tabular-classification",
        "voice-activity-detection",
        "zero-shot-object-detection",
        "other",
        "text-to-3d",
        "image-text-to-image",
        "tabular-regression",
        "graph-ml",
        "table-question-answering",
        "document-question-answering",
        "image-text-to-video"
    ]

    searchTerms = [
        "transformers",
        "diffusers",
        "sentence-transformers",
        "bert",
        "llama",
        "t5",
        "vit",
        "whisper",
        "stable-diffusion",
    ]

    for term in tasks + searchTerms:
        populateDataDBBySearch(term, returnLimit)

def makeTables():
    conn = sqlite3.connect(dbPath)
    conn.execute("DROP TABLE IF EXISTS model_tags;")
    conn.execute("DROP TABLE IF EXISTS tags;")
    conn.execute("DROP TABLE IF EXISTS fullJSON;")
    conn.execute("DROP TABLE IF EXISTS models;")
    conn.commit()
    command = "CREATE TABLE models (id TEXT PRIMARY KEY, author TEXT, createdAt TEXT, lastModified TEXT, downloads INTEGER, likes INTEGER, pipelineTag TEXT, libraryName TEXT);"
    conn.execute(command)
    command = "CREATE TABLE tags (tag TEXT PRIMARY KEY);"
    conn.execute(command)
    command = "CREATE TABLE model_tags (modelID TEXT, tag TEXT, PRIMARY KEY (modelID, tag), FOREIGN KEY (modelID) REFERENCES models(id),FOREIGN KEY (tag) REFERENCES tags(tag));"
    conn.execute(command)
    command = "CREATE TABLE fullJSON (id TEXT PRIMARY KEY,fullJSON TEXT,FOREIGN KEY (id) REFERENCES models(id));"
    conn.execute(command)

    conn.commit()
    conn.close()
    
def fullPopulateDB():
    makeTables()
    populateDataBase(10000)
    populateDataDBGlobal(10000)


    