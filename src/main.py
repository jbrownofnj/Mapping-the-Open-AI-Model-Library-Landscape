import os
import sqlite3
from dotenv import load_dotenv
from huggingface_hub import HfApi
import json
import matplotlib.pyplot as plt
import pandas as pd
from db import *
from analysis import *
from render import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dbPath = "/home/john/Documents/dataManagement/hf-library-landscape/data/huggingface_models.db"

load_dotenv()
hfToken = os.getenv("HF_TOKEN")
api = HfApi(token=hfToken)

#fullPopulateDB()
runFullAnalysis()
produceGraphs()
runPredictionModelsNoLikes()


