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
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from render import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

dbPath = "/home/john/Documents/dataManagement/hf-library-landscape/data/huggingface_models.db"
def getDF(command):
    connection = sqlite3.connect(dbPath)
    df = pd.read_sql_query(command, connection)
    connection.close()
    return df

def printTable(tableName, limit):
    connection = sqlite3.connect(dbPath)
    command = f"SELECT * FROM {tableName} LIMIT {limit};"
    cursor = connection.execute(command)
    tableResults = cursor.fetchall()
    print(f"The first {limit} rows of the {tableName} table:")
    for result in tableResults:
        print(result)

    connection.close()

def tableSummary():
    conn = sqlite3.connect(dbPath)
    tableNames = ["models", "tags", "model_tags", "fullJSON"]
    for tableName in tableNames:
        print(f"{tableName} table info:")
        command = f"PRAGMA table_info({tableName});"
        cursor = conn.execute(command)
        tableInfo = cursor.fetchall()
        print(f"Columns in {tableName} table:")
        for column in tableInfo:
            print(column)
    conn.close()

def printTableSize(tableName):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT COUNT(*) FROM {tableName};"
    cursor = conn.execute(command)
    tableSize = cursor.fetchall()
    print(f"{tableName} table size: {tableSize[0][0]} rows")
    conn.close()

def getPipelineTagCounts():
    conn = sqlite3.connect(dbPath)
    command = "SELECT pipelineTag, COUNT(*) AS count FROM modelsWHERE pipelineTag IS NOT NULL GROUP BY pipelineTag ORDER BY count DESC;"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    conn.close()
    return results

def getTagCounts(limit):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT tag, COUNT(*) AS count FROM model_tags GROUP BY tag ORDER BY count DESC LIMIT {limit};"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    conn.close()
    return results

def getTopDownloadedTags(limit):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT model_tags.tag, SUM(models.downloads) AS totalDownloads FROM model_tags JOIN models ON model_tags.modelID = models.id WHERE models.downloads IS NOT NULL GROUP BY model_tags.tag ORDER BY totalDownloads DESC LIMIT {limit};"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    conn.close()
    return results

def topModelsByDownloads(limit):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT * FROM models ORDER BY downloads DESC LIMIT {limit};"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    conn.close()
    return results

def topModelsByLikes(limit):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT * FROM models ORDER BY likes DESC LIMIT {limit};"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    conn.close()
    return results

def topModelInYear(year):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT * FROM models WHERE strftime('%Y', createdAt) = '{year}' ORDER BY downloads DESC LIMIT 1;"
    cursor = conn.execute(command)
    result = cursor.fetchone()
    conn.close()
    return result

def topModelInEachYear():
    for year in range(2020, 2026):
        topModel = topModelInYear(year)
        print(f"Top model in {year}:")
        print(topModel)

def authorLeaderboard(limit):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT author, COUNT(*) AS modelCount FROM models WHERE author IS NOT NULL GROUP BY author ORDER BY modelCount DESC LIMIT {limit};"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    conn.close()
    return results

def percentDownloadsByTopXPercent(x):
    conn = sqlite3.connect(dbPath)
    command = "SELECT downloads FROM models WHERE downloads IS NOT NULL ORDER BY downloads DESC;"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    #gets total downloads
    command = "SELECT SUM(downloads) FROM models WHERE downloads IS NOT NULL;"
    cursor = conn.execute(command)
    totalDownloads = cursor.fetchone()[0]
    #finds top x percent num models.
    topXCountIndex = int(len(results) * x / 100)
    topXDownloadsfromTopXpercent = sum([row[0] for row in results[:topXCountIndex] if row[0] is not None])

    percent = (topXDownloadsfromTopXpercent / totalDownloads) * 100 if totalDownloads and totalDownloads > 0 else 0

    conn.close()
    return percent

def topModelInEachPipelineTag():
    conn = sqlite3.connect(dbPath)
    command = "SELECT pipelineTag, id, downloads FROM models WHERE pipelineTag IS NOT NULL AND downloads IS NOT NULL AND downloads > 0 ORDER BY pipelineTag, downloads DESC;"
    
    cursor = conn.execute(command)
    results = cursor.fetchall()
    topModels = {}

    for pipelineTag, model_id, downloads in results:
        if pipelineTag not in topModels:
            topModels[pipelineTag] = (model_id, downloads)

    conn.close()
    return topModels

def getLikesDownloadRatio(limit=20):
    conn = sqlite3.connect(dbPath)
    cursor = conn.execute(f"SELECT libraryName, SUM(likes) * 1.0 / SUM(downloads) AS ratio FROM models WHERE downloads > 1000 AND libraryName IS NOT NULL GROUP BY libraryName ORDER BY ratio DESC LIMIT {limit};")
    results = cursor.fetchall()
    conn.close()
    return results

def runBasicStats():
    print("Table sizes:")
    printTableSize("fullJSON")
    printTableSize("models")
    printTableSize("tags")
    printTableSize("model_tags")
    print()
    print("table summaries:")
    tableSummary()

def runTopModels():
    print("top models by downloads:")
    results = topModelsByDownloads(10)
    for model in results:
        print(model)

    print("top models by likes:")
    results = topModelsByLikes(10)
    for model in results:
        print(model)

def runYearAnalysis():
    print("top model in each year:")
    topModelInEachYear()

def runAuthorStats():
    print("author leaderboard:")
    for author in authorLeaderboard(10):
        print(author)

def runPipelineAnalysis():
    print("top model in each pipeline tag:")
    results = topModelInEachPipelineTag()
    for pipelineTag, (model_id, downloads) in results.items():
        print(f"{pipelineTag}: {model_id} ({downloads})")

def runLibraryStats():
    print("likes/downloads ratio by library:")
    results = getLikesDownloadRatio(20)
    for libraryName, ratio in results:
        print(f"{libraryName}: {ratio:.4f}")

def nullSummary():
    conn = sqlite3.connect(dbPath)
    command = """
        SELECT COUNT(*) AS totalModels, SUM(CASE WHEN libraryName IS NULL THEN 1 ELSE 0 END) AS missingLibrary, SUM(CASE WHEN pipelineTag IS NULL THEN 1 ELSE 0 END) AS missingPipelineTag,
        SUM(CASE WHEN createdAt IS NULL THEN 1 ELSE 0 END) AS missingCreatedAt, SUM(CASE WHEN lastModified IS NULL THEN 1 ELSE 0 END) AS missingLastModified,
        SUM(CASE WHEN downloads IS NULL THEN 1 ELSE 0 END) AS missingDownloads, SUM(CASE WHEN likes IS NULL THEN 1 ELSE 0 END) AS missingLikes FROM models;"""
    result = conn.execute(command).fetchone()
    conn.close()
    total=result[0]
    libraryName=result[1]
    pipelineTag=result[2]
    careatedAt=result[3]
    lastmodified=result[4]
    downloads=result[5]
    likes=result[6]
    print(f"missing libraryName: {libraryName} / {total} or ({(libraryName/total)*100:.2f}%)")
    print(f"missing pipelineTag: {pipelineTag} / {total} or ({(pipelineTag/total)*100:.2f}%)")
    print(f"missing createdAt: {careatedAt} / {total} or ({(careatedAt/total)*100:.2f}%)")
    print(f"missing lastModified: {lastmodified} / {total} or ({(lastmodified/total)*100:.2f}%)")
    print(f"missing downloads: {downloads} / {total} or ({(downloads/total)*100:.2f}%)")
    print(f"missing likes: {likes} / {total} or ({(likes/total)*100:.2f}%)")    
    return result

def runTagAnalysis():
    print("Most common tags:")
    for tag, count in getTagCounts(20):
        print(f"{tag}: {count}")

    print("Top downloaded tags:")
    for tag, downloads in getTopDownloadedTags(20):
        print(f"{tag}: {downloads}")

def buildModelDatasetNoLikes():
    df = getDF("SELECT id, downloads, libraryName, pipelineTag, createdAt, likes FROM models WHERE downloads IS NOT NULL AND libraryName IS NOT NULL AND createdAt IS NOT NULL")
    # add age
    df["year"] = pd.to_datetime(df["createdAt"]).dt.year
    df["age"] = 2026 - df["year"]
    return df

def trainModel(x, y):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=35)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(xTrain, yTrain)
    preds = model.predict(xTest)
    accuracy = accuracy_score(yTest, preds)
    print(f"Model Accuracy: {accuracy:.4f}")
    return model, x

def prepareFeaturesMedianPopularity(df):
    df = df.copy()
        #clip for no divide by zero
    
    df["age_days"] = df["age"].clip(lower=1)
    df["downloads_per_day"] = df["downloads"] / df["age_days"]
    df["LikesPerDay"] = df["likes"] / df["age_days"].clip(lower=1)
    median_val = df["downloads_per_day"].median()
    y = (df["downloads_per_day"] > median_val).astype(int)
    x = df[["libraryName", "pipelineTag", "LikesPerDay"]]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)
    numPipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),("scaler", StandardScaler())])
    catPipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[("num", numPipe, ["LikesPerDay"]),("cat", catPipe, ["libraryName", "pipelineTag"])])
    model = Pipeline(steps=[("preprocessor", preprocessor),("model", LogisticRegression(max_iter=1000))])

    return xTrain, xTest, yTrain, yTest, model

def runPredictionModelsNoLikes():
    print("Running prediction models.")
    df = buildModelDatasetNoLikes()

    xTrain, xTest, yTrain, yTest, logModelPipe = prepareFeaturesMedianPopularity(df)

    logModelPipe.fit(xTrain, yTrain)
    logPreds = logModelPipe.predict(xTest)
    logAcc = accuracy_score(yTest, logPreds)

    print("logistic regression accuracy:")
    print(f"Accuracy: {logAcc:.4f}")

    rfModelPipe = Pipeline(steps=[("preprocessor", logModelPipe.named_steps["preprocessor"]),("model", RandomForestClassifier(n_estimators=1000))])
    rfModelPipe.fit(xTrain, yTrain)
    rfPreds = rfModelPipe.predict(xTest)
    rfAcc = accuracy_score(yTest, rfPreds)

    print("Random Forest accuracy:")
    print(f"Accuracy: {rfAcc:.4f}")
    getFeatureImportance(rfModelPipe)
    plotConfusionMatrix(rfModelPipe, xTest, yTest)
    
def plotLogisticCoefficients(model, X, topN=10):
    #pulls coefficients
    coefs = model.coef_[0]
    feature_names = X.columns

    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_coef"] = df["coefficient"].abs()
    df = df.sort_values(by="abs_coef", ascending=False).head(topN)
    plt.figure(figsize=(10,6))
    plt.barh(df["feature"], df["coefficient"])
    plt.title("Logistic Regression Feature Impact")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("logistic_coefficients.png")

def getFeatureImportance(modelPipe):
    preprocessor = modelPipe.named_steps["preprocessor"]
    model = modelPipe.named_steps["model"]

    featureNames = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": featureNames, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(15)
    print(df)

def plotConfusionMatrix(model, xTest, yTest):
    preds = model.predict(xTest)
    cm = confusion_matrix(yTest, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusionMatrix.png")


def runFullAnalysis():
    print(percentDownloadsByTopXPercent(0.1))
    runBasicStats()
    runTopModels()
    runYearAnalysis()
    runAuthorStats()
    runPipelineAnalysis()
    runLibraryStats()
    runTagAnalysis()
    nullSummary()
