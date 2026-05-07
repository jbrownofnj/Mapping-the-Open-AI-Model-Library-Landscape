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
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from render import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from collections import Counter

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
    print(f"\nTHE FIRST {limit} ROWS OF THE {tableName.upper()} TABLE:")
    for result in tableResults:
        print(result)

    connection.close()

def tableSummary():
    conn = sqlite3.connect(dbPath)
    tableNames = ["models", "tags", "model_tags", "fullJSON"]
    for tableName in tableNames:
        print(f"\nINFO FOR THE {tableName.upper()} TABLE:")
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
    print(f"{tableName.upper()} TABLE SIZE IS: {tableSize[0][0]} ROWS")
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
    command = f"SELECT id FROM models ORDER BY downloads DESC LIMIT {limit};"
    cursor = conn.execute(command)
    results = cursor.fetchall()
    conn.close()
    return results

def topModelsByLikes(limit):
    conn = sqlite3.connect(dbPath)
    command = f"SELECT id FROM models ORDER BY likes DESC LIMIT {limit};"
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
        print(f"\nTHE TOP MODEL IN THE YEAR {year} WAS:")
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
    print("\nTHESE ARE THE SIZES OF EACH TABLE:")
    printTableSize("fullJSON")
    printTableSize("models")
    printTableSize("tags")
    printTableSize("model_tags")
    print("\nHere you can see a summary of each tables columns and their types:")
    tableSummary()

def runTopModels():
    print("\nTHESE ARE THE MODELS WITH THE MOST DOWNLOADS OVERALL:")
    results = topModelsByDownloads(10)
    for model in results:
        print(model)

    print("\nTHESES ARE THE MODELS WITH THE MOST LIKES OVERALL:")
    results = topModelsByLikes(10)
    for model in results:
        print(model)

def runYearAnalysis():
    print("\nTOP MODELS IN EACH YEAR:")
    topModelInEachYear()

def runAuthorStats():
    print("\nAUTHOR LEADERBOARD:")
    for author in authorLeaderboard(10):
        print(author)

def runPipelineAnalysis():
    print("\nTOP MODELS IN EACH PIPELINE TAG:")
    results = topModelInEachPipelineTag()
    for pipelineTag, (model_id, downloads) in results.items():
        print(f"{pipelineTag}: {model_id} ({downloads})")

def runLibraryStats():
    print("\nLIKES/DOWNLOADS RATIO BY LIBRARY:")
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
    print("\n AS YOU CAN SEE IN THE TABLE BELOW LIBRARY NAME AND PIPELINE TAG ARE MISSING ALOT OF ENTRIES.")
    print("\n NULL VALUE SUMMARY FOR MODELS TABLE:")
    print(f"MISSING libraryName: {libraryName} / {total} or ({(libraryName/total)*100:.2f}%)")
    print(f"MISSING pipelineTag: {pipelineTag} / {total} or ({(pipelineTag/total)*100:.2f}%)")
    print(f"MISSING createdAt: {careatedAt} / {total} or ({(careatedAt/total)*100:.2f}%)")
    print(f"MISSING lastModified: {lastmodified} / {total} or ({(lastmodified/total)*100:.2f}%)")
    print(f"MISSING downloads: {downloads} / {total} or ({(downloads/total)*100:.2f}%)")
    print(f"MISSING likes: {likes} / {total} or ({(likes/total)*100:.2f}%)")    

    return result

def runTagAnalysis():
    print("\nMOST COMMON TAGS:")
    for tag, count in getTagCounts(20):
        print(f"{tag}: {count}")

    print("\nTOP DOWNLOADED TAGS:")
    for tag, downloads in getTopDownloadedTags(20):
        print(f"{tag}: {downloads}")

def buildModelDatasetNoLikes():
    df = getDF("""
        SELECT 
            models.id,
            models.downloads,
            models.libraryName,
            models.pipelineTag,
            models.createdAt,
            tags.tag AS tagName
        FROM models
        LEFT JOIN model_tags 
            ON models.id = model_tags.modelId
        LEFT JOIN tags 
            ON model_tags.tag = tags.tag
        WHERE models.downloads IS NOT NULL
            AND models.libraryName IS NOT NULL
            AND models.createdAt IS NOT NULL;
    """)

    df = df.groupby(
        ["id", "downloads", "libraryName", "pipelineTag", "createdAt"],
        dropna=False
    )["tagName"].apply(
        lambda tags: [tag for tag in tags if pd.notna(tag)]
    ).reset_index()

    df["year"] = pd.to_datetime(df["createdAt"]).dt.year
    df["age"] = 2026 - df["year"]

    return df

def trainModel(x, y):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=35)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(xTrain, yTrain)
    preds = model.predict(xTest)
    accuracy = accuracy_score(yTest, preds)
    print(f"\n RF MODEL ACCURACY: {accuracy:.4f}")
    return model, x

def prepareFeaturesMedianPopularity(df):
    df = df.copy()

    df["age_days"] = df["age"].clip(lower=1)
    df["downloads_per_day"] = df["downloads"] / df["age_days"]

    median_val = df["downloads_per_day"].median()
    y = (df["downloads_per_day"] > median_val).astype(int)

    # Keep only top 50 tags so the model does not explode into thousands of columns
    allTags = [tag for tagList in df["tagName"] for tag in tagList]
    topTags = set([tag for tag, count in Counter(allTags).most_common(50)])

    df["tagName"] = df["tagName"].apply(
        lambda tagList: [tag for tag in tagList if tag in topTags]
    )

    mlb = MultiLabelBinarizer()
    tagEncoded = mlb.fit_transform(df["tagName"])

    tagDF = pd.DataFrame(
        tagEncoded,
        columns=["tag_" + tag for tag in mlb.classes_],
        index=df.index
    )

    x = pd.concat(
        [df[["libraryName", "pipelineTag"]], tagDF],
        axis=1
    )

    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    catPipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    tagColumns = list(tagDF.columns)

    preprocessor = ColumnTransformer(transformers=[
        ("cat", catPipe, ["libraryName", "pipelineTag"]),
        ("tags", "passthrough", tagColumns)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    return xTrain, xTest, yTrain, yTest, model

def runPredictionModelsNoLikes():
    print("RUNNING PREDICTION MODELS.")
    df = buildModelDatasetNoLikes()

    xTrain, xTest, yTrain, yTest, logModelPipe = prepareFeaturesMedianPopularity(df)

    logModelPipe.fit(xTrain, yTrain)
    logPreds = logModelPipe.predict(xTest)
    logAcc = accuracy_score(yTest, logPreds)
    
    print("\nTHE ACCURACY OF THE LOGISTIC REGRESSION MODEL IS:")
    print(f"ACCURACY: {logAcc:.4f}\n")

    rfModelPipe = Pipeline(steps=[("preprocessor", logModelPipe.named_steps["preprocessor"]),("model", RandomForestClassifier(n_estimators=10))])
    rfModelPipe.fit(xTrain, yTrain)
    rfPreds = rfModelPipe.predict(xTest)
    rfAcc = accuracy_score(yTest, rfPreds)
    plotLogisticCoefficients(logModelPipe)
    
    print("THE ACCURACY OF THE RANDOM FOREST MODEL IS:")
    print(f"ACCURACY: {rfAcc:.4f}\n")

    print("\nTHESE ARE THE TOP RANDOM FOREST FEATURES:")
    getFeatureImportance(rfModelPipe)
    plotConfusionMatrix(rfModelPipe, xTest, yTest)
    
def plotLogisticCoefficients(modelPipe, topN=10):
    preprocessor = modelPipe.named_steps["preprocessor"]
    model = modelPipe.named_steps["model"]

    featureNames = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]

    df = pd.DataFrame({
        "feature": featureNames,
        "coefficient": coefs
    })

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
    
