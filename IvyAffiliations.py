import pandas as pd
#Loading in dataset
df = pd.read_csv("C:/Users/coles/OneDrive/Documents/IvyAffiliationsDataSetBalanced.csv")

#Defining function to preprocessing name strings
import re
def preprocess_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z\s]', '', name)  # Remove special characters
    return name

#Combining names into single variable
df["Processed_First"] = df["First"].apply(preprocess_name)
df["Processed_Last"] = df["Last"].apply(preprocess_name)

df["Full_Name"] = df["Processed_First"].astype(str) + " " + df["Processed_Last"].astype(str)

#Vectorize name & affiliation strings
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Full_Name"])
y = df["Affiliation1"]

#Dividing data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Building / training the RF model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_train, y_train)

#Evaluation phase (highkey does not matter at all...the worse, the better)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
#print("Accuracy:", accuracy_score(y_test, y_pred))

#Defining predictive function (output = affiliation guess)
def predict_affiliation(name):
    name = preprocess_name(name)
    name_vectorized = vectorizer.transform([name])
    prediction = model.predict(name_vectorized)
    return prediction[0]

#Saving the trained model
import joblib
joblib.dump(model, "affiliation_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


