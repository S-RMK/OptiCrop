\# Import Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



\# Load Dataset

df = pd.read\_csv("Crop\_recommendation.csv")



\# Display Dataset

print("First 5 Rows:")

print(df.head())



print("\\nLast 5 Rows:")

print(df.tail())



print("\\nRandom 5 Rows:")

print(df.sample(5))



\# Data Functions

print("\\nShape of Dataset:")

print(df.shape)



print("\\nDataset Information:")

print(df.info())



print("\\nCount of Non-Null Values:")

print(df.count())



print("\\nStatistical Summary:")

print(df.describe())



print("\\nColumn Names:")

print(df.columns)



print("\\nData Types:")

print(df.dtypes)



print("\\nMissing Values:")

print(df.isnull().sum())



\# Graphical Representation



\# Count Plot

plt.figure(figsize=(12, 6))

sns.countplot(x='label', data=df)

plt.xticks(rotation=90)

plt.title("Count of Crops")

plt.show()



\# Histograms

df.hist(figsize=(12, 10))

plt.show()



\# Correlation Heatmap

plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(numeric\_only=True),

&#x20;           annot=True,

&#x20;           cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()



\# Box Plot

plt.figure(figsize=(12, 6))

sns.boxplot(data=df)

plt.xticks(rotation=90)

plt.title("Box Plot")

plt.show()



\# Violin Plots

columns = \['N', 'P', 'K', 'temperature',

&#x20;          'humidity', 'ph', 'rainfall']



for col in columns:

&#x20;   plt.figure(figsize=(12, 5))

&#x20;   sns.violinplot(x='label', y=col, data=df)

&#x20;   plt.xticks(rotation=90)

&#x20;   plt.title(f"Violin Plot of {col}")

&#x20;   plt.show()



\# Handling Missing Values

for col in df.select\_dtypes(include=np.number):

&#x20;   df\[col].fillna(df\[col].mean(), inplace=True)



\# Elimination of Outliers using IQR

numeric\_cols = df.select\_dtypes(include=np.number).columns



for col in numeric\_cols:

&#x20;   Q1 = df\[col].quantile(0.25)

&#x20;   Q3 = df\[col].quantile(0.75)

&#x20;   IQR = Q3 - Q1



&#x20;   lower = Q1 - 1.5 \* IQR

&#x20;   upper = Q3 + 1.5 \* IQR



&#x20;   df = df\[(df\[col] >= lower) \&

&#x20;           (df\[col] <= upper)]



print("\\nShape After Removing Outliers:")

print(df.shape)



\# Feature and Target Separation

X = df.drop('label', axis=1)

y = df\['label']



\# Label Encoding

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

y = le.fit\_transform(y)



\# Feature Scaling

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X = scaler.fit\_transform(X)



print("\\nFeatures Shape:", X.shape)

print("Target Shape:", y.shape)





\# Train-Test Split



from sklearn.model\_selection import train\_test\_split



X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&#x20;   X,

&#x20;   y,

&#x20;   test\_size=0.2,

&#x20;   random\_state=42,

&#x20;   stratify=y

)



print("Training Features Shape :", X\_train.shape)

print("Testing Features Shape  :", X\_test.shape)

print("Training Labels Shape   :", y\_train.shape)

print("Testing Labels Shape    :", y\_test.shape)





\# Elbow Method



from sklearn.cluster import KMeans

wcss = \[]



for i in range(1,11):

&#x20;   kmeans = KMeans(

&#x20;       n\_clusters=i,

&#x20;       random\_state=42,

&#x20;       n\_init=10

&#x20;   )

&#x20;   kmeans.fit(X)

&#x20;   wcss.append(kmeans.inertia\_)



plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.show()





\# k-means model



kmeans = KMeans(

&#x20;   n\_clusters=5,

&#x20;   random\_state=42,

&#x20;   n\_init=10

)



clusters = kmeans.fit\_predict(X)



print("Cluster Labels")

print(clusters)





\#kmeans clustering visualization



from sklearn.decomposition import PCA



pca = PCA(n\_components=2)



X\_pca = pca.fit\_transform(X)



plt.figure(figsize=(8,6))

plt.scatter(

&#x20;   X\_pca\[:,0],

&#x20;   X\_pca\[:,1],

&#x20;   c=clusters,

&#x20;   cmap='viridis'

)



plt.title("KMeans Clustering")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(label="Cluster")

plt.show()



\#logistic regression



from sklearn.linear\_model import LogisticRegression



model = LogisticRegression(

&#x20;   max\_iter=1000,

&#x20;   random\_state=42

)



model.fit(X\_train, y\_train)



y\_pred = model.predict(X\_test)



print("Predicted Labels")

print(y\_pred)



\#Evaluating Model Performance



from sklearn.metrics import accuracy\_score

from sklearn.metrics import confusion\_matrix

from sklearn.metrics import classification\_report

from sklearn.metrics import precision\_score

from sklearn.metrics import recall\_score

from sklearn.metrics import f1\_score



accuracy = accuracy\_score(y\_test, y\_pred)



precision = precision\_score(

&#x20;   y\_test,

&#x20;   y\_pred,

&#x20;   average='weighted'

)



recall = recall\_score(

&#x20;   y\_test,

&#x20;   y\_pred,

&#x20;   average='weighted'

)



f1 = f1\_score(

&#x20;   y\_test,

&#x20;   y\_pred,

&#x20;   average='weighted'

)



print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)



\#confusion matrix



plt.figure(figsize=(10,8))



cm = confusion\_matrix(y\_test, y\_pred)



sns.heatmap(

&#x20;   cm,

&#x20;   annot=True,

&#x20;   fmt='d',

&#x20;   cmap='Blues'

)



plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()



\#Classification report



print("Classification Report")



print(classification\_report(

&#x20;   y\_test,

&#x20;   y\_pred,

&#x20;   target\_names=le.classes\_

))







#### \# Save the trained Logistic Regression model



import pickle

with open("best\_crop\_model.pkl", "wb") as file:

&#x20;   pickle.dump(model, file)



print("Model saved successfully as best\_crop\_model.pkl")





with open("scaler.pkl", "wb") as file:

&#x20;   pickle.dump(scaler, file)



print("Scaler saved successfully as scaler.pkl")

