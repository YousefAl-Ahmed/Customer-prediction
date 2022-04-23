import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns


import pandas as pd
import numpy as np
from sklearn import svm


TEST_SIZE = 0.4


def main():
    
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data("shopping.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
 
    predictions = model.predict(X_test)

    sensitivity, specificity = evaluate(y_test, predictions)
    print("Number of customers tested:",len(y_test))

    # Print results
    print_plot("shopping.csv",y_test,"Testing data")
    print_plot("shopping.csv",predictions,"Prediction data")

    print(f"Correct prediction: {(y_test == predictions).sum()}")
    print(f"Incorrect predicion: {(y_test != predictions).sum()}")
    print(f"Customers who bought and were accurately identified (sensitivity): {100 * sensitivity:.2f}%")
    print(f"Customers who did not buy and were accurately identified (specificity): {100 * specificity:.2f}%")

def print_plot(filename,revenue,title):
    df = pd.read_csv(filename, delimiter=',')
    sns.countplot(y=revenue,data=df);
    plt.ylabel("Did/Did not buy");
    plt.title(title)
    plt.show()
    

    
    
#prepare the data
def load_data(filename):

    df = pd.read_csv(filename, delimiter=',')
    
    df["Weekend"] = df["Weekend"].apply(lambda x:1 if x == True else 0)
    df["Revenue"] = df["Revenue"].apply(lambda x:1 if x == True else 0)

    df["VisitorType"] = df["VisitorType"].apply(lambda x:1 if x == "Returning_Visitor" else 0)
    monthMap = {"Jan":0,"Feb":1,"Mar":2,"Apr":3,"May":4,"June":5,"Jul":6,"Aug":7,"Sep":8,"Oct":9,"Nov":10,"Dec":11}
    df["Month"] = df["Month"].map(monthMap)
    #print(df[df['Month'].isna()])
    evidence = df.iloc[:, :-1].values
    labels = df.iloc[:, 17].values

    #display(df.head())

    print_plot("shopping.csv","Revenue","Actual data")
    return (evidence, labels)

   

#prepare the model
def train_model(X_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=1)
    return classifier.fit(X_train, y_train)



#test the results
def evaluate(labels, predictions): 
    confusionMatrix = confusion_matrix(labels, predictions)
    accuracy = metrics.accuracy_score(labels, predictions)
    print("Accuracy =",round(accuracy,4)*100,"%")

    TP = confusionMatrix[1][1]
    TN = confusionMatrix[0][0]
    FP = confusionMatrix[0][1]
    FN = confusionMatrix[1][0]
    
    sensitivity = (TP / float(TP + FN))
    specificity = (TN / float(TN + FP))
    
    return (sensitivity,specificity)







if __name__ == "__main__":
     main()

