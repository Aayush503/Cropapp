import pickle
import bz2
N=float(("28"))
P=float(("67"))
K=float(("21"))
temp=float(("21.7"))
Hum=float(("63"))
Ph=float(("6"))
Rain=float(("46"))
sfile = bz2.BZ2File('All Model', 'r')
model=pickle.load(sfile)
names = ["K-Nearest Neighbors", "SVM",
         "Decision Tree", "Random Forest",
         "Naive Bayes","ExtraTreesClassifier","VotingClassifier"]
for i in range(len(model)):
    print(names[i])
    test_prediction = model[i].predict([[N,P,K,temp,Hum,Ph,Rain]])
    le=pickle.load(open('le.pkl', 'rb'))
    label=le.inverse_transform(test_prediction)
    print(label[0])