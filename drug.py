import azureml.core
from azureml.core import Workspace
from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration
import json
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.conda_dependencies import CondaDependencies
#from sklearn.datasets import load_diabetes
from azureml.core.image import ContainerImage
#from sklearn.linear_model import Ridge
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

from flask import Flask

from flask import request, jsonify

import datetime
import joblib
import pickle
import pandas as pan
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

ws = Workspace.from_config()
exp = Experiment(workspace = ws, name = 'Drug_pred')
run = exp.start_logging()
run.log("Starting time", str(datetime.datetime.now()))

dataframe_1 = pan.read_csv('https://raw.githubusercontent.com/deep-matrix5229582/masters_project-v2/caf5893f93bbaf0ee4af01131988bef72c31f173/drug200.csv')
dataframe_1.head()
le = LabelEncoder()


ctf = [fe for fe in dataframe_1.columns if dataframe_1[fe].dtypes == 'O']
for fe in ctf:
    dataframe_1[fe]=le.fit_transform(dataframe_1[fe])
var_1 = dataframe_1.drop("Drug", axis = 1)
var_2 = dataframe_1["Drug"]

ml_mod = DecisionTreeClassifier(criterion = "entropy")
ml_mod.fit(var_1, var_2)
k_f = KFold(random_state = 42, shuffle= True)
val_res = cross_val_score(ml_mod, var_1, var_2, cv=k_f, scoring = "accuracy")

"""pickle_file = open('outputs/drug_mod.pkl', 'ab')
pickle.dump(ml_mod, pickle_file)                     
pickle_file.close()"""

filename = 'outputs/drug_mod.pkl'
joblib.dump(ml_mod, filename)
app = Flask(__name__)

gender_map = {"F": 0, "M": 1}
bp_map = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
cholestol_map = {"HIGH": 0, "NORMAL": 1}
drug_map = {0: "DrugY", 3: "drugC", 4: "drugX", 1: "drugA", 2: "drugB"}

def predict_drug(Age, 
                 Sex, 
                 BP, 
                 Cholesterol, 
                 Na_to_K):

    # 1. Read the machine learning model from its saved state ...
    #pickle_file = open('outputs/drug_mod.pkl', 'rb')     
    model = joblib.load('outputs/drug_mod.pkl')
    
    # 2. Transform the "raw data" passed into the function to the encoded / numerical values using the maps / dictionaries
    Sex = gender_map[Sex]
    BP = bp_map[BP]
    Cholesterol = cholestol_map[Cholesterol]

    # 3. Make an individual prediction for this set of data
    y_predict = model.predict([[Age, Sex, BP, Cholesterol, Na_to_K]])[0]

    # 4. Return the "raw" version of the prediction i.e. the actual name of the drug rather than the numerical encoded version
    return drug_map[y_predict] 
#predict_drug(47, "F", "LOW",  "HIGH", 14)

#predict_drug(60, "F", "LOW",  "HIGH", 20)
@app.route("/")
def hello():
    return "A test web service for accessing a machine learning model to make drug recommendations v2."

@app.route('/drug', methods=['GET'])
def api_all():
#    return jsonify(data_science_books)

    Age = int(request.args['Age'])
    Sex = request.args['Sex']
    BP = request.args['BP']
    Cholesterol = request.args['Cholesterol']
    Na_to_K = float(request.args['Na_to_K'])

    drug = predict_drug(Age, Sex, BP, Cholesterol, Na_to_K)

    #return(jsonify(drug))
    return(jsonify(recommended_drug = drug))
  
  
