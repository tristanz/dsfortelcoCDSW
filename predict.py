import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))

def predict(args):
  account=np.array(args["features"])
  return {"result" : model.predict(account)[0]}

data = [[0.0, 1.0, 0.0, 74.0, 29.78, 79.0, 12.89, 109.0, 10.37, 3.0, 1.43, 1.0]]
predict(dict(features = data))