
from dataset import preprocess_data
from train import train_model
from evaluate import evaluate_model



X_train, X_val, X_test, Y_train, Y_val, Y_test = preprocess_data()
model = train_model(X_train, Y_train, X_val, Y_val)
evaluate_model(model, X_test, Y_test)


import joblib 
joblib.dump(model , 'MRI_MODEL.pkl')