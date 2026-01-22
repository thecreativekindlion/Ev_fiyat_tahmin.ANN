# Ev_fiyat_tahmin.ANN
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

data = fetch_california_housing()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_test = scaler.transform(x_test)

model = Sequential([
    Dense(64,activation ="relu",input_shape=(x_train.shape[1],)),
    Dense(64,activation="relu"),
    Dense(1)
    ])

model.compile(optimizer = "adam",loss = "mse",metrics = ["mae"])

es= EarlyStopping(monitor ="val_loss",patience = 10,restore_best_weights = True)

history = model.fit(
    x_train,y_train,
    validation_split = 0.2,
    epochs = 200,
    batch_size = 32,
    callbacks = [es],
    verbose = 1
    )

pred = model.predict(x_test).flatten()  # Eğitilmis ANN verilerine test verilerini verir ve her ev için tahmini fiyatı hesaplatır.
 

mae = mean_absolute_error(y_test,pred)
rmse = np.sqrt(mean_squared_error(y_test,pred))

print("MAE: ", mae)
print("RMSE: ",rmse)

i = 0
print("Gerçek: ",y_test[i],"Tahmin: ",pred[i])
