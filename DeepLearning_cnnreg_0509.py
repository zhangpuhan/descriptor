import numpy as np
import torch
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import keras.backend as kb

from sklearn.metrics import mean_absolute_error

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data file

train_X = torch.load('testdata/trainaev_tensor.pt').cpu().numpy()
train_Y = torch.load('testdata/trainforce_tensor.pt').cpu().numpy()

test_X = torch.load('testdata/testaev_tensor.pt').cpu().numpy()
test_Y = torch.load('testdata/testforce_tensor.pt').cpu().numpy()

num_input = train_X.shape[1]
# print(num_input)

num_output = train_Y.shape[1]
# print(num_output)

scalar = StandardScaler()
train_scale_X = scalar.fit_transform(train_X)
test_scale_X = scalar.fit_transform(test_X)
# train_scale_Y = scalar.fit_transform(train_Y)
# test_scale_Y = scalar.fit_transform(test_Y)

train_scale_Y = train_Y
test_scale_Y = test_Y

# create model
# def baseline_model():
model = Sequential()
model.add(Dense(400, input_dim=num_input, activation='elu'))
model.add(Dense(256, activation='elu', kernel_initializer='glorot_uniform'))
model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform'))
model.add(Dense(64, activation='elu', kernel_initializer='glorot_uniform'))
model.add(Dense(3))
# model.add(Dense(num_output, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
kb.set_value(model.optimizer.lr, .001)
print(kb.get_value(model.optimizer.lr))

# early_stop = EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10, verbose=1, mode='auto')

# return model
history = model.fit(train_scale_X, train_scale_Y, validation_split=0.2, epochs=100, batch_size=32,
                    verbose=2)

predicted_m = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]
print("\nm=%.2f b=%.2f\n" % (predicted_m, predicted_b))

# plot metrics
# his_mse = history.history['mean_squared_error']
# his_mae = history.history['mean_absolute_error']
# his_map = history.history['mean_absolute_percentage_error']
# his_cosine = history.history['cosine_proximity']

# np.savetxt('his_mse.out', his_mse)
# np.savetxt('his_mae.out', his_mae)
# np.savetxt('his_map.out', his_map)
# np.savetxt('his_cosine.out', his_cosine)

model.save('my_model_1layerLinear.h5')

# evaluate model
y_pred = model.predict(test_scale_X)

# y_pred = scalar.inverse_transform(y_pred)
# test_inverse_Y = scalar.inverse_transform(test_scale_Y)

print(y_pred)

meanAbsError_test = mean_absolute_error(test_Y, y_pred)

print(meanAbsError_test)

np.savetxt('test.out', y_pred)

y_train_pred = model.predict(train_scale_X)

# y_train_pred = scalar.inverse_transform(y_train_pred)
# train_inverse_Y = scalar.inverse_transform(train_scale_Y)
meanAbsError_train = mean_absolute_error(train_Y, y_train_pred)

print(meanAbsError_train)

print(y_train_pred)
np.savetxt('train.out', y_train_pred)

