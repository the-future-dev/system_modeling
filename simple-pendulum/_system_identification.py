#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from dataset_creation import generate_pendulum_dataset, read_pendulum_data

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def single_dataset_plotting(time, theta, image_name):
    plt.figure(figsize=(10,6))
    plt.plot(time, theta)
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Displacement (radians)')
    plt.title('Simple Pendulum Motion')
    plt.grid(True)
    #plt.text(0.5 * max(time), max(theta), params_str, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"))

    plt.tight_layout()
    plt.savefig(image_name)
    plt.show()


def plot_angular_displacement(time, theta, parameters, save_path='./images/last_simulation.png'):
    """
    Function to plot angular displacement of the pendulum over time.
    """
    plt.figure(figsize=(10,6))
    plt.plot(time, theta)
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Displacement (radians)')
    plt.title('Simple Pendulum Motion')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred, sample_indices, title="Actual vs. Predicted Angular Displacement"):
    """
    Function to plot actual vs. predicted angular displacements for given samples.
    """
    n_samples = len(sample_indices)
    
    fig, axs = plt.subplots(n_samples, figsize=(12, 6 * n_samples))
    
    # If there's only one sample, make axs a list
    if n_samples == 1:
        axs = [axs]
    
    for idx, ax in zip(sample_indices, axs):
        ax.plot(y_test[idx], label="Actual")
        ax.plot(y_pred[idx], label="Predicted")
        ax.set_title(f"{title} - Sample {idx}")
    
    plt.show()


def normalize_data(data):
    """Normalize data to [0, 1] range."""
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data], min_val, max_val

def denormalize_data(normalized_data, min_val, max_val):
    """Denormalize data from [0, 1] range to original range."""
    return [x * (max_val - min_val) + min_val for x in normalized_data]

def normalize_dataset(dataset):
    # Normalize parameters
    params_data = [sample['parameters'] for sample in dataset]
    normalized_params, params_min, params_max = zip(*[normalize_data(params) for params in params_data])

    # Normalize theta_data
    theta_data = [sample['data']['theta_data'] for sample in dataset]
    normalized_theta, theta_min, theta_max = zip(*[normalize_data(theta) for theta in theta_data])

    # Store normalized data back
    for idx, sample in enumerate(dataset):
        sample['parameters'] = normalized_params[idx]
        sample['data']['theta_data'] = normalized_theta[idx]

    # Return normalization info for each dataset for possible denormalization
    normalization_info = {
        'params_min': params_min,
        'params_max': params_max,
        'theta_min': theta_min,
        'theta_max': theta_max
    }

    return dataset, normalization_info


def get_best_model_path(models_dir="models"):
    """
    Get the path of the model with the smallest validation loss.
    
    Assumes model filenames are in the format "weights-{epoch:02d}-{val_loss}.hdf5".
    """

    # List all files in the models directory
    files = os.listdir(models_dir)
    
    # Filter out files that don't match the expected pattern
    valid_files = [f for f in files if f.startswith("weights-") and f.endswith(".hdf5")]
    
    # Extract validation loss from each valid filename and get the filename with the minimum loss
    best_file = min(valid_files, key=lambda x: float(x.split('-')[2].rstrip('.hdf5')))
    print(best_file)

    return os.path.join(models_dir, best_file)


# Generate dataset
n_samples = 30
time_s = 200
dataset = generate_pendulum_dataset(n_samples, time_s)


# In[2]:


dataset = read_pendulum_data(30)
time = dataset[-1]['data']['time_data']
theta = dataset[-1]['data']['theta_data']
#params = dataset[-1]['parameters']
#params_str = "\n".join([f"{key}: {value:.3f}" for key, value in params.items()])

single_dataset_plotting(time, theta, './images/last_simulation.png')


# 

# #### Data Preparation

# In[3]:


normalized_dataset, _ = normalize_dataset(dataset)  # Use your normalization function
time = normalized_dataset[-1]['data']['time_data']
theta = normalized_dataset[-1]['data']['theta_data']
single_dataset_plotting(time, theta, './images/last_simulation_normalized')


# In[4]:


X = [item['parameters'] for item in normalized_dataset] #+ [item['data']['time_data'][-1]]
y = [item['data']['theta_data'] for item in normalized_dataset]

print(f'Normalized parameters (X): {X[0]}')  # Print the first sample
print(f'Normalized theta data (y): {len(y[0])}')  # Print the first sample


# In[5]:


# Splitting data into training, validation, and testing
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# Reshape X_train, X_val, X_test for LSTM
X_train = np.array(X_train).reshape((len(X_train), len(X_train[0]), 1))
X_val = np.array(X_val).reshape((len(X_val), len(X_val[0]), 1))
X_test = np.array(X_test).reshape((len(X_test), len(X_test[0]), 1))

y_train = pad_sequences(y_train, dtype='float32', padding='post')
y_val = pad_sequences(y_val, dtype='float32', padding='post')
y_test = pad_sequences(y_test, dtype='float32', padding='post')

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)


# In[6]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(len(X[0]), 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(len(y[0])))

model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mse'])


# In[7]:


# Step 3: Train and Save Models
checkpoint = ModelCheckpoint(os.path.join('models', "best_model.hdf5"), 
                             save_best_only=True, 
                             monitor='val_loss', 
                             mode='min')

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val), 
          epochs=200, batch_size=1, 
          callbacks=[checkpoint])


# In[8]:


# Step 4: Evaluate and Plot Results
best_model_path = './models/best_model.hdf5' # best_model_path = get_best_model_path()
best_model = load_model(best_model_path)

# Predict
y_pred = best_model.predict(X_test)

# Reshaping y_pred and y_test for the comparison plots
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])


# In[9]:


sample_indices = [1, 2, 3, 4]  # choose any indices you want
#y_pred = np.squeeze(y_pred)
plot_actual_vs_predicted(y_test, y_pred, sample_indices)


# In[12]:


print(f'Shape of y_test: {np.array(y_test).shape}')
print(f'Shape of y_pred: {np.array(y_pred).shape}')

