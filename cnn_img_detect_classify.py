############################################################################################
# This Python code is designed to use CNN to have the optimum architecture for predicting  #
# lung condition as being normal or having pneumonia.                                      #
# The lung images are JPG, grayscale & are distributed as follows:                         #
# Normal = 1,576 images                                                                    #
# Bacterial Pneumonia = 1,593                                                              #
# Viral Pneumonia = 999                                                                    #
# Total images = 4,168                                                                     #
#                                                                                          #
# Possible predictions:                                                                    #
# 1. Normal                                                                                #
# 2. Bacterial Pneumonia                                                                   #
# 3. Viral Pneumonia                                                                       #
#                                                                                          #
# Version -                                                                                #
# Date: Jan 20, 2020                                                                       #
# Author: Francis Bello                                                                    #
############################################################################################


# Import libraries
from datetime import datetime
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os # For local file system manipulation
import cv2 # For image operations
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss, recall_score, precision_score, \
    roc_auc_score, roc_curve, auc
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # used for CNN GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize
from tensorflow.keras.regularizers import l1, l2
from tensorflow.python.keras import backend as k


"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
"""



##########################################################################################
# Variables declaration
##########################################################################################
DATADIR = r"C:\Queens\MMAI894\720"
SUBDIRS = ["NORMAL", "BACTERIA", "VIRUS"]
IMG_SIZE = 150 # size in pixels
Upsample_Count = 3000 # this is the up-sample count of the images per category
Training_Data = []
Error_Files = []
X = []
y = []

hidden_layers = [1, 3, 5] # number of hidden layers
batch_size = [50, 100, 150, 200]
epochs = [5,10,15,20]
optimizer = ['Adam']
learn_rate = [0.1, 0.01, 0.001]
init_mode = ['random_uniform']
activation = ['relu']
dropout_rate = [0.0, 0.1, 0.25]
input_layer = [64, 128, 256] # neuron count in input layer
hidden_neurons = [64, 128, 256] # neuron count in a hidden layer
dense_output_layer = [500, 1000, 1500, 2000] # neuron count on output layer
input_kernel_size = [9] # kernel size in input layer
input_strides = [(2, 2), (3, 3)] # number of strides in input layer
input_pool_size = [(2, 2), (3, 3)] # pool size in input layer
hidden_kernel_size = [1, 3, 5] # kernel size in hidden layer
hidden_strides = [(2, 2), (3, 3)] # strides in hidden layer
hidden_pool_size = [(2, 2), (3, 3)] # pool size in hidden layer

l1l2_switch = ['none', 'l1', 'l2']
l1_value = [0.1, 0.01, 0.001]
l2_value = [0.1, 0.01, 0.001]

if k.image_data_format() == 'channels_first':
    input_shape = (1, IMG_SIZE, IMG_SIZE)
else:
    input_shape = (IMG_SIZE, IMG_SIZE, 1)


##########################################################################################
# Methods definitions

##########################################################################################
# Method to randomly generate files from real files using ImageDataGenerator
def generate_random_imges(upsample_count):
    datagen = ImageDataGenerator(
            rotation_range=10,
            shear_range=0.75,
            zoom_range=0.1,
            horizontal_flip=True
    )

    for subdir in SUBDIRS:
        path = os.path.join(DATADIR, subdir)
        files = os.listdir(path)
        num_files_in_dir = len(files)

        if upsample_count<=num_files_in_dir: break

        for ctr in range(upsample_count-num_files_in_dir):
            # get a random file
            index = random.randrange(0, len(files))
            filename = files[index]
            fullfilepath = os.path.join(path, filename)

            # generate random files up to File_to_Generate_Count
            img = load_img(fullfilepath)
            file_to_generate = img_to_array(img)
            file_to_generate = file_to_generate.reshape((1,) + file_to_generate.shape)

            for batch in datagen.flow(file_to_generate,
                                      batch_size=1,
                                      save_to_dir=path,
                                      save_prefix='generated',
                                      save_format='jpeg'):
                break


##########################################################################################
# Method to populate  Training_Data, assigning each image according to it's classification
def create_training_data():
    for subdir in SUBDIRS:
        path = os.path.join(DATADIR, subdir)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                classifier_num = SUBDIRS.index(subdir)
                Training_Data.append([img_resized, classifier_num])
            except Exception as e:
                # Accumulate unreadable files, force loop to continue
                Error_Files.append(img)
                pass


##########################################################################################
# Returns the features & labels from a list
def create_features_labels(list_array):
    features_array = []
    label_array = []
    for features, label in list_array:
        features_array.append(features)
        label_array.append(label)
    return features_array, label_array

##########################################################################################
# This method is used for CNN hyperparameter tuning
def create_model(optimizer='adam', init_mode='uniform', activation='relu', dropout_rate=0.0, hidden_neurons=1,
                 learn_rate=0.01, hidden_layers=1, dense_output_layer=1, input_layer=1, input_kernel_size=9,
                 input_strides=(1, 1), input_pool_size=(1, 1), hidden_kernel_size=1, hidden_strides=(1, 1),
                 hidden_pool_size=(1, 1), l1l2_switch='none', l1_value=0.0, l2_value=0.0):
    s_model = Sequential()

    # create the input layer
    if l1l2_switch == 'none':
        s_model.add(Conv2D(filters=input_layer,
                             kernel_size=input_kernel_size,
                             strides=input_strides,
                             activation=activation,
                             kernel_initializer=init_mode,
                             padding='same',
                             input_shape=input_shape))
    elif l1l2_switch == 'l1':
        s_model.add(Conv2D(filters=input_layer,
                             kernel_size=input_kernel_size,
                             strides=input_strides,
                             activation=activation,
                             kernel_initializer=init_mode,
                             kernel_regularizer=l1(l1_value),
                             padding='same',
                             input_shape=input_shape))
    elif l1l2_switch == 'l2':
        s_model.add(Conv2D(filters=input_layer,
                             kernel_size=input_kernel_size,
                             strides=input_strides,
                             activation=activation,
                             kernel_initializer=init_mode,
                             kernel_regularizer=l2(l2_value),
                             padding='same',
                             input_shape=input_shape))
    s_model.add(MaxPooling2D(pool_size=input_pool_size, padding='same'))
    s_model.add(Dropout(dropout_rate))

    # dynamically add hidden layers
    for i in range(hidden_layers):
        if l1l2_switch == 'none':
            s_model.add(Conv2D(filters=hidden_neurons,
                               kernel_size=hidden_kernel_size,
                               strides=hidden_strides,
                               activation=activation,
                               kernel_initializer=init_mode,
                               padding='same',
                               input_shape=input_shape))
        elif l1l2_switch == 'l1':
            s_model.add(Conv2D(filters=input_layer,
                               kernel_size=hidden_kernel_size,
                               strides=hidden_strides,
                               activation=activation,
                               kernel_initializer=init_mode,
                               kernel_regularizer=l1(l1_value),
                               padding='same',
                               input_shape=input_shape))
        elif l1l2_switch == 'l2':
            s_model.add(Conv2D(filters=input_layer,
                               kernel_size=hidden_kernel_size,
                               strides=hidden_strides,
                               activation=activation,
                               kernel_initializer=init_mode,
                               kernel_regularizer=l2(l2_value),
                               padding='same',
                               input_shape=input_shape))
        s_model.add(MaxPooling2D(pool_size=hidden_pool_size, padding='same'))
        s_model.add(Dropout(dropout_rate))

    # create the output layer
    s_model.add(Flatten())
    s_model.add(Dense(dense_output_layer, activation=activation))

    s_model.add(Dense(3, activation='softmax'))  # keep activation as is, there are 3 choices to choose from

    s_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    #s_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    s_model.optimizer.lr = learn_rate

    return s_model

##########################################################################################
# Perform GridSearchCV on CNN to find best parameters combination
def check_best_parameters(s_X_train, s_y_train, s_batch_size, s_epochs, s_optimizer, s_init_mode, s_activation,
                          s_dropout_rate, s_hidden_neurons, s_learn_rate, s_hidden_layers, s_dense_output_layer,
                          s_input_layer, s_input_kernel_size, s_input_strides, s_input_pool_size, s_hidden_kernel_size,
                          s_hidden_strides, s_hidden_pool_size, s_l1l2_switch, s_l1_val, s_l2_val):
    model = KerasClassifier(build_fn=create_model)

    param_grid = dict(batch_size=s_batch_size, epochs=s_epochs, optimizer=s_optimizer,
                      init_mode=s_init_mode, activation=s_activation, dropout_rate=s_dropout_rate,
                      hidden_neurons=s_hidden_neurons, learn_rate=s_learn_rate, hidden_layers=s_hidden_layers,
                      dense_output_layer=s_dense_output_layer, input_layer=s_input_layer,
                      input_kernel_size=s_input_kernel_size, input_strides=s_input_strides,
                      input_pool_size=s_input_pool_size, hidden_kernel_size=s_hidden_kernel_size,
                      hidden_strides=s_hidden_strides, hidden_pool_size=s_hidden_pool_size, l1l2_switch=s_l1l2_switch,
                      l1_value=s_l1_val, l2_value=s_l2_val)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=0)

    s_grid_result = grid.fit(s_X_train, s_y_train)

    return s_grid_result

##########################################################################################
# Display the different combinations of hyperparameters and respective score
def list_hyperparameters_score(s_grid_result):
    means = s_grid_result.cv_results_['mean_test_score']
    stddevs = s_grid_result.cv_results_['std_test_score']
    params = s_grid_result.cv_results_['params']
    for mean, stddev, param in zip(means, stddevs, params):
        print("%f (%f) with: %r" % (mean, stddev, param))

##########################################################################################
# Get the best hyperparameter values from grid_result
def get_best_params(s_best_params):
    s_best_batch_size = s_best_params['batch_size']
    s_best_epochs = s_best_params['epochs']
    s_best_optimizer = s_best_params['optimizer']
    s_best_learn_rate = s_best_params['learn_rate']
    s_best_init_mode = s_best_params['init_mode']
    s_best_activation = s_best_params['activation']
    s_best_dropout_rate = s_best_params['dropout_rate']
    s_best_input_layer = s_best_params['input_layer']
    s_best_hidden_neurons = s_best_params['hidden_neurons']
    s_best_dense_output_layer = s_best_params['dense_output_layer']
    s_best_hidden_layers = s_best_params['hidden_layers']
    s_best_input_kernel_size = s_best_params['input_kernel_size']
    s_best_input_strides = s_best_params['input_strides']
    s_best_input_pool_size = s_best_params['input_pool_size']
    s_best_hidden_kernel_size = s_best_params['hidden_kernel_size']
    s_best_hidden_strides = s_best_params['hidden_strides']
    s_best_hidden_pool_size = s_best_params['hidden_pool_size']
    s_l1l2_switch = s_best_params['l1l2_switch']
    s_l1_val = s_best_params['l1_value']
    s_l2_val = s_best_params['l2_value']

    return s_best_batch_size, s_best_epochs, s_best_optimizer, s_best_learn_rate, s_best_init_mode, s_best_activation, \
            s_best_dropout_rate, s_best_input_layer, s_best_hidden_neurons, s_best_dense_output_layer, \
            s_best_hidden_layers, s_best_input_kernel_size, s_best_input_strides, s_best_input_pool_size,\
            s_best_hidden_kernel_size, s_best_hidden_strides, s_best_hidden_pool_size, s_l1l2_switch, s_l1_val, s_l2_val

##########################################################################################
# Plot AUC curve
def plot_roc(clf, s_X_test, s_y_test, name, ax, show_threshold=False):
    y_pred_a = clf.predict_proba(s_X_test)[:, 1]
    fpr, tpr, thr = roc_curve(s_y_test, y_pred_a)

    ax.plot([0, 1], [0, 1])
    ax.plot(fpr, tpr, label='{}, AUC={:.5f}'.format(name, auc(fpr, tpr)));

    if show_threshold:
        for i, th in enumerate(thr):
            ax.text(x=fpr[i], y=tpr[i], s="{:.2f}".format(th), fontsize=9, horizontalalignment='left',
                    verticalalignment='top', color='black',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.1', alpha=0.1));

    ax.set_xlabel('False positive rate', fontsize=18);
    ax.set_ylabel('True positive rate', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=18);
    ax.grid(True);
    ax.set_title('ROC Curve', fontsize=18)





Starttime = datetime.now()


print("k.image_data_format() = ", k.image_data_format())

##########################################################################################
# to disable: "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


##########################################################################################
# Create augmented files
print("Performing data augmentation on images.")
print()
generate_random_imges(Upsample_Count)



##########################################################################################
# Create the training data
print("Reading the files.")
print()
create_training_data()



##########################################################################################
# Display how many files failed during reading
print("Error files count: ", len(Error_Files))
print()



##########################################################################################
# Shuffle the order
print("Perform shuffle.")
print()
random.shuffle(Training_Data)



##########################################################################################
# Split the training data into X & y
print("Separated X & y.")
print()
X, y = create_features_labels(Training_Data)



##########################################################################################
# Convert training sets to numpy arrays
X_reshaped = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)



##########################################################################################
# Scale the training set
X_norm = normalize(X_reshaped, axis=1)

print("X normalized.")
print()



##########################################################################################
# Split into training and test sets; there's not many training data, choosing 80%-20% ratio
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, random_state=42, test_size=0.2, stratify=y)

print("X & y split into X_train, X_test, y_train, y_test.")
print()



##########################################################################################
# Find the optimum combinations of hyperparameters
print("- Check the optimum combinations of hyperparameters -")
print()

grid_result = check_best_parameters(X_train, y_train, batch_size, epochs, optimizer, init_mode, activation,
                                    dropout_rate, hidden_neurons, learn_rate, hidden_layers, dense_output_layer,
                                    input_layer, input_kernel_size, input_strides, input_pool_size, hidden_kernel_size,
                                    hidden_strides, hidden_pool_size, l1l2_switch, l1_value, l2_value)
best_params = grid_result.best_params_


print("- Record count -")
print("X count: ", len(X))
print("y count: ", len(y))
print()

print("- Data distribution -")
print("0: normal | 1: bacterial | 2: viral")
print(sorted(Counter(y).items()))
print()



##########################################################################################
# Get the best hyerparameter values
best_batch_size, best_epochs, best_optimizer, best_learn_rate, best_init_mode, best_activation, best_dropout_rate, \
    best_input_layer, best_hidden_neurons, best_dense_output_layer, best_hidden_layers, best_input_kernel_size, \
    best_input_strides, best_input_pool_size, best_hidden_kernel_size, best_hidden_strides, best_hidden_pool_size, \
    best_l1l2_switch, best_l1_value, best_l2_value = get_best_params(best_params)



##########################################################################################
# Build the CNN model using the best parameters
early_stopping_monitor = EarlyStopping(patience=3, monitor='loss', mode='auto')
weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

cnn_model = create_model(best_optimizer, best_init_mode, best_activation, best_dropout_rate, best_hidden_neurons,
                         best_learn_rate, best_hidden_layers, best_dense_output_layer, best_input_layer,
                         best_input_kernel_size, best_input_strides, best_input_pool_size, best_hidden_kernel_size,
                         best_hidden_strides, best_hidden_pool_size, best_l1l2_switch, best_l1_value, best_l2_value)

#cnn_model.fit(X_train, y_train, epochs=best_epochs, batch_size=best_batch_size, callbacks=[early_stopping_monitor],
#              class_weight=weights, validation_data=(X_test, y_test), verbose=1)
cnn_model.fit(X_train, y_train, epochs=best_epochs, batch_size=best_batch_size, callbacks=[early_stopping_monitor],
              class_weight=weights, verbose=0)

print("- Model summary -")
cnn_model.summary()



##########################################################################################
# Display the different combinations of hyperparameters and respective score
print()
print("Best: %f using %s" % (grid_result.best_score_, best_params))
print()
print("Different combinations:")
list_hyperparameters_score(grid_result)



print()
print("Start: ", Starttime)
print("End: ", datetime.now())



##########################################################################################
# Test prediction
print()
print("Test prediction")

cnn_pred = cnn_model.predict(X_test)



##########################################################################################
# Print prediction performance
print()
print("Prediction performance")

cnn_cm = confusion_matrix(y_test, cnn_pred)
cnn_auc = roc_auc_score(y_test, cnn_pred)

print("Confusion Matrix:")
print(cnn_cm)
print()
print("Classification Report:")
print(metrics.classification_report(y_test, cnn_pred))
print()
print("CNN F1 Score : %5.5f" %(round(f1_score(y_test, cnn_pred), 3)))
print("CNN Accuracy : %5.5f" %(round(accuracy_score(y_test, cnn_pred), 3)))
print("CNN Log Loss : %5.5f" %(round(log_loss(y_test, cnn_pred), 3)))
print("CNN Recall : %5.5f" %(round(recall_score(y_test, cnn_pred), 3)))
print("CNN Precision : %5.5f" %(round(precision_score(y_test, cnn_pred), 3)))
print("CNN AUC : %5.5f" %(cnn_auc))

plt.style.use('default')
figure = plt.figure(figsize=(10, 6))
ax1 = plt.subplot(1, 1, 1)
plot_roc(cnn_model, X_test, y_test, "CNN Test", ax1)
plot_roc(cnn_model, X_train, y_train, "CNN Train", ax1)
plt.legend(loc='lower right', fontsize=18)
plt.tight_layout()
