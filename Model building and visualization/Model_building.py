# model.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import gc
import os
import glob
import shutil
import time
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import metrics
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import load_model

def read_and_prepare_data(validation_mode, k=5, augment=True):

    if validation_mode == 'k_fold':

        # Read in data from files
        images = np.load('images.npy')
        labels = np.load('labels.npy')

        # Split the image and label datasets into k number of subsets
        folded_images = []
        folded_labels = []
        for i in range(k):
            start = int((i / k) * len(images))
            end = int(((i + 1) / k) * len(images))
            folded_images.append(images[start:end])
            folded_labels.append(labels[start:end])

        # Generate augmented images for each fold
        folded_augmented_images = []
        folded_augmented_labels = []
        for i in range(k):
            if augment:
                print('\nAugmenting Fold ' + str(i + 1) + ' of ' + str(k))
                augmented_images, augmented_labels = augment_images(folded_images[i], folded_labels[i])
                folded_augmented_images.append(augmented_images)
                folded_augmented_labels.append(augmented_labels)


        # Combine the folds into sets for each iteration of the model and standardize the data
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        for i in range(k):
            train_images.append(np.concatenate(folded_images[:i] + folded_images[(i+1):]))
            train_labels.append(np.concatenate(folded_labels[:i] + folded_labels[(i+1):]))
            if augment:
                train_images[i] = np.concatenate(([train_images[i]] + folded_augmented_images[:i] + folded_augmented_images[(i + 1):]))
                train_labels[i] = np.concatenate(([train_labels[i]] + folded_augmented_labels[:i] + folded_augmented_labels[(i + 1):]))
            test_images.append(folded_images[i])
            test_labels.append(folded_labels[i])
            train_images[i], test_images[i] = standardize_data(train_images[i], test_images[i])

        return train_images, train_labels, test_images, test_labels

def augment_images(images, labels):

    # Create generators to augment images
    from tensorflow.keras.preprocessing import image
    flip_generator = image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True
    )
    rotate_generator = image.ImageDataGenerator(
        rotation_range=360,
        fill_mode='nearest'
    )

    # Accumulate augmented images and labels
    augmented_images = []
    augmented_labels = []

    # Loop each images in the set to augment
    for i in range(len(images)):

        # Reshape image for generator
        image = np.reshape(images[i], (1, images[i].shape[0], images[i].shape[1], 1))
        label = labels[i]

        # Reset the number of augmented images have been created to zero
        num_new_images = 0

        # Generate 2 new images if the image is of a tropical cyclone between 50 and 75 knots
        if 50 < label < 75:
            for batch in flip_generator.flow(image, batch_size=1):
                gc.collect()
                new_image = np.reshape(batch[0], (batch[0].shape[0], batch[0].shape[1], 1))
                augmented_images.append(new_image)
                augmented_labels.append(label)
                num_new_images += 1
                if num_new_images == 2:
                    break

        # Generate 6 new images if the image is of a tropical cyclone between 75 and 100 knots
        elif 75 < label < 100:
            for batch in rotate_generator.flow(image, batch_size=1):
                gc.collect()
                new_image = np.reshape(batch[0], (batch[0].shape[0], batch[0].shape[1], 1))
                augmented_images.append(new_image)
                augmented_labels.append(label)
                num_new_images += 1
                if num_new_images == 6:
                    break

        # Generate 12 new images if the image is of a tropical cyclone greater than or equal to 100 knots
        elif 100 <= label:
            for batch in rotate_generator.flow(image, batch_size=1):
                gc.collect()
                new_image = np.reshape(batch[0], (batch[0].shape[0], batch[0].shape[1], 1))
                augmented_images.append(new_image)
                augmented_labels.append(label)
                num_new_images += 1
                if num_new_images == 12:
                    break

        print_progress('Augmenting Images', i + 1, len(images))

    # Convert lists of images/labels into numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels


def build_model():

    # Build network architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(50, 50, 1)))
    model.add(layers.BatchNormalization(axis=1))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation=None))

    # Configure model optimization
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()]
    )

    return model


def train_model(model, train_images, train_labels, test_images, test_labels, show_performance_by_epoch=False):
    start = time.process_time()
    # Run model and get metrics for each epoch
    performance_log = model.fit(
        train_images,
        train_labels,
        callbacks=[EarlyStopping(monitor='val_mean_absolute_error', patience=5, restore_best_weights=True)],
        epochs=100,
        batch_size=60,
        validation_data=(test_images, test_labels))

    print(time.process_time() - start)

    if show_performance_by_epoch:
        performance_by_epoch(performance_log)

    return model

def performance_by_epoch(performance_log):

    # Get metrics for each epoch after model finishes training
    train_loss = performance_log.history['loss']
    test_loss = performance_log.history['val_loss']
    train_mae = performance_log.history['mean_absolute_error']
    test_mae = performance_log.history['val_mean_absolute_error']
    epochs = range(1, len(train_loss) + 1)

    # Build a dataframe storing epoch metrics
    performance_df = pd.DataFrame(columns=['epoch', 'train_or_test', 'loss_or_mae', 'value'])
    for i in range(len(train_loss)):
        new_row = {'epoch': epochs[i], 'train_or_test': 'train', 'loss_or_mae': 'loss', 'value': train_loss[i]}
        new_row = pd.DataFrame(new_row, index=[0])
        performance_df = pd.concat([performance_df, new_row], ignore_index=True)
        new_row = {'epoch': epochs[i], 'train_or_test': 'test', 'loss_or_mae': 'loss', 'value': test_loss[i]}
        new_row = pd.DataFrame(new_row, index=[0])
        performance_df = pd.concat([performance_df, new_row], ignore_index=True)
        new_row = {'epoch': epochs[i], 'train_or_test': 'train', 'loss_or_mae': 'mae', 'value': train_mae[i]}
        new_row = pd.DataFrame(new_row, index=[0])
        performance_df = pd.concat([performance_df, new_row], ignore_index=True)
        new_row = {'epoch': epochs[i], 'train_or_test': 'test', 'loss_or_mae': 'mae', 'value': test_mae[i]}
        new_row = pd.DataFrame(new_row, index=[0])
        performance_df = pd.concat([performance_df, new_row], ignore_index=True)
    performance_df = performance_df.astype({'epoch': np.int64})
    print(performance_df)

    '''
    # Plot metrics on graph, fitted with exponential decay curves
    lm = sns.lmplot(
        x='epoch',
        y='value',
        data=performance_df,
        row='loss_or_mae',
        hue='train_or_test',  # Note: If epoch = 1, this line causes an error. Make sure epoch >= 2
        logx=True,
        truncate=False,
        sharey=False)
    axes = lm.axes
    max_mae = performance_df.loc[performance_df.loss_or_mae == 'mae']['value'].max()
    min_mae = performance_df.loc[performance_df.loss_or_mae == 'mae']['value'].min()
    axes[1, 0].set_ylim(min_mae - min_mae * 0.2, max_mae + max_mae * 0.2)
    plt.show()
    '''

def generate_predictions(model, test_images, test_labels):

    # Run validation data through model and print mean absolute error
    raw_predictions = model.predict(test_images)
    raw_predictions = raw_predictions.flatten()

    # Build a dataframe storing data for each prediction made by the model
    processed_predictions = pd.DataFrame(columns=['prediction', 'actual', 'abs_error', 'category'])
    for i in range(len(raw_predictions)):
        abs_error = abs(raw_predictions[i] - test_labels[i])
        new_row = {
            'prediction': raw_predictions[i],
            'actual': test_labels[i],
            'abs_error': abs_error,
            'abs_error_squared': abs_error ** 2,
            'category': category_of(test_labels[i])}
        new_row = pd.DataFrame(new_row, index=[0])
        processed_predictions = pd.concat([processed_predictions, new_row], ignore_index=True)
        print_progress('Processing Predictions', i + 1, len(raw_predictions))
        print(f"prediction: {raw_predictions[i]} and diff: {abs_error}")

    return processed_predictions


def show_validation_results(predictions, show_plots=True, print_error=True):

    print('\n\nRESULTS')

    if print_error:
        mae = predictions['abs_error'].mean()
        print('\nMean Absolute Error: ' + str(round(float(mae), 2)) + ' knots')
        rmse = predictions['abs_error_squared'].mean() ** 0.5
        print('Root Mean Square Error: ' + str(round(float(rmse), 2)) + ' knots')

    if show_plots:
        # List of categories in order of ascending strength
        categories = ['T. Depression', 'T. Storm', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']

        # Show bar graph of median absolute error for each category
        plt.figure(figsize=(10, 5), dpi=300)
        sns.barplot(
            x='category',
            y='abs_error',
            data=predictions,
            estimator=np.median,
            order=categories)
        sns.despine()
        plt.xlabel("Hurricane Strength")
        plt.ylabel("Absolute Error")
        plt.title("Median Absolute Error in Neural Network's Predictions By Category")
        plt.savefig('median_abs_error_by_category.png')
        print('Graph of median absolute error by category saved as median_abs_error_by_category.png')
        plt.clf()

        # Show density plot of error for each category
        for category in categories:
            num_samples_tested = len(predictions.loc[predictions.category == category]['abs_error'])
            sns.distplot(
                predictions.loc[predictions.category == category]['abs_error'],
                label=category + ' (' + str(num_samples_tested) + ' samples tested)',
                hist=False,
                kde_kws={"shade": True})
            sns.despine()
        plt.xlabel("Absolute Error")
        plt.title("Distribution of Absolute Error By Category")
        plt.legend()
        plt.xlim(0, None)
        plt.ylim(0, None)
        plt.savefig('error_dist_by_category.png')
        print('Graph of error distribution by category saved as error_dist_by_category.png')


def standardize_data(train_images, test_images):
    train_images[train_images < 0] = 0
    test_images[test_images < 0] = 0
    st_dev = np.std(train_images)
    mean = np.mean(train_images)
    train_images = np.divide(np.subtract(train_images, mean), st_dev)
    test_images = np.divide(np.subtract(test_images, mean), st_dev)
    return train_images, test_images

def standardize_data1(test_images):
    test_images[test_images < 0] = 0
    st_dev = np.std(test_images)
    mean = np.mean(test_images)
    test_images = np.divide(np.subtract(test_images, mean), st_dev)
    return test_images

def print_progress(action, progress, total):
    percent_progress = round((progress / total) * 100, 1)
    print('\r' + action + '... ' + str(percent_progress) + '% (' + str(progress) + ' of ' + str(total) + ')', end='')


def category_of(wind_speed):
    if wind_speed <= 33:
        return 'T. Depression'
    elif wind_speed <= 64:
        return 'T. Storm'
    elif wind_speed <= 83:
        return 'Category 1'
    elif wind_speed <= 95:
        return 'Category 2'
    elif wind_speed <= 113:
        return 'Category 3'
    elif wind_speed <= 134:
        return 'Category 4'
    else:
        return 'Category 5'

def save_model_results(*argv):
    if not os.path.exists('model_results'):
        os.makedirs('model_results')

    file_path = "_"
    file_path = file_path.join(argv)
    file_path = "model_results/" + file_path

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        return

    content = ['MeanAbsoluteError: ' + str(round(float(predictions['abs_error'].mean()), 2)) + 'knots',
               'RootMeanSquareError: ' + str(round(float(predictions['abs_error_squared'].mean() ** 0.5), 2)) + 'knots']

    for pngfile in glob.iglob(os.path.join("", "*.PNG")):
        shutil.copy(pngfile, file_path)

    file = open(f"{file_path}/results.txt", "a+")
    file.writelines(content)
    file.close()

def predict_image():
    '''
    from tensorflow.keras.preprocessing import image

    images = image.load_img("./test_image/image1.jpg", color_mode='grayscale', target_size=(50, 50))
    images = image.img_to_array(images)
    print(images.shape)
    images = np.expand_dims(images, axis=0)
    print(images.shape)
    images = standardize_data1(images)
    print(model.predict(images))
    '''
    model = load_model('Model.h5')
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()]
    )
    images = np.load('images.npy')
    labels = np.load('labels.npy')
    image = []
    for x in range(10):
        image.append(images[x])
        print(f"labels{x}: {labels[x]}")

    image = np.array(image)
    image = standardize_data1(image)
    image = model.predict(image)
    image = image.flatten()

    for i in range(len(image)):
        print(f"images{i}: {image[i]}")


# save database--------------------------------------------------------------------------------

def save_database():
    import psycopg2
    import datetime
    import netCDF4
    #from PIL import Image as im

    model = load_model('Model.h5')
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()]
    )
    intensity_arr = []
    category_arr = []
    label = []

    images = np.load('images.npy')
    labels = np.load('labels.npy')
    labels = labels.astype('float64')

    image = []
    for x in range(len(images)):
        image.append(images[x])
        label.append(labels[x])

    image = np.array(image)
    image = standardize_data1(image)
    intensity = model.predict(image)
    intensity = intensity.flatten()
    intensity = intensity.astype('float64')

    for x in range(len(intensity)):
        intensity_arr.append(intensity[x])

        if labels[x] <= 33:
            category_arr.append(1)
        elif labels[x] <= 64:
            category_arr.append(2)
        elif labels[x] <= 83:
            category_arr.append(3)
        elif labels[x] <= 95:
            category_arr.append(4)
        elif labels[x] <= 113:
            category_arr.append(5)
        elif labels[x] <= 134:
            category_arr.append(6)
        else:
            category_arr.append(7)

    conn = psycopg2.connect(user="postgres",
                            password="root",
                            host="127.0.0.1",
                            port="5432",
                            database="postgre")
    cur = conn.cursor()
    cur.execute("DELETE FROM intensity_app_stormdata;")
    cur.execute("DELETE FROM intensity_app_stormtrack;")

    index = 0
    id = 1
    for filename in os.listdir("Satellite Imagery"):
        storm_name = filename.split(".")[1]
        time = filename.split(".")[5]
        time1 = ""
        for i in time:
            if i != "0":
                time1 = time1 + i
            else:
                continue
        if time1 == "":
            time1 = "0"
        date = filename.split(".")[2] + filename.split(".")[3] + filename.split(".")[4]
        latitude = filename[8:10]
        longitude = filename[10:13]
        date = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(time1))

        cur.execute(f"SELECT * FROM intensity_app_stormdata WHERE storm_name = '{storm_name}';")
        storm_exists = cur.fetchall()
        if storm_exists:
            cur.execute(f"""INSERT INTO intensity_app_stormtrack (intensity, labels, latitude, longitude, date, storm_data_id) VALUES (%s, %s, %s, %s, %s, %s);""", (intensity_arr[index], label[index], latitude, longitude, date, storm_exists[0][0]))
            conn.commit()
        else:
            cur.execute(f"""INSERT INTO intensity_app_stormdata (storm_id, storm_name, category) VALUES (%s, %s, %s);""", (id, storm_name, category_arr[index]))
            conn.commit()
            cur.execute(f"""INSERT INTO intensity_app_stormtrack (intensity, labels, latitude, longitude, date, storm_data_id) VALUES (%s, %s, %s, %s, %s, %s);""", (intensity_arr[index], label[index], latitude, longitude, date, id))
            conn.commit()
            url = f"Satellite Imagery/{filename}"
            nc = netCDF4.Dataset(url)
            '''
            data = im.fromarray(nc.variables['IRWIN'][0])
            data = data.convert("L")
            data.save(f'../media/images{id}.png')
            '''
            plt.imsave(f'../media/images{id}.png', nc.variables['IRWIN'][0], cmap="binary")
            id = id + 1

        index = index + 1

    if (conn):
        cur.close()
        conn.close()
        print("PostgreSQL connection is closed")

if __name__ == "__main__":

    # Specify whether the script should use Keras's ImageDataGenerator to augment the training dataset. Assigning
    # this variable to True will improve accuracy, but will also increase execution time.
    save_database = False
    AUGMENT = True
    save_result = False
    save_model = True

    # Specify how many folds in the k-fold validation process. Can be any integer greater than or equal to 2. Larger
    # integers will increase execution time.
    NUM_FOLDS = 2

    train_images, train_labels, test_images, test_labels = read_and_prepare_data('k_fold', NUM_FOLDS, augment=AUGMENT)
    model = build_model()
    predictions = pd.DataFrame(columns=['prediction', 'actual', 'abs_error', 'category'])
    for i in range(NUM_FOLDS):
        print('\n\nTraining Fold ' + str(i + 1) + ' of ' + str(NUM_FOLDS) + '\n')
        print(test_images[i].shape)
        model = train_model(model, train_images[i], train_labels[i], test_images[i], test_labels[i])
        print(test_images[i].shape)
        kth_fold_predictions = generate_predictions(model, test_images[i], test_labels[i])
        predictions = pd.concat([predictions, kth_fold_predictions], ignore_index=True)
    show_validation_results(predictions)
    if save_result:
        save_model_results("novel")
    if save_model:
        model.save('Model.h5')
        del model
        print('Model Saved!')

# Default hyperparameter values: batch_size=64, epoch=100, optimiser=rmsprop, loss=mse, Augmentation=True, NUM_FOLDS=5
# Hyperparameter values for model testing: epoch=10, NUM_FOLDS=2, Augmentation=False

#predict_image()

if save_database:

    save_database()