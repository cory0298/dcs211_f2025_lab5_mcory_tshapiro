import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split  # for Question 7
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


###########################################################################
def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = True)

###########################################################################
def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]: #type: ignore
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the reprsented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple

###########################################################################
def predictiveModel(training_set: np.ndarray, features: np.ndarray) -> int:
    ''' Implements a 1-NN classifier to predict the digit for a given set of features.
    Parameters:
        training_set: a numpy array where each row contains features (pixels)
                      and the last column contains the label (digit)
        features: a 1D numpy array of pixel values for a single digit to classify
    Returns:
        predicted_digit: an integer representing the predicted digit (0-9)
    '''
    min_distance = float('inf')  # start with infinite distance
    predicted_digit = -1 # initialize predicted digit
    
    # Iterate through each row in the training set
    for row in training_set:
        # Extract the features (all columns except the last one, which is the label)
        training_features = row[:-1] # all but last column
        training_label = int(row[-1])
        
        # Calculate Euclidean distance between test features and training features
        distance = np.linalg.norm(features - training_features)
        
        # If this is the closest digit so far, update our prediction
        if distance < min_distance:
            min_distance = distance
            predicted_digit = training_label
    
    return predicted_digit  

###########################################################################
def cleanTheData(data: pd.core.frame.DataFrame) -> np.ndarray: #type: ignore
    '''
    Cleans the dataframe made from digits.csv by removing useless values, 
    converting all digits to float and returning a numpy array
    Parameters:
        data: pandas dataframe to clean
    Returns:
        data_array: cleaned numpy array
    '''
    # Drop any columns that have NaN values
    data = data.dropna(axis=1)
    # Convert to numpy array
    data_array = data.to_numpy()
    # Convert to int
    data_array = data_array.astype(int)
    return data_array

###########################################################################
def splitData(data: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Question 7: Splits data into training and testing sets using sklearn.
    Parameters:
        data: numpy array with features in all columns except last, and labels in last column
        random_state: seed for random number generator (default 42)
    Returns:
        tuple containing (X_test, y_test, X_train, y_train) in that order
    '''
    # Separate features (X) and labels (y)
    # Use all columns except the last one for features
    X = data[:, :-1]  # all columns except the last one are features
    y = data[:, -1]   # last column is the label
    
    # Use sklearn to split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Return in the order specified by assignment: X_test, y_test, X_train, y_train
    return (X_test, y_test, X_train, y_train)

###########################################################################
def modelTrainingAndTesting(filename: str, training_size: float, show_misclassified: bool = False) -> None: 
    '''
    Helper function for training and testing the model with different split points
    Parameters:
        Filename: filename of the file to be read
        Training size: 0 - 1 representing how what percent of the file is to be training
        show_misclassified: if True, shows heatmaps of first 5 misclassified digits
    '''
    df = pd.read_csv(filename, header = 0)
    print("\n=== Question 3: Testing 1-NN classifier ===")

    # Check the data shape and info
    print(f"Original data shape: {df.shape}")
    print(f"Number of NaN values per column:\n{df.isna().sum()}")

    # Keep only the first 65 columns (64 pixels + 1 label)
    df = df.iloc[:, :65]

    # Convert dataframe to numpy array
    data = df.values
    split_point = int(len(data) * (training_size))
    training_set = data[:split_point]  # first 80%
    test_set = data[split_point:]      # last 20%

    print(f"Training set size: {len(training_set)}")
    print(f"Test set size: {len(test_set)}")

    # Test each digit in test set and count correct predictions
    correct = 0
    total = len(test_set)
    wrong_digits = 0
    print("Predicting test digits...")
    # Simple progress bar using a loop
    for i in range(total):
        # Get the test features (pixels) and actual label
        test_features = test_set[i, 0:64]  # columns 0-63 are pixels
        actual_label = int(test_set[i, 64])  # column 64 is the label
        
        # Predict using 1-NN
        predicted_label = predictiveModel(training_set, test_features)
        # Check if correct
        if predicted_label == actual_label:
            correct += 1
        else:
            # Question 5: Visualize first 5 incorrectly predicted digits (only if requested)
            wrong_digits += 1
            if show_misclassified and wrong_digits <= 5:
                print(f"\nMisclassified digit #{wrong_digits}:")
                print(f"  Actual label: {actual_label}")
                print(f"  Predicted label: {predicted_label}")
                pixels = np.reshape(test_features, (8, 8))
                drawDigitHeatmap(pixels)                
        
        # Show progress every 10%
        if (i + 1) % (total // 10) == 0:
            print(f"Progress: {i + 1}/{total} ({100 * (i + 1) / total:.0f}%)")

    # Calculate and report accuracy
    accuracy = correct / total
    print(f"\nResults:")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")

###########################################################################
def compareLabels(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> int:
    ''' a more neatly formatted comparison, returning the number correct '''
    num_labels = len(predicted_labels)
    num_correct = 0

    for i in range(num_labels):
        predicted = int(round(predicted_labels[i]))  # round-to-int protects from float imprecision
        actual    = int(round(actual_labels[i]))
        result = "incorrect"
        if predicted == actual:  # if they match,
            result = ""       # no longer incorrect
            num_correct += 1  # and we count a match!

            print(f"{i:>5d} {predicted:>10d} {actual:>10d} {result:>12s}")


    accuracy = num_correct / num_labels
    print(f"Correct: {num_correct} out of {num_labels}")
    print(f"Accuracy: {accuracy:.3f}")
    
    return num_correct

###########################################################################
def findBestK(X_train: np.ndarray, y_train: np.ndarray, random_seed: int = 42) -> int:
    '''
    Question 9: Finds the best k value by testing on a validation set.
    Parameters:
        X_train: numpy array of training features
        y_train: numpy array of training labels
        random_seed: seed for reproducible splits

    Returns:
        best_k: the k value from 1 to 85 that gives highest accuracy on validation data
    '''
    # Split training data into sub-training and validation sets
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_seed
    )
    
    best_k = 1
    best_accuracy = 0.0

    print(f"Testing k values from 1 to 85 with seed {random_seed}...")
    for k in range(1, 86):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_subtrain, y_subtrain)  # Train on sub-training set
        acc = knn.score(X_val, y_val)     # Test on validation set (NOT training!)

        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k

    print(f"Best k = {best_k} with validation accuracy = {best_accuracy:.3f}")
    return best_k

###########################################################################
def trainAndTest(X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, best_k: int) -> np.ndarray:
    '''
    Question 10: Train and test the model using the best k value.
    Parameters:
        X_train: numpy array of training features
        y_train: numpy array of training labels
        X_test: numpy array of test features
        best_k: best k value determined from validation
    Returns:
        predicted_labels: numpy array of predicted labels for X_test
    '''
    print(f"\n=== Question 10: Training and testing with best k={best_k} ===")

    # Create and train k-NN model with best k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    # Predict labels for the test set
    predicted_labels = knn.predict(X_test)
    return predicted_labels

########################################################################### 

def main() -> None:
    # for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe...")

    # Initial heatmap drawing to visualize the data
    num_to_draw = 5
    for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)

        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")  
        drawDigitHeatmap(pixels)
        plt.show()

    #
    # OK!  Onward to knn for digits! (based on your iris work...)

    # Questions 3 & 4: running training and testing model with manual split
    # Question 5: Show misclassified digits only on first run
    modelTrainingAndTesting('digits.csv', .8, show_misclassified=True)
    modelTrainingAndTesting('digits.csv', .2, show_misclassified=False)
    
    # Question 7: Test the splitData function with sklearn
    print("\n=== Question 7: Testing sklearn splitData function ===")
    
    # Prepare data
    df_clean = df.iloc[:, :65]  # keep only first 65 columns
    data = df_clean.values
    
    # Call splitData function (returns X_test, y_test, X_train, y_train)
    X_test, y_test, X_train, y_train = splitData(data)
    
    # Print the results to verify it worked
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Question 8: Test k-NN with sklearn using a guessed k value
    k = 3  # Choosing k=3 as initial guess (common starting point)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_prediction = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    print(f"Accuracy for k={k}: {accuracy:.3f}")
    # Question 9: Find best k using three different random seeds
    print("\nQuestion 9: Finding best k with different random seeds:")
    seeds = [8675309, 5551212, 42]  # two given + one custom
    best_k_values = []

    for seed in seeds:
        print(f"\n--- Testing with seed {seed} ---")
        best_k = findBestK(X_train, y_train, random_seed=seed)
        best_k_values.append(best_k)

    print(f"Best k for seed 8675309: {best_k_values[0]}")
    print(f"Best k for seed 5551212: {best_k_values[1]}")
    print(f"Best k for seed 42: {best_k_values[2]}")
    
    # Check if they're all the same
    if best_k_values[0] == best_k_values[1] == best_k_values[2]:
        print("Are they all the same? YES")
    else:
        print("Are they all the same? NO")
    
    # Pick the most common k value
    count_0 = best_k_values.count(best_k_values[0])
    count_1 = best_k_values.count(best_k_values[1])
    count_2 = best_k_values.count(best_k_values[2])
    
    if count_0 >= count_1 and count_0 >= count_2:
        chosen_k = best_k_values[0]
    elif count_1 >= count_2:
        chosen_k = best_k_values[1]
    else:
        chosen_k = best_k_values[2]
    
    print(f"Chosen best k: {chosen_k}")
    
    # Question 10: Train and test with the chosen best k
    predicted_labels = trainAndTest(X_train, y_train, X_test, chosen_k)
    
    # Calculate accuracy and show comparison
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"\nAccuracy with k={chosen_k}: {accuracy:.3f}")
    print("\nComparison of predicted vs actual labels:")
    compareLabels(predicted_labels, y_test)
    
###############################################################################

# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()
