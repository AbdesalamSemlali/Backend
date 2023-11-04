import numpy as np

# Create the desired vector using np.arange()
def myArrange(n) :
    vector = np.arange(n, -1, -1)

    # Pad the vector with zeros to achieve the desired length
    desired_length = 2*n+1
    padded_vector = np.pad(vector, (0, desired_length - len(vector)), mode='constant')

    return padded_vector


