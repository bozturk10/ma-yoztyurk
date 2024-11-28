#just as an idea now, not used in the project
from skmultilearn.dataset import load_dataset
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')

from scipy.sparse import csr_matrix

# Example y_labels (replace this with your actual data)
# Assume y_labels is a 2D array or list of lists
y_labels = [
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1],
    # Add more rows as needed
]

# Convert y_labels to a NumPy array
y_labels_array = np.array(y_labels)

# Create a sparse matrix (CSR format)
sparse_matrix = csr_matrix(y_labels_array)
