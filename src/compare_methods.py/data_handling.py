import scipy.io
import numpy

# Helper function to load data
def load_data(prefix, idx_range, jdx_range):
    data_list = []
    for i in idx_range:
        for j in jdx_range:
            file_path = f"/content/drive/MyDrive/data_project/predicted_{prefix}_{i}_{j}.mat"
            data_list.append(scipy.io.loadmat(file_path))
    return data_list

# Helper function to compute accuracy
def compute_accuracy(data_list, label_key):
    count_true, count_total = 0, 0
    for data in data_list:
        prob = data[label_key]
        true_labels = data["true_labels"].squeeze()
        pred_labels = numpy.argmax(prob, axis=-1) + 1
        count_total += len(true_labels)
        count_true += numpy.sum(true_labels == pred_labels)
    return count_true, count_total
