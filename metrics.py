from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def evaluate_unweighted_macro_avg(pred_labels, true_labels):
    num_classes = pred_labels.size(1)  # Get the number of classes
    
    # Initialize lists to store class-wise metrics
    precisions = []
    recalls = []
    f1s = []

    for class_idx in range(num_classes):
        pred_labels_class = pred_labels[:, class_idx, :]  # Select predictions for the current class
        true_labels_class = true_labels[:, class_idx, :]  # Select true labels for the current class
        
        pred_labels_flat = pred_labels_class.reshape(-1)
        true_labels_flat = true_labels_class.reshape(-1)
        
        print(true_labels_flat.cpu().numpy())
        print(pred_labels_flat.cpu().numpy())
        precision = precision_score(true_labels_flat.cpu().numpy(), pred_labels_flat.cpu().numpy())
        recall = recall_score(true_labels_flat.cpu().numpy(), pred_labels_flat.cpu().numpy())
        f1 = f1_score(true_labels_flat.cpu().numpy(), pred_labels_flat.cpu().numpy())
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Calculate unweighted macro averages
    precision_avg = np.mean(precisions)
    recall_avg = np.mean(recalls)
    f1_avg = np.mean(f1s)
    
    return precision_avg, recall_avg, f1_avg

def evaluate_metrics(pred_labels, true_labels):
    true_labels = true_labels.flatten()
    pred_labels = pred_labels.flatten()
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    
    return precision, recall, f1

if __name__ == "main":
    pass