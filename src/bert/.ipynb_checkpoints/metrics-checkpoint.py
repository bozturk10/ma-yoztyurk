
from datetime import datetime
import json
import os
import numpy as np
from sklearn.metrics import classification_report
import wandb

from src.paths import CLSF_LOGS_DIR
from src.utils import load_lookup_data


def define_metrics():
    from evaluate import load
    accuracy = load("accuracy",'multilabel',trust_remote_code=True)
    precision = load("precision",'multilabel',trust_remote_code=True)
    recall = load("recall",'multilabel',trust_remote_code=True)
    f1 = load("f1",'multilabel',trust_remote_code=True)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def get_compute_metrics_function(metrics,target_names=None):
    
    def compute_metrics(eval_pred):
        nonlocal target_names
        predictions, labels = eval_pred
        print('predictions.shape',predictions.shape)
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int)#.reshape(-1)
        references = labels.astype(int)#.reshape(-1)

        average = 'samples'
        acc = metrics['accuracy'].compute(predictions=predictions, references=references)
        f1 = metrics['f1'].compute(predictions=predictions, references=references,average=average)
        precision = metrics['precision'].compute(predictions=predictions, references=references,average=average)
        recall = metrics['recall'].compute(predictions=predictions, references=references,average=average)
        print('accuracy:',acc, 'precision',precision, 'recall',recall, 'f1',f1)
        if target_names and acc["accuracy"]>0.5:
            print(classification_report(references, predictions,target_names=target_names, output_dict=False,zero_division=0))
            clsf_dict=classification_report(references, predictions,target_names=target_names, output_dict=True,zero_division=0)
            fname= os.path.join(CLSF_LOGS_DIR,f'classification_report_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
            with open(fname, 'w') as f:
                json.dump(clsf_dict, f)
        
        return {
            "accuracy": acc["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"]
        }
        
    return compute_metrics

