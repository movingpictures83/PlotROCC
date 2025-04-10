
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')

import PyIO
import PyPluMA
class PlotROCCPlugin:
 def input(self, inputfile):
     self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  stat1_test = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["teststat"], sep="\t")
  origin_test = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["testorigin"], sep="\t")
  model = XGBClassifier()
  model.load_model(PyPluMA.prefix()+"/"+self.parameters["model"])


  # Get predicted probabilities for positive class
  y_prob = model.predict_proba(stat1_test)[:, 1]

  # Compute ROC curve and ROC area
  fpr, tpr, _ = roc_curve(origin_test, y_prob)
  roc_auc = auc(fpr, tpr)

  # Plot ROC curve
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic Curve')
  plt.legend(loc="lower right")
  plt.show()
  plt.savefig(outputfile, dpi=1200, bbox_inches='tight')


