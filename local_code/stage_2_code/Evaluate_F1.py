'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score


class Evaluate_F1_None(evaluate):
    data = None
    
    def evaluate(self):
        # print('evaluating multilabel classification...')
        return f1_score(self.data['true_y'], self.data['pred_y'], average=None)
    
class Evaluate_F1_Weighted(evaluate):
    data = None
    
    def evaluate(self):
        # print('evaluating weighted f1...')
        return f1_score(self.data['true_y'], self.data['pred_y'], average="weighted")

class Evaluate_F1_Macro(evaluate):
    data = None
    
    def evaluate(self):
        # print('evaluating macro f1...')
        return f1_score(self.data['true_y'], self.data['pred_y'], average="macro")

class Evaluate_F1_Micro(evaluate):
    data = None
    
    def evaluate(self):
        # print('evaluating micro f1...')
        return f1_score(self.data['true_y'], self.data['pred_y'], average="micro")
     
