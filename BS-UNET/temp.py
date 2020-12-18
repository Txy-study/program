import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
 # 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum() #PA=OA
    print("acc:"+acc)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)   #acc
    acc_cls = np.nanmean(acc_cls)  #MPA
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()  #axis=1计算行和  axis=0计算列和
    return ({"OA:\t":acc,
             "AA:\t":acc_cls,
             "Mean IOU:\t":mean_iu,
             }
    )



# def kappa(self,y_test,y_predicted):
#     return round(cohen_kappa_score(y_test,y_predicted),3)
#
# def score(self,y_test,y_predicted):
#     oa=accuracy_score(y_test,y_predicted)
#     # n_classes=max([np.unique(y_test).__len__(),np.unique(y_predicted).__len__()])
#     ca=[]
#     for c in np.unique(y_test):
#         y_c=y_test[np.nonzero(y_test==c)]
#         y_c_p=y_predicted[np.nonzero(y_test==c)]
#         accuracy=accuracy_score(y_c,y_c_p)
#         ca.append(accuracy)
#     ca=np.array(ca)
#     aa=ca.mean()
#     'kappa'
#     kappa=self.kappa(y_test,y_predicted)
#     return oa,aa,kappa
