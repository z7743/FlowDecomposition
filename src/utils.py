import torch

def get_td_embedding_torch(ts, dim, stride, return_pred=False, tp=0):
    tdemb = ts.unfold(0,(dim-1) * stride + 1,1)[...,::stride]
    tdemb = torch.swapaxes(tdemb,-1,-2)
    if return_pred:
        return tdemb[:tdemb.shape[0]-tp], ts[(dim-1) * stride + tp:]
    else:
        return tdemb