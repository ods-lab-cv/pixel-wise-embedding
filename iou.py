import torch

class IoU(torch.nn.Module):
    __name__="IoU"
    def __init__(self, mode='MULTILABEL_MODE', threshold=0.5):
        assert mode in ['MULTILABEL_MODE']
        super(IoU, self).__init__()
        self.mode = mode
        self.threshold = threshold


    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            X - model prediction, (batch_size, num_classes, height, width) 
            Y - label, (batch_size, num_classes, height, width) 
        '''
        batch_size, num_classes, height, width = X.shape

        X = torch.where(X >= self.threshold, 1, 0)

        X = X.view(batch_size, num_classes, -1)
        Y = Y.view(batch_size, num_classes, -1)

        tp = (X * Y).sum(2)
        fp = X.sum(2) - tp
        fn = Y.sum(2) - tp
        tn = torch.tensor(height*width) - (tp + fp + fn)
        
        return torch.nanmean(tp / (tp+fp+fn), axis=0)
