"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
# import torchvision.models as models
import torch.nn.functional as F
import torchvision


from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPPHead
# from torch_model_zoo.models.mobilenet_v3 import lr_aspp_mobilenet_v3_large

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.n_classes = num_classes
        self.model = lraspp_mobilenet_v3_large(pretrained=True)
        
        for param in self.model.parameters():
            param.requiresGrad = False
            
        self.model.classifier = LRASPPHead(40, 960, hparams['n_classes'], 128)
        

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        x = self.model(x)
        x = x['out']

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
