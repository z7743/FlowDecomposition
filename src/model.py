
import torch
import torch.nn as nn
import torch.nn.init as init

class LinearModel(nn.Module):
    def __init__(self, input_dim, proj_dim, n_comp, device="cuda",random_state=None):
        """
        Initializes the linear projection module.
        
        Args:
            input_dim (int): The dimension of the input data.
            proj_dim (int): The dimension of the embedding.
            n_comp (int): The dimension of the output projection.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        super(LinearModel, self).__init__()
        self.device = device

        self.proj_dim = proj_dim
        self.n_comp = n_comp

        if random_state != None:
            torch.manual_seed(random_state)
        self.model = nn.Linear(input_dim, n_comp*proj_dim, bias=False,device=self.device)
    
    def _init_weights(self, m):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

    def forward(self, x):
        x_shape = x.shape
        x = self.model(x).reshape(*x_shape[:-1], self.proj_dim, self.n_comp)

        return x
    
    def get_weights(self):
        """
        Retrieves the weights of the model.
        
        Returns:
            numpy.ndarray: Weights of the model reshaped and permuted.
        """
        return torch.permute(self.model.weight.T.reshape(-1,self.proj_dim,self.n_comp), dims=(0,2,1)).cpu().detach().numpy()

