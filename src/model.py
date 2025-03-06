
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, proj_dim, n_comp, device="cuda", dtype=torch.float32, random_state=None):
        """
        Initializes the linear projection module.
        
        Args:
            input_dim (int): The dimension of the input data.
            proj_dim (int): The dimension of the output projection.
            n_comp (int): The number of the output components.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        super(LinearModel, self).__init__()
        self.device = device

        self.proj_dim = proj_dim
        self.n_comp = n_comp

        if random_state != None:
            torch.manual_seed(random_state)
        self.model = nn.Linear(input_dim, n_comp*proj_dim, bias=False,device=device, dtype=dtype)

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

class NonlinearModel(nn.Module):
    def __init__(self, input_dim, proj_dim, n_comp, device="cuda", dtype=torch.float32, random_state=None):
        """
        Initializes the nonlinear projection module.

        Args:
            input_dim (int): The dimension of the input data.
            proj_dim (int): The dimension of the output projection.
            n_comp (int): The number of the output components.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        super(NonlinearModel, self).__init__()
        self.device = device
        self.proj_dim = proj_dim
        self.n_comp = n_comp

        if random_state is not None:
            torch.manual_seed(random_state)

        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim // 2, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(input_dim // 2, proj_dim * n_comp, bias=False, device=device, dtype=dtype)
        )

    def forward(self, x):
        """
        Forward pass: Applies the nonlinear transformations and reshapes the output.

        Args:
            x (torch.Tensor): Input tensor with shape (..., input_dim).

        Returns:
            torch.Tensor: Output tensor reshaped to (..., proj_dim, n_comp).
        """
        x_shape = x.shape
        x = self.model(x).reshape(*x_shape[:-1], self.proj_dim, self.n_comp)
        return x

    def get_weights(self):
        """
        Retrieves the weights of the final linear layer.

        Returns:
            numpy.ndarray: Weights of the final linear layer reshaped and permuted.
        """
        final_layer = self.model[-1]
        # Permute and reshape to mirror the LinearModel's get_weights() method.
        weights = torch.permute(final_layer.weight.T.reshape(-1, self.proj_dim, self.n_comp), dims=(0, 2, 1))
        return weights.cpu().detach().numpy()
