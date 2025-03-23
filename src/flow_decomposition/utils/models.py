import torch
import torch.nn as nn

class BaseDecompositionModel(nn.Module):
    """
    BaseProjectionModel serves as an abstract base class for projection models in the flow framework.
    This class provides the basic structure for models that project input data into a different dimensional space.
    All subclasses must implement the forward method.
    Parameters
    ----------
    input_dim : int
        The dimension of the input data.
    proj_dim : int
        The dimension of the projected space.
    n_comp : int
        The number of components in the projection.
    Notes
    -----
    All subclasses inheriting from this base class must:
    1. Initialize with input_dim, proj_dim, and n_comp parameters
    2. Implement the forward method to define the projection logic
    """
    def __init__(self, input_dim, proj_dim, n_comp):
        super().__init__()
        # Check if required parameters are provided
        if input_dim is None:
            raise ValueError("input_dim must be provided")
        if proj_dim is None:
            raise ValueError("proj_dim must be provided")
        if n_comp is None:
            raise ValueError("n_comp must be provided")
            
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.n_comp = n_comp

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

class BaseRegressionModel(nn.Module):
    """
    BaseRegressionModel serves as an abstract base class for regression models in the flow framework.
    This class provides the basic structure for models that map input data to output targets.
    All subclasses must implement the forward method.
    
    Parameters
    ----------
    input_dim : int
        The dimension of the input data.
    output_dim : int
        The dimension of the output targets.
    
    Notes
    -----
    All subclasses inheriting from this base class must:
    1. Initialize with input_dim and output_dim parameters
    2. Implement the forward method to define the regression logic
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Check if required parameters are provided
        if input_dim is None:
            raise ValueError("input_dim must be provided")
        if output_dim is None:
            raise ValueError("output_dim must be provided")
            
        self.input_dim = input_dim
        self.proj_dim = output_dim
        self.n_comp = 1  # Always set to 1 for regression models

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method.")  
    
        
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
            nn.Linear(input_dim, proj_dim * n_comp * 2, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(proj_dim * n_comp * 2, proj_dim * n_comp * 2, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(proj_dim * n_comp * 2, proj_dim * n_comp, bias=False, device=device, dtype=dtype)
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
