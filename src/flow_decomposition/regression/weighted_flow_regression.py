from flow_decomposition.utils.utils import get_td_embedding_torch
from flow_decomposition.utils.models import LinearModel, NonlinearModel
from .flow_regression import FlowRegression

import torch
import torch.optim as optim

DATA_DTYPE = torch.float32
MODEL_DTYPE = torch.float32

class WeightedFlowRegression(FlowRegression):
    
    def __init__(self, input_dim, proj_dim, 
                 num_delays=None, delay_step=None, model="linear", subtract_autocorr=False, 
                 device="cuda",data_device="cpu", optimizer="Adagrad", learning_rate=0.01, random_state=None):
        """
        Initializes the WeightedFlowRegression model.
        
        Args:
        """
        super(WeightedFlowRegression, self).__init__(input_dim, 
                                                     proj_dim, 
                                                     num_delays, 
                                                     delay_step,
                                                     model, 
                                                     subtract_autocorr, 
                                                     device, 
                                                     data_device, 
                                                     optimizer, 
                                                     learning_rate, 
                                                     random_state)
        
        if self.use_delay:
            E = self.proj_dim * self.num_delays
        else:
            E = self.proj_dim

        self.weight_model = torch.nn.Sequential(torch.nn.Linear(E, E, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Linear(E, 1, bias=True),
                                                torch.nn.Sigmoid()).to(device)
        
        self.optimizer = getattr(optim, optimizer)(list(self.model.parameters()) + list(self.weight_model.parameters()), lr=learning_rate)

    def _get_smap_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        A, B = super()._get_smap_prediction(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad)
        
        batch_size, sample_size, _, _ = sample_X.shape
        sample_weights = self.weight_model(sample_X[:,:,:,0]).reshape(batch_size,sample_size, 1, 1, 1)
        
        r_AB = self.__get_batch_weighted_corr(A,B,sample_weights)
        return r_AB
    
    def _get_knn_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
        A, B = super()._get_knn_prediction(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad)
        
        batch_size, sample_size, _, _ = sample_X.shape
        sample_weights = self.weight_model(sample_X[:,:,:,0]).reshape(batch_size,sample_size, 1, 1, 1)
        
        r_AB = self.__get_batch_weighted_corr(A,B,sample_weights)
        return r_AB
        
    def __get_batch_weighted_corr(self, A, B, weights, eps=1e-8):
        # Compute the weighted mean along axis 1 (same as your unweighted version but with weights)
        sum_w = weights.sum(dim=1, keepdim=True)  # shape: [batch, 1, ...]
        mean_A = (weights * A).sum(dim=1, keepdim=True) / sum_w
        mean_B = (weights * B).sum(dim=1, keepdim=True) / sum_w

        # Compute the weighted covariance and variances along axis 1
        sum_AB = torch.sum(weights * (A - mean_A) * (B - mean_B), dim=1)
        sum_AA = torch.sum(weights * (A - mean_A) ** 2, dim=1)
        sum_BB = torch.sum(weights * (B - mean_B) ** 2, dim=1)

        # Compute the weighted correlation coefficient
        r_AB = sum_AB / (torch.sqrt(sum_AA * sum_BB) + eps)
        return r_AB

    def predict_weights(self, X, pad_nan=True, device="cpu"):
        """
        Predicts weights from input X.
        
        Args:
            X (np.array): Input data.
            pad_nan (bool): If True, pads output with NaNs to match the length of X.
            device (str): Device for computation ("cpu", "cuda", etc.).
        
        Returns:
            np.array: Predicted weights of samples.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=MODEL_DTYPE, device=device)
            outputs = self.model.to(device)(inputs)[:, :, 0]
            if self.use_delay:
                outputs = get_td_embedding_torch(outputs, self.num_delays, self.delay_step)
                outputs = outputs.reshape(outputs.size(0), -1)
            outputs = self.weight_model.to(device)(outputs)
            
            # Pad outputs with NaNs at the beginning if pad_nan is True.
            if pad_nan and self.use_delay:
                pad_length = X.shape[0] - outputs.shape[0]
                if pad_length > 0:
                    nan_pad = torch.full((pad_length, outputs.shape[1]),
                                        float('nan'),
                                        dtype=outputs.dtype,
                                        device=device)
                    outputs = torch.cat([nan_pad, outputs], dim=0)
        return outputs.cpu().numpy()
    
    def save(self, filepath: str):
        """
        Saves the model, optimizer state, and other attributes to disk,
        including the additional weight_model.
        """
        # 1) Collect constructor / hyperparams needed to re-initialize
        init_params = {
            "input_dim": self.input_dim,
            "proj_dim": self.proj_dim,
            "num_delays": self.num_delays,
            "delay_step": self.delay_step,
            "model": "linear" if isinstance(self.model, LinearModel) else "nonlinear",
            "subtract_autocorr": self.subtract_autocorr,
            "device": self.device,  # might be overridden on load
            "data_device": self.data_device,
            "optimizer": type(self.optimizer).__name__,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "random_state": self.random_state
        }

        # 2) Create checkpoint dict
        checkpoint = {
            "init_params": init_params,
            "model_state": self.model.state_dict(),
            "weight_model_state": self.weight_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
        }

        # 3) Save checkpoint
        torch.save(checkpoint, filepath)
        print(f"WeightedFlowRegression model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, map_location: str = "cpu"):
        """
        Loads a saved WeightedFlowRegression model from disk.
        """
        # 1) Load checkpoint
        checkpoint = torch.load(filepath, map_location=map_location)
        init_params = checkpoint["init_params"]

        # 2) Instantiate a new instance with the stored hyperparams
        #    Override device/data_device for new session if desired
        init_params["device"] = map_location
        init_params["data_device"] = map_location
        new_instance = cls(**init_params)

        # 3) Load states
        new_instance.model.load_state_dict(checkpoint["model_state"])
        new_instance.weight_model.load_state_dict(checkpoint["weight_model_state"])
        new_instance.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # 4) Restore training history
        new_instance.loss_history = checkpoint["loss_history"]

        print(f"WeightedFlowRegression model loaded from {filepath}")
        return new_instance