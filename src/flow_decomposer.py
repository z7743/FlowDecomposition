from model import LinearModel
from data_sampler import RandomSampleSubsetPairDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class FlowDecomposition:
    def __init__(self, input_dim, proj_dim, n_components, 
                 num_delays=None, delay_step=None, subtract_autocorr=False, 
                 device="cuda", optimizer="SGD", learning_rate=0.01, random_state=None):
        """
        Initialize the FlowDecomposition model.

        Args:
            input_dim (int): Dimension of the input time series.
            proj_dim (int): Projection dimension.
            n_components (int): Number of components.
            num_delays (Optional[int]): Number of delays.
            delay_step (Optional[int]): Delay step.
            subtract_autocorr (bool): Whether to subtract autocorrelation.
            device (str): Device to run on ("cuda" or "cpu").
            optimizer (str): Optimizer name (e.g. "SGD").
            learning_rate (float): Learning rate for optimizer.
            random_state (Optional[int]): Random state for reproducibility.
        """
        self.device = device
        self.random_state = random_state
        self.proj_dim = proj_dim
        self.n_comp = n_components
        self.num_delays = num_delays
        self.delay_step = delay_step
        self.subtract_autocorr = subtract_autocorr
        self.loss_history = []

        self.model = LinearModel(input_dim, proj_dim, n_components, device, random_state)
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate)

        self.use_delay = bool(num_delays) and bool(delay_step)

    def fit(self, X_tensor, 
            sample_size, 
            library_size, 
            exclusion_rad=0,
            theta=None, 
            time_intv=1, 
            num_epochs=100, 
            num_rand_samples=32, 
            batch_size=1,
            beta=0,
            optim_policy="range", 
            mask_size=None):
        """
        Fit the model using the provided data.

        Args:
            X (Tensor or np.ndarray): Input data.
            sample_size (int): Sample size for dataset.
            library_size (int): Library (subset) size for dataset.
            exclusion_rad (int, optional): Exclusion radius.
            theta (Optional[float], optional): Parameter for local weights.
            time_intv (int, optional): Time interval.
            num_epochs (int, optional): Number of training epochs.
            num_rand_samples (int, optional): Number of random samples.
            batch_size (int, optional): Size of the batch (<= num_rand_samples).
            beta (float, optional): Projection regularization coefficient.
            optim_policy (str, optional): Policy for optimization ("fixed" or "range").
            mask_size (Optional[int], optional): Mask size for selecting dimensions.
        """
        X_tensor = torch.tensor(X_tensor,requires_grad=True, device=self.device, dtype=torch.float32)

        if optim_policy == "fixed":
            tp_range = (time_intv, time_intv)
        elif optim_policy == "range":
            tp_range = (1, time_intv)
        else:
            raise ValueError(f"Unknown optim_policy: {optim_policy}")
        
        dataset = RandomSampleSubsetPairDataset(
                    X=X_tensor,
                    sample_size=sample_size,
                    subset_size=library_size,
                    E=self.num_delays,
                    tau=self.delay_step,
                    num_batches=num_rand_samples,
                    tp_range=tp_range,
                    device=self.device,
                    random_state=self.random_state,
                )

        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False)

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            ccm_loss = 0
            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                #Shape: [batch, subset/sample size, E, data_dim]
                subset_X_z = self.model(subset_X)
                sample_X_z = self.model(sample_X)
                subset_y_z = self.model(subset_y)
                sample_y_z = self.model(sample_y)
                #Shape: [batch, subset/sample size, E, proj_dim, n_components]

                subset_X_z = subset_X_z.reshape(subset_X_z.size(0),subset_X_z.size(1), -1, subset_X_z.size(4))
                sample_X_z = sample_X_z.reshape(sample_X_z.size(0),sample_X_z.size(1), -1, sample_X_z.size(4))
                subset_y_z = subset_y_z.reshape(subset_y_z.size(0),subset_y_z.size(1), -1, subset_y_z.size(4))
                sample_y_z = sample_y_z.reshape(sample_y_z.size(0),sample_y_z.size(1), -1, sample_y_z.size(4))
                #Shape: [batch, subset/sample size, E * proj_dim, n_components]

                loss = self.__compute_loss(subset_idx, sample_idx,
                                      sample_X_z, sample_y_z, subset_X_z, subset_y_z, 
                                      theta, exclusion_rad, mask_size)
                
                loss /= num_rand_samples
                ccm_loss += loss.sum()

            h_norm = self.__compute_h_norm()
            
            total_loss = ccm_loss + beta * h_norm

            total_loss.backward()

            self.optimizer.step()

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Loss: {total_loss.item():.4f}, "
                f"ccm_loss: {ccm_loss.item():.4f}, "
                f"h_norm_loss: {h_norm.item():.4f}"
            )
            self.loss_history.append(total_loss.item())

    def predict(self, X):
        """
        Calculates embeddings using the trained model.
        
        Args:
            X (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Decomposed outputs.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            outputs = torch.permute(self.model(inputs),dims=(0,2,1)) #Easier to interpret
        return outputs.cpu().numpy()

    def __compute_h_norm(self):
        l1 = sum(p.abs().sum() 
                for p in self.model.parameters() if p.requires_grad)
        l2 = sum(p.norm(2)
                for p in self.model.parameters() if p.requires_grad)
        num_w = torch.tensor(sum(p.numel()
                for p in self.model.parameters() if p.requires_grad),dtype=torch.float32)
        h_norm = 1-(torch.sqrt(num_w) - (l1 / (l2 + 1e-6))) / (torch.sqrt(num_w) - 1)

        return h_norm

    def __compute_loss(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad, mask_size):
        dim = self.n_comp

        if mask_size is not None:
            rand_idx = torch.argsort(torch.rand(dim),device=self.device)[:mask_size]
            sample_X = sample_X[:,:,:,rand_idx]
            sample_y = sample_y[:,:,:,rand_idx]
            subset_X = subset_X[:,:,:,rand_idx]
            subset_y = subset_y[:,:,:,rand_idx]

            dim = mask_size

        ccm = (self.__get_ccm_matrix_approx(subset_idx, 
                                            sample_idx, 
                                            sample_X, sample_y, 
                                            subset_X, subset_y, theta, exclusion_rad))
        #Shape: (B, E, dim, dim)
        mask = torch.eye(dim,dtype=bool,device=self.device)

        ccm = ccm**2

        if self.subtract_autocorr:
            corr = torch.abs(self.__get_autoreg_matrix_approx(sample_X,sample_y))**2
            if dim > 1:
                score = 1 + torch.abs(ccm[:,:,~mask]).mean(axis=(1,2)) - (ccm[:,:,mask]).mean(axis=(1,2)) + (corr[:,:,mask]).mean(axis=(1,2))
            else:
                score = 1 + (-ccm[:,:,0,0] + corr[:,:,0,0]).mean(axis=1)
            return score
        else:
            if dim > 1:
                score = 1 + torch.abs(ccm[:,:,~mask]).mean(axis=(1,2)) - (ccm[:,:,mask]).mean(axis=(1,2)) 
            else:
                score = 1 + (-ccm[:,:,0,0]).mean(axis=1)
            return score
        
    def __get_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        batch_size, sample_size, dim_x, n_comp = sample_X.shape
        _, subset_size, _, _ = subset_X.shape
        _, _, dim_y, _ = sample_y.shape

        sample_X_t = sample_X.permute(0, 3, 1, 2)
        subset_X_t = subset_X.permute(0, 3, 1, 2)
        subset_y_t = subset_y.permute(0, 3, 1, 2)
        #Shape: [batch, comp, points, proj_dim]
        
        weights = self.__get_local_weights(subset_X_t,sample_X_t,subset_idx, sample_idx, exclusion_rad, theta)
        #Shape: [batch, comp, sample_size, subset_size]
        W = (weights.unsqueeze(2)
             .expand(batch_size, n_comp, n_comp, sample_size, subset_size)
             .reshape(batch_size * n_comp * n_comp * sample_size, subset_size, 1))
        #Shape: [batch * comp * comp, subset_size, 1]

        X = (subset_X_t.unsqueeze(2).unsqueeze(2)
             .expand(batch_size, n_comp, n_comp, sample_size, subset_size, dim_x)
             .reshape(batch_size * n_comp * n_comp * sample_size, subset_size, dim_x))

        Y = (subset_y_t.unsqueeze(2).unsqueeze(1)
             .expand(batch_size, n_comp, n_comp, sample_size, subset_size, dim_y)
             .reshape(batch_size * n_comp * n_comp * sample_size, subset_size, dim_y))

        X_intercept = torch.cat([torch.ones((batch_size * n_comp * n_comp * sample_size, subset_size, 1),device=self.device), X], dim=2)
        
        X_intercept_weighted = X_intercept * W
        Y_weighted = Y * W

        XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
        XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)
        beta = torch.bmm(torch.pinverse(XTWX), XTWy)

        X_ = (sample_X_t.unsqueeze(2)
              .expand(batch_size, n_comp, n_comp, sample_size, dim_x)
              .reshape(batch_size * n_comp * n_comp * sample_size, dim_x))
        X_ = torch.cat([torch.ones((batch_size * n_comp * n_comp * sample_size, 1),device=self.device), X_], dim=1)
        X_ = X_.reshape(batch_size * n_comp * n_comp * sample_size, 1, dim_x+1)
        
        A = torch.bmm(X_, beta).reshape(batch_size, n_comp, n_comp, sample_size, dim_y)
        A = torch.permute(A,(0,3,4,2,1))

        B = sample_y.unsqueeze(-1).expand(batch_size, sample_size, dim_y, n_comp, n_comp)
        
        r_AB = self.__get_batch_corr(A,B)
        return r_AB
    
    def __get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        #[batch, comp, points, proj_dim]
        dist = torch.cdist(sublib,lib)
        if theta == None:
            weights = torch.exp(-(dist))
        else:
            weights = torch.exp(-(theta*dist/(dist.mean(dim=3, keepdim=True) + 1e-6)))

        if exclusion_rad > 0:
            exclusion_matrix = (torch.abs(subset_idx.unsqueeze(-2) - sample_idx.unsqueeze(-1)) > exclusion_rad)
            weights = weights * exclusion_matrix[:,None]
        
        return weights
    
    def __get_autoreg_matrix_approx(self, A, B):
        batch_size, _, _, _ = A.shape
        dim = self.n_comp
        if self.use_delay:
            E = self.proj_dim * self.num_delays
        else:
            E = self.proj_dim
        
        A = A[:,:,:,None,:].expand(batch_size,-1, E, dim, dim)
        B = B[:,:,:,:,None].expand(batch_size,-1, E, dim, dim)

        r_AB = self.__get_batch_corr(A,B)
        return r_AB
    
    def __get_batch_corr(self, A, B):
        mean_A = torch.mean(A,axis=1).unsqueeze(1)
        mean_B = torch.mean(B,axis=1).unsqueeze(1)
        
        sum_AB = torch.sum((A - mean_A) * (B - mean_B),axis=1)
        sum_AA = torch.sum((A - mean_A) ** 2,axis=1)
        sum_BB = torch.sum((B - mean_B) ** 2,axis=1)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB    
    

