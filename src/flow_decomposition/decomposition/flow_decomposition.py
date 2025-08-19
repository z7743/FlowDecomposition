from flow_decomposition.utils.models import LinearModel, NonlinearModel
from flow_decomposition.utils.data_samplers import RandomSampleSubsetPairDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Union

import torch.nn.functional as F

DATA_DTYPE = torch.float32
MODEL_DTYPE = torch.float32

class FlowDecomposition:
    def __init__(self, input_dim, proj_dim, n_components, 
                 num_delays=None, delay_step=None, model="linear", subtract_autocorr=False, 
                 device="cpu",data_device="cpu", optimizer="Adagrad", learning_rate=0.01, random_state=None, verbose=1):
        """
        Initialize the FlowDecomposition model.

        Args:
            input_dim (int): Dimension of the input time series.
            proj_dim (int): Projection dimension.
            n_components (int): Number of components.
            model (str): Projection type ("linear" or "nonlinear").
            num_delays (Optional[int]): Number of delays.
            delay_step (Optional[int]): Delay step.
            subtract_autocorr (bool): Whether to subtract autocorrelation.
            device (str): Device to run on ("cuda" or "cpu").
            data_device (str): Device to store data on ("cuda" or "cpu").
            optimizer (str): Optimizer name (e.g. "Adagrad").
            learning_rate (float): Learning rate for optimizer.
            random_state (Optional[int]): Random state for reproducibility.
            verbose (int): Verbosity flag (1: print epoch info, 0: don't print).
        """
        self.device = device
        self.data_device = data_device
        self.random_state = random_state
        self.proj_dim = proj_dim
        self.n_comp = n_components
        self.num_delays = num_delays
        self.delay_step = delay_step
        self.subtract_autocorr = subtract_autocorr
        self.loss_history = []
        self.input_dim = input_dim
        self.verbose = verbose

        if model == "linear":
            self.model = LinearModel(input_dim, proj_dim, n_components, 
                                     device=device, dtype=MODEL_DTYPE, random_state=random_state)
        elif model == "nonlinear":
            self.model = NonlinearModel(input_dim, proj_dim, n_components, 
                                        device=device, dtype=MODEL_DTYPE, random_state=random_state)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate)

        self.use_delay = bool(num_delays) and bool(delay_step)

    def fit(self, X,
            sample_size, library_size, exclusion_rad=0, method="knn",metric="corr",
            theta=None, nbrs_num=None, time_intv=1, num_epochs=100, num_rand_samples=32,
            batch_size=1, beta=0, optim_policy="range",
            mask_size=None):
        """
        Fit the model using the provided data.
        """
        # Create a DataLoader once, reuse it each epoch

        if self.random_state is not None:
            old_rng_state = torch.get_rng_state()
            torch.manual_seed(self.random_state) #TODO: Not safe for multiple workers

        dataloader = self._create_dataloader(X, sample_size, library_size,
                                            time_intv, num_rand_samples, batch_size,
                                            optim_policy)
        
        if metric == "corr":
            self.metric = self.__get_batch_corr
        elif metric == "dcorr":
            self.metric = self.__get_batch_distance_corr
        elif metric == "ccc":
            self.metric = self.__get_batch_ccc
        elif metric == "dccc":
            self.metric = self.__get_batch_distance_concordance
        elif metric == "abs_corr":
            self.metric = self.__get_batch_abs_corr
        else:
            raise ValueError(f"Unknown metric: {metric}")

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Compute CCM loss
            ccm_loss = self._compute_loss_from_dataloader(dataloader, num_rand_samples, method, theta,
                                                        nbrs_num, exclusion_rad, mask_size)
            # Projection norm penalty
            h_norm = self.__compute_h_norm()
            total_loss = ccm_loss + beta * h_norm
            
            total_loss.backward()
            self.optimizer.step()
            
            if self.verbose == 1:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {total_loss.item():.4f}, "
                    f"ccm_loss: {ccm_loss.item():.4f}, "
                    f"h_norm: {h_norm.item():.4f}"
                )
            self.loss_history.append(total_loss.item())

            #self.model.model.orthonormalise()
        
        if self.random_state is not None:
            torch.set_rng_state(old_rng_state)

    def evaluate_loss(self, X_test, 
                      sample_size, library_size, exclusion_rad=0, method="knn",
                      theta=None, nbrs_num=None, time_intv=1,
                      num_epochs=None, num_rand_samples=32, batch_size=1, beta=None,
                      optim_policy="range", mask_size=None):
        """
        Evaluate the CCM-based loss on test data, with no parameter updates.
        """
        
        if self.random_state is not None:
            old_rng_state = torch.get_rng_state()
            torch.manual_seed(self.random_state) #TODO: Not safe for multiple workers

        with torch.no_grad():
            # Build a DataLoader for test data
            dataloader = self._create_dataloader(X_test,
                                                sample_size,
                                                library_size,
                                                time_intv,
                                                num_rand_samples,
                                                batch_size,
                                                optim_policy)
            # Accumulate CCM-based loss
            ccm_loss = self._compute_loss_from_dataloader(dataloader,
                                                        num_rand_samples,
                                                        method,
                                                        theta,
                                                        nbrs_num,
                                                        exclusion_rad,
                                                        mask_size)
        
        if self.random_state is not None:
            torch.set_rng_state(old_rng_state)
        # Return as a scalar
        return ccm_loss.item()

    def transform(self, X, device="cpu"):
        """
        Calculates embeddings using the trained model.
        
        Args:
            X (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Decomposed outputs.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=MODEL_DTYPE,device=device)
            outputs = torch.permute(self.model.to(device)(inputs),dims=(0,2,1)) #Easier to interpret
        return outputs.cpu().numpy()

    def _create_dataloader(self, X, sample_size, library_size, 
                       time_intv, num_rand_samples, batch_size, 
                       optim_policy):
        """
        Internal helper to build a DataLoader from the input data and the
        specified sampling policy.
        """
        # Move numpy or other input into a torch.Tensor
        X_tensor = torch.tensor(X, requires_grad=False, device=self.data_device, dtype=DATA_DTYPE)
        
        # Decide which range of time lags to use
        if optim_policy == "fixed":
            tp_range = (time_intv, time_intv)
        elif optim_policy == "range":
            tp_range = (1, time_intv)
        else:
            raise ValueError(f"Unknown optim_policy: {optim_policy}")

        dataset = RandomSampleSubsetPairDataset(
            X = X_tensor,
            sample_size = sample_size,
            subset_size = library_size,
            E = self.num_delays,
            tau = self.delay_step,
            num_batches = num_rand_samples,
            tp_range = tp_range,
            device = self.data_device,
            #random_state = self.random_state,
        )
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                pin_memory=False, 
                                num_workers=0)
        return dataloader

    def _compute_loss_from_dataloader(self, dataloader, num_rand_samples, method,
                                    theta, nbrs_num, exclusion_rad, mask_size):
        """
        Internal helper to accumulate CCM-based loss over all DataLoader batches.
        """
        ccm_loss = 0.0
        for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
            # Move everything to the right device/dtype
            subset_idx = subset_idx.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            sample_idx = sample_idx.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            subset_X   = subset_X.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            subset_y   = subset_y.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            sample_X   = sample_X.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            sample_y   = sample_y.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)

            # Project data with the trained model
            # => shape: [batch, subset_size, E, proj_dim, n_components]
            subset_X_z = self.model(subset_X)
            sample_X_z = self.model(sample_X)
            subset_y_z = self.model(subset_y)
            sample_y_z = self.model(sample_y)
            
            # Flatten the E,proj_dim => single embedding dimension
            # => shape: [batch, subset_size, E*proj_dim, n_components]
            subset_X_z = subset_X_z.reshape(*subset_X_z.shape[:2], -1, subset_X_z.shape[-1])
            sample_X_z = sample_X_z.reshape(*sample_X_z.shape[:2], -1, sample_X_z.shape[-1])
            subset_y_z = subset_y_z.reshape(*subset_y_z.shape[:2], -1, subset_y_z.shape[-1])
            sample_y_z = sample_y_z.reshape(*sample_y_z.shape[:2], -1, sample_y_z.shape[-1])

            # Compute CCM-based loss for this batch
            loss = self.__compute_loss(
                subset_idx, sample_idx,
                sample_X_z, sample_y_z,
                subset_X_z, subset_y_z,
                method, theta, nbrs_num,
                exclusion_rad, mask_size
            )
            # Average across the random draws
            loss /= num_rand_samples
            # Accumulate
            ccm_loss += loss.sum()

        return ccm_loss

    def __compute_loss(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y,
                   method, theta=None, nbrs_num=None, exclusion_rad=0, mask_size=None):
 
        dim = self.n_comp

        if mask_size is not None:
            rand_idx = torch.argsort(torch.rand(dim,device=self.device))[:mask_size]
            sample_X = sample_X[:,:,:,rand_idx]
            sample_y = sample_y[:,:,:,rand_idx]
            subset_X = subset_X[:,:,:,rand_idx]
            subset_y = subset_y[:,:,:,rand_idx]

            dim = mask_size
        
        if method == "smap":
            if theta is None:
                raise ValueError("`theta` must be provided when using the 'smap' method.")
            ccm = self.__get_smap_ccm_matrix_approx(
                subset_idx, sample_idx, 
                sample_X, sample_y, 
                subset_X, subset_y, 
                theta, exclusion_rad
            )
        elif method == "knn":
            if nbrs_num is None:
                raise ValueError("`nbrs_num` must be provided when using the 'nrst_nbrs' method.")
            ccm = self.__get_knn_ccm_matrix_approx(
                subset_idx, sample_idx, 
                sample_X, sample_y, 
                subset_X, subset_y, 
                nbrs_num, exclusion_rad
            )
        elif method == "dcorr":
            if theta is None:
                raise ValueError("`theta` must be provided when using the 'dcorr' method.")
            ccm = self.__get_dcorr_ccm_matrix_approx(
                subset_idx, sample_idx, 
                sample_X, sample_y, 
                subset_X, subset_y, 
                theta, exclusion_rad
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        #Shape: (B, E, dim, dim)
        mask = torch.eye(dim,dtype=bool,device=self.device)

        if self.subtract_autocorr:
            corr = torch.abs(self.__get_autoreg_matrix_approx(sample_X,sample_y))
            if dim > 1:
                score = 1 + (ccm[:,:,~mask]).mean(axis=(1,2)) - (ccm[:,:,mask]).mean(axis=(1,2)) + (corr[:,:,mask]).mean(axis=(1,2))
            else:
                score = 1 + (-ccm[:,:,0,0] + corr[:,:,0,0]).mean(axis=1)
            return score
        else:
            if dim > 1:
                score = 1 + (ccm[:,:,~mask]).mean(axis=(1,2)) - (ccm[:,:,mask]).mean(axis=(1,2)) 
            else:
                score = 1 + (-ccm[:,:,0,0]).mean(axis=1)
            return score
    
    def __compute_h_norm(self):
        num_w = torch.tensor(sum(p.numel() 
                        for p in self.model.parameters() if p.requires_grad), dtype=MODEL_DTYPE)/self.n_comp
        l1 = torch.stack([p.abs().view(self.proj_dim,self.n_comp, -1).sum(axis=(0,2))
                for p in self.model.parameters() if p.requires_grad]).squeeze(0)
        l2 = torch.stack([p.view(self.proj_dim,self.n_comp, -1).norm(2, dim=(0,2))
                for p in self.model.parameters() if p.requires_grad]).squeeze(0)

        h_norm = 1-(torch.sqrt(num_w) - (l1 / (l2 + 1e-6))) / (torch.sqrt(num_w) - 1)

        h_norm = h_norm.mean()  
        return h_norm
    

    def __get_dcorr_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        batch_size, sample_size, dim_x, n_comp = sample_X.shape
        _, subset_size, _, _ = subset_X.shape
        _, _, dim_y, _ = sample_y.shape

        A = subset_X.unsqueeze(-2).expand(batch_size, subset_size, dim_x, n_comp, n_comp)
        a = sample_X.unsqueeze(-2).expand(batch_size, sample_size, dim_x, n_comp, n_comp)
        # [batch, num_points, dim, n_comp, n_comp]

        B = subset_y.unsqueeze(-1).expand(batch_size, subset_size, dim_y, n_comp, n_comp)
        b = sample_y.unsqueeze(-1).expand(batch_size, sample_size, dim_y, n_comp, n_comp)
    
        r_AB = self.__get_batch_distance_corr_weighted(A,a,B,b,theta=theta)
        #r_AB__ = self.__get_batch_distance_corr_weighted(A,a,B,b,theta=theta//2)
        #r_AB_ = self.__get_batch_distance_corr_weighted(A,a,B,b,theta=0)
        return r_AB 
    
    def __get_knn_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
        batch_size, sample_size, dim_x, n_comp = sample_X.shape
        _, subset_size, _, _ = subset_X.shape
        _, _, dim_y, _ = sample_y.shape

        sample_X_t = sample_X.permute(0, 3, 1, 2)
        subset_X_t = subset_X.permute(0, 3, 1, 2)
        
        weights, indices = self.__get_nbrs_indices(subset_X_t,sample_X_t, nbrs_num, subset_idx, sample_idx, exclusion_rad)
        # Shape: [batch, n_comp, sample_size, nbrs_num]

        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1, 1, 1)
        # Shape: [batch, 1, 1, 1]

        selected = subset_y[batch_idx, indices, :, :]
        # [batch, n_comp, sample_size, nbrs_num, dim_y, n_comp]

        result = selected.permute(0, 2, 4, 3, 5, 1)
        # [batch, sample_size, dim_y, nbrs_num, n_comp, n_comp]
        # Start with weights of shape: [B, n_comp, sample_size, nbrs_num]
        w = weights.permute(0, 2, 3, 1)  # Now shape: [B, sample_size, nbrs_num, n_comp]
        w = w.unsqueeze(2).unsqueeze(-2) # Final shape: [B, sample_size, 1, nbrs_num, 1, n_comp] #TODO: w.unsqueeze(2).unsqueeze(-1) ?

        A = (result * w).sum(dim=3) 
        # [batch, num_points, dim, n_comp, n_comp]

        B = sample_y.unsqueeze(-1).expand(batch_size, sample_size, dim_y, n_comp, n_comp)
        
        r_AB = self.metric(A,B)
        return r_AB

    def __get_smap_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        batch_size, sample_size, dim_x, n_comp = sample_X.shape
        _, subset_size, _, _ = subset_X.shape
        _, _, dim_y, _ = sample_y.shape

        sample_X_t = sample_X.permute(0, 3, 1, 2)
        subset_X_t = subset_X.permute(0, 3, 1, 2)
        subset_y_t = subset_y.permute(0, 3, 1, 2)
        #Shape: [batch, comp, points, proj_dim]
        
        weights = self.__get_local_weights(subset_X_t,sample_X_t,subset_idx, sample_idx, exclusion_rad, theta)
        #Shape: [batch, comp, sample_size, subset_size]
        W = (weights.unsqueeze(1)
             .expand(batch_size, n_comp, n_comp, sample_size, subset_size)
             .reshape(batch_size * n_comp * n_comp * sample_size, subset_size, 1))
        #Shape: [batch * {comp} * comp * sample_size, subset_size, 1]

        X = (subset_X_t.unsqueeze(2).unsqueeze(1)
             .expand(batch_size, n_comp, n_comp, sample_size, subset_size, dim_x)
             .reshape(batch_size * n_comp * n_comp * sample_size, subset_size, dim_x))
        #Shape: [batch * {comp} * comp * {sample_size}, subset_size, dim_x]

        Y = (subset_y_t.unsqueeze(2).unsqueeze(2)
             .expand(batch_size, n_comp, n_comp, sample_size, subset_size, dim_y)
             .reshape(batch_size * n_comp * n_comp * sample_size, subset_size, dim_y))
        #Shape: [batch * comp * {comp} * {sample_size}, subset_size, dim_y]

        X_intercept = torch.cat([torch.ones((batch_size * n_comp * n_comp * sample_size, subset_size, 1),device=self.device), X], dim=2)
        #Shape: [batch * comp * comp * sample_size, subset_size, dim_x + 1]

        X_intercept_weighted = X_intercept * W
        Y_weighted = Y * W

        XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
        XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)
        beta = torch.bmm(torch.inverse(XTWX), XTWy)
        #Shape: [batch * comp{y} * comp{x} * sample_size, dim_x + 1, dim_y]

        X_ = (sample_X_t.unsqueeze(1)
              .expand(batch_size, n_comp, n_comp, sample_size, dim_x)
              .reshape(batch_size * n_comp * n_comp * sample_size, dim_x))
        X_ = torch.cat([torch.ones((batch_size * n_comp * n_comp * sample_size, 1),device=self.device), X_], dim=1)
        X_ = X_.reshape(batch_size * n_comp * n_comp * sample_size, 1, dim_x+1)
        
        A = torch.bmm(X_, beta).reshape(batch_size, n_comp, n_comp, sample_size, dim_y)
        A = torch.permute(A,(0,3,4,1,2))

        B = sample_y.unsqueeze(-1).expand(batch_size, sample_size, dim_y, n_comp, n_comp)

        r_AB = self.metric(A,B)
        return r_AB
    
    def __get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        #[batch, comp, points, proj_dim]
        dist = torch.cdist(sublib,lib,compute_mode="use_mm_for_euclid_dist")
        if theta == None:
            weights = torch.exp(-(dist))
        else:
            weights = torch.exp(-(theta*dist/(dist.mean(dim=3, keepdim=True) + 1e-6)))

        if exclusion_rad > 0:
            exclusion_matrix = (torch.abs(subset_idx.unsqueeze(-2) - sample_idx.unsqueeze(-1)) > exclusion_rad)
            weights = weights * exclusion_matrix[:,None]
        
        return weights
    
    def __get_nbrs_indices(self, lib, sublib, n_nbrs, subset_idx, sample_idx, exclusion_rad):
        #[batch, comp, points, proj_dim]
        eps = 1e-6
        dist = torch.cdist(sublib,lib,compute_mode="use_mm_for_euclid_dist")
        exclusion_matrix = torch.where(
            torch.abs(subset_idx.unsqueeze(-2) - sample_idx.unsqueeze(-1)) > exclusion_rad,
            0,
            float('inf')
        ).unsqueeze(1)
        dist = dist + exclusion_matrix

        near_dist, indices = torch.topk(dist, n_nbrs, largest=False)

        # Calculate weights
        near_dist_0 = near_dist[:, :, :, 0][:, :, :, None]
        near_dist_0[near_dist_0 < eps] = eps
        weights = torch.exp(-near_dist / near_dist_0)
        weights = weights / weights.sum(dim=3, keepdim=True)

        return weights, indices
    
    def __get_autoreg_matrix_approx(self, A, B):
        batch_size, _, _, dim = A.shape
        if self.use_delay:
            E = self.proj_dim * self.num_delays
        else:
            E = self.proj_dim
        
        A = A[:,:,:,None,:].expand(batch_size,-1, E, dim, dim)
        B = B[:,:,:,:,None].expand(batch_size,-1, E, dim, dim)

        r_AB = self.metric(A,B)
        return r_AB
    
    def __get_batch_abs_corr(self, A, B):
        mean_A = torch.mean(A,axis=1).unsqueeze(1)
        mean_B = torch.mean(B,axis=1).unsqueeze(1)
        
        sum_AB = torch.sum(torch.abs(A - mean_A) * torch.abs(B - mean_B),axis=1)
        sum_AA = torch.sum(torch.abs(A - mean_A) ** 2,axis=1)
        sum_BB = torch.sum(torch.abs(B - mean_B) ** 2,axis=1)

        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB   

    def __get_batch_corr(self, A, B):
        mean_A = torch.mean(A,axis=1).unsqueeze(1)
        mean_B = torch.mean(B,axis=1).unsqueeze(1)
        
        sum_AB = torch.sum((A - mean_A) * (B - mean_B),axis=1)
        sum_AA = torch.sum((A - mean_A) ** 2,axis=1)
        sum_BB = torch.sum((B - mean_B) ** 2,axis=1)

        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB    
    
    def __get_batch_ccc(self, A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12, unbiased: bool = True):
        """
        Concordance Correlation Coefficient (Lin's CCC) between A and B.

        Parameters
        ----------
        A, B : torch.Tensor
            Shape [batch, samples, dim, comp, comp].

        Returns
        -------
        torch.Tensor
            Shape [batch, 1, comp, comp].
        """
        # Flatten samples and dim -> observations axis
        # Resulting shape: [batch, N, comp, comp], where N = samples * dim
        A2 = A.flatten(start_dim=1, end_dim=2)
        B2 = B.flatten(start_dim=1, end_dim=2)
        N = A2.shape[1]

        # Means along observations
        mu_A = A2.mean(dim=1, keepdim=True)            # [batch, 1, comp, comp]
        mu_B = B2.mean(dim=1, keepdim=True)            # [batch, 1, comp, comp]

        # Variances and covariance along observations
        if unbiased and N > 1:
            var_A = ((A2 - mu_A) ** 2).sum(dim=1, keepdim=True) / (N - 1)
            var_B = ((B2 - mu_B) ** 2).sum(dim=1, keepdim=True) / (N - 1)
            cov_AB = (((A2 - mu_A) * (B2 - mu_B)).sum(dim=1, keepdim=True)) / (N - 1)
        else:
            var_A = ((A2 - mu_A) ** 2).mean(dim=1, keepdim=True)
            var_B = ((B2 - mu_B) ** 2).mean(dim=1, keepdim=True)
            cov_AB = ((A2 - mu_A) * (B2 - mu_B)).mean(dim=1, keepdim=True)

        # Lin's CCC
        denom = var_A + var_B + (mu_A - mu_B).pow(2)
        ccc = (2.0 * cov_AB) / (denom + eps)           # [batch, 1, comp, comp]

        return ccc
    
    def __get_batch_distance_concordance(self, A, B, eps: float = 1e-8, signed: bool = False):
        """
        Distance Concordance Correlation Coefficient (dCCC) between A and B.
        An analog of Lin's CCC using distance covariance/variance plus a mean-difference penalty.

        Parameters
        ----------
        A, B : torch.Tensor
            Shape [batch, samples, dim, comp, comp].
        eps : float
            Numerical stability.
        signed : bool
            If True, multiply by the sign of (Pearson) correlation between A and B
            so the coefficient can be negative for anti-concordance.

        Returns
        -------
        torch.Tensor
            Shape [batch, 1, comp, comp].
        """
        import torch
        batch, samples, dim, comp, _ = A.shape
        device = A.device
        dtype = A.dtype

        # Flatten to [batch*comp*comp, samples, dim]
        A_flat = A.permute(0, 3, 4, 1, 2).reshape(batch * comp * comp, samples, dim)
        B_flat = B.permute(0, 3, 4, 1, 2).reshape(batch * comp * comp, samples, dim)

        # Sample means across the "observations" axis (samples)
        mu_A = A_flat.mean(dim=1)                         # [N, dim]
        mu_B = B_flat.mean(dim=1)                         # [N, dim]
        mean_diff_sq = (mu_A - mu_B).pow(2).sum(dim=1)    # [N]

        # Pairwise distances and double-centering (same as your dCor)
        def pairwise_distances(X):
            return torch.cdist(X, X, p=2, compute_mode="use_mm_for_euclid_dist")  # [N, samples, samples]

        def double_center(D):
            mean_row = D.mean(dim=-1, keepdim=True)
            mean_col = D.mean(dim=-2, keepdim=True)
            mean_all = D.mean(dim=(-1, -2), keepdim=True)
            return D - mean_row - mean_col + mean_all

        A_dc = double_center(pairwise_distances(A_flat))
        B_dc = double_center(pairwise_distances(B_flat))

        # Distance covariance/variance (biased, differentiable, nonnegative up to num. noise)
        dCov   = (A_dc * B_dc).mean(dim=(-1, -2))         # [N]
        dVar_A = (A_dc.pow(2)).mean(dim=(-1, -2))         # [N]
        dVar_B = (B_dc.pow(2)).mean(dim=(-1, -2))         # [N]

        # dCCC numerator/denominator
        denom = dVar_A + dVar_B + mean_diff_sq + eps       # [N]
        dccc = (2.0 * dCov) / denom                        # [N]

        # Optional signed version (attach a sign like CCC can be negative)
        if signed:
            A0 = A_flat - mu_A.unsqueeze(1)                # [N, samples, dim]
            B0 = B_flat - mu_B.unsqueeze(1)
            num = (A0 * B0).mean(dim=(1, 2))               # inner product mean
            den = torch.sqrt(A0.pow(2).mean(dim=(1, 2)) * B0.pow(2).mean(dim=(1, 2)) + eps)
            sign = torch.sign(num / den)                   # [-1, 0, 1]
            dccc = dccc * sign

        # Handle degenerate identical-constant case: define dCCC=1
        deg = (dVar_A + dVar_B + mean_diff_sq) < eps
        if torch.any(deg):
            dccc = dccc.clone()
            dccc[deg] = torch.tensor(1.0, dtype=dtype, device=device)

        # Reshape back to [batch, 1, comp, comp]
        dccc = dccc.reshape(batch, comp, comp).unsqueeze(1)
        return dccc

    def save(self, filepath: str):
        """
        Saves the model, optimizer state, and other attributes to disk.
        """
        # 1) Collect constructor / hyperparams needed to re-initialize
        init_params = {
            "input_dim": self.input_dim,
            "proj_dim": self.proj_dim,
            "n_components": self.n_comp,
            "num_delays": self.num_delays,
            "delay_step": self.delay_step,
            "model": "linear" if isinstance(self.model, LinearModel) else "nonlinear",
            "subtract_autocorr": self.subtract_autocorr,
            "device": self.device,       # might be overridden on load
            "data_device": self.data_device,
            "optimizer": type(self.optimizer).__name__,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "random_state": self.random_state
        }

        # 2) Create checkpoint dictionary
        checkpoint = {
            "init_params": init_params,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
        }

        # 3) Save checkpoint
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, map_location: str = "cpu"):
        """
        Loads a saved model from disk.
        Args:
            filepath (str): Path to the saved file.
            map_location (str): Device mapping for the checkpoint (e.g. "cpu" or "cuda").
        Returns:
            FlowDecomposition: A new FlowDecomposition instance with loaded state.
        """
        # 1) Load checkpoint
        checkpoint = torch.load(filepath, map_location=map_location)
        init_params = checkpoint["init_params"]

        # 2) Instantiate a new instance with the stored hyperparams
        #    Note: we override `device` in init_params so that the new instance
        #    is created on `map_location`, if desired. 
        init_params["device"] = map_location
        init_params["data_device"] = map_location

        new_instance = cls(**init_params)

        # 3) Load model and optimizer states
        new_instance.model.load_state_dict(checkpoint["model_state"])
        new_instance.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # 4) Restore any extra info such as training history
        new_instance.loss_history = checkpoint["loss_history"]

        print(f"Model loaded from {filepath}")
        return new_instance
    
    def __get_batch_distance_corr(self, A, B, eps=1e-8):
        """
        Differentiable distance correlation for batches.

        Parameters
        ----------
        A, B : torch.Tensor
            Shape [batch, samples, dim, comp, comp].

        eps : float
            Small value to avoid division by zero.

        Returns
        -------
        torch.Tensor
            Shape [batch, 1, comp, comp].
        """
        batch, samples, dim, comp, _ = A.shape

        # flatten to [batch * comp * comp, samples, dim]
        A_flat = (
            A.permute(0, 3, 4, 1, 2)
            .reshape(batch * comp * comp, samples, dim)
        )
        B_flat = (
            B.permute(0, 3, 4, 1, 2)
            .reshape(batch * comp * comp, samples, dim)
        )

        def pairwise_distances(X):
            """Compute pairwise Euclidean distances."""
            D = torch.cdist(X, X, p=2,compute_mode="use_mm_for_euclid_dist")
            return D

        def double_center(D):
            """Double-center distance matrix."""
            mean_row = D.mean(dim=-1, keepdim=True)
            mean_col = D.mean(dim=-2, keepdim=True)
            mean_all = D.mean(dim=(-1, -2), keepdim=True)
            return D - mean_row - mean_col + mean_all

        # compute pairwise distances
        A_dist = pairwise_distances(A_flat)
        B_dist = pairwise_distances(B_flat)

        # double centering
        A_dc = double_center(A_dist)
        B_dc = double_center(B_dist)
    
        # compute dCov and dVar
        dCov = (A_dc * B_dc).mean(dim=(-1, -2))
        dVar_A = (A_dc ** 2).mean(dim=(-1, -2))
        dVar_B = (B_dc ** 2).mean(dim=(-1, -2))

        # calculate distance correlation
        dCor = dCov / (torch.sqrt(dVar_A * dVar_B) + eps)

        # reshape back to [batch, 1, comp, comp]
        dCor = dCor.reshape(batch, comp, comp).unsqueeze(1)

        return dCor

    def __get_batch_distance_corr_weighted1(
            self,
            A: torch.Tensor,a: torch.Tensor,
            B: torch.Tensor,b: torch.Tensor,
            theta: float = 1.0,
            eps: float = 1e-8,
        ):
        """
        Locally-weighted distance correlation for batched inputs.

        Parameters
        ----------
        A, B : torch.Tensor
            Shape [batch, samples, dim, comp, comp].

        theta : float
            Controls how fast the weight decays with distance.  
            Larger theta ⇒ stronger emphasis on near-by samples.

        eps : float
            Numerical jitter.

        Returns
        -------
        torch.Tensor
            Shape [batch, 1, comp, comp].
        """
        batch, samples, dim, comp, _ = a.shape
        batch, subset, dim, comp, _ = A.shape

        # flatten to [batch * comp * comp, samples, dim]
        A_flat = (
            A.permute(0, 3, 4, 1, 2)
            .reshape(batch * comp * comp, subset, dim)
        )

        a_flat = (
            a.permute(0, 3, 4, 1, 2)
            .reshape(batch * comp * comp, samples, dim)
        )
        
        B_flat = (
            B.permute(0, 3, 4, 1, 2)
            .reshape(batch * comp * comp, subset, dim)
        )

        b_flat = (
            b.permute(0, 3, 4, 1, 2)
            .reshape(batch * comp * comp, samples, dim)
        )

        def pairwise_distances(X,Y):
            """Pairwise Euclidean distances."""
            return torch.cdist(
                X, Y, p=2, compute_mode="use_mm_for_euclid_dist"
            )

        def double_center(D):
            """Gower double-centering."""
            mean_row = D.mean(dim=-1, keepdim=True)
            mean_col = D.mean(dim=-2, keepdim=True)
            mean_all = D.mean(dim=(-1, -2), keepdim=True)
            return D - mean_row - mean_col + mean_all

        # ── pairwise distances ────────────────────────────────────────────────────
        A_dist = pairwise_distances(a_flat, A_flat)
        B_dist = pairwise_distances(b_flat, B_flat)

        # You can base the weights on A_dist, B_dist, or any other metric.
        # Using the symmetric average is a cheap, sensible default:

        # (optionally) set self-weights to zero to ignore diagonal terms
        # weights = weights.fill_diagonal_(0.)
        dist_base = B_dist
        avg_dist = dist_base.mean(dim=(-1, -2), keepdim=True)
        weights = torch.exp(-theta * dist_base / (avg_dist + eps))

        # ── double centering ─────────────────────────────────────────────────────
        A_dc = double_center(A_dist)
        B_dc = double_center(B_dist)

        # ── weighted statistics ──────────────────────────────────────────────────
        w_sum = weights.sum(dim=(-1, -2), keepdim=False) + eps

        dCov   = (weights * A_dc * B_dc).sum(dim=(-1, -2)) / w_sum
        dVar_A = (weights * A_dc.pow(2)).sum(dim=(-1, -2)) / w_sum
        dVar_B = (weights * B_dc.pow(2)).sum(dim=(-1, -2)) / w_sum

        # ── distance correlation ─────────────────────────────────────────────────
        dCor = dCov / (torch.sqrt(dVar_A * dVar_B) + eps)

        # reshape back to [batch, 1, comp, comp]
        dCor = dCor.reshape(batch, comp, comp).unsqueeze(1)
        return dCor
    

    def __get_batch_distance_corr_weighted(
            self,
            A: torch.Tensor, a: torch.Tensor,
            B: torch.Tensor, b: torch.Tensor,
            theta: float = 1.0,
            eps: float = 1e-8,
            subtract_unweighted: bool = True,
        ):

        batch, samples, dim, comp, _ = a.shape
        batch, subset,  _,   _,  _ = A.shape

        def _flat(x):  
            return x.permute(0, 3, 4, 1, 2).reshape(batch*comp*comp, -1, dim)

        a_f, A_f, b_f, B_f = map(_flat, (a, A, b, B))

        pairwise = lambda X, Y: torch.cdist(X, Y, p=2,
                                            compute_mode="use_mm_for_euclid_dist")

        def double_center(D):
            mr, mc = D.mean(-1, True), D.mean(-2, True)
            return D - mr - mc + D.mean((-1, -2), True)

        A_dist, B_dist = pairwise(a_f, A_f), pairwise(b_f, B_f)

        dist_base   = B_dist                    
        avg_dist    = dist_base.mean((-1, -2), True)
        W_local     = torch.exp(-theta * dist_base / (avg_dist + eps))

        A_dc, B_dc = map(double_center, (A_dist, B_dist))

        def _dcor(W):
            w_sum   = W.sum((-1, -2)) + eps
            dCov    = (W * A_dc * B_dc).sum((-1, -2)) / w_sum
            dVar_A  = (W * A_dc.pow(2)).sum((-1, -2)) / w_sum
            dVar_B  = (W * B_dc.pow(2)).sum((-1, -2)) / w_sum   
            return dCov / (torch.sqrt(dVar_A * dVar_B) + eps)

        dCor_w = _dcor(W_local)

        if subtract_unweighted:
            W_uniform = torch.ones_like(W_local)

            dCor_u    = _dcor(W_uniform)
            dCor_u = dCor_u.reshape(batch, comp, comp).unsqueeze(1)
            dCor_w = dCor_w.reshape(batch, comp, comp).unsqueeze(1)
            eye = torch.eye(comp, device=dCor_w.device).view(1, 1, comp, comp)
            dCor = dCor_w - dCor_u * eye
        else:
            dCor      = dCor_w.reshape(batch, comp, comp).unsqueeze(1)

        # reshape to [batch, 1, comp, comp]
        return dCor