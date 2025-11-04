from flow_decomposition.utils.models import LinearModel, NonlinearModel
from flow_decomposition.utils.data_samplers import RandomSampleSubsetPairDataset
from flow_decomposition.utils.metrics import get_metric

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
            mask_size=None,include_opposite_tp=False):
        """
        Fit the model using the provided data.
        """
        # Create a DataLoader once, reuse it each epoch

        if self.random_state is not None:
            old_rng_state = torch.get_rng_state()
            torch.manual_seed(self.random_state) #TODO: Not safe for multiple workers

        dataloader = self._create_dataloader(X, sample_size, library_size,
                                            time_intv, num_rand_samples, batch_size,
                                            optim_policy,include_opposite_tp=include_opposite_tp)
        
        self.metric_name = metric
        self.metric = get_metric(metric) 

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
                      optim_policy="range", mask_size=None,include_opposite_tp=False):
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
                                                optim_policy,
                                                include_opposite_tp)
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
    
    def transform(self, X, device="cpu", pca: bool = False):
        """
        Returns embeddings with shape [N, n_components, proj_dim].
        If pca=True, apply PCA independently for each component along the projection dimension.
        """
        with torch.no_grad():
            inputs = torch.as_tensor(X, dtype=MODEL_DTYPE, device=device)

            # Model outputs [N, D, C]
            Z_raw = self.model.to(device)(inputs)

            if not pca:
                # Keep your original transpose: -> [N, C, D]
                return torch.permute(Z_raw, (0, 2, 1)).cpu().numpy()

            # --- per-component PCA over proj_dim ---
            N, D, C = Z_raw.shape
            Zp_raw = torch.empty_like(Z_raw, dtype=torch.float64)  # work in float64 for SVD/eigh
            Z64 = Z_raw.to(torch.float64)

            for c in range(C):
                Xc = Z64[:, :, c]                      # [N, D]  (component c)
                Xc = Xc - Xc.mean(0, keepdim=True)     # center (translation only)

                # Full D×D orthonormal basis via covariance eigendecomposition
                Ccov = (Xc.T @ Xc) / max(N - 1, 1)     # [D, D]
                evals, V = torch.linalg.eigh(Ccov)     # columns are eigvecs (ascending)
                V = V[:, torch.argsort(evals, descending=True)]  # sort by variance desc

                # Scores in PCA coords, shape [N, D]
                Zp_raw[:, :, c] = (Xc @ V)

            # Back to your public shape: [N, C, D]
            return torch.permute(Zp_raw.to(Z_raw.dtype), (0, 2, 1)).cpu().numpy()

    def _create_dataloader(self, X, sample_size, library_size, 
                       time_intv, num_rand_samples, batch_size, 
                       optim_policy,include_opposite_tp=False):
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
            include_opposite = include_opposite_tp
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
            #corr = torch.abs(self.__get_autoreg_matrix_approx(sample_X,sample_y))

            #corr = (self.__get_autoreg_matrix_approx(sample_X,sample_y))
            corr = torch.abs(self.__get_autoreg_matrix_approx(subset_X, subset_y, sample_X, sample_y))
            #print(torch.stack([(corr[:,:,mask]).mean(axis=(1,2)).detach(),(ccm[:,:,mask]).mean(axis=(1,2)).detach()]).T)
            #print(ccm.mean(axis=(1))[0].detach())
            if dim > 1:
                score = 1 + (ccm[:,:,~mask]).mean(axis=(1,2)) - (ccm[:,:,mask]).mean(axis=(1,2)) + (corr[:,:,mask]).mean(axis=(1,2))
                #score = ((ccm[:,:,~mask]).mean(axis=(1,2)) + (corr[:,:,mask]).mean(axis=(1,2)))/(2 * ccm[:,:,mask]).mean(axis=(1,2))
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


        #r = l2 + 1e-6
        #r = r.detach()           
        #h_norm = (r ** 1) * h_norm
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

    def __get_knnloao_ccm_matrix_approx1(
        self,
        subset_idx, sample_idx,
        sample_X, sample_y,
        subset_X, subset_y,
        nbrs_num, exclusion_rad
    ):
        """
        LOAO KNN-CCM approximation.

        Shapes (same as your current method):
        sample_X: [B, S, dim_x, n_comp]
        subset_X: [B, T, dim_x, n_comp]
        sample_y: [B, S, dim_y, n_comp]
        subset_y: [B, T, dim_y, n_comp]

        Returns:
        r_AB: [B, 1, n_comp, n_comp]
        """

        device = self.device

        Bsz, S, dim_x, n_comp = sample_X.shape
        _,  T, _, _            = subset_X.shape
        _, _, dim_y, _         = sample_y.shape

        # --- 1) Base (full-dim) KNN for all pairs; we'll keep off-diagonals from this ---
        sample_X_t = sample_X.permute(0, 3, 1, 2)  # [B, n_comp, S, dim_x]
        subset_X_t = subset_X.permute(0, 3, 1, 2)  # [B, n_comp, T, dim_x]

        weights_full, indices_full = self.__get_nbrs_indices(
            subset_X_t, sample_X_t, nbrs_num, subset_idx, sample_idx, exclusion_rad
        )  # [B, n_comp, S, K]

        batch_idx = torch.arange(Bsz, device=device).view(Bsz, 1, 1, 1)  # [B,1,1,1]
        selected_full = subset_y[batch_idx, indices_full, :, :]          # [B, n_comp, S, K, dim_y, n_comp]

        # shape -> [B, S, dim_y, K, n_comp(tgt), n_comp(src)]
        result_full = selected_full.permute(0, 2, 4, 3, 5, 1)

        # weights -> [B, S, 1, K, 1, n_comp(src)]
        w_full = weights_full.permute(0, 2, 3, 1).unsqueeze(2).unsqueeze(-2)

        # A_full: [B, S, dim_y, n_comp(tgt), n_comp(src)]
        A_full = (result_full * w_full).sum(dim=3)

        # Start from full predictions; we'll overwrite the diagonal with LOAO results.
        A = A_full.clone()

        # --- 2) LOAO for diagonal (i == j): repeat for every input axis d ---
        # We only overwrite A[:, :, d, i, i] using neighbors computed without axis d.
        # (If dim_y != dim_x, we only fill the min(dim_x, dim_y) leading dims.)
        nd_fill = min(dim_x, dim_y)

        for d in range(nd_fill):
            # Zero out axis d to "remove" it from distance computations.
            # (Zeroing in both query and candidate sets makes that axis contribute 0 to all pairwise distances.)
            sample_X_mask = sample_X_t.clone()
            subset_X_mask = subset_X_t.clone()
            sample_X_mask[..., d] = 0.0
            subset_X_mask[..., d] = 0.0

            weights_d, indices_d = self.__get_nbrs_indices(
                subset_X_mask, sample_X_mask, nbrs_num, subset_idx, sample_idx, exclusion_rad
            )  # [B, n_comp, S, K]

            # Gather ONLY the d-th output dimension for efficiency: [:, :, :, :, d:d+1, :]
            selected_d = subset_y[batch_idx, indices_d, d:d+1, :]  # [B, n_comp, S, K, 1, n_comp]

            # -> [B, S, 1, K, n_comp(tgt), n_comp(src)]
            result_d = selected_d.permute(0, 2, 4, 3, 5, 1)

            # weights -> [B, S, 1, K, 1, n_comp(src)]
            w_d = weights_d.permute(0, 2, 3, 1).unsqueeze(2).unsqueeze(-2)

            # A_d_full: [B, S, 1, n_comp(tgt), n_comp(src)]
            A_d_full = (result_d * w_d).sum(dim=3)

            # We only want diagonal (i==j): take diag over the last two dims.
            # A_d_diag: [B, S, 1, n_comp]  (value for each component i on its own diagonal slot)
            A_d_diag = A_d_full.diagonal(offset=0, dim1=3, dim2=4)  # diag over (tgt, src)

            # Write into A[:, :, d, i, i] for all i.
            # Loop is simple & safe (avoids tricky advanced-index assign semantics).
            for i in range(n_comp):
                # A_d_diag[..., i] is [B, S, 1]
                A[:, :, d, i, i] = A_d_diag[..., i].squeeze(-1)

        # --- 3) Compare to ground truth and return ---
        # B: [B, S, dim_y, n_comp, n_comp] (broadcasted along last axis)
        B = sample_y.unsqueeze(-1).expand(Bsz, S, dim_y, n_comp, n_comp)

        r_AB = self.metric(A, B)  # [B, 1, n_comp, n_comp]
        return r_AB

    def __get_smap_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad,ridge=0.0):
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
        if ridge and ridge > 0.0:
            I = torch.eye(dim_x + 1, device=XTWX.device, dtype=XTWX.dtype).unsqueeze(0).expand(XTWX.size(0), -1, -1)
            I = I.clone()
            I[:, 0, 0] = 0.0
            XTWX = XTWX + ridge * I
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
    
    def __get_autoreg_matrix_approx1(self, A, B):
        batch_size, _, _, dim = A.shape
        if self.use_delay:
            E = self.proj_dim * self.num_delays
        else:
            E = self.proj_dim
        
        A = A[:,:,:,None,:].expand(batch_size,-1, E, dim, dim)
        B = B[:,:,:,:,None].expand(batch_size,-1, E, dim, dim)

        r_AB = self.metric(A,B)
        return r_AB
    
    def __get_autoreg_matrix_approx(self,
                            subset_X, subset_y,
                            sample_X, sample_y,
                            ridge: float = 0.0,
                            use_intercept: bool = True):
        """
        Fit global linear map on the LIBRARY, predict the SAMPLES.
        Shapes:
        subset_X: [B, L, Dx, Cx]
        subset_y: [B, L, Dy, Cy]
        sample_X: [B, S, Dx, Cx]
        sample_y: [B, S, Dy, Cy]
        Returns: [B, 1, Cy, Cx] via self.metric(pred, sample_y)
        """
        Bch, L, Dx, Cx = subset_X.shape
        _,  _, Dy, Cy = subset_y.shape
        S = sample_X.shape[1]

        # Train design: [B, Cx, L, Dx], Targets: [B, Cy, L, Dy]
        Xtr = subset_X.permute(0, 3, 1, 2)
        Ytr = subset_y.permute(0, 3, 1, 2)

        if use_intercept:
            ones_tr = torch.ones(Bch, Cx, L, 1, device=subset_X.device, dtype=subset_X.dtype)
            Xtr = torch.cat([ones_tr, Xtr], dim=-1)
            Dx1 = Dx + 1
        else:
            Dx1 = Dx

        # Normal equations per (Cy, Cx)
        XtX = torch.matmul(Xtr.transpose(-2, -1), Xtr)  # [B, Cx, Dx1, Dx1]
        if ridge > 0:
            I = torch.eye(Dx1, device=Xtr.device, dtype=Xtr.dtype).view(1,1,Dx1,Dx1)
            XtX = XtX + ridge * I

        XtY = torch.matmul(                                   # [B, Cy, Cx, Dx1, Dy]
            Xtr.transpose(-2, -1).unsqueeze(1),               # [B, 1,  Cx, Dx1, L]
            Ytr.unsqueeze(2)                                  # [B, Cy, 1,  L,   Dy]
        )
        W = torch.linalg.solve(                               # [B, Cy, Cx, Dx1, Dy]
            XtX.unsqueeze(1).expand(-1, Cy, -1, -1, -1),
            XtY
        )

        # Predict on samples
        Xte = sample_X.permute(0, 3, 1, 2)                    # [B, Cx, S, Dx]
        if use_intercept:
            ones_te = torch.ones(Bch, Cx, S, 1, device=sample_X.device, dtype=sample_X.dtype)
            Xte = torch.cat([ones_te, Xte], dim=-1)          # [B, Cx, S, Dx1]

        Y_hat = torch.matmul(Xte.unsqueeze(1), W)             # [B, Cy, Cx, S, Dy]

        # Score with the same metric as CCM: expects [B, S, D, C, C]
        B_pred = Y_hat.permute(0, 3, 4, 1, 2)                 # [B, S, Dy, Cy, Cx]
        B_true = sample_y.unsqueeze(-1).expand(Bch, S, Dy, Cy, Cx)
        return self.metric(B_pred, B_true)                    # [B, 1, Cy, Cx]

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