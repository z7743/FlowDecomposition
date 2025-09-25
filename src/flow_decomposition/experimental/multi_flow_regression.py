from flow_decomposition.utils.models import LinearModel, NonlinearModel
from flow_decomposition.utils.data_samplers import RandomSampleSubsetPairDataset
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from flow_decomposition.utils.metrics import get_metric


DATA_DTYPE = torch.float32
MODEL_DTYPE = torch.float32

class MultiFlowRegression:

    def __init__(self, input_dim, proj_dim, n_components,
                 num_delays=None, delay_step=None, model="linear", subtract_autocorr=False,  tol=None,
                 device="cuda",data_device="cpu", optimizer="Adagrad", learning_rate=0.01, random_state=None, verbose=1):
        """
        Initializes the FlowRegression model.
        
        Args:
        """
        self.device = device
        self.input_dim = input_dim
        self.data_device = data_device
        self.random_state = random_state
        self.proj_dim = proj_dim
        self.n_comp = n_components
        self.num_delays = num_delays
        self.delay_step = delay_step
        self.subtract_autocorr = subtract_autocorr
        self.loss_history = []
        self.verbose = verbose
        self.tol = tol

        if model == "linear":
            self.model = LinearModel(input_dim=input_dim, proj_dim=proj_dim, n_comp=n_components, device=device,random_state=random_state)
        elif model == "nonlinear":
            self.model = NonlinearModel(input_dim=input_dim, proj_dim=proj_dim, n_comp=n_components, device=device,random_state=random_state)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        self.use_delay = bool(num_delays) and bool(delay_step)


        self.component_mixing = torch.nn.Sequential(torch.nn.Linear(n_components, 1, bias=False)).to(device)

        self.optimizer = getattr(optim, optimizer)(list(self.model.parameters()) + list(self.component_mixing.parameters()), lr=learning_rate)

        self.metric_name = None
        self.metric = None

    def fit(self, X, Y,
            sample_size, library_size, exclusion_rad=0, 
            method="knn",metric="corr",
            theta=None, nbrs_num=None, time_intv=1, 
            num_epochs=100, num_rand_samples=32, batch_size=1, beta=0,
            optim_policy="range"):
        """
        Fit the model using the provided data.
        """

        if self.random_state is not None:
            old_rng_state = torch.get_rng_state()
            torch.manual_seed(self.random_state) #TODO: Not safe for multiple workers

        dataloader = self._create_dataloader(
            X, Y,
            sample_size=sample_size,
            library_size=library_size,
            time_intv=time_intv,
            num_rand_samples=num_rand_samples,
            optim_policy=optim_policy,
            batch_size=batch_size
        )

        self.metric_name = metric
        self.metric = None if metric == "bce" else get_metric(metric)
        
        ema = None
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            ccm_loss = self._compute_loss_from_dataloader(
                dataloader, num_rand_samples, method, theta, nbrs_num, exclusion_rad
            )
            
            h_norm_val = self.__compute_h_norm()
            total_loss = ccm_loss + beta * h_norm_val

            # Backprop and update
            total_loss.backward()
            self.optimizer.step()

            if self.verbose == 1:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"CCM Loss: {ccm_loss.item():.4f}, "
                    f"h_norm: {h_norm_val.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}"
                )

            
            if self.tol is not None:
                cur = total_loss.item()
                if ema is None:
                    ema = cur
                else:
                    delta = abs(cur - ema)
                    if delta < self.tol * max(1.0, abs(ema)):
                        if self.verbose:
                            print(f"Early-stopped at epoch {epoch+1} "
                                  f"(Δloss={delta:.3g} < tol={self.tol})")
                        break
                    ema = 0.1 * cur + 0.9 * ema

            self.loss_history.append(total_loss.item())

        if self.random_state is not None:
            torch.set_rng_state(old_rng_state)

    def evaluate_loss(self, X_test, Y_test,
                      sample_size, library_size, exclusion_rad=0, method="knn",metric="corr",
                      theta=None, nbrs_num=None, time_intv=1, 
                      num_epochs=None, num_rand_samples=32, batch_size=1, beta=None,
                      optim_policy="range"):
        """
        Computes only the raw CCM loss on a test set. Importantly, this method does not 
        calculate predictions based on the training data. Instead, it builds the model using 
        the test data and calculates the loss as if it were during the training cycle.
        num_epochs and beta are included for compatibility with the fit method, but are not used.

        Returns:
            float: The total CCM loss on the test dataset.
        """

        if self.random_state is not None:
            old_rng_state = torch.get_rng_state()
            torch.manual_seed(self.random_state) #TODO: Not safe for multiple workers

        self.metric_name = metric
        self.metric = None if metric == "bce" else get_metric(metric)

        with torch.no_grad():
            # Create dataloader for test data
            dataloader = self._create_dataloader(
                X_test, Y_test,
                sample_size=sample_size,
                library_size=library_size,
                time_intv=time_intv,
                num_rand_samples=num_rand_samples,
                optim_policy=optim_policy,
                batch_size=batch_size
            )

            ccm_loss = self._compute_loss_from_dataloader(
                dataloader, num_rand_samples, method, theta, nbrs_num, exclusion_rad
            )
        
        if self.random_state is not None:
            torch.set_rng_state(old_rng_state)

        return ccm_loss.item()

    def transform(self, X, device="cpu", pca: bool = False):
        """
        Returns embeddings with shape [N, n_components, proj_dim].
        If pca=True, apply PCA independently per component along the projection dimension
        (same implementation as FlowDecomposition).
        """
        with torch.no_grad():
            inputs = torch.as_tensor(X, dtype=MODEL_DTYPE, device=device)
            Z_raw = self.model.to(device)(inputs)            # [N, D, C]
            if not pca:
                return torch.permute(Z_raw, (0, 2, 1)).cpu().numpy()

            N, D, C = Z_raw.shape
            Z64 = Z_raw.to(torch.float64)
            Zp_raw = torch.empty_like(Z64)
            for c in range(C):
                Xc = Z64[:, :, c]
                Xc = Xc - Xc.mean(0, keepdim=True)
                Ccov = (Xc.T @ Xc) / max(N - 1, 1)
                evals, V = torch.linalg.eigh(Ccov)
                V = V[:, torch.argsort(evals, descending=True)]
                Zp_raw[:, :, c] = Xc @ V
            return torch.permute(Zp_raw.to(Z_raw.dtype), (0, 2, 1)).cpu().numpy()

    def save(self, filepath: str):
        init_params = {
            "input_dim": self.input_dim,               
            "proj_dim": self.proj_dim,
            "n_components": self.n_comp_x,
            "num_delays": self.num_delays,
            "delay_step": self.delay_step,
            "model": "linear" if isinstance(self.model, LinearModel) else (
                     "nonlinear" if isinstance(self.model, NonlinearModel) else "custom"),
            "subtract_autocorr": self.subtract_autocorr,
            "regularizer": self.regularizer,
            "tol": self.tol,
            "device": self.device,
            "data_device": self.data_device,
            "optimizer": type(self.optimizer).__name__,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "random_state": self.random_state,
        }
        checkpoint = {
            "init_params": init_params,
            "model_state": self.model.state_dict(),
            "mix_state": self.component_mixing.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
        }
        torch.save(checkpoint, filepath)
        if self.verbose:
            print(f"MultiFlowRegression saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, map_location: str = "cpu"):
        checkpoint = torch.load(filepath, map_location=map_location)
        init_params = checkpoint["init_params"]
        init_params["device"] = map_location
        init_params["data_device"] = map_location

        new = cls(**init_params)

        new.model.load_state_dict(checkpoint["model_state"])
        new.component_mixing.load_state_dict(checkpoint["mix_state"])
        new.optimizer.load_state_dict(checkpoint["optimizer_state"])
        new.loss_history = checkpoint.get("loss_history", [])
        print(f"MultiFlowRegression loaded from {filepath}")
        return new



    def _create_dataloader(self, X, Y, sample_size, library_size,
                           time_intv, num_rand_samples, optim_policy, batch_size):
        """
        Creates a DataLoader for the given dataset and parameters.
        """
        if optim_policy == "fixed":
            tp_range = (time_intv, time_intv)
        elif optim_policy == "range":
            tp_range = (0, time_intv)
        else:
            raise ValueError(f"Unknown optim_policy: {optim_policy}")

        X_tensor = torch.tensor(X, requires_grad=False, device=self.data_device, dtype=DATA_DTYPE)
        Y_tensor = torch.tensor(Y, requires_grad=False, device=self.data_device, dtype=DATA_DTYPE)

        dataset = RandomSampleSubsetPairDataset(
            X=X_tensor,
            y=Y_tensor,
            sample_size=sample_size,
            subset_size=library_size,
            E=self.num_delays,
            tau=self.delay_step,
            num_batches=num_rand_samples,
            tp_range=tp_range,
            device=self.data_device,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=0
        )
        return dataloader

    def _compute_loss_from_dataloader(self, dataloader, num_rand_samples, method, theta, nbrs_num, exclusion_rad):
        """
        Computes only the CCM-based loss over the batches.
        """
        total_loss = 0.0
        for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
            # Move tensors to the appropriate device and type.
            subset_idx = subset_idx.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            sample_idx = sample_idx.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            subset_X = subset_X.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            subset_y = subset_y.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            sample_X = sample_X.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            sample_y = sample_y.to(self.device, dtype=MODEL_DTYPE, non_blocking=True)
            
            # Compute embeddings for the subset and sample
            subset_X_z = self.model(subset_X)
            sample_X_z = self.model(sample_X)
            
            # Use the last element of y as in the fit method.
            subset_y_z = subset_y[:, :, [-1]].unsqueeze(-1)
            sample_y_z = sample_y[:, :, [-1]].unsqueeze(-1)
            
            # Reshape to match expected dimensions:
            subset_X_z = subset_X_z.reshape(subset_X_z.size(0), subset_X_z.size(1), -1, subset_X_z.size(4))
            sample_X_z = sample_X_z.reshape(sample_X_z.size(0), sample_X_z.size(1), -1, sample_X_z.size(4))
            subset_y_z = subset_y_z.reshape(subset_y_z.size(0), subset_y_z.size(1), -1, subset_y_z.size(4))
            sample_y_z = sample_y_z.reshape(sample_y_z.size(0), sample_y_z.size(1), -1, sample_y_z.size(4))
            
            # Compute the CCM-based loss for this batch.
            loss = self.__compute_loss(
                subset_idx, sample_idx,
                sample_X_z, sample_y_z,
                subset_X_z, subset_y_z,
                method, theta, nbrs_num, exclusion_rad
            )
            # Divide by num_rand_samples to average across random draws
            loss /= num_rand_samples
            total_loss += loss.sum()
        
        return total_loss
    
    def __compute_loss(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y,
                       method, theta=None, nbrs_num=None, exclusion_rad=0):
        """
        Internal method computing the correlation-based loss for a given batch.
        """
        if method == "smap":
            if theta is None:
                raise ValueError("`theta` must be provided when using the 'smap' method.")
            ccm = self._get_smap_ccm_matrix_approx(
                subset_idx, sample_idx, 
                sample_X, sample_y, 
                subset_X, subset_y, 
                theta, exclusion_rad
            )
        elif method == "knn":
            if nbrs_num is None:
                raise ValueError("`nbrs_num` must be provided when using the 'knn' method.")
            ccm = self._get_knn_ccm_matrix_approx(
                subset_idx, sample_idx, 
                sample_X, sample_y, 
                subset_X, subset_y, 
                nbrs_num, exclusion_rad
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # ccm has shape: (batch, E_y, n_comp_y, n_comp_x)
        #ccm = torch.abs(ccm)
        #ccm_r = ccm[:,:,mask]
        #ccm_R = ccm.clone()
        #ccm_R[:,:,mask] = 1
        ccm = ccm[:,:,0]
        #score = ccm_r.unsqueeze(-2) @ torch.inverse(ccm_R) @ ccm_r.unsqueeze(-1)
        #return 1 - score.mean(axis=(1,2,3))
        if self.subtract_autocorr:
            corr = torch.abs(self.__get_autoreg_matrix_approx(sample_X,sample_y))
            if self.n_comp > 1:
                score = 1  - ccm.mean(axis=1) + corr.mean(axis=1)
                #score = 1 - (ccm[:,:,mask]).mean(axis=(1,2)) + (corr[:,:,mask]).mean(axis=(1,2))
            else:
                score = 1 + (-ccm[:,:,0,0] + corr[:,:,0,0]).mean(axis=1)
            return score
        else:
            if self.n_comp > 1:
                score = 1 - ccm.mean(axis=1)#.mean(axis=(1,2))#.max(axis=2)[0].mean(axis=1)
                #score = 1 - (ccm[:,:,mask]).mean(axis=(1,2))
            else:
                score = 1 + (-ccm[:,:,0,0]).mean(axis=1)
            return score
        
    def __compute_h_norm(self):
        l1 = sum(p.abs().sum() 
                for p in self.model.parameters() if p.requires_grad)
        l2 = sum(p.norm(2)
                for p in self.model.parameters() if p.requires_grad)
        num_w = torch.tensor(sum(p.numel()
                for p in self.model.parameters() if p.requires_grad),dtype=MODEL_DTYPE)
        h_norm = 1-(torch.sqrt(num_w) - (l1 / (l2 + 1e-6))) / (torch.sqrt(num_w) - 1)

        return h_norm

    def _get_knn_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
        A, B = self._get_knn_prediction(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad)
        
        r_AB = self.__get_batch_corr(A,B)
        return r_AB
    
    def _get_knn_prediction1(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
        batch_size, sample_size, dim_x, n_comp_x = sample_X.shape
        _, subset_size, _, _ = subset_X.shape
        _, _, dim_y, n_comp_y = sample_y.shape

        sample_X_t = sample_X.permute(0, 3, 1, 2)
        subset_X_t = subset_X.permute(0, 3, 1, 2)
        
        weights, indices = self.__get_nbrs_indices(subset_X_t,sample_X_t, nbrs_num, subset_idx, sample_idx, exclusion_rad)
        # Shape: [batch, n_comp_x, sample_size, nbrs_num]

        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1, 1, 1)
        # Shape: [batch, 1, 1, 1]
        comp_idx = torch.arange(n_comp_x, device=self.device).view(1, n_comp_x, 1, 1)
        # Shape: [1, n_comp_x, 1, 1]
       

        #selected = subset_y[batch_idx, indices, :, :]
        # [batch, n_comp_x, sample_size, nbrs_num, dim_y, n_comp_y]
        selected = subset_y.permute(0,2,1,3)[batch_idx, comp_idx,indices]
        # [batch, n_comp_x, sample_size, nbrs_num, n_comp_y]

        #result = selected.permute(0, 2, 4, 3, 5, 1)
        # [batch, sample_size, dim_y, nbrs_num, n_comp_y, n_comp_x]
        result = selected.permute(0, 2, 1, 3, 4)
        # [batch, sample_size, n_comp_x, nbrs_num, n_comp_y]

        # Start with weights of shape: [batch, n_comp_x, sample_size, nbrs_num]
        #w = weights.permute(0, 2, 3, 1)  # Now shape: [batch, sample_size, nbrs_num, n_comp_x]
        #w = w.unsqueeze(2).unsqueeze(-2) # Final shape: [batch, sample_size, 1, nbrs_num, 1, n_comp_x]

        w = weights.permute(0, 2, 1, 3).unsqueeze(-1)
        # [batch, sample_size, n_comp_x, nbrs_num, n_comp_y]

        A = (result * w).sum(dim=3) 
        # [batch, num_points, dim_y, n_comp_y, n_comp_x]
        #A = A.mean(axis=-1).unsqueeze(-1)
        #A = self.component_mixing(A)
        #A = A.expand(batch_size,sample_size,dim_y, n_comp_x, n_comp_x)
        B = sample_y#.unsqueeze(-1).expand(batch_size, sample_size, dim_y, n_comp_y)

        #B = sample_y.unsqueeze(-1).expand(batch_size, sample_size, dim_y, n_comp_y, n_comp_x)
        #B = sample_y.unsqueeze(-1)#.expand(batch_size, sample_size, dim_y, n_comp_y, n_comp_x)

        return A, B
    
    def _get_knn_prediction(self, subset_idx, sample_idx,
                            sample_X, sample_y,
                            subset_X, subset_y,
                            nbrs_num, exclusion_rad):
        """
        Diagonal kNN: component i in X predicts y[:, :, i, :]
        Returns:
        if metric=="bce":  A,B -> [B, S, Dy, Cy]
        else (corr):       A,B -> [B, S, Dy, Cy]  (scored per component later)
        """
        B, S, Dx, Cx = sample_X.shape
        _, L, Dy, Cy = subset_y.shape
        if Dy != Cx:
            raise ValueError(f"Diagonal mapping requires Y.shape[2] (Dy) == n_components (got Dy={Dy}, Cx={Cx}).")

        # Distances per X component
        sample_X_t = sample_X.permute(0, 3, 1, 2)   # [B, Cx, S, Dx]
        subset_X_t = subset_X.permute(0, 3, 1, 2)   # [B, Cx, L, Dx]
        weights, indices = self.__get_nbrs_indices(subset_X_t, sample_X_t, nbrs_num,
                                                subset_idx, sample_idx, exclusion_rad)   # [B,Cx,S,K], [B,Cx,S,K]

        # For each comp i, gather y at dimension i
        # subset_y: [B, L, Dy, Cy] -> [B, Dy, L, Cy]
        suby = subset_y.permute(0, 2, 1, 3)  # [B, Dy, L, Cy]
        bidx = torch.arange(B, device=self.device).view(B, 1, 1, 1)
        cidx = torch.arange(Cx, device=self.device).view(1, Cx, 1, 1)

        # Gather over L using per-comp indices -> [B, Cx, S, K, Cy]
        gathered = suby[bidx, cidx, indices, :]
        w = weights.unsqueeze(-1)                     # [B, Cx, S, K, 1]
        pred = (gathered * w).sum(dim=3)              # [B, Cx, S, Cy]

        # Place into [B, S, Dy, Cy] with Dy==Cx
        A = pred.permute(0, 2, 1, 3)                  # [B, S, Dy, Cy]
        B_true = sample_y                              # [B, S, Dy, Cy]
        return A, B_true
    
    def _get_smap_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        A, B = self._get_smap_prediction(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad)
        
        r_AB = self.__get_batch_corr(A,B)
        return r_AB

    def _get_smap_prediction1(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        batch_size, sample_size, dim_x, n_comp_x = sample_X.shape
        _, subset_size, _, _ = subset_X.shape
        _, _, dim_y, n_comp_y = sample_y.shape

        sample_X_t = sample_X.permute(0, 3, 1, 2)
        subset_X_t = subset_X.permute(0, 3, 1, 2)
        subset_y_t = subset_y.permute(0, 3, 1, 2)
        #Shape: [batch, comp, points, proj_dim]
        
        weights = self.__get_local_weights(subset_X_t,sample_X_t,subset_idx, sample_idx, exclusion_rad, theta)
        #Shape: [batch, n_comp_x, sample_size, subset_size]
        W = (weights.unsqueeze(1)
             .expand(batch_size, n_comp_y, n_comp_x, sample_size, subset_size)
             .reshape(batch_size * n_comp_y * n_comp_x * sample_size, subset_size, 1))
        #Shape: [batch * {comp} * comp * sample_size, subset_size, 1]

        X = (subset_X_t.unsqueeze(2).unsqueeze(1)
             .expand(batch_size, n_comp_y, n_comp_x, sample_size, subset_size, dim_x)
             .reshape(batch_size * n_comp_y * n_comp_x * sample_size, subset_size, dim_x))
        #Shape: [batch * {comp} * comp * {sample_size}, subset_size, dim_x]

        Y = (subset_y_t.unsqueeze(2)
             .expand(batch_size, n_comp_y, sample_size, subset_size, dim_y)
             .permute(0, 1, 4, 2, 3)
             .reshape(batch_size * n_comp_y * dim_y * sample_size, subset_size, 1))
        #Shape: [batch * comp * {dim_y} * {sample_size}, subset_size, 1]

        X_intercept = torch.cat([torch.ones((batch_size * n_comp_y * n_comp_x * sample_size, subset_size, 1),device=self.device), X], dim=2)
        #Shape: [batch * comp * comp * sample_size, subset_size, dim_x + 1]

        X_intercept_weighted = X_intercept * W
        Y_weighted = Y * W

        XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
        XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)
        beta = torch.bmm(torch.inverse(XTWX), XTWy)
        #Shape: [batch * comp{y} * comp{x} * sample_size, dim_x + 1, 1]

        X_ = (sample_X_t.unsqueeze(1)
              .expand(batch_size, n_comp_y, n_comp_x, sample_size, dim_x)
              .reshape(batch_size * n_comp_y * n_comp_x * sample_size, dim_x))
        X_ = torch.cat([torch.ones((batch_size * n_comp_y * n_comp_x * sample_size, 1),device=self.device), X_], dim=1)
        X_ = X_.reshape(batch_size * n_comp_y * n_comp_x * sample_size, 1, dim_x+1)
        
        A = torch.bmm(X_, beta).reshape(batch_size, n_comp_y, dim_y, sample_size)
        A = A.permute(0, 3, 2, 1)
        # [batch, sample_size, dim_y, n_comp_y]

        #A = torch.permute(A,(0,3,4,1,2))

        B = sample_y

        
        return A, B
    
    def _get_smap_prediction(self, subset_idx, sample_idx,
                            sample_X, sample_y,
                            subset_X, subset_y,
                            theta, exclusion_rad):
        """
        Diagonal SMAP: component i in X predicts y[:, :, i, :]
        Returns A,B with shape [B, S, Dy, Cy] (Dy==n_components)
        """
        B, S, Dx, Cx = sample_X.shape
        _, L, Dy, Cy = subset_y.shape
        if Dy != Cx:
            raise ValueError(f"Diagonal mapping requires Y.shape[2] (Dy) == n_components (got Dy={Dy}, Cx={Cx}).")

        # [B, Cx, S, Dx] / [B, Cx, L, Dx]
        sample_X_t = sample_X.permute(0, 3, 1, 2)
        subset_X_t = subset_X.permute(0, 3, 1, 2)
        weights = self.__get_local_weights(subset_X_t, sample_X_t,
                                        subset_idx, sample_idx, exclusion_rad, theta)     # [B, Cx, S, L]

        # Build weighted local linear models per (comp i, sample s), predict y_dim=i (all channels Cy)
        # Design matrices
        W = weights.reshape(B * Cx * S, L, 1)                                                 # [B*Cx*S, L, 1]
        X = subset_X_t.reshape(B * Cx, L, Dx).unsqueeze(2).expand(B * Cx, L, S, Dx)\
                    .reshape(B * Cx * S, L, Dx)                                            # [B*Cx*S, L, Dx]

        # Target Y: take y_dim == comp i
        suby = subset_y.permute(0, 2, 1, 3)                                                   # [B, Dy, L, Cy]
        # repeat along S and pack -> [B*Cx*S, L, Cy]
        Y = suby[:, torch.arange(Cx, device=self.device)].permute(1, 0, 2, 3)\
            .reshape(Cx, B, L, Cy).permute(1, 0, 2, 3)\
            .unsqueeze(2).expand(B, Cx, S, L, Cy).reshape(B * Cx * S, L, Cy)

        X1 = torch.cat([torch.ones((B * Cx * S, L, 1), device=self.device), X], dim=2)        # [*, L, Dx+1]
        Xw, Yw = X1 * W, Y * W
        XTWX = torch.bmm(Xw.transpose(1, 2), Xw)                                              # [*, Dx+1, Dx+1]
        XTWy = torch.bmm(Xw.transpose(1, 2), Yw)                                              # [*, Dx+1, Cy]
        eye = torch.eye(XTWX.size(-1), device=XTWX.device) * 1e-6
        beta = torch.bmm(torch.inverse(XTWX + eye), XTWy)                                     # [*, Dx+1, Cy]

        Xq = torch.cat([torch.ones((B * Cx * S, 1), device=self.device),
                        sample_X_t.reshape(B * Cx, S, Dx).reshape(B * Cx * S, Dx)], dim=1)\
                        .reshape(B * Cx * S, 1, Dx + 1)                                       # [*, 1, Dx+1]
        pred = torch.bmm(Xq, beta).reshape(B, Cx, S, Cy)                                      # [B, Cx, S, Cy]
        A = pred.permute(0, 2, 1, 3)                                                          # [B, S, Dy, Cy]
        B_true = sample_y
        return A, B_true

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
    
    def __get_nbrs_indices(self, lib, sublib, n_nbrs, subset_idx, sample_idx, exclusion_rad):
        #[batch, comp, points, proj_dim]
        eps = 1e-6
        dist = torch.cdist(sublib,lib)
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

    def __neg_cross_entropy_per_dim(self, A, B):
        A_clamped = A.clamp(min=1e-7, max=1. - 1e-7)
        elementwise_loss = F.binary_cross_entropy(
            A_clamped, B.float(), reduction="none"
        )
        # Now average over the 'num_points' dimension => shape (batch, num_dimensions)
        loss_per_dim = elementwise_loss.mean(dim=1)
        
        return -loss_per_dim