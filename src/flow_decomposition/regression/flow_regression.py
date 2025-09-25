from flow_decomposition.utils.models import LinearModel, NonlinearModel
from flow_decomposition.utils.data_samplers import RandomSampleSubsetPairDataset
from flow_decomposition.utils.metrics import get_metric

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

DATA_DTYPE = torch.float32
MODEL_DTYPE = torch.float32

class FlowRegression:

    def __init__(self, input_dim, proj_dim, 
                 num_delays=None, delay_step=None, model="linear", subtract_autocorr=False, 
                 regularizer="h_norm", tol=None,
                 device="cuda", data_device="cpu", optimizer="Adagrad", learning_rate=0.01, random_state=None, verbose=1):
        """
        Initializes the FlowRegression model.
        
        Args:
            input_dim: Dimensionality of the input data
            proj_dim: Dimensionality of the projection
            num_delays: Number of delay embeddings to use
            delay_step: Step size between delays
            model: Either "linear", "nonlinear", or a custom torch.nn.Module instance
            subtract_autocorr: Whether to subtract autocorrelation from the CCM score
            device: Device to run the model on ("cuda" or "cpu")
            data_device: Device to store the data on
            optimizer: Optimizer to use (string name from torch.optim)
            learning_rate: Learning rate for the optimizer
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0 or 1)
        """
        self.device = device
        self.data_device = data_device
        self.random_state = random_state
        self.proj_dim = proj_dim
        self.num_delays = num_delays
        self.delay_step = delay_step
        self.subtract_autocorr = subtract_autocorr
        self.loss_history = []
        self.input_dim = input_dim
        self.verbose = verbose
        self.regularizer = regularizer
        self.tol = tol

        if isinstance(model, torch.nn.Module):
            self.model = model
        elif model == "linear":
            self.model = LinearModel(input_dim=input_dim, proj_dim=proj_dim, n_comp=1, device=device, random_state=random_state)
        elif model == "nonlinear":
            self.model = NonlinearModel(input_dim=input_dim, proj_dim=proj_dim, n_comp=1, device=device, random_state=random_state)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate)
        self.use_delay = bool(num_delays) and bool(delay_step)


    def fit(self, X, Y,
            sample_size, library_size, exclusion_rad=0, method="knn",metric="corr",
            theta=None, nbrs_num=None, time_intv=1, beta=0,
            num_epochs=100, num_rand_samples=32, batch_size=1,
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
        self.metric = get_metric(metric) 
        
        ema_loss = None
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            ccm_loss = self._compute_prediction_loss(
                dataloader, num_rand_samples, method, theta, nbrs_num, exclusion_rad
            )
            
            h_norm_val = self.__regularization_loss()
            total_loss = (ccm_loss) + beta * (h_norm_val)

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

            self.loss_history.append(total_loss.item())

            loss_val = total_loss.item()                    
            if self.tol is not None:
                if ema_loss is None:                  
                    ema_loss = loss_val
                else:                                     
                    delta = abs(loss_val - ema_loss)

                    if delta < self.tol * max(1.0, abs(ema_loss)):
                        if self.verbose:
                            print(f"Early-stopped at epoch {epoch + 1} "
                                f"(Δloss={delta:.3g} < tol={self.tol})")
                        break

                    ema_loss = 0.1 * loss_val + 0.9 * ema_loss
            
                    
        if self.random_state is not None:
            torch.set_rng_state(old_rng_state)

    def evaluate_loss(self, X, Y,
                      sample_size, library_size, exclusion_rad=0, method="knn",
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
            torch.manual_seed(self.random_state)

        with torch.no_grad():
            # Create dataloader for test data
            dataloader = self._create_dataloader(
                X, Y,
                sample_size=sample_size,
                library_size=library_size,
                time_intv=time_intv,
                num_rand_samples=num_rand_samples,
                optim_policy=optim_policy,
                batch_size=batch_size
            )

            ccm_loss = self._compute_prediction_loss(
                dataloader, num_rand_samples, method, theta, nbrs_num, exclusion_rad
            )
        
        if self.random_state is not None:
            torch.set_rng_state(old_rng_state)

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
            outputs = self.model.to(device)(inputs)[:,:,0] 
        return outputs.cpu().numpy()

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
            device=self.data_device
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=0
        )
        return dataloader

    def _compute_prediction_loss(self, dataloader, num_rand_samples, method, theta, nbrs_num, exclusion_rad):
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
            loss = self.__crossmapping_loss(
                subset_idx, sample_idx,
                sample_X_z, sample_y_z,
                subset_X_z, subset_y_z,
                method, theta, nbrs_num, exclusion_rad
            )
            # Divide by num_rand_samples to average across random draws
            loss /= num_rand_samples
            total_loss += loss.sum()
        
        return total_loss
    
    def __crossmapping_loss(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y,
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
        
        # ccm has shape: (B, E, dim, dim)
        ccm = ccm.squeeze(-2).squeeze(-1)

        # Possibly subtract auto-correlation
        if self.subtract_autocorr:
            #corr = torch.abs(self.__get_linpred_matrix_approx(sample_X, sample_y))
            corr = self.__get_linpred_matrix_approx(
                subset_X, subset_y,
                sample_X, sample_y
            )
            corr = corr.squeeze(-2).squeeze(-1)
            print(corr.mean())
            score = 1 + (-(ccm) + corr).mean(axis=1)
            return score
        else:
            score = 1 + (-(ccm)).mean(axis=1)
            return score
        
    def __regularization_loss(self):

        if self.regularizer == "h_norm":
            l1 = sum(p.abs().sum() 
                    for p in self.model.parameters() if p.requires_grad)
            l2 = sum(p.norm(2)
                    for p in self.model.parameters() if p.requires_grad)
            num_w = torch.tensor(sum(p.numel()
                    for p in self.model.parameters() if p.requires_grad),dtype=MODEL_DTYPE)
            h_norm = 1-(torch.sqrt(num_w) - (l1 / (l2 + 1e-12))) / (torch.sqrt(num_w) - 1)

            return h_norm
        
        elif self.regularizer == "gini":
            w = torch.cat([p.view(-1).abs()                      
                        for p in self.model.parameters()
                        if p.requires_grad])

            if w.numel() == 0:             
                return torch.tensor(0., dtype=MODEL_DTYPE,
                                    device=next(self.model.parameters()).device)

            w_sorted, _ = torch.sort(w)    
            n = w_sorted.numel()

            index = torch.arange(1, n + 1, device=w_sorted.device, dtype=MODEL_DTYPE)
            gini = (2.0 * (index * w_sorted).sum()) / (n * w_sorted.sum() + 1e-12) - (n + 1) / n

            g_norm = 1.0 - gini

            return g_norm
        
        elif self.regularizer == "l1_l2":
            
            l1 = sum(p.abs().sum() 
                    for p in self.model.parameters() if p.requires_grad)
            l2 = sum(p.norm(2)
                    for p in self.model.parameters() if p.requires_grad)
            
            reg = (l1 / (l2 + 1e-12))
            #reg = torch.log1p(l1 / (l2 + 1e-12))

            return reg
        
        else:
            raise ValueError(f"Unknown regularizer: {self.regularizer}")

    def _get_knn_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
        A, B = self._get_knn_prediction(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad)
        
        r_AB = self.metric(A,B)
        return r_AB
    
    def _get_knn_prediction(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, nbrs_num, exclusion_rad):
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
        w = w.unsqueeze(2).unsqueeze(-2) # Final shape: [B, sample_size, 1, nbrs_num, 1,  n_comp]

        A = (result * w).sum(dim=3) 
        # [batch, num_points, dim, n_comp, n_comp]

        B = sample_y.unsqueeze(-1).expand(batch_size, sample_size, dim_y, n_comp, n_comp)

        return A, B
    
    def _get_smap_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        A, B = self._get_smap_prediction(subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad)
        
        r_AB = self.metric(A,B)
        return r_AB

    def _get_smap_prediction(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad, ridge=0.0):
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
        
        return A, B
    
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
    
    def __get_autoreg_matrix_approx1(self, A, B):
        batch_size, sample_size, dim_x, n_comp = A.shape
        _, _, dim_y, _ = B.shape
        # n_comp = 1 always
        if self.use_delay:
            E = self.proj_dim * self.num_delays
        else:
            E = self.proj_dim
        
        A = A[:,:,:,None,:].expand(batch_size,-1, dim_x, n_comp, n_comp) 
        B = B[:,:,:,:,None].expand(batch_size,-1, dim_y, n_comp, n_comp)

        r_AB = self.metric(A,B)
        return r_AB
    
    def __get_linpred_matrix_approx(self,
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
        # 1) Gather constructor params
        init_params = {
            "input_dim": self.input_dim,
            "proj_dim": self.proj_dim,
            "num_delays": self.num_delays,
            "delay_step": self.delay_step,
            "model": "linear" if isinstance(self.model, LinearModel) else "nonlinear",
            "subtract_autocorr": self.subtract_autocorr,
            "regularizer": self.regularizer,
            "tol": self.tol,
            "device": self.device,
            "data_device": self.data_device,
            "optimizer": type(self.optimizer).__name__,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "random_state": self.random_state
        }

        # 2) Create checkpoint
        checkpoint = {
            "init_params": init_params,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
        }

        # 3) Save to disk
        torch.save(checkpoint, filepath)
        print(f"FlowRegression model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, map_location: str = "cpu"):
        """
        Loads a saved FlowRegression model from disk.
        """
        # 1) Load checkpoint
        checkpoint = torch.load(filepath, map_location=map_location)
        init_params = checkpoint["init_params"]

        # 2) Instantiate new instance
        init_params["device"] = map_location
        init_params["data_device"] = map_location
        new_instance = cls(**init_params)

        # 3) Load model and optimizer states
        new_instance.model.load_state_dict(checkpoint["model_state"])
        new_instance.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # 4) Restore additional data
        new_instance.loss_history = checkpoint["loss_history"]

        print(f"FlowRegression model loaded from {filepath}")
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
            D = torch.cdist(X, X, p=2)
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