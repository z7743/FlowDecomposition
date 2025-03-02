from model import LinearModel
from data_sampler import RandomSampleSubsetPairDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class FlowDecomposition:
    def __init__(self, input_dim, proj_dim, n_components, 
                 num_delays=None, delay_step=None, subtract_corr=False, 
                 device="cuda", optimizer="Adam", learning_rate=0.01, random_state=None):

        self.device = device
        
        self.model = LinearModel(input_dim, proj_dim, n_components, device, random_state)

        self.num_delays = num_delays
        self.delay_step = delay_step
        self.proj_dim = proj_dim
        self.n_comp = n_components
        self.random_state = random_state

        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate)
        self.subtract_corr = subtract_corr
        self.loss_history = []

        if (num_delays is None) or (delay_step is None) or (num_delays == 0) or (delay_step == 0):
            self.use_delay = False
        else:
            self.use_delay = True

    def fit(self, X, 
            sample_len, 
            library_len, 
            exclusion_rad=0,
            theta=None, 
            tp=1, 
            epochs=100, 
            num_batches=32, 
            beta=0,
            tp_policy="range", 
            loss_mask_size=None):
        
        X = torch.tensor(X,requires_grad=True, device=self.device, dtype=torch.float32)

        if tp_policy == "fixed":
            dataset = RandomSampleSubsetPairDataset(X=X, sample_size = sample_len,
                                                    subset_size = library_len, E = self.num_delays, tau=self.delay_step,
                                                    num_batches = num_batches, tp_range=(tp, tp),
                                                   device=self.device,random_state=self.random_state)
        elif tp_policy == "range":
            dataset = RandomSampleSubsetPairDataset(X=X, sample_size = sample_len,
                                                    subset_size = library_len, E = self.num_delays, tau=self.delay_step,
                                                    num_batches = num_batches, tp_range=(1, tp),
                                                   device=self.device,random_state=self.random_state)
        else:
            pass #TODO: pass an exception

        dataloader = DataLoader(dataset, batch_size=1, pin_memory=False)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            ccm_loss = 0
            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                #Shape: [1, E, subset/sample size, data_dim]
                subset_X_z = self.model(subset_X[0])
                sample_X_z = self.model(sample_X[0])
                subset_y_z = self.model(subset_y[0])
                sample_y_z = self.model(sample_y[0])
                #Shape: [E, subset/sample size, proj_dim, n_components]

                subset_X_z = subset_X_z.permute(1, 0, 2, 3).reshape(subset_X_z.size(1), -1, subset_X_z.size(3))
                sample_X_z = sample_X_z.permute(1, 0, 2, 3).reshape(sample_X_z.size(1), -1, sample_X_z.size(3))
                subset_y_z = subset_y_z.permute(1, 0, 2, 3).reshape(subset_y_z.size(1), -1, subset_y_z.size(3))
                sample_y_z = sample_y_z.permute(1, 0, 2, 3).reshape(sample_y_z.size(1), -1, sample_y_z.size(3))
                #Shape: [subset/sample size, E * proj_dim, n_components]

                loss = self.__loss_fn(subset_idx, sample_idx,
                                      sample_X_z, sample_y_z, subset_X_z, subset_y_z, 
                                      theta, exclusion_rad, loss_mask_size)

                loss /= num_batches
                ccm_loss += loss

            l1 = sum(p.abs().sum() 
                    for p in self.model.parameters() if p.requires_grad)
            l2 = sum(p.norm(2)
                    for p in self.model.parameters() if p.requires_grad)
            num_w = torch.tensor(sum(p.numel()
                    for p in self.model.parameters() if p.requires_grad),dtype=torch.float32)
            h_norm = 1-(torch.sqrt(num_w) - (l1 / (l2 + 1e-6))) / (torch.sqrt(num_w) - 1)
            
            total_loss = ccm_loss + beta * h_norm

            total_loss.backward()

            self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}, ccm_loss: {ccm_loss.item():.4f}, h_norm_loss: {h_norm.item():.4f}')
            self.loss_history += [total_loss.item()]

    def predict(self, X):
        """
        Calculates embeddings using the trained model.
        
        Args:
            X (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Predicted outputs.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            outputs = torch.permute(self.model(inputs),dims=(0,2,1)) #Easier to interpret
        return outputs.cpu().numpy()

    def __loss_fn(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad, loss_mask_size):
        dim = self.n_comp

        if loss_mask_size is not None:
            rand_idx = torch.argsort(torch.rand(dim))[:loss_mask_size]
            sample_X = sample_X[:,:,rand_idx]
            sample_y = sample_y[:,:,rand_idx]
            subset_X = subset_X[:,:,rand_idx]
            subset_y = subset_y[:,:,rand_idx]

            dim = loss_mask_size

        ccm = (self.__get_ccm_matrix_approx(subset_idx, 
                                            sample_idx, 
                                            sample_X, sample_y, 
                                            subset_X, subset_y, theta, exclusion_rad))
        
        mask = torch.eye(dim,dtype=bool,device=self.device)

        ccm = ccm**2

        if self.subtract_corr:
            corr = torch.abs(self.__get_autoreg_matrix_approx(sample_X,sample_y))**2
            if dim > 1:
                score = 1 + torch.abs(ccm[:,~mask]).mean() - (ccm[:,mask]).mean() + (corr[:,mask]).mean()
            else:
                score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
            return score
        else:
            if dim > 1:
                score = 1 + torch.abs(ccm[:,~mask]).mean() - (ccm[:,mask]).mean() 
            else:
                score = 1 + (-ccm[:,0,0]).mean()
            return score
        
    def __get_ccm_matrix_approx(self, subset_idx, sample_idx, sample_X, sample_y, subset_X, subset_y, theta, exclusion_rad):
        dim = self.n_comp
        if self.use_delay:
            E_x = self.proj_dim * self.num_delays
            E_y = self.proj_dim * self.num_delays
        else:
            E_x = self.proj_dim
            E_y = self.proj_dim
        sample_size = sample_X.shape[0]
        subset_size = subset_X.shape[0]

        sample_X_t = sample_X.permute(2, 0, 1)
        subset_X_t = subset_X.permute(2, 0, 1)
        subset_y_t = subset_y.permute(2, 0, 1)
        
        weights = self.__get_local_weights(subset_X_t,sample_X_t,subset_idx, sample_idx, exclusion_rad, theta)
        W = weights.unsqueeze(1).expand(dim, dim, sample_size, subset_size).reshape(dim * dim * sample_size, subset_size, 1)

        X = subset_X_t.unsqueeze(1).unsqueeze(1).expand(dim, dim, sample_size, subset_size, E_x)
        X = X.reshape(dim * dim * sample_size, subset_size, E_x)

        Y = subset_y_t.unsqueeze(1).unsqueeze(0).expand(dim, dim, sample_size, subset_size, E_y)
        Y = Y.reshape(dim * dim * sample_size, subset_size, E_y)

        X_intercept = torch.cat([torch.ones((dim * dim * sample_size, subset_size, 1),device=self.device), X], dim=2)
        
        X_intercept_weighted = X_intercept * W
        Y_weighted = Y * W

        XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
        XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)
        beta = torch.bmm(torch.pinverse(XTWX), XTWy)

        X_ = sample_X_t.unsqueeze(1).expand(dim, dim, sample_size, E_x)
        X_ = X_.reshape(dim * dim * sample_size, E_x)
        X_ = torch.cat([torch.ones((dim * dim * sample_size, 1),device=self.device), X_], dim=1)
        X_ = X_.reshape(dim * dim * sample_size, 1, E_x+1)
        
        A = torch.bmm(X_, beta).reshape(dim, dim, sample_size, E_y)
        A = torch.permute(A,(2,3,1,0))

        B = sample_y.unsqueeze(-1).expand(sample_size, E_y, dim, dim)
        #TODO: test whether B = sample_y.unsqueeze(-2).expand(sample_size, E_y, dim, dim)
        
        r_AB = self.__get_batch_corr(A,B)
        return r_AB
    
    def __get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        dist = torch.cdist(sublib,lib)
        if theta == None:
            weights = torch.exp(-(dist))
        else:
            weights = torch.exp(-(theta*dist/dist.mean(axis=2)[:,:,None]))

        if exclusion_rad > 0:
            exclusion_matrix = (torch.abs(subset_idx - sample_idx.T) > exclusion_rad)
            weights = weights * exclusion_matrix
        
        return weights
    
    def __get_autoreg_matrix_approx(self, A, B):
        dim = A.shape[-1]
        E = A.shape[-2]
        
        A = A[:,:,None,:].expand(-1, E, dim, dim)
        B = B[:,:,:,None].expand(-1, E, dim, dim)

        r_AB = self.__get_batch_corr(A,B)
        return r_AB
    
    def __get_batch_corr(self, A, B):
        mean_A = torch.mean(A,axis=0)
        mean_B = torch.mean(B,axis=0)
        
        sum_AB = torch.sum((A - mean_A[None,:,:]) * (B - mean_B[None,:,:]),axis=0)
        sum_AA = torch.sum((A - mean_A[None,:,:]) ** 2,axis=0)
        sum_BB = torch.sum((B - mean_B[None,:,:]) ** 2,axis=0)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB    
    

