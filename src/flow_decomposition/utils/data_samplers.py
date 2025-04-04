import torch
from torch.utils.data import Dataset
import time

class RandomSampleSubsetPairDataset_depricated(Dataset):
    """
    Dataset that samples two disjoint sets of indices from a multivariate time series (or point cloud)
    and returns their corresponding data along with shifted target values.

    For each __getitem__ call:
      - A time shift is selected from a generated tp_range (cycling through its values).
      - Two independent random sets of indices are drawn from a valid range.
      - Any overlap between the "sample" and "subset" indices is removed.

    Delay embedding is applied if delay parameters are provided and not equal to (E, tau) = (1, 0). For a given base
    index b, the delay embedding is defined as:

          [b - (E-1)*tau, b - (E-2)*tau, ..., b]

    resulting in output data of shape [E, num_indices, data_dim]. When delay embedding is not used,
    the sampled indices are unsqueezed to yield an output shape of [1, num_indices, data_dim].

    If a target tensor y is not provided, X is used as the target.
    """
    def __init__(self, X, sample_size, subset_size, y=None, num_batches=32, 
                 tp_range=(1, 2), device="cuda", 
                 E=None, tau=None, random_state=None):
        """
        Initializes the RandomSampleSubsetPairDataset instance.
        
        Parameters:
            X (torch.Tensor): Input data tensor of shape [num_datapoints, ...].
            sample_size (int): Number of indices to sample for the "sample" set.
            subset_size (int): Number of indices to sample for the "subset" set.
            num_batches (int): Number of random batches to produce.
            tp_range (torch.Tensor): A 1D tensor of time shifts; the maximum value is used to adjust the valid sampling range.
            device (str): The device on which to perform computations (e.g., "cuda" or "cpu").
            y (torch.Tensor, optional): Target tensor; if not provided, X is used as the target.
            E (int, optional): Embedding dimension (number of delays). If omitted or set to 1, no delay embedding is applied.
            tau (int, optional): Time delay between consecutive elements in the embedding.
            random_state (int, optional): Seed for reproducibility; if provided, ensures consistent random sampling for each index.
        """
        self.device = device
        self.X = X
        self.y = y if y is not None else self.X
        self.sample_size = sample_size
        self.subset_size = subset_size
        self.num_batches = num_batches
        self.num_datapoints = X.shape[0]
        self.random_state = random_state

        # Generate tp_range as a linspace from tp_min to tp_max with num_batches elements.
        if isinstance(tp_range, (list, tuple)) and len(tp_range) == 2:
            tp_min, tp_max = tp_range
            self.tp_range = torch.linspace(tp_min, tp_max + (1 - 1e-5), num_batches, device=self.device).to(torch.int)
        else:
            raise ValueError("tp_range must be a tuple of two numbers (tp_min, tp_max).")
        self.tp_max = int(self.tp_range.max().item())

        # Determine whether delay embedding should be used.
        if (E is None) or (tau is None) or (E == 0) or (tau == 0):
            self.use_delay = False
            self.E = 1
            self.tau = 0
            self.offset = 0
        else:
            self.use_delay = True
            self.E = E
            self.tau = tau
            self.offset = (self.E - 1) * self.tau

        # Compute the valid range for sampling base indices.
        self.valid_range = self.num_datapoints - self.tp_max - self.offset
        if self.valid_range < (sample_size + subset_size):
            raise ValueError("Not enough valid data points given sample_size, subset_size, tp_range, and delay parameters.")
    
    def __len__(self):
        # The dataset length is defined by the number of time shifts.
        return self.num_batches
    
    def _get_local_generator(self, idx):
        """Creates and returns a local random generator if random_state is set."""
        if self.random_state is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.random_state + idx)
            return gen
        return None
    
    def _sample_candidates_depricated(self, num, generator):
        """Samples 'num' indices from the valid range using torch.randperm."""
        if generator is not None:
            return torch.randperm(self.valid_range, device=self.device, generator=generator)[:num]
        return torch.randperm(self.valid_range, device=self.device)[:num]
    
    def _sample_candidates(self, num, generator=None):
        return torch.randint(
            high=self.valid_range,
            size=(num,),
            device=self.device,
            generator=generator
        )

    def _get_delay_embedding(self, base_indices):
        """
        Constructs delay embeddings for a set of base indices.
        
        Args:
            base_indices (torch.Tensor): Tensor of base indices, shape [N].
        
        Returns:
            torch.Tensor: Delay embedding indices of shape [E, N] where each column is
                          [b - (E-1)*tau, b - (E-2)*tau, ..., b].
        """
        delays = self.tau * torch.arange(self.E - 1, -1, -1, device=self.device)
        return base_indices.unsqueeze(0) - delays.unsqueeze(1)
    
    def __getitem__(self, idx):
        local_gen = self._get_local_generator(idx)
        # Select the time shift (cycling through tp_range).
        shift = int(self.tp_range[idx % len(self.tp_range)].item())
        
        # Sample indices for the sample and subset sets.
        subset_candidates = self._sample_candidates(self.sample_size + self.subset_size, local_gen)
        sample_candidates = subset_candidates[self.subset_size:]
        subset_candidates = subset_candidates[:self.subset_size]
        
        if self.use_delay:
            # Adjust base indices by adding the offset.
            sample_base = sample_candidates + self.offset
            subset_base = subset_candidates + self.offset

            sample_indices = self._get_delay_embedding(sample_base)
            subset_indices = self._get_delay_embedding(subset_base)
        else:
            # Without delay, add a singleton dimension to ensure shape [1, num_indices].
            sample_base = sample_candidates
            subset_base = subset_candidates

            sample_indices = sample_candidates.unsqueeze(0)
            subset_indices = subset_candidates.unsqueeze(0)

        sample_indices = sample_indices.T
        subset_indices = subset_indices.T
        # Retrieve the corresponding data from X and shifted targets from y.
        #st = time.time()
        X_sample = self.X[sample_indices]
        X_subset = self.X[subset_indices]
        y_sample = self.y[sample_indices + shift]
        y_subset = self.y[subset_indices + shift]
        #print("ss",time.time()-st)
        return subset_base, sample_base, X_subset, y_subset, X_sample, y_sample

    def __getitem1__(self, idx):
        local_gen = self._get_local_generator(idx)
        # Select the time shift (cycling through tp_range).
        shift = int(self.tp_range[idx % len(self.tp_range)].item())
        
        # Sample indices for the sample and subset sets.
        #sample_candidates = self._sample_candidates(self.sample_size, local_gen)
        #subset_candidates = self._sample_candidates(self.sample_size + self.subset_size, local_gen)
        # Ensure that subset_candidates are disjoint from sample_candidates.
        #mask = ~(subset_candidates.unsqueeze(0) == sample_candidates.unsqueeze(1)).any(dim=0)

        idx_candidates = self._sample_candidates(self.sample_size + self.subset_size, local_gen)
        
        if self.use_delay:
            # Adjust base indices by adding the offset.
            indices = self._get_delay_embedding(idx_candidates + self.offset)
        else:
            indices = idx_candidates.unsqueeze(0)

        indices = indices.T
        # Retrieve the corresponding data from X and shifted targets from y.
        st = time.time()
        X_sample = self.X[indices]
        y_sample = self.y[indices + shift]

        X_subset = X_sample[:self.subset_size]
        y_subset = y_sample[:self.subset_size]
        X_sample = X_sample[self.subset_size:]
        y_sample = y_sample[self.subset_size:]

        subset_base = idx_candidates[:self.subset_size]
        sample_base = idx_candidates[self.subset_size:]
        print("ss", time.time()-st)
        
        return subset_base, sample_base, X_subset, y_subset, X_sample, y_sample
    

class RandomSampleSubsetPairDataset(Dataset):
    """
    Dataset that samples two disjoint sets of indices from a multivariate time series (or point cloud)
    and returns their corresponding data along with shifted target values.

    For each __getitem__ call:
      - A time shift is selected from a generated tp_range (cycling through its values).
      - Two independent random sets of indices are drawn from a valid range.
      - Any overlap between the "sample" and "subset" indices is removed.

    Delay embedding is applied if delay parameters are provided and not equal to (E, tau) = (1, 0). 
    """
    def __init__(
        self,
        X,
        sample_size,
        subset_size,
        y = None,
        num_batches = 32,
        tp_range = (1, 2),
        device = "cpu", 
        E = None,
        tau = None
    ):
        """
        Parameters:
            X (torch.Tensor): Input data tensor of shape [num_datapoints, ...].
            sample_size (int): Number of indices to sample for the "sample" set.
            subset_size (int): Number of indices to sample for the "subset" set.
            num_batches (int): Number of random batches to produce.
            tp_range (tuple): (tp_min, tp_max) specifying the time-shift range.
            device (str): The device to which *sliced* mini-batches should be moved.
            y (torch.Tensor, optional): Target tensor; if not provided, uses X as target.
            E (int, optional): Embedding dimension (number of delays).
            tau (int, optional): Time delay between consecutive elements in the embedding.
            random_state (int, optional): Seed for reproducibility.
        """
        super().__init__()
        
        self.device = device         # the device we eventually want to move slices to

        self.X = X.to(device)
        if y is None:
            self.y = X.to(device)
        else:
            self.y = y.to(device)
        
        self.sample_size = sample_size
        self.subset_size = subset_size
        self.num_batches = num_batches
        self.num_datapoints = self.X.shape[0]

        if isinstance(tp_range, (list, tuple)) and len(tp_range) == 2:
            tp_min, tp_max = tp_range
            # Store time shifts on CPU or data_device so it doesn't matter:
            self.tp_range = torch.linspace(tp_min, tp_max + (1 - 1e-5), num_batches, device=self.device).to(torch.int)
        else:
            raise ValueError("tp_range must be a tuple of two numbers (tp_min, tp_max).")
        self.tp_max = int(self.tp_range.max().item())

        # Determine whether delay embedding should be used.
        if (E is None) or (tau is None) or (E == 0) or (tau == 0):
            self.use_delay = False
            self.E = 1
            self.tau = 0
            self.offset = 0
        else:
            self.use_delay = True
            self.E = E
            self.tau = tau
            self.offset = (self.E - 1) * self.tau

        # Compute the valid range for sampling base indices.
        self.valid_range = self.num_datapoints - self.tp_max - self.offset
        if self.valid_range < (sample_size + subset_size):
            raise ValueError(
                "Not enough valid data points given sample_size, subset_size, "
                "tp_range, and delay parameters."
            )

    def __len__(self):
        return self.num_batches

    def _get_local_generator(self, idx):
        #if self.random_state is not None:
        #    gen = torch.Generator(device=self.device)
        #    gen.manual_seed(self.random_state + idx)
        #    return gen
        return None

    def _sample_candidates(self, num, generator=None):
        return torch.randint(
            high=self.valid_range,
            size=(num,),
            device=self.device,
            generator=generator
        )

    def _get_delay_embedding(self, base_indices: torch.Tensor):
        """
        Constructs delay embeddings for a set of base indices.
        Returns shape [E, N], each col is b - [E-1, ..., 0]*tau
        """
        delays = self.tau * torch.arange(self.E - 1, -1, -1, device=self.device)
        return base_indices.unsqueeze(0) - delays.unsqueeze(1)
    
    def __getitem__(self, idx):
        # Local random generator
        local_gen = self._get_local_generator(idx)

        # 1) time shift
        shift = int(self.tp_range[idx % len(self.tp_range)].item())

        # 2) sample random indices
        subset_candidates = self._sample_candidates(self.sample_size + self.subset_size, local_gen)
        sample_candidates = subset_candidates[self.subset_size:]
        subset_candidates = subset_candidates[:self.subset_size]

        if self.use_delay:
            sample_base = sample_candidates + self.offset
            subset_base = subset_candidates + self.offset
            sample_indices = self._get_delay_embedding(sample_base)
            subset_indices = self._get_delay_embedding(subset_base)
        else:
            sample_base = sample_candidates
            subset_base = subset_candidates

            # Without delays, shape [1, N]
            sample_indices = sample_candidates.unsqueeze(0)
            subset_indices = subset_candidates.unsqueeze(0)

        # Convert from [E, N] -> [N, E]
        sample_indices = sample_indices.T
        subset_indices = subset_indices.T

        # 3) Slice from the big dataset (on self.data_device)...
        X_subset = self.X[subset_indices] 
        X_sample = self.X[sample_indices]
        y_subset = self.y[subset_indices + shift]
        y_sample = self.y[sample_indices + shift]

        return subset_base, sample_base, X_subset, y_subset, X_sample, y_sample
