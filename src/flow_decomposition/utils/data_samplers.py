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
    def __init__(
        self,
        X: torch.Tensor,
        sample_size: int,
        subset_size: int,
        y: torch.Tensor = None,
        num_batches: int = 32,
        tp_range = (1, 2),
        device: str = "cpu",
        E: int = None,
        tau: int = None,
        include_opposite: bool = False,
        even_tp: bool = True,
        seed: int = None,
    ):
        super().__init__()

        self.device = device
        self.X = X.to(device)
        self.y = (X if y is None else y).to(device)

        self.sample_size = int(sample_size)
        self.subset_size = int(subset_size)
        self.num_batches = int(num_batches)
        self.num_datapoints = self.X.shape[0]

        # ---- Build integer tp values ----
        if not (isinstance(tp_range, (list, tuple)) and len(tp_range) == 2):
            raise ValueError("tp_range must be a tuple (tp_min, tp_max).")

        tp_min, tp_max = map(int, tp_range)
        # allow zero when forward-only; forbid negatives unless symmetric
        if not include_opposite and tp_min < 0:
            raise ValueError("Negative shifts require include_opposite=True.")
        if tp_max < 0:
            raise ValueError("tp_max must be >= 0.")
        if include_opposite:
            self.tp_values = torch.arange(-tp_max, tp_max + 1, device=device, dtype=torch.long)
        else:
            self.tp_values = torch.arange(tp_min, tp_max + 1, device=device, dtype=torch.long)
        if self.tp_values.numel() == 0:
            raise ValueError("Empty shift range: check tp_range/include_opposite.")

        self.tp_max = int(self.tp_values.abs().max().item())
        self.include_opposite = bool(include_opposite)

        # ---- RNG (optional reproducibility) ----
        self._g = torch.Generator(device=self.device)
        if seed is not None:
            self._g.manual_seed(int(seed))

        # ---- Delay embedding params ----
        if (E is None) or (tau is None) or (E == 0) or (tau == 0):
            self.use_delay = False
            self.E = 1
            self.tau = 0
            self.offset = 0
        else:
            self.use_delay = True
            self.E = int(E)
            self.tau = int(tau)
            self.offset = (self.E - 1) * self.tau

        # ---- Safe window (handles ±tp and delays) ----
        left_margin  = self.tp_max if self.include_opposite else 0
        right_margin = self.tp_max
        self._base_low  = left_margin
        self._base_high = self.num_datapoints - right_margin - self.offset
        if self._base_high <= self._base_low:
            raise ValueError(
                "Not enough data for given (tp_range, E, tau). Reduce tp_max/E/tau."
            )

        self.valid_range = self._base_high - self._base_low
        if self.valid_range < (self.sample_size + self.subset_size):
            raise ValueError(
                "Not enough valid points for sample_size+subset_size within the safe window. "
                f"Window size={self.valid_range}, required>={self.sample_size+self.subset_size}."
            )

        # ---- Build tp schedule (even coverage) ----
        self.even_tp = bool(even_tp)
        if self.even_tp:
            self._build_even_tp_schedule(epoch_offset=0)
        else:
            # fallback: simple cyclic schedule
            k = self.tp_values.numel()
            reps = (self.num_batches + k - 1) // k
            self._tp_schedule = self.tp_values.repeat(reps)[: self.num_batches]

    def __len__(self):
        return self.num_batches

    # --- Public: reseed & reshuffle per epoch for fairness (optional) ---
    def set_epoch(self, epoch: int = 0):
        """Call this at each epoch start to rotate which tps get the extra count."""
        if self.even_tp:
            self._build_even_tp_schedule(epoch_offset=int(epoch))

    def reseed(self, seed: int):
        self._g = torch.Generator(device=self.device)
        self._g.manual_seed(int(seed))

    # --- Even schedule builder: counts differ by at most 1, interleaved ---
    def _build_even_tp_schedule(self, epoch_offset: int = 0):
        k = self.tp_values.numel()
        T = self.num_batches
        q, r = divmod(T, k)  # base repeats and remainder

        # Start order rotated by epoch_offset for fairness across epochs
        base_order = torch.arange(k, device=self.device)
        if k > 0 and epoch_offset % k:
            base_order = torch.roll(base_order, shifts=epoch_offset % k)

        counts = torch.full((k,), q, dtype=torch.long, device=self.device)
        # Distribute remainder to r categories; rotate start so "extra" rotates by epoch
        counts[base_order[:r]] += 1

        # Interleave in round-robin to avoid long blocks
        schedule = []
        remaining = counts.clone()
        while int(remaining.sum().item()) > 0:
            for i in base_order:
                if remaining[i] > 0:
                    schedule.append(self.tp_values[i].item())
                    remaining[i] -= 1

        self._tp_schedule = torch.tensor(schedule, device=self.device, dtype=torch.long)

    def _get_delay_embedding(self, base_indices: torch.Tensor) -> torch.Tensor:
        delays = self.tau * torch.arange(self.E - 1, -1, -1, device=self.device)
        return base_indices.unsqueeze(0) - delays.unsqueeze(1)

    def _sample_candidates(self, num: int) -> torch.Tensor:
        # without-replacement for lower variance; switch to randint if you prefer with-replacement
        pool = torch.randperm(self.valid_range, device=self.device, generator=self._g)[:num]
        return self._base_low + pool

    def __getitem__(self, idx: int):
        # Even schedule: pick the predetermined tp for this batch
        shift = int(self._tp_schedule[idx].item())

        cands = self._sample_candidates(self.sample_size + self.subset_size)
        subset_candidates = cands[:self.subset_size]
        sample_candidates = cands[self.subset_size:]

        if self.use_delay:
            subset_base = subset_candidates + self.offset
            sample_base = sample_candidates + self.offset
            subset_indices = self._get_delay_embedding(subset_base).T
            sample_indices = self._get_delay_embedding(sample_base).T
        else:
            subset_base = subset_candidates
            sample_base = sample_candidates
            subset_indices = subset_candidates.unsqueeze(1)
            sample_indices = sample_candidates.unsqueeze(1)

        X_subset = self.X[subset_indices]
        X_sample = self.X[sample_indices]
        y_subset = self.y[subset_indices + shift]
        y_sample = self.y[sample_indices + shift]
        return subset_base, sample_base, X_subset, y_subset, X_sample, y_sample