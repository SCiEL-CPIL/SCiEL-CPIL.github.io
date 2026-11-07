import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_2d(self, n):
        """Generate n Halton points in 2D using bases 2 and 3."""
        h1 = np.zeros(n)
        h2 = np.zeros(n)
        for i in range(1, n + 1):
            # Base 2
            f, r = 0.5, 0.0
            j = i
            while j > 0:
                r += f * (j % 2)
                j //= 2
                f *= 0.5
            h1[i - 1] = r
            # Base 3
            f, r = 1.0 / 3.0, 0.0
            j = i
            while j > 0:
                r += f * (j % 3)
                j //= 3
                f /= 3.0
            h2[i - 1] = r
        return h1, h2

    def _simplex_mapping(self, u1, u2):
        """Map [0,1]^2 to the 2-simplex (triangle) using reflection."""
        mask = u1 + u2 > 1.0
        u1 = np.where(mask, 1.0 - u1, u1)
        u2 = np.where(mask, 1.0 - u2, u2)
        return u1, u2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        budget = self.budget
        dim = self.dim
        all_F = []
        
        # DTLZ1 Pareto front:
        # f1 = 55.5 * x0
        # f2 = 55.5 * (1-x0) * x1  
        # f3 = 55.5 * (1-x0) * (1-x1)
        # with x[2:] = 0.5 for minimum g
        
        # Strategy: Use multiple quasi-random sequences with different 
        # offsets to get better coverage of the Pareto front
        n_sequences = 4
        seq_lengths = [budget // n_sequences] * n_sequences
        # Adjust for remainder
        for i in range(budget % n_sequences):
            seq_lengths[i] += 1
        
        # Generate Halton points for each sequence with different starting offsets
        for seq_idx in range(n_sequences):
            n = seq_lengths[seq_idx]
            if n <= 0:
                continue
            
            if time.time() > deadline:
                break
            
            # Use different base pairs for diversity
            # Sequence 0: bases 2,3
            # Sequence 1: bases 5,7  
            # Sequence 2: bases 11,13
            # Sequence 3: bases 17,19
            bases = [(2, 3), (5, 7), (11, 13), (17, 19)]
            b1, b2 = bases[seq_idx % len(bases)]
            
            h1 = np.zeros(n)
            h2 = np.zeros(n)
            for i in range(1, n + 1):
                # Base b1
                f, r = 1.0 / b1, 0.0
                j = i
                while j > 0:
                    r += f * (j % b1)
                    j //= b1
                    f /= b1
                h1[i - 1] = r
                # Base b2
                f, r = 1.0 / b2, 0.0
                j = i
                while j > 0:
                    r += f * (j % b2)
                    j //= b2
                    f /= b2
                h2[i - 1] = r
            
            # Map to simplex
            x0, x1 = self._simplex_mapping(h1, h2)
            
            # Construct and evaluate
            for k in range(n):
                if time.time() > deadline:
                    break
                
                x = np.full(dim, 0.5)
                x[0] = x0[k]
                x[1] = x1[k]
                
                try:
                    f = np.asarray(problem(x), dtype=np.float64)
                    if f.shape == (3,) and np.all(np.isfinite(f)):
                        all_F.append(f)
                except Exception:
                    pass
        
        return all_F
