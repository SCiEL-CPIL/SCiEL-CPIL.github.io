import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
        # Parameters for this specific task
        # q=3, n=8, radius=1
        if q != 3 or n != 8 or radius != 1:
            # Fallback to seed for unexpected inputs
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

        N = q ** n
        powers = [q**i for i in range(n)]
        
        # Precompute neighbors
        neighbors_list = []
        for i in range(N):
            nbrs = [i]
            for k in range(n):
                current_digit = (i // powers[k]) % q
                for v in range(q):
                    if v == current_digit: continue
                    new_i = i - current_digit * powers[k] + v * powers[k]
                    nbrs.append(new_i)
            neighbors_list.append(nbrs)
        neighbors = np.array(neighbors_list, dtype=np.int32)
        
        def index_to_digits(idx):
            digits = []
            temp = idx
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            return digits

        def run_greedy(random_seed=None, use_noise=False):
            if random_seed is not None:
                np.random.seed(random_seed)
                random.seed(random_seed)
                
            covered = np.zeros(N, dtype=bool)
            code = []
            
            # Pre-allocate noise array if needed
            noise = np.zeros(N) if use_noise else None
            
            while True:
                if use_noise:
                    # Add small random noise to break ties
                    # Max count is 17. Noise in [-0.1, 0.1] is enough to break ties without changing order significantly
                    noise = np.random.uniform(-0.1, 0.1, N)
                
                mask = ~covered[neighbors]
                counts = mask.sum(axis=1)
                
                if use_noise:
                    counts = counts + noise
                
                best_idx = counts.argmax()
                best_count = counts[best_idx]
                
                # If best_count is 0 (or close to 0 if noise), we are done
                # Check actual count without noise for termination
                actual_count = mask[best_idx].sum()
                if actual_count == 0:
                    break
                
                code.append(best_idx)
                covered[neighbors[best_idx]] = True
                
            return code

        start_time = time.time()
        deadline = 110 # 10s margin
        
        best_code = None
        best_len = float('inf')
        
        # Run 1: Deterministic
        code = run_greedy(random_seed=42, use_noise=False)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
        # Run multiple times with noise
        run_id = 0
        while time.time() < deadline:
            run_id += 1
            code = run_greedy(random_seed=run_id, use_noise=True)
            if len(code) < best_len:
                best_len = len(code)
                best_code = code.copy()
                
        # Convert best_code to required format
        result = [index_to_digits(idx) for idx in best_code]
        return result
