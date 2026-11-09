import time
import random
import math
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55  # 5s margin
    
    # Thue-Morse initialization
    def thue_morse(n):
        seq = []
        for i in range(n):
            c = bin(i).count('1')
            if c % 2 == 0:
                seq.append(-1)
            else:
                seq.append(1)
        return seq

    seq = thue_morse(n)
    arr = np.array(seq, dtype=np.int8)
    
    def get_cost(arr):
        """Compute max absolute signed sum over all arithmetic progressions using vectorized ops."""
        max_disc = 0
        nn = n
        # For each step size d, compute prefix sums along each residue class
        # then find max-min of prefix sums (which equals max subarray sum in absolute value)
        for d in range(1, nn + 1):
            for r in range(d):
                # Get indices r, r+d, r+2d, ...
                idx = np.arange(r, nn, d)
                if len(idx) == 0:
                    continue
                # Compute cumulative sum
                cumsum = np.cumsum(arr[idx])
                # The max absolute sum over any contiguous subsequence of this progression
                # equals max(cumsum) - min(cumsum) if we consider prefix sums starting from 0
                # Actually: max subarray sum = max(cumsum) - min(cumsum) where we prepend 0
                # Let's prepend 0 to cumsum
                cs = np.concatenate(([0], cumsum))
                disc = cs.max() - cs.min()
                if disc > max_disc:
                    max_disc = disc
        return max_disc

    best_seq = seq[:]
    best_cost = get_cost(arr)
    
    rng = random.Random(42)
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling = 0.998
    min_T = 0.1
    
    # We will run multiple restarts if time permits
    restarts = 0
    max_restarts = 3
    
    while time.time() < deadline:
        T = 10.0
        current_seq = best_seq[:]
        current_arr = np.array(current_seq, dtype=np.int8)
        current_cost = best_cost
        
        no_improve = 0
        patience = 3000
        
        while time.time() < deadline:
            idx = rng.randrange(n)
            current_seq[idx] *= -1
            current_arr[idx] = current_seq[idx]
            
            new_cost = get_cost(current_arr)
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = current_seq[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if delta == 0:
                    current_cost = new_cost
                    no_improve += 1
                elif rng.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    current_seq[idx] *= -1
                    current_arr[idx] = current_seq[idx]
                    no_improve += 1
            
            T *= cooling
            if T < min_T:
                break
            
            if no_improve > patience:
                break
        
        if time.time() >= deadline:
            break
        
        restarts += 1
        if restarts > max_restarts:
            break
    
    # Final greedy polish pass: try flipping each position, keep if it improves
    polish_arr = np.array(best_seq, dtype=np.int8)
    polish_cost = best_cost
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            polish_arr[i] *= -1
            new_c = get_cost(polish_arr)
            if new_c < polish_cost:
                polish_cost = new_c
                best_seq = polish_arr.tolist()
                improved = True
            else:
                polish_arr[i] *= -1
    
    best_cost = polish_cost
    return best_seq
