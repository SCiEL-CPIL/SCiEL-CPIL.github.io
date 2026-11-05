import time
import random
import numpy as np
import math

class _APEvaluator:
    """
    Fast evaluator for max |sum| over all arithmetic progressions.
    Uses precomputed index arrays and vectorized cumulative sums.
    """
    def __init__(self, n):
        self.n = n
        self.ap_data = []
        for d in range(1, n):
            max_len_d = (n - 1) // d + 1
            r_idx = np.arange(d)[:, None]
            k_idx = np.arange(max_len_d)[None, :]
            flat_indices = r_idx + k_idx * d
            valid_mask = flat_indices < n
            safe_indices = np.where(valid_mask, flat_indices, 0)
            self.ap_data.append((safe_indices, valid_mask, d))
    
    def eval(self, signs):
        """
        Returns (max_disc, worst_d, worst_r).
        signs: list or array of +1/-1 of length n.
        """
        n = self.n
        signs_arr = np.asarray(signs, dtype=np.int32)
        global_max = 1
        worst_d = 1
        worst_r = 0
        
        for safe_indices, valid_mask, d in self.ap_data:
            S_vals = signs_arr[safe_indices].copy()
            S_vals[~valid_mask] = 0
            
            C = np.cumsum(S_vals, axis=1)
            
            # Max subarray per row
            MinPref = np.minimum.accumulate(C, axis=1)
            MaxSub = C - MinPref
            row_maxs = np.max(MaxSub, axis=1)
            
            # Min subarray per row
            MaxPref = np.maximum.accumulate(C, axis=1)
            MinSub = C - MaxPref
            row_mins = np.min(MinSub, axis=1)
            
            current_max_d = int(np.max(row_maxs))
            current_min_d = int(np.min(row_mins))
            
            disc_d = max(current_max_d, -current_min_d)
            
            if disc_d > global_max:
                global_max = disc_d
                if current_max_d >= -current_min_d:
                    worst_r = int(np.argmax(row_maxs))
                else:
                    worst_r = int(np.argmin(row_mins))
                worst_d = d
            
            if d > n // 2 and global_max >= 2:
                break
        
        return global_max, worst_d, worst_r

def _get_worst_ap_indices(n, d, r):
    return list(range(r, n, d))

def _polish(signs, evaluator, deadline, rng, n):
    """
    Final polish: targeted descent on worst APs.
    Returns improved signs.
    """
    current_disc, worst_d, worst_r = evaluator.eval(signs)
    best_disc = current_disc
    best_signs = signs[:]
    
    while time.time() < deadline:
        # Collect all worst APs (indices where disc == current_disc)
        # For now, just use the single worst
        ap_indices = _get_worst_ap_indices(n, worst_d, worst_r)
        
        if len(ap_indices) == 0:
            break
        
        # Try flipping each element in the worst AP, keep best
        best_flip = None
        best_flip_disc = current_disc
        
        for idx in ap_indices:
            old_val = signs[idx]
            signs[idx] = -old_val
            d, wd, wr = evaluator.eval(signs)
            signs[idx] = old_val
            
            if d < best_flip_disc:
                best_flip_disc = d
                best_flip = idx
        
        if best_flip is not None:
            signs[best_flip] = -signs[best_flip]
            current_disc = best_flip_disc
            worst_d, worst_r = evaluator.eval(signs)
            worst_d, worst_r = worst_d, worst_r
            
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
            
            if best_disc == 1:
                break
        else:
            # No single flip in worst AP helps; try a few random flips
            improved = False
            for _ in range(20):
                idx = rng.randint(0, n - 1)
                old_val = signs[idx]
                signs[idx] = -old_val
                d, wd, wr = evaluator.eval(signs)
                if d < current_disc:
                    current_disc = d
                    worst_d, worst_r = wd, wr
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                        if best_disc == 1:
                            return best_signs
                else:
                    signs[idx] = old_val
            
            if not improved:
                break
    
    return best_signs

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    n = 256
    rng = random.Random(42)
    
    evaluator = _APEvaluator(n)
    
    # Thue-Morse initialization
    signs = [0] * n
    for i in range(n):
        cnt = bin(i).count('1')
        signs[i] = 1 if cnt % 2 == 0 else -1
    
    current_disc, worst_d, worst_r = evaluator.eval(signs)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Simulated Annealing
    temp = 5.0
    cool_rate = 0.9995
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline - 5:  # Leave 5s for polish
        
        # Decide move
        if rng.random() < 0.7 and worst_d > 0:
            ap_indices = _get_worst_ap_indices(n, worst_d, worst_r)
            if len(ap_indices) > 0:
                flip_idx = ap_indices[rng.randint(0, len(ap_indices) - 1)]
            else:
                flip_idx = rng.randint(0, n - 1)
        else:
            flip_idx = rng.randint(0, n - 1)
            
        old_val = signs[flip_idx]
        new_val = -old_val
        signs[flip_idx] = new_val
        
        new_disc, new_worst_d, new_worst_r = evaluator.eval(signs)
        
        delta = new_disc - current_disc
        
        if delta < 0 or (delta <= 2 and rng.random() < np.exp(-delta / max(temp, 0.1))):
            current_disc = new_disc
            worst_d = new_worst_d
            worst_r = new_worst_r
            
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
                no_improve = 0
            else:
                no_improve += 1
                
            temp *= cool_rate
        else:
            signs[flip_idx] = old_val
            no_improve += 1
            temp *= cool_rate
            
        if best_disc == 1:
            break
        
        if no_improve > max_no_improve:
            temp = 5.0
            no_improve = 0
    
    # Final polish
    polish_deadline = min(deadline, time.time() + 4)
    best_signs = _polish(best_signs, evaluator, polish_deadline, rng, n)
    
    return best_signs
