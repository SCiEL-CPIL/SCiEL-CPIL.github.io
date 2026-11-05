import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit = 140.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
    
    def count_mono(coloring):
        c = coloring[aps_arr]  # shape (total_aps, 4)
        return int(np.all(c == c[:, 0:1], axis=1).sum())
    
    def compute_deltas(coloring):
        """Compute delta for flipping each position. Vectorized."""
        c = coloring[aps_arr]  # (total_aps, 4)
        n0 = (c == 0).sum(axis=1)  # (total_aps,)
        n1 = 4 - n0
        
        # mono_mask: APs that are currently mono
        mono_mask = (n0 == 4) | (n1 == 4)
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # Process mono APs: for each mono AP, subtract 1 from all 4 positions
        if mono_mask.any():
            mono_aps = aps_arr[mono_mask]  # (k, 4)
            mono_flat = mono_aps.flatten()
            deltas -= np.bincount(mono_flat, minlength=n).astype(np.int32)
        
        # Process 3-1 APs (n0==3, n1==1): the odd one out is the position with color 1
        mask_31 = (n0 == 3) & (n1 == 1)
        if mask_31.any():
            ap_indices_31 = np.where(mask_31)[0]
            for idx in ap_indices_31:
                ap = aps_arr[idx]
                colors = c[idx]
                odd_pos = ap[colors == 1][0]
                deltas[odd_pos] += 1
        
        # Process 1-3 APs (n0==1, n1==3): the odd one out is the position with color 0
        mask_13 = (n0 == 1) & (n1 == 3)
        if mask_13.any():
            ap_indices_13 = np.where(mask_13)[0]
            for idx in ap_indices_13:
                ap = aps_arr[idx]
                colors = c[idx]
                odd_pos = ap[colors == 0][0]
                deltas[odd_pos] += 1
        
        return deltas
    
    def greedy_improve(coloring, max_sweeps=200, time_limit=None):
        """Greedy local search: repeatedly flip the position with the most negative delta."""
        current_score = count_mono(coloring)
        
        for sweep in range(max_sweeps):
            if time_limit is not None and time.time() > time_limit:
                break
            
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                break  # No improvement possible
            
            # Flip all positions with delta == min_delta (most negative)
            flip_mask = (deltas == min_delta)
            flip_positions = np.where(flip_mask)[0]
            
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_score += min_delta * len(flip_positions)
        
        return coloring, current_score
    
    def simulated_annealing(coloring, time_limit=None):
        """Simulated annealing with acceptance of worse moves."""
        current_score = count_mono(coloring)
        best_coloring = coloring.copy()
        best_score = current_score
        
        T = 5.0  # initial temperature
        T_min = 0.1
        cooling = 0.995
        
        rng_local = np.random.default_rng(int(time.time() * 1000) % 2**32)
        
        while time_limit is None or time.time() < time_limit:
            if T < T_min:
                break
            
            # Pick a random position to flip
            pos = rng_local.integers(0, n)
            old_color = coloring[pos]
            coloring[pos] = 1 - old_color
            
            # Compute new score
            new_score = count_mono(coloring)
            delta = new_score - current_score
            
            if delta <= 0:
                # Accept improvement
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
            else:
                # Accept worse move with probability exp(-delta/T)
                if rng_local.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Reject: revert
                    coloring[pos] = old_color
            
            T *= cooling
        
        return best_coloring, best_score
    
    def multi_start_search(time_limit):
        """Combine multiple strategies: random restarts with greedy, then simulated annealing."""
        best_coloring = None
        best_score = float('inf')
        
        rng = np.random.default_rng(12345)
        
        # Phase 1: Random restarts with greedy improvement
        while time.time() < time_limit - 30.0:
            coloring = rng.integers(0, 2, n)
            coloring, score = greedy_improve(coloring, max_sweeps=100, time_limit=time_limit - 25.0)
            
            if score < best_score:
                best_score = score
                best_coloring = coloring.copy()
            
            if time.time() > time_limit - 35.0:
                break
        
        # Phase 2: Simulated annealing from the best solution found
        if best_coloring is not None and time.time() < time_limit - 10.0:
            sa_coloring, sa_score = simulated_annealing(best_coloring.copy(), time_limit=time_limit - 5.0)
            if sa_score < best_score:
                best_score = sa_score
                best_coloring = sa_coloring.copy()
        
        # Phase 3: Final greedy polish
        if best_coloring is not None and time.time() < time_limit - 2.0:
            final_coloring, final_score = greedy_improve(best_coloring.copy(), max_sweeps=500, time_limit=time_limit - 1.0)
            if final_score < best_score:
                best_score = final_score
                best_coloring = final_coloring.copy()
        
        if best_coloring is None:
            return [int(x) for x in np.random.randint(0, 2, n)]
        return [int(x) for x in best_coloring]
    
    result = multi_start_search(limit)
    return result
# EVOLVE-BLOCK-END
