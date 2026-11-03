"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def ratio(h):
        h = np.asarray(h, dtype=float)
        l2_sq = float(np.dot(h, h))
        l1 = float(np.sum(h))
        if l1 == 0.0:
            return 0.0
        # autoconvolution (linear, full)
        ac = np.convolve(h, h, mode='full')
        sup = float(np.max(ac))
        if sup == 0.0:
            return 0.0
        return l2_sq / (l1 * sup)

    def autoconvolution_peak(h):
        """Compute the maximum value of the autoconvolution."""
        ac = np.convolve(h, h, mode='full')
        return float(np.max(ac))

    # Strategy: sparse, exponentially decaying sequence on a Sidon set
    # The idea is to place large weights at positions that minimize autoconvolution overlap
    # with existing support, while using exponential decay to concentrate L2 norm
    
    # First, generate a Sidon set (set where all pairwise sums are distinct)
    def generate_sidon_set(max_size, n_positions):
        """Generate a Sidon set of size up to max_size from positions 0..n_positions-1"""
        S = [0]
        used_sums = {0}
        for x in range(1, n_positions):
            new_sums = [x + s for s in S]
            new_sums.append(2 * x)
            collision = False
            for s in new_sums:
                if s in used_sums:
                    collision = True
                    break
            if not collision:
                S.append(x)
                for s in new_sums:
                    used_sums.add(s)
                if len(S) >= max_size:
                    break
        return S

    # Try multiple configurations with different Sidon set sizes and decay rates
    best_h = np.zeros(N)
    best_r = 0.0

    # Generate several Sidon sets of different sizes
    sidon_sizes = [8, 12, 16, 20, 25]
    
    for sidon_size in sidon_sizes:
        if time.time() > deadline - 30:
            break
            
        S = generate_sidon_set(sidon_size, N)
        k = len(S)
        if k < 2:
            continue
            
        # Try different exponential decay rates
        for decay in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline - 20:
                    break
                    
            # Create exponentially decaying weights on the Sidon set
            # Sort S and assign decaying weights
            S_sorted = sorted(S)
            weights = np.array([decay ** i for i in range(k)])
            
            # Normalize to unit L1
            weights = weights / np.sum(weights)
            
            h = np.zeros(N)
            for idx, wt in zip(S_sorted, weights):
                h[idx] = wt
            
            r = ratio(h)
            if r > best_r:
                best_r = r
                best_h = h.copy()
        
        # Also try with the largest weight at position 0 (center of autoconvolution)
        # and smaller weights elsewhere
        S_sorted = sorted(S)
        if S_sorted[0] == 0:
            # Put largest weight at 0
            weights = np.zeros(k)
            weights[0] = 1.0
            for i in range(1, k):
                weights[i] = 0.1 * (0.5 ** i)
            weights = weights / np.sum(weights)
            
            h = np.zeros(N)
            for idx, wt in zip(S_sorted, weights):
                h[idx] = wt
            
            r = ratio(h)
            if r > best_r:
                best_r = r
                best_h = h.copy()

    # Now refine the best solution with local search
    if best_r > 0:
        support = [i for i in range(N) if best_h[i] > 1e-10]
        if len(support) > 0:
            current_support = support
            current_w = np.array([best_h[i] for i in support])
            current_r = best_r
            
            # Local coordinate ascent on weights
            for iteration in range(500):
                if time.time() > deadline - 10:
                    break
                    
                improved = False
                n_s = len(current_support)
                
                for i in range(n_s):
                    base = current_w[i]
                    if base < 1e-10:
                        continue
                        
                    for factor in [0.7, 0.85, 0.95, 1.05, 1.15, 1.3]:
                        new_w = current_w.copy()
                        new_w[i] = base * factor
                        if np.sum(new_w) > 0:
                            new_w = new_w / np.sum(new_w) * np.sum(current_w)
                        
                        h = np.zeros(N)
                        for idx, wt in zip(current_support, new_w):
                            h[idx] = wt
                        r = ratio(h)
                        
                        if r > current_r + 1e-12:
                            current_w = new_w
                            current_r = r
                            improved = True
                            break
                
                # Try adding a new element
                if time.time() < deadline - 15:
                    candidates = [i for i in range(N) if i not in current_support]
                    random.shuffle(candidates)
                    for cand in candidates[:10]:
                        new_support = current_support + [cand]
                        new_support.sort()
                        new_w = np.ones(len(new_support)) / len(new_support)
                        
                        h = np.zeros(N)
                        for idx, wt in zip(new_support, new_w):
                            h[idx] = wt
                        r = ratio(h)
                        
                        if r > current_r + 1e-12:
                            current_support = new_support
                            current_w = new_w
                            current_r = r
                            break
                
                # Try removing an element
                if len(current_support) > 2 and time.time() < deadline - 15:
                    for i in range(len(current_support)):
                        new_support = [x for j, x in enumerate(current_support) if j != i]
                        new_w = np.ones(len(new_support)) / len(new_support)
                        
                        h = np.zeros(N)
                        for idx, wt in zip(new_support, new_w):
                            h[idx] = wt
                        r = ratio(h)
                        
                        if r > current_r + 1e-12:
                            current_support = new_support
                            current_w = new_w
                            current_r = r
                            break
                
                if not improved and time.time() < deadline - 15:
                    # Try a random perturbation
                    if len(current_support) > 2:
                        idx_to_remove = random.randint(0, len(current_support) - 1)
                        new_support = [x for j, x in enumerate(current_support) if j != idx_to_remove]
                        if len(new_support) > 0:
                            new_w = np.ones(len(new_support)) / len(new_support)
                            h = np.zeros(N)
                            for idx, wt in zip(new_support, new_w):
                                h[idx] = wt
                            r = ratio(h)
                            if r > current_r:
                                current_support = new_support
                                current_w = new_w
                                current_r = r
                
                if current_r > best_r:
                    best_r = current_r
                    best_h = np.zeros(N)
                    for idx, wt in zip(current_support, current_w):
                        best_h[idx] = wt

    # Final check
    if best_r <= 0:
        # Fallback: uniform
        best_h = np.ones(N) / N
        best_r = ratio(best_h)

    return list(best_h)
# EVOLVE-BLOCK-END
