"""
Strategy: Spectral Alignment + Combinatorial Swap Local Search.

The current program uses smooth parametric families (Gaussian, etc.). 
This strategy recognizes that the ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) 
is scale invariant and bounded. To maximize it, we want to maximize ||h||_2^2 
relative to the peak of the convolution. 

A key insight is that for non-negative h, the maximum of the autoconvolution 
||h*h||_inf is at least ||h||_1^2 / N (by averaging) but can be much larger 
if h is concentrated. However, if h is a delta function, ||h||_2^2 = 1, 
||h||_1 = 1, ||h*h||_inf = 1, so R=1. 
If h is uniform, ||h||_2^2 = 1/N, ||h||_1 = 1, ||h*h||_inf = 1/N (peak of triangle), 
so R = (1/N) / (1 * 1/N) = 1.

Wait, let's re-evaluate.
h = delta: h=[1,0,..]. L2^2=1, L1=1, Conv=[0,..,1,..0], Max=1. R=1.
h = uniform: h=[1/N,..,1/N]. L2^2=1/N, L1=1, Conv is triangle with peak 1/N. R=1.

Where does the improvement come from?
The inspiration mentions "Sidon sets" or "sparse support".
If h is sparse, say support S. 
Let h_i = 1 for i in S, 0 otherwise.
L2^2 = |S|. L1 = |S|.
Conv is the autocorrelation of the indicator of S.
Max conv is the maximum number of pairs (i,j) in S such that i-j = k for some k.
Actually, conv_k = sum h_i h_{i-k}. This is the number of pairs in S with difference k.
The peak is the maximum difference count.
R = |S| / (|S| * max_diff_count) = 1 / max_diff_count.
To maximize R, we want to MINIMIZE the maximum difference count.
This is the problem of finding a set S with minimal maximum multiplicity of differences.
This is related to Sidon sets (where max_diff_count is small, ideally 1 or 2).
If max_diff_count = 1, R=1.
If we can get max_diff_count < 1? No, counts are integers.
Wait, if h is not 0/1 but real-valued?

Let's look at the seed score 1.1576. This is > 1.
How can R > 1?
R = L2^2 / (L1 * ConvMax).
Let h = [a, b].
L2^2 = a^2+b^2. L1 = a+b.
Conv: [a^2, ab, b^2, ab, a^2] (full). Max = max(a^2, ab, b^2).
If a=b, Max = a^2. R = 2a^2 / (2a * a^2) = 1/a. If a=1, R=1.
If a=1, b=0. R=1.
If a=1, b=0.5.
L2^2 = 1.25. L1 = 1.5.
Conv: [1, 0.5, 0.25, 0.5, 1]. Max = 1.
R = 1.25 / 1.5 = 0.833.

How to get > 1?
We need L2^2 / (L1 * ConvMax) > 1 => L2^2 > L1 * ConvMax.
Note that ConvMax >= L1^2 / N (by Cauchy-Schwarz or Jensen on the average of conv values? 
Sum of conv values = L1^2. There are 2N-1 values. So average is L1^2/(2N-1).
So ConvMax >= L1^2/(2N-1).
Then R <= L2^2 / (L1 * L1^2/(2N-1)) = (2N-1) * L2^2 / L1^3.
By power mean, L2^2 >= L1^2 / N.
So R <= (2N-1)/N * (L1^2/N) / L1^3 ? No.

Let's test a specific case.
If h is a "comb" or has specific structure.
Actually, the "Sidon" idea suggests that if we pick a set of indices where differences are unique, the conv peaks are low.
But for real values, we can balance the peaks.

The strategy:
1. Start with a random sparse support (Sidon-like).
2. Use local search (swapping values, moving mass) to minimize ConvMax while keeping L2^2 high relative to L1.
3. Since it's scale invariant, fix L1=1. Maximize L2^2 / ConvMax.

We will implement a local search on the vector h.
Initial guess: A Sidon set of indices with equal weights.
Then refine weights and support.
"""

import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Initial best: Uniform
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    # Strategy: Combinatorial Local Search
    # 1. Generate random sparse supports (Sidon-like)
    # 2. Optimize weights on that support
    # 3. Try moving support elements
    
    rng = np.random.default_rng(42)
    
    def optimize_weights_on_support(support_indices, h_vals):
        # h_vals are the values at support_indices, others 0
        # We want to maximize ratio. 
        # Simple gradient-free local search on the support values.
        # Since N is small, we can do a few iterations of coordinate ascent.
        # Fix L1=1 for optimization.
        
        current_h = np.zeros(N)
        current_h[support_indices] = h_vals
        current_l1 = np.sum(current_h)
        if current_l1 < 1e-10:
            return current_h, 0.0
        
        current_h /= current_l1 # Normalize
        current_score = compute_ratio(current_h)
        
        # Coordinate ascent on support values
        for _ in range(20):
            improved = False
            for idx in support_indices:
                # Try perturbing h[idx]
                old_val = current_h[idx]
                if old_val < 1e-10:
                    continue
                # Try small changes
                for delta_factor in [0.5, 0.8, 1.2, 1.5, 0.0]:
                    new_val = old_val * delta_factor
                    if new_val < 0:
                        continue
                    new_h = current_h.copy()
                    new_h[idx] = new_val
                    new_l1 = np.sum(new_h)
                    if new_l1 < 1e-10:
                        continue
                    new_h /= new_l1
                    new_score = compute_ratio(new_h)
                    if new_score > current_score + 1e-12:
                        current_h = new_h
                        current_score = new_score
                        improved = True
                        break
            if not improved:
                break
        return current_h, current_score

    def generate_sidon_support(k):
        # Generate a set of k indices with minimal difference collisions
        # Simple heuristic: Random selection, check collisions
        best_set = None
        best_collisions = float('inf')
        for _ in range(100):
            S = set(rng.choice(N, k, replace=False))
            diffs = {}
            for i in S:
                for j in S:
                    if i != j:
                        d = abs(i - j)
                        diffs[d] = diffs.get(d, 0) + 1
            max_coll = max(diffs.values()) if diffs else 0
            if max_coll < best_collisions:
                best_collisions = max_coll
                best_set = list(S)
        return sorted(best_set) if best_set else []

    # Try different sparsities
    for k in range(5, 26):
        if time.time() > deadline:
            break
        
        # Generate multiple candidate supports for this k
        for trial in range(5):
            if time.time() > deadline:
                break
            
            support = generate_sidon_support(k)
            if len(support) < k:
                continue
            
            # Initialize weights: equal
            h_init = np.zeros(N)
            h_init[support] = 1.0 / k
            
            # Optimize weights
            h_opt, score_opt = optimize_weights_on_support(support, h_init[support])
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
    # Final global refinement: 
    # Take the best_h found, and perform a broader local search allowing support changes.
    # We will try to swap a non-zero element with a zero element, or adjust values.
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Sort indices by value to identify "core" support
    # Perform random swaps
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        # Randomly pick two indices
        i, j = rng.integers(0, N, 2)
        if i == j:
            continue
            
        # Try moving mass from i to j
        # Or just swap values? 
        # Let's try: new_h[i] = current_h[i] * (1-eps), new_h[j] = current_h[j] + eps*current_h[i]
        # This is a random walk in the simplex.
        
        # Better: Try to set one of them to 0 or adjust ratio.
        # Let's try a few specific moves:
        # 1. Zero out i
        # 2. Zero out j
        # 3. Equalize i and j
        # 4. Random perturbation
        
        moves = []
        
        # Move 1: Zero out i
        h1 = current_h.copy()
        h1[i] = 0.0
        moves.append(h1)
        
        # Move 2: Zero out j
        h2 = current_h.copy()
        h2[j] = 0.0
        moves.append(h2)
        
        # Move 3: Equalize
        h3 = current_h.copy()
        avg = (current_h[i] + current_h[j]) / 2.0
        h3[i] = avg
        h3[j] = avg
        moves.append(h3)
        
        # Move 4: Random scale
        h4 = current_h.copy()
        scale = rng.uniform(0.5, 1.5)
        h4[i] *= scale
        h4[j] /= scale
        h4 = np.maximum(h4, 0)
        moves.append(h4)
        
        for mh in moves:
            l1 = np.sum(mh)
            if l1 < 1e-10:
                continue
            mh /= l1
            s = compute_ratio(mh)
            if s > current_score + 1e-12:
                current_h = mh
                current_score = s
                best_h = mh.copy()
                best_score = s
                break # Accept best move and restart loop
            
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
