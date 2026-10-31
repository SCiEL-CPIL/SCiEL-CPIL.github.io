"""Strategy: Geometric Gapped Support Optimization for Second Autocorrelation Inequality."""
import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h is a 1D array of non-negative floats.
    """
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
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    # The problem asks to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # Scale invariance allows us to fix ||h||_1 = 1.
    # Then R = ||h||_2^2 / ||h*h||_inf.
    #
    # Key Insight from "Second Autocorrelation Inequality":
    # For non-negative sequences, the peak of the autocorrelation ||h*h||_inf 
    # is minimized (relative to ||h||_2^2) when the mass is spread out 
    # but in a way that avoids large constructive interference in the convolution.
    #
    # A uniform distribution (all ones) gives R = 1/N.
    # A delta distribution (one 1, rest 0) gives R = 1.
    # Intermediate structures can do better than 1/N but worse than 1? 
    # Wait, let's check the seed score. Seed is 1.0.
    # If seed is 1.0, it likely corresponds to a specific baseline.
    # Let's test simple cases:
    # h = [1, 0, ..., 0] -> L1=1, L2^2=1, conv=[1], max=1. R=1.
    # h = [1/50, ..., 1/50] -> L1=1, L2^2=50*(1/50)^2=1/50. 
    # conv is triangular. Peak is at center. 
    # conv[k] = sum h_i h_{i+k}. For uniform, peak is 50 * (1/50)^2 = 1/50.
    # R = (1/50) / (1 * 1/50) = 1.
    #
    # So both Delta and Uniform give R=1.
    # The current program scores 1.1576. So it found something better than 1.
    #
    # Let's look at "Gapped" structures.
    # Suppose we have a block of K ones, and the rest zeros.
    # L1 = K * h_val. Let's fix L1=1, so h_val = 1/K.
    # L2^2 = K * (1/K)^2 = 1/K.
    # Convolution of a block of K ones (scaled by 1/K):
    # The convolution is a triangle of height K * (1/K)^2 = 1/K at the center, 
    # decaying linearly to 0 at the edges.
    # Max conv = 1/K.
    # R = (1/K) / (1 * 1/K) = 1.
    #
    # So any contiguous block of constant height also gives R=1.
    #
    # What if we have TWO separated blocks?
    # Block 1: length K1, height h1. Block 2: length K2, height h2.
    # Separation G.
    # L1 = K1*h1 + K2*h2 = 1.
    # L2^2 = K1*h1^2 + K2*h2^2.
    # Convolution:
    # Self-conv of Block 1: Peak P1 = K1*h1^2. Width ~ 2K1.
    # Self-conv of Block 2: Peak P2 = K2*h2^2. Width ~ 2K2.
    # Cross-conv: Two humps. Max cross term roughly 2 * (sum Block 1) * (sum Block 2) / (something)?
    # Actually, cross-conv max is bounded by (sum h1) * (sum h2) = (K1 h1) (K2 h2).
    # If blocks are far apart, the peaks of self-conv and cross-conv are separated.
    # The global max of convolution will be max(P1, P2, MaxCross).
    #
    # Let's try to make P1 and P2 small, but L2^2 large.
    # This suggests making h_i large in few places? But that increases L2^2.
    #
    # Let's re-evaluate the ratio for a specific "Two Spikes" case.
    # h[0] = a, h[G] = b. Others 0.
    # L1 = a+b = 1.
    # L2^2 = a^2 + b^2.
    # Conv:
    # (0,0) -> a^2
    # (G,G) -> b^2
    # (0,G) and (G,0) -> ab
    # Max conv = max(a^2, b^2, ab).
    # Since a,b > 0, a^2+b^2 >= 2ab.
    # If a=b=0.5, L2^2 = 0.5. Max conv = 0.25. R = 0.5/0.25 = 2.
    #
    # Wait, R=2 is much better than 1.1576.
    # Let's verify.
    # h = [0.5, 0, ..., 0, 0.5] (at indices 0 and G).
    # L1 = 1.
    # L2^2 = 0.25 + 0.25 = 0.5.
    # Conv:
    # Index 0: 0.5*0.5 = 0.25
    # Index G: 0.5*0.5 = 0.25
    # Index 2G? No, indices are 0 and G.
    # Conv indices:
    # 0: h[0]h[0] = 0.25
    # G: h[0]h[G] + h[G]h[0] = 0.25 + 0.25 = 0.5? 
    # NO. Convolution definition: (h*h)[k] = sum_i h[i] h[k-i].
    # If h[0]=a, h[G]=b.
    # k=0: h[0]h[0] = a^2.
    # k=G: h[0]h[G] + h[G]h[0] = ab + ba = 2ab.
    # k=2G: h[G]h[G] = b^2.
    #
    # So Max Conv = max(a^2, 2ab, b^2).
    # If a=b=0.5, Max Conv = max(0.25, 0.5, 0.25) = 0.5.
    # R = 0.5 / 0.5 = 1.
    #
    # My previous calculation of cross-conv was wrong for discrete convolution with two non-zero points.
    # The cross term adds up!
    #
    # Let's try 3 points?
    # h[0]=a, h[G]=b, h[2G]=c.
    # L1 = a+b+c = 1.
    # L2^2 = a^2+b^2+c^2.
    # Conv peaks:
    # k=0: a^2
    # k=G: 2ab
    # k=2G: 2ac + b^2
    # k=3G: 2bc
    # k=4G: c^2
    #
    # To maximize R, we want L2^2 large (mass concentrated) but Conv Max small (mass spread).
    # Concentrating mass increases L2^2 but also increases Conv peaks quadratically.
    # Spreading mass decreases L2^2 but decreases Conv peaks.
    #
    # Let's try equal spacing: a=b=c=1/3.
    # L2^2 = 3*(1/9) = 1/3.
    # Conv:
    # k=0: 1/9
    # k=G: 2/9
    # k=2G: 2/9 + 1/9 = 3/9 = 1/3
    # k=3G: 2/9
    # k=4G: 1/9
    # Max Conv = 1/3.
    # R = (1/3) / (1/3) = 1.
    #
    # It seems for any "sparse" set of points with equal weights, R=1?
    # Let's check unequal weights.
    # a=0.8, b=0.1, c=0.1.
    # L2^2 = 0.64 + 0.01 + 0.01 = 0.66.
    # Conv:
    # k=0: 0.64
    # k=G: 0.16
    # k=2G: 0.16 + 0.01 = 0.17
    # Max Conv = 0.64.
    # R = 0.66 / 0.64 = 1.03125.
    #
    # Improvement!
    # Let's optimize weights for 3 points.
    # Maximize (a^2+b^2+c^2) / max(a^2, 2ab, 2ac+b^2, 2bc, c^2) subject to a+b+c=1.
    #
    # The term 2ac+b^2 is interesting.
    # If we make a large, a^2 dominates. R -> 1.
    # If we make a,b,c balanced, R=1.
    # The "sweet spot" is when the max convolution term is NOT the largest square, but a cross term?
    # No, a^2 is always >= 2ac+b^2? 
    # a^2 - (2ac+b^2) = (a-b)^2 - 2c(a+b)? No.
    #
    # Let's just do a numerical search over the "spike" configurations.
    # Since N=50, we can try all subsets of indices? 2^50 is too big.
    # But we can try regular patterns: Spacing G, Number of spikes K.
    #
    # Strategy:
    # 1. Iterate over K (number of non-zero elements) from 1 to 50.
    # 2. Iterate over Spacing G from 1 to N//K.
    # 3. For each (K, G), we have K positions: 0, G, 2G, ..., (K-1)G.
    # 4. We need to find optimal weights w_0..w_{K-1} summing to 1.
    # 5. This is a small optimization problem (K dims).
    #    We can use a simple random search or gradient ascent for weights.
    #    Since K is small (<=50), and we have time, we can do a grid search or random restarts.
    
    best_h = np.zeros(N)
    best_h[0] = 1.0 # Initial
    best_score = compute_ratio(best_h)
    
    # Precompute indices for potential spike patterns
    # We will test patterns defined by (K, G)
    
    # To speed up weight optimization, we'll use a simple randomized hill climbing
    # or just a dense random search for weights, as K is small.
    
    max_k = 50
    max_g = N // 2 # Max gap
    
    # We can't iterate all K,G and do heavy optimization for each if time is tight.
    # But N=50 is small.
    
    # Let's prioritize K=2, K=3, K=4, K=5 with various G.
    # And also "dense" blocks which we know give R=1, so skip if K is large and G=1?
    # Actually, G=1 is contiguous block. We know R=1 for uniform weights.
    # But non-uniform weights in a contiguous block might do better?
    # Let's include G=1 in the search.
    
    # Time budget check
    if time.time() > deadline:
        return list(best_h)
        
    # Search space:
    # K: 2 to 10 (Sparse is better for R>1)
    # G: 1 to N//K
    
    # For each (K, G), run a local optimization on weights.
    # Weights w[0..K-1].
    
    rng = np.random.default_rng(42)
    
    # We will do a coarse grid of K and G, then refine the best ones.
    
    candidates = []
    
    for K in range(2, 11):
        if time.time() > deadline - 10:
            break
        max_g = N // K
        for G in range(1, max_g + 1):
            # Positions
            pos = np.arange(K) * G
            if pos[-1] >= N:
                continue
            
            # Optimize weights for this fixed geometry
            # Use a few random restarts with local search
            best_w = np.ones(K) / K
            best_local_score = -1
            best_local_conv_max = 0
            best_local_l2 = 0
            
            # We can vectorize the convolution calculation for fixed geometry?
            # Convolution of spikes at pos with weights w:
            # (h*h)[k] = sum_{i,j: pos_i - pos_j = k} w_i w_j
            # This is a quadratic form.
            # We want to maximize w^T D w / max_k (w^T A_k w)
            # where D is identity (for L2^2) and A_k are matrices for each lag.
            
            # For small K, we can just compute directly.
            
            num_restarts = 50
            for r in range(num_restarts):
                # Dirichlet random start
                w = rng.dirichlet(np.ones(K))
                
                # Local search: 200 steps
                for step in range(200):
                    # Compute ratio
                    h = np.zeros(N)
                    h[pos] = w
                    score = compute_ratio(h)
                    
                    if score > best_local_score:
                        best_local_score = score
                        best_w = w.copy()
                        
                    # Perturb
                    # Move mass from i to j
                    i = rng.integers(K)
                    j = rng.integers(K)
                    if i == j: continue
                    delta = rng.uniform(-0.1, 0.1)
                    new_w = w.copy()
                    new_w[i] -= delta
                    new_w[j] += delta
                    if new_w[i] < 0 or new_w[j] < 0:
                        continue
                    new_w = new_w / np.sum(new_w)
                    w = new_w
                    
            if best_local_score > best_score:
                best_score = best_local_score
                best_h = np.zeros(N)
                best_h[pos] = best_w
                candidates.append((best_score, K, G, best_w))
                
    # Refine the top candidates
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:5]
        
        for score, K, G, w_init in top_candidates:
            if time.time() > deadline - 5:
                break
            pos = np.arange(K) * G
            if pos[-1] >= N:
                continue
                
            w = w_init.copy()
            # Finer local search
            for step in range(500):
                h = np.zeros(N)
                h[pos] = w
                current_score = compute_ratio(h)
                
                if current_score > best_score:
                    best_score = current_score
                    best_h = h.copy()
                    
                # Smaller perturbations
                i = rng.integers(K)
                j = rng.integers(K)
                if i == j: continue
                delta = rng.uniform(-0.05, 0.05)
                new_w = w.copy()
                new_w[i] -= delta
                new_w[j] += delta
                if new_w[i] < 0 or new_w[j] < 0:
                    continue
                new_w = new_w / np.sum(new_w)
                w = new_w
                
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
