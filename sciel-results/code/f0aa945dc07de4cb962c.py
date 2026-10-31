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
    # Use FFT for convolution? N=50 is small, direct convolution is fine and faster overhead-wise.
    # np.convolve(h, h, mode='full')
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Initialize with a bimodal distribution, which is a strong candidate for minimizing 
    # the central peak of the autoconvolution while maintaining high L2 norm.
    # Two peaks at the ends of the array.
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_h[N-1] = 1.0
    best_score = compute_ratio(best_h)
    
    # Also consider a single peak at the edge (uniform is 1.0, single peak is 1.0? 
    # Let's check: h=[1,0..0]. L2=1, L1=1, conv=[1,0..0], max=1. Ratio=1.
    # h=[1,1,0..0]. L2=2, L1=2. conv=[1,2,1], max=2. Ratio=2/(2*2)=0.5.
    # h=[1,0,1,0..]. L2=2, L1=2. conv=[1,0,1,0,1], max=1. Ratio=2/(2*1)=1.
    # h=[1,0,0,1,0..]. L2=2, L1=2. conv=[1,0,0,1,0,0,1], max=1. Ratio=1.
    # h=[1,0,0,0,1,0..]. L2=2, L1=2. conv=[1,0,0,0,1,0,0,0,1], max=1. Ratio=1.
    # It seems for sparse separated peaks, the ratio is 1.
    # To get > 1, we need L2^2 / (L1 * MaxConv) > 1 => L2^2 > L1 * MaxConv.
    # Since MaxConv >= L2^2 / L1 is not generally true? 
    # Actually, by Cauchy-Schwarz or similar, is there an upper bound?
    # The task says "maximize the ratio... score is the lower bound on C2".
    # Wait, the seed score is 1.0. The current best is 1.157.
    # So values > 1 exist.
    
    # Let's start with a random sparse initialization and refine.
    rng = np.random.default_rng(42)
    
    # Strategy: Simulated Annealing on the vector h.
    # Move types:
    # 1. Move mass from i to j.
    # 2. Set a random bin to 0 (sparsification).
    # 3. Add mass to a random 0-bin (densification).
    # 4. Swap values of two bins.
    
    # We maintain sum(h) = 1 for normalization, but the ratio is scale invariant.
    # It's easier to work with unnormalized h and just compute the ratio.
    # However, scaling doesn't change the ratio.
    # Let's fix L1 = 1 to keep numbers stable.
    
    h = best_h.copy()
    h = h / np.sum(h)
    current_score = best_score
    
    # Annealing parameters
    T = 1.0
    cooling_rate = 0.9999
    min_T = 1e-5
    
    # Precompute indices for efficiency? 
    # N=50 is small.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Generate a neighbor
        move_type = rng.integers(0, 4)
        h_new = h.copy()
        
        if move_type == 0:
            # Move mass from i to j
            i = rng.integers(N)
            j = rng.integers(N)
            if i == j:
                continue
            # Move a fraction of mass
            delta = rng.uniform(0.01, 0.1) * h[i]
            if h[i] < delta:
                delta = h[i]
            h_new[i] -= delta
            h_new[j] += delta
            
        elif move_type == 1:
            # Sparsify: set a random positive bin to 0 and redistribute its mass to a neighbor
            positive_indices = np.where(h > 1e-6)[0]
            if len(positive_indices) == 0:
                continue
            i = positive_indices[rng.integers(len(positive_indices))]
            mass = h[i]
            h_new[i] = 0.0
            # Redistribute to a random other bin
            j = rng.integers(N)
            if j != i:
                h_new[j] += mass
                
        elif move_type == 2:
            # Densify: take mass from a random positive bin and add to a random zero bin
            positive_indices = np.where(h > 1e-6)[0]
            zero_indices = np.where(h < 1e-6)[0]
            if len(positive_indices) == 0 or len(zero_indices) == 0:
                continue
            i = positive_indices[rng.integers(len(positive_indices))]
            j = zero_indices[rng.integers(len(zero_indices))]
            delta = rng.uniform(0.01, 0.1) * h[i]
            if h[i] < delta:
                delta = h[i]
            h_new[i] -= delta
            h_new[j] += delta
            
        else:
            # Swap or mix two bins
            i = rng.integers(N)
            j = rng.integers(N)
            if i == j:
                continue
            # Mix: h_new[i] = (1-a)h[i] + a h[j], h_new[j] = (1-a)h[j] + a h[i]
            a = rng.uniform(0.1, 0.9)
            h_new[i] = (1-a)*h[i] + a*h[j]
            h_new[j] = (1-a)*h[j] + a*h[i]

        # Ensure non-negativity
        h_new = np.maximum(h_new, 0)
        
        # Normalize to sum 1 (optional, but keeps scale consistent)
        s = np.sum(h_new)
        if s < 1e-300:
            continue
        h_new = h_new / s
        
        new_score = compute_ratio(h_new)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score > 0:
            h = h_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Accept with probability exp(delta_score / T)
            # Note: delta_score is negative.
            # We need to be careful with the scale of the score.
            # Score is around 1.0-1.2. 
            # T starts at 1.0, so exp(-0.1/1.0) ~ 0.9. Good.
            if rng.random() < np.exp(delta_score / T):
                h = h_new
                current_score = new_score
        
        # Cool down
        T = max(T * cooling_rate, min_T)
        
        # If we have cooled down significantly and haven't improved in a while, restart?
        # Simple restart logic: if T < 0.1, and no improvement in last 1000 steps, restart T.
        # For simplicity, let's just keep cooling.
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
