"""Erdos minimum overlap - find step heights minimizing max overlap with shifted complement."""
import numpy as np
import time

N = 100

def compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    N = len(h)
    width = 2.0 / N
    max_overlap = 0.0
    for s in range(1, N):
        overlap = np.sum(h[:N-s] * (1 - h[s:])) * width
        max_overlap = max(max_overlap, overlap)
    return max_overlap

def compute_max_overlap_fast(h):
    """Faster computation using convolution-like approach for binary h."""
    N = len(h)
    width = 2.0 / N
    max_overlap = 0.0
    # For binary h, overlap at shift s = count of i where h[i]=1 and h[i+s]=0
    # = sum(h[i] * (1-h[i+s])) for i=0..N-s-1
    # = sum(h[i]) for i=0..N-s-1 minus sum(h[i]*h[i+s]) for i=0..N-s-1
    # Precompute prefix sums of h
    h_int = h.astype(np.int32)
    for s in range(1, N):
        # Count positions where h[i]=1 and h[i+s]=0
        # = number of 1s in h[0:N-s] - number of 11 pairs at distance s
        ones_in_range = np.sum(h_int[:N-s])
        ones_ones = np.sum(h_int[:N-s] * h_int[s:])
        overlap = (ones_in_range - ones_ones) * width
        max_overlap = max(max_overlap, overlap)
    return max_overlap

def local_search_binary(h, max_iters=2000):
    """Local search: flip bits to reduce max overlap while maintaining sum=N/2."""
    h = h.copy().astype(np.float64)
    N = len(h)
    current_score = compute_max_overlap_fast(h)
    
    # Track which positions are 1
    ones_mask = h > 0.5
    
    for iteration in range(max_iters):
        # Try flipping a 1 to 0 and a 0 to 1 (maintain sum)
        ones_indices = np.where(ones_mask)[0]
        zeros_indices = np.where(~ones_mask)[0]
        
        if len(ones_indices) == 0 or len(zeros_indices) == 0:
            break
        
        # Try a random pair of flips
        best_pair = None
        best_new_score = current_score
        
        # Try several random pairs
        for _ in range(50):
            i = ones_indices[np.random.randint(len(ones_indices))]
            j = zeros_indices[np.random.randint(len(zeros_indices))]
            
            if i == j:
                continue
            
            # Flip i: 1->0, flip j: 0->1
            h[i] = 0.0
            h[j] = 1.0
            new_score = compute_max_overlap_fast(h)
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_pair = (i, j)
            
            # Revert
            h[i] = 1.0
            h[j] = 0.0
        
        if best_pair is not None:
            i, j = best_pair
            h[i] = 0.0
            h[j] = 1.0
            ones_mask[i] = False
            ones_mask[j] = True
            current_score = best_new_score
        else:
            # No improvement found in this iteration, try a larger neighborhood
            # Try flipping 2 ones to 0s and 2 zeros to 1s
            improved = False
            for _ in range(20):
                if len(ones_indices) < 2 or len(zeros_indices) < 2:
                    break
                i1 = ones_indices[np.random.randint(len(ones_indices))]
                i2 = ones_indices[np.random.randint(len(ones_indices))]
                if i1 == i2:
                    continue
                j1 = zeros_indices[np.random.randint(len(zeros_indices))]
                j2 = zeros_indices[np.random.randint(len(zeros_indices))]
                if j1 == j2:
                    continue
                
                h[i1] = 0.0
                h[i2] = 0.0
                h[j1] = 1.0
                h[j2] = 1.0
                new_score = compute_max_overlap_fast(h)
                
                if new_score < current_score:
                    ones_mask[i1] = False
                    ones_mask[i2] = False
                    ones_mask[j1] = True
                    ones_mask[j2] = True
                    current_score = new_score
                    improved = True
                    ones_indices = np.where(ones_mask)[0]
                    zeros_indices = np.where(~ones_mask)[0]
                    break
                else:
                    # Revert
                    h[i1] = 1.0
                    h[i2] = 1.0
                    h[j1] = 0.0
                    h[j2] = 0.0
            
            if not improved:
                break
    
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap(best_h)
    
    # Try periodic binary patterns as initial seeds
    initial_patterns = []
    
    # 1100 pattern (period 4)
    if N % 4 == 0:
        pattern_1100 = np.array([1.0, 1.0, 0.0, 0.0])
        h_1100 = np.tile(pattern_1100, N // 4)
        initial_patterns.append(h_1100)
    
    # 10 pattern (period 2)
    if N % 2 == 0:
        pattern_10 = np.array([1.0, 0.0])
        h_10 = np.tile(pattern_10, N // 2)
        initial_patterns.append(h_10)
    
    # 111000 pattern (period 6)
    if N % 6 == 0:
        pattern_111000 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        h_111000 = np.tile(pattern_111000, N // 6)
        initial_patterns.append(h_111000)
    
    # 11110000 pattern (period 8)
    if N % 8 == 0:
        pattern_11110000 = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        h_11110000 = np.tile(pattern_11110000, N // 8)
        initial_patterns.append(h_11110000)
    
    # Thue-Morse inspired pattern
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        tm = [0]
        for _ in range(int(np.log2(n)) + 1):
            tm = tm + [1 - x for x in tm]
        return np.array(tm[:n])
    
    tm_seq = thue_morse(N)
    # Make it balanced: ensure sum = N/2
    # Thue-Morse has roughly half 0s and half 1s
    if np.sum(tm_seq) != N / 2:
        # Adjust
        diff = N / 2 - np.sum(tm_seq)
        if diff > 0:
            # Need more 1s
            zeros_idx = np.where(tm_seq == 0)[0]
            for i in range(int(abs(diff))):
                tm_seq[zeros_idx[i]] = 1
        else:
            ones_idx = np.where(tm_seq == 1)[0]
            for i in range(int(abs(diff))):
                tm_seq[ones_idx[i]] = 0
    initial_patterns.append(tm_seq.astype(np.float64))
    
    # Random balanced binary sequences
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.zeros(N)
        ones_positions = np.random.choice(N, N//2, replace=False)
        h_rand[ones_positions] = 1.0
        initial_patterns.append(h_rand)
    
    # Local search from each initial pattern
    for h_init in initial_patterns:
        if time.time() > deadline - 5:
            break
        
        # Ensure sum is N/2
        if np.sum(h_init) != N / 2:
            continue
        
        h_optimized, score = local_search_binary(h_init, max_iters=500)
        
        if score < best_score:
            best_score = score
            best_h = h_optimized.copy()
    
    # Try more local search iterations from the best solution
    if time.time() < deadline - 10:
        h_final, score_final = local_search_binary(best_h, max_iters=1000)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    # Ensure sum is exactly N/2
    if np.sum(best_h) != N / 2:
        diff = N / 2 - np.sum(best_h)
        if abs(diff) < 1e-10:
            best_h = np.round(best_h, 10)
        else:
            best_h = np.full(N, 0.5)
    
    return best_h
