import numpy as np

N = 400

def construct():
    # 1. Select Support
    # Aim for M around 20-30.
    M = 25
    
    # We will optimize the support to minimize the maximum difference collision count.
    # This is a proxy for minimizing the potential peak of the autoconvolution.
    
    def get_diff_counts(supp):
        diffs = []
        for i in range(len(supp)):
            for j in range(i+1, len(supp)):
                diffs.append(abs(supp[i] - supp[j]))
        counts = np.bincount(diffs, minlength=N)
        return counts

    # Initialize with random permutation
    support = np.random.permutation(N)[:M]
    counts = get_diff_counts(support)
    max_count = np.max(counts)
    score = np.sum(counts**2)
    
    # Local search on support: swap two elements to reduce collision score
    # We prioritize reducing max_count, then total squared collisions.
    for _ in range(200):
        if max_count <= 1:
            break
            
        # Pick two random indices in support to swap
        i1, i2 = np.random.choice(M, 2, replace=False)
        old_i1, old_i2 = support[i1], support[i2]
        
        # Pick two random indices NOT in support
        # To be efficient, we can precompute the set of available indices
        # But M is small, so setdiff1d is okay, or we can just pick randoms and check.
        # Let's use a set for O(1) lookup of available indices.
        support_set = set(support.tolist())
        all_indices = set(range(N))
        available = list(all_indices - support_set)
        
        if len(available) < 2:
            continue
            
        new_i1, new_i2 = np.random.choice(available, 2, replace=False)
        
        support[i1] = new_i1
        support[i2] = new_i2
        
        new_counts = get_diff_counts(support)
        new_max = np.max(new_counts)
        new_score = np.sum(new_counts**2)
        
        if new_max < max_count or (new_max == max_count and new_score < score):
            counts = new_counts
            max_count = new_max
            score = new_score
        else:
            support[i1] = old_i1
            support[i2] = old_i2

    # 2. Assign Signs via Local Search
    # Now that support is fixed (or as good as we can get), optimize signs.
    signs = np.random.choice([-1.0, 1.0], size=M)
    
    def calc_peak(signs, support):
        Y = np.zeros(N)
        for k in range(M):
            for l in range(k+1, M):
                d = abs(support[k] - support[l])
                Y[d] += signs[k] * signs[l]
        return np.max(np.abs(Y))

    current_peak = calc_peak(signs, support)
    
    # Local Search: Flip a sign if it reduces the peak
    # Use a more aggressive search with multiple restarts if stuck?
    # For now, simple hill climbing.
    for _ in range(200):
        improved = False
        for k in range(M):
            old_sign = signs[k]
            signs[k] = -old_sign
            new_peak = calc_peak(signs, support)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                signs[k] = old_sign
        if not improved:
            break

    # 3. Construct Final Array
    # Scale by 1/sqrt(N) to keep L2 norm 1.
    # The task asks for an upper bound on C3. 
    # If C3 is defined as ||x * x||_inf / ||x||_2^2, scaling cancels out.
    # If it's just ||x * x||_inf, we want it small.
    # With L2 norm 1, the values are 1/sqrt(N).
    # The convolution of a sparse signal with M non-zeros will have magnitude ~ M / N.
    # Minimizing the peak collision count and using sign cancellation is key.
    
    x = np.zeros(N)
    x[support] = signs / np.sqrt(N)
    
    return x
