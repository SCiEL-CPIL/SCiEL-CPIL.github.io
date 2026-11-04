import numpy as np

N = 400

def construct():
    # 1. Select Support Size M
    # Smaller M reduces the number of terms in the convolution sum.
    M = 25
    
    # We will optimize the support to minimize the maximum difference collision count.
    # This is a proxy for minimizing the potential peak of the autoconvolution.
    
    def get_diff_counts(supp):
        # Vectorized difference calculation
        diffs = np.abs(supp[:, None] - supp[None, :])
        tri = np.triu_indices(M, k=1)
        d_vals = diffs[tri]
        counts = np.bincount(d_vals, minlength=N)
        return counts

    # Initialize with random permutation
    # Use a fixed seed for reproducibility of the algorithm's behavior
    rng = np.random.default_rng(42)
    
    support = rng.choice(N, M, replace=False)
    counts = get_diff_counts(support)
    max_count = np.max(counts)
    score = np.sum(counts**2)
    
    # Local search on support: swap two elements to reduce collision score
    # We prioritize reducing max_count, then total squared collisions.
    support_set = set(support.tolist())
    available = np.setdiff1d(np.arange(N), support, assume_unique=True)
    
    for _ in range(500):
        if max_count <= 1:
            break
            
        # Pick two random indices in support to swap
        i1, i2 = rng.choice(M, 2, replace=False)
        old_i1, old_i2 = support[i1], support[i2]
        
        # Pick two random indices NOT in support
        if len(available) < 2:
            continue
            
        new_i1, new_i2 = rng.choice(available, 2, replace=False)
        
        # Update support
        support[i1] = new_i1
        support[i2] = new_i2
        
        new_counts = get_diff_counts(support)
        new_max = np.max(new_counts)
        new_score = np.sum(new_counts**2)
        
        if new_max < max_count or (new_max == max_count and new_score < score):
            counts = new_counts
            max_count = new_max
            score = new_score
            # Update available set
            support_set.remove(old_i1)
            support_set.remove(old_i2)
            support_set.add(new_i1)
            support_set.add(new_i2)
            available = np.setdiff1d(np.arange(N), support, assume_unique=True)
        else:
            support[i1] = old_i1
            support[i2] = old_i2

    # 2. Assign Signs via Local Search
    # Now that support is fixed (or as good as we can get), optimize signs.
    signs = rng.choice([-1.0, 1.0], size=M)
    
    # Precompute differences for faster peak calculation
    diffs = np.abs(support[:, None] - support[None, :])
    tri = np.triu_indices(M, k=1)
    d_vals = diffs[tri]
    
    def calc_peak(signs):
        Y = np.zeros(N)
        # Use np.add.at for vectorized accumulation of signed products
        products = signs[tri[0]] * signs[tri[1]]
        np.add.at(Y, d_vals, products)
        return np.max(np.abs(Y))

    current_peak = calc_peak(signs)
    
    # Local Search: Flip a sign if it reduces the peak
    improved = True
    while improved:
        improved = False
        for k in range(M):
            old_sign = signs[k]
            signs[k] = -old_sign
            new_peak = calc_peak(signs)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                signs[k] = old_sign

    # 3. Construct Final Array
    # Scale by 1/sqrt(N) to keep L2 norm 1.
    x = np.zeros(N)
    x[support] = signs / np.sqrt(N)
    
    return x
