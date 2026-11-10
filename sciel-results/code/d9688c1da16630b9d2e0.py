# EVOLVE-BLOCK-START
import numpy as np
import random


def _entropy_from_counts_fast(counts, total):
    """
    Compute Shannon entropy in bits from counts and total probability mass.
    """
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, k):
    if k == 0:
        return 0.0
    D = _entropy_from_counts_fast(counts_D, k)
    H0 = _entropy_from_counts_fast(counts_H0, k)
    H1 = _entropy_from_counts_fast(counts_H1, k)
    H2 = _entropy_from_counts_fast(counts_H2, k)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M


def _greedy_build(n, idx_D, idx_H0, idx_H1, idx_H2, size_D, size_H0, size_H1, size_H2, start_set=None):
    """
    Greedily build a support set starting from start_set (or empty).
    Returns the best support set found during the process.
    """
    if start_set is None:
        start_set = set()
    
    selected = np.zeros((n, n), dtype=bool)
    for y, x in start_set:
        selected[y, x] = True
        
    counts_D = np.zeros(size_D, dtype=np.int32)
    counts_H0 = np.zeros(size_H0, dtype=np.int32)
    counts_H1 = np.zeros(size_H1, dtype=np.int32)
    counts_H2 = np.zeros(size_H2, dtype=np.int32)
    
    for y, x in start_set:
        counts_D[idx_D[y, x]] += 1
        counts_H0[idx_H0[y, x]] += 1
        counts_H1[idx_H1[y, x]] += 1
        counts_H2[idx_H2[y, x]] += 1
        
    current_k = len(start_set)
    best_ratio = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, current_k)
    best_support = set(start_set)
    
    # List of all cells
    all_cells = [(y, x) for y in range(n) for x in range(n)]
    unselected = [c for c in all_cells if not selected[c[0], c[1]]]
    
    # Greedy addition
    for _ in range(n * n):
        if not unselected:
            break
            
        best_gain = -np.inf
        best_cell = None
        
        # Shuffle to break ties randomly
        random.shuffle(unselected)
        
        for y, x in unselected:
            iD = idx_D[y, x]
            iH0 = idx_H0[y, x]
            iH1 = idx_H1[y, x]
            iH2 = idx_H2[y, x]
            
            counts_D[iD] += 1
            counts_H0[iH0] += 1
            counts_H1[iH1] += 1
            counts_H2[iH2] += 1
            new_k = current_k + 1
            
            ratio = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, new_k)
            
            counts_D[iD] -= 1
            counts_H0[iH0] -= 1
            counts_H1[iH1] -= 1
            counts_H2[iH2] -= 1
            
            if ratio > best_gain:
                best_gain = ratio
                best_cell = (y, x)
                
        if best_cell is None:
            break
            
        # Add the best cell
        y, x = best_cell
        iD = idx_D[y, x]
        iH0 = idx_H0[y, x]
        iH1 = idx_H1[y, x]
        iH2 = idx_H2[y, x]
        
        selected[y, x] = True
        counts_D[iD] += 1
        counts_H0[iH0] += 1
        counts_H1[iH1] += 1
        counts_H2[iH2] += 1
        current_k += 1
        unselected.remove((y, x))
        
        if best_gain > best_ratio:
            best_ratio = best_gain
            best_support = set(selected)
            
        # Early stopping if ratio drops significantly
        if best_ratio > 0 and best_gain < best_ratio * 0.8 and current_k > 5:
            break
            
    return best_support, best_ratio


def _local_search(support, n, idx_D, idx_H0, idx_H1, idx_H2, size_D, size_H0, size_H1, size_H2, max_iters=1000):
    """
    Try to improve the support set by swapping cells.
    """
    support = set(support)
    all_cells = [(y, x) for y in range(n) for x in range(n)]
    not_in_support = [c for c in all_cells if c not in support]
    
    counts_D = np.zeros(size_D, dtype=np.int32)
    counts_H0 = np.zeros(size_H0, dtype=np.int32)
    counts_H1 = np.zeros(size_H1, dtype=np.int32)
    counts_H2 = np.zeros(size_H2, dtype=np.int32)
    
    for y, x in support:
        counts_D[idx_D[y, x]] += 1
        counts_H0[idx_H0[y, x]] += 1
        counts_H1[idx_H1[y, x]] += 1
        counts_H2[idx_H2[y, x]] += 1
        
    current_k = len(support)
    current_ratio = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, current_k)
    
    for _ in range(max_iters):
        improved = False
        random.shuffle(not_in_support)
        
        # Try swapping one cell out and one in
        for y_out, x_out in list(support):
            # Remove out cell
            iD_out = idx_D[y_out, x_out]
            iH0_out = idx_H0[y_out, x_out]
            iH1_out = idx_H1[y_out, x_out]
            iH2_out = idx_H2[y_out, x_out]
            
            counts_D[iD_out] -= 1
            counts_H0[iH0_out] -= 1
            counts_H1[iH1_out] -= 1
            counts_H2[iH2_out] -= 1
            
            best_swap_ratio = -np.inf
            best_swap_in = None
            
            for y_in, x_in in not_in_support:
                iD_in = idx_D[y_in, x_in]
                iH0_in = idx_H0[y_in, x_in]
                iH1_in = idx_H1[y_in, x_in]
                iH2_in = idx_H2[y_in, x_in]
                
                counts_D[iD_in] += 1
                counts_H0[iH0_in] += 1
                counts_H1[iH1_in] += 1
                counts_H2[iH2_in] += 1
                
                ratio = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, current_k)
                
                counts_D[iD_in] -= 1
                counts_H0[iH0_in] -= 1
                counts_H1[iH1_in] -= 1
                counts_H2[iH2_in] -= 1
                
                if ratio > best_swap_ratio:
                    best_swap_ratio = ratio
                    best_swap_in = (y_in, x_in)
                    
            if best_swap_in is not None and best_swap_ratio > current_ratio + 1e-6:
                # Perform swap
                y_in, x_in = best_swap_in
                iD_in = idx_D[y_in, x_in]
                iH0_in = idx_H0[y_in, x_in]
                iH1_in = idx_H1[y_in, x_in]
                iH2_in = idx_H2[y_in, x_in]
                
                # Add in
                counts_D[iD_in] += 1
                counts_H0[iH0_in] += 1
                counts_H1[iH1_in] += 1
                counts_H2[iH2_in] += 1
                
                # Remove out
                support.remove((y_out, x_out))
                support.add((y_in, x_in))
                not_in_support.remove((y_in, x_in))
                not_in_support.append((y_out, x_out))
                
                current_ratio = best_swap_ratio
                improved = True
                break # Restart search after a swap
                
            # Restore out cell
            counts_D[iD_out] += 1
            counts_H0[iH0_out] += 1
            counts_H1[iH1_out] += 1
            counts_H2[iH2_out] += 1
            
        if not improved:
            break
            
    return support, current_ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    total_cells = n * n
    
    idx_D = np.zeros((n, n), dtype=np.int32)
    idx_H0 = np.zeros((n, n), dtype=np.int32)
    idx_H1 = np.zeros((n, n), dtype=np.int32)
    idx_H2 = np.zeros((n, n), dtype=np.int32)
    
    for y in range(n):
        for x in range(n):
            idx_D[y, x] = (x - y) + (n - 1)
            idx_H0[y, x] = x
            idx_H1[y, x] = x + y
            idx_H2[y, x] = x + 2 * y
            
    size_D = 2 * n - 1
    size_H0 = n
    size_H1 = 2 * n - 1
    size_H2 = 3 * n - 1
    
    best_global_ratio = 0.0
    best_global_support = None
    
    # Run multiple restarts with different random initializations
    for restart in range(50):
        random.seed(42 + restart)
        
        # Start with a random small set or empty
        if restart == 0:
            start_set = set()
        else:
            # Random start of size 2-5
            k_start = random.randint(2, 5)
            start_set = set(random.sample([(y, x) for y in range(n) for x in range(n)], k_start))
            
        support, ratio = _greedy_build(n, idx_D, idx_H0, idx_H1, idx_H2, size_D, size_H0, size_H1, size_H2, start_set)
        
        if ratio > best_global_ratio:
            best_global_ratio = ratio
            best_global_support = support
            
        # Local search
        if support:
            support, ratio = _local_search(support, n, idx_D, idx_H0, idx_H1, idx_H2, size_D, size_H0, size_H1, size_H2, max_iters=200)
            
        if ratio > best_global_ratio:
            best_global_ratio = ratio
            best_global_support = support
            
    if best_global_support is None or best_global_ratio < 1e-9:
        # Fallback to uniform
        P = np.ones((n, n), dtype=np.float64) / total_cells
    else:
        P = np.zeros((n, n), dtype=np.float64)
        for y, x in best_global_support:
            P[y, x] = 1.0
        P = P / np.sum(P)
        
    return P


# EVOLVE-BLOCK-END
