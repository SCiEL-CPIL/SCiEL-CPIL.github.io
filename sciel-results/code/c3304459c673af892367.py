# EVOLVE-BLOCK-START
import numpy as np


def _entropy_from_counts(counts, total):
    """
    Compute Shannon entropy in bits from counts and total probability mass.
    counts: array of integers (counts of items in each bin)
    total: total number of items (sum of counts)
    """
    if total <= 0:
        return 0.0
    probs = counts / total
    # Ignore zero probabilities
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, k):
    """Compute D/M ratio from current counts."""
    if k <= 0:
        return 0.0
    D = _entropy_from_counts(counts_D, k)
    H0 = _entropy_from_counts(counts_H0, k)
    H1 = _entropy_from_counts(counts_H1, k)
    H2 = _entropy_from_counts(counts_H2, k)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M


def _local_search(selected, idx_D, idx_H0, idx_H1, idx_H2, 
                  size_D, size_H0, size_H1, size_H2, n, max_iter=50):
    """
    Local search: try removing one cell and adding another to improve ratio.
    Returns improved selected mask and ratio.
    """
    counts_D = np.zeros(size_D, dtype=np.int32)
    counts_H0 = np.zeros(size_H0, dtype=np.int32)
    counts_H1 = np.zeros(size_H1, dtype=np.int32)
    counts_H2 = np.zeros(size_H2, dtype=np.int32)
    
    for y in range(n):
        for x in range(n):
            if selected[y, x]:
                counts_D[idx_D[y, x]] += 1
                counts_H0[idx_H0[y, x]] += 1
                counts_H1[idx_H1[y, x]] += 1
                counts_H2[idx_H2[y, x]] += 1
    
    current_k = np.sum(selected)
    if current_k == 0:
        return selected, 0.0
    
    current_ratio = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, current_k)
    best_ratio = current_ratio
    best_selected = selected.copy()
    
    for iteration in range(max_iter):
        improved = False
        
        # Try removing one selected cell
        for y in range(n):
            for x in range(n):
                if not selected[y, x]:
                    continue
                
                iD = idx_D[y, x]
                iH0 = idx_H0[y, x]
                iH1 = idx_H1[y, x]
                iH2 = idx_H2[y, x]
                
                # Tentatively remove
                counts_D[iD] -= 1
                counts_H0[iH0] -= 1
                counts_H1[iH1] -= 1
                counts_H2[iH2] -= 1
                new_k = current_k - 1
                
                if new_k > 0:
                    ratio_new = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, new_k)
                    if ratio_new > best_ratio + 1e-10:
                        best_ratio = ratio_new
                        best_selected = selected.copy()
                        best_selected[y, x] = False
                        improved = True
                        # Update state
                        selected[y, x] = False
                        current_k = new_k
                        # Don't revert - we accept this move
                        break
                else:
                    # Revert
                    counts_D[iD] += 1
                    counts_H0[iH0] += 1
                    counts_H1[iH1] += 1
                    counts_H2[iH2] += 1
            else:
                continue
            break
        
        if improved:
            continue
        
        # Try adding one unselected cell
        for y in range(n):
            for x in range(n):
                if selected[y, x]:
                    continue
                
                iD = idx_D[y, x]
                iH0 = idx_H0[y, x]
                iH1 = idx_H1[y, x]
                iH2 = idx_H2[y, x]
                
                # Tentatively add
                counts_D[iD] += 1
                counts_H0[iH0] += 1
                counts_H1[iH1] += 1
                counts_H2[iH2] += 1
                new_k = current_k + 1
                
                ratio_new = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, new_k)
                
                # Revert
                counts_D[iD] -= 1
                counts_H0[iH0] -= 1
                counts_H1[iH1] -= 1
                counts_H2[iH2] -= 1
                
                if ratio_new > best_ratio + 1e-10:
                    best_ratio = ratio_new
                    best_selected = selected.copy()
                    best_selected[y, x] = True
                    improved = True
                    # Update state
                    selected[y, x] = True
                    counts_D[iD] += 1
                    counts_H0[iH0] += 1
                    counts_H1[iH1] += 1
                    counts_H2[iH2] += 1
                    current_k = new_k
                    break
            else:
                continue
            break
        
        if not improved:
            break
    
    return best_selected, best_ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Uses greedy construction followed by local search to find a sparse support set S
    that maximizes D/M, where D is entropy of x-y and M is max entropy of x, x+y, x+2y.
    """
    n = 10
    total_cells = n * n
    
    # Precompute bin indices for each cell (y, x)
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
    
    # Try multiple greedy runs with different random seeds
    best_ratio = 0.0
    best_support = None
    
    np.random.seed(42)
    
    for trial in range(5):
        # Greedy selection
        selected = np.zeros((n, n), dtype=bool)
        
        counts_D = np.zeros(size_D, dtype=np.int32)
        counts_H0 = np.zeros(size_H0, dtype=np.int32)
        counts_H1 = np.zeros(size_H1, dtype=np.int32)
        counts_H2 = np.zeros(size_H2, dtype=np.int32)
        
        current_k = 0
        trial_best_ratio = 0.0
        trial_best_support = None
        
        # Add initial cells: start with a few random cells to break symmetry
        if trial == 0:
            # Deterministic: start with diagonal
            for i in range(n):
                selected[i, i] = True
                counts_D[idx_D[i, i]] += 1
                counts_H0[idx_H0[i, i]] += 1
                counts_H1[idx_H1[i, i]] += 1
                counts_H2[idx_H2[i, i]] += 1
                current_k += 1
        else:
            # Random initialization
            num_init = np.random.randint(3, 8)
            cells = np.random.choice(total_cells, size=num_init, replace=False)
            for c in cells:
                y, x = divmod(c, n)
                selected[y, x] = True
                counts_D[idx_D[y, x]] += 1
                counts_H0[idx_H0[y, x]] += 1
                counts_H1[idx_H1[y, x]] += 1
                counts_H2[idx_H2[y, x]] += 1
                current_k += 1
        
        # Compute initial ratio
        if current_k > 0:
            trial_best_ratio = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, current_k)
            P_temp = np.zeros((n, n), dtype=np.float64)
            P_temp[selected] = 1.0 / current_k
            trial_best_support = P_temp.copy()
        
        # Greedy addition
        for step in range(total_cells - current_k):
            best_gain = -np.inf
            best_cell = None
            best_new_ratio = -np.inf
            
            for y in range(n):
                for x in range(n):
                    if selected[y, x]:
                        continue
                    
                    iD = idx_D[y, x]
                    iH0 = idx_H0[y, x]
                    iH1 = idx_H1[y, x]
                    iH2 = idx_H2[y, x]
                    
                    # Tentatively add
                    counts_D[iD] += 1
                    counts_H0[iH0] += 1
                    counts_H1[iH1] += 1
                    counts_H2[iH2] += 1
                    new_k = current_k + 1
                    
                    ratio_new = _compute_ratio(counts_D, counts_H0, counts_H1, counts_H2, new_k)
                    
                    # Revert
                    counts_D[iD] -= 1
                    counts_H0[iH0] -= 1
                    counts_H1[iH1] -= 1
                    counts_H2[iH2] -= 1
                    
                    if ratio_new > best_gain:
                        best_gain = ratio_new
                        best_cell = (y, x)
                        best_new_ratio = ratio_new
            
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
            
            if best_new_ratio > trial_best_ratio:
                trial_best_ratio = best_new_ratio
                P_temp = np.zeros((n, n), dtype=np.float64)
                P_temp[selected] = 1.0 / current_k
                trial_best_support = P_temp.copy()
            
            # Early stopping if ratio dropped significantly
            if trial_best_ratio > 0.5 and best_new_ratio < trial_best_ratio * 0.7 and current_k > 15:
                break
        
        # Local search to refine
        if trial_best_support is not None:
            improved_selected, improved_ratio = _local_search(
                selected, idx_D, idx_H0, idx_H1, idx_H2,
                size_D, size_H0, size_H1, size_H2, n, max_iter=30
            )
            
            if improved_ratio > trial_best_ratio:
                trial_best_ratio = improved_ratio
                P_temp = np.zeros((n, n), dtype=np.float64)
                P_temp[improved_selected] = 1.0 / np.sum(improved_selected)
                trial_best_support = P_temp.copy()
        
        # Update global best
        if trial_best_ratio > best_ratio:
            best_ratio = trial_best_ratio
            best_support = trial_best_support
    
    # Fallback if no valid support found
    if best_ratio < 1e-9 or best_support is None:
        best_support = np.ones((n, n), dtype=np.float64) / total_cells
    else:
        # Normalize
        total_sum = np.sum(best_support)
        if total_sum > 1e-9:
            best_support = best_support / total_sum
        else:
            best_support = np.ones((n, n), dtype=np.float64) / total_cells
    
    # Clip and return
    best_support = np.clip(best_support, 0, None)
    
    return best_support


# EVOLVE-BLOCK-END
