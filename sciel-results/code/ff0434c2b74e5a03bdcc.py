"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import time
import numpy as np

_DEADLINE = None


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy vectorization."""
    n = len(labels)
    # Convert sequences to numeric matrix: A=0, C=1, G=2, T=3
    seq_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            arr[i, j] = seq_dict[ch]
    
    # Compute pairwise Hamming distances
    # diff[i,j,k] = 1 if arr[i,k] != arr[j,k]
    # Use broadcasting: arr[:, np.newaxis, :] vs arr[np.newaxis, :, :]
    # This creates (n, n, L) which might be large; do it in chunks if needed
    L = arr.shape[1]
    hamming = np.zeros((n, n), dtype=np.int32)
    
    # For memory efficiency, process in row chunks
    chunk_size = max(1, min(n, 5000000 // L))  # keep intermediate ~5MB
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # diff shape: (end-start, n, L)
        diff = (arr[start:end, np.newaxis, :] != arr[np.newaxis, :, :])
        hamming[start:end, :] = diff.sum(axis=2)
    
    # Convert to JC-corrected distances
    p = hamming.astype(np.float64) / L
    # p >= 0.75 -> cap at 3.0
    mask = p >= 0.75
    p_safe = np.where(mask, 0.74, p)
    dist = -0.75 * np.log1p(-4.0 * p_safe / 3.0)
    dist[mask] = 3.0
    # Zero diagonal
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_newick(dist, labels, deadline=None):
    """Run NJ and return Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        # Any unrooted tree with 3 taxa is the same topology
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Work with a copy
    D = dist.copy()
    active = list(range(n))
    # Node labels: start with original labels
    node_labels = list(labels)
    
    # We'll track the tree structure via a list of "clusters"
    # Each cluster is either a leaf (original label) or a composite
    # Use a list where index corresponds to position in active list
    # Actually, let's track: for each active node, its Newick string so far
    cluster_newick = list(labels)
    
    while len(active) > 3:
        m = len(active)
        if deadline is not None and time.time() > deadline:
            break
        
        # Row sums
        row_sums = D[np.ix_(active, active)].sum(axis=1)
        
        # Q matrix: Q[i,j] = (m-2)*D[i,j] - row_sums[i] - row_sums[j]
        sub_D = D[np.ix_(active, active)]
        Q = (m - 2) * sub_D - row_sums[:, np.newaxis] - row_sums[np.newaxis, :]
        
        # Find minimum Q (excluding diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_local, j_local = min_idx
        i, j = active[i_local], active[j_local]
        
        # Merge i and j
        # New distance to k: 0.5*(D[i,k] + D[j,k] - D[i,j])
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        # Zero out self
        new_dist[i] = 0.0
        new_dist[j] = 0.0
        # For nodes being removed, set to 0
        new_dist[np.array(active)] = 0.0  # will be overwritten properly
        # Actually, let's be more careful:
        for k in active:
            if k != i and k != j:
                new_dist[k] = 0.5 * (D[i, k] + D[j, k] - D[i, j])
        new_dist[i] = 0.0
        new_dist[j] = 0.0
        
        # Update D
        # Add new row/col
        D = np.vstack([D, new_dist[np.newaxis, :]])
        D = np.hstack([D, new_dist[:, np.newaxis]])
        
        # New node index
        new_idx = len(active)
        new_label = f"({cluster_newick[i]},{cluster_newick[j]})"
        
        # Remove i and j from active, add new_idx
        active.remove(i)
        active.remove(j)
        active.append(new_idx)
        cluster_newick.append(new_label)
    
    # Final tree: join remaining 3 nodes
    if len(active) == 3:
        result = f"({cluster_newick[active[0]]},({cluster_newick[active[1]]},{cluster_newick[active[2]]}));"
    else:
        # Shouldn't happen, but safety
        result = "(" + ",".join(cluster_newick[i] for i in active) + ");"
    
    return result


def reconstruct(sequences):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 250  # Safety margin from 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    try:
        result = _nj_newick(dist, labels, deadline=_DEADLINE)
    except Exception:
        # Fallback: return a star tree
        result = "(" + ",".join(labels) + ");"
    
    return result
# EVOLVE-BLOCK-END
