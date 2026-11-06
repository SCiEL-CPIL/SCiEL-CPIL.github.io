import math
import time
import numpy as np

_DEADLINE = None

def _compute_jc_distances(sequences, labels):
    n = len(labels)
    seq_arrays = []
    for lab in labels:
        s = sequences[lab]
        arr = np.array([0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in s], dtype=np.int8)
        seq_arrays.append(arr)
    
    mat = np.stack(seq_arrays)
    L = mat.shape[1]
    
    dist = np.zeros((n, n), dtype=np.float64)
    var = np.zeros((n, n), dtype=np.float64)
    
    diff = (mat[:, None, :] != mat[None, :, :])
    counts = diff.sum(axis=2)
    p = counts.astype(np.float64) / L
    
    p_safe = np.clip(p, 0, 0.7499)
    jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    jc[p >= 0.7499] = 3.0
    
    denom = (1.0 - 4.0 * p_safe / 3.0)**2
    denom[denom < 1e-10] = 1e-10
    var = (p_safe * (1.0 - p_safe)) / (L * denom)
    var[p >= 0.7499] = 1.0
    
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(var, 0.0)
    
    return dist, var

def _bionj_correction(dist, var, labels):
    n = len(labels)
    if n < 3:
        return dist
    
    v = var.sum(axis=1) / (n - 1)
    v_diff = v[:, None] - v[None, :]
    dist_bionj = dist + 0.5 * v_diff
    
    dist_bionj = (dist_bionj + dist_bionj.T) / 2.0
    np.fill_diagonal(dist_bionj, 0.0)
    
    return dist_bionj

def _nj_newick_fast(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist

    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    next_id = n

    while len(active) > 3:
        m = len(active)
        idx = np.array(active)
        D_sub = D[idx[:, None], idx[None, :]].copy()
        
        np.fill_diagonal(D_sub, 0.0)
        r = D_sub.sum(axis=1)
        
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        flat_idx = np.argmin(Q)
        ii = flat_idx // m
        jj = flat_idx % m
        
        a = active[ii]
        b = active[jj]
        
        new_id = next_id
        next_id += 1
        
        mask = np.ones(m, dtype=bool)
        mask[ii] = False
        mask[jj] = False
        active_others = active[mask]
        
        if len(active_others) > 0:
            D_new = 0.5 * (D[a, active_others] + D[b, active_others] - D[a, b])
            D[new_id, active_others] = D_new
            D[active_others, new_id] = D_new
        
        tree_str[new_id] = f"({tree_str[a]},{tree_str[b]})"
        
        active = [x for x in active if x != a and x != b]
        active.append(new_id)

    a, b, c = active
    return f"({tree_str[a]},{tree_str[b]},{tree_str[c]})"

def reconstruct(sequences):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 250

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    if time.time() > _DEADLINE - 5:
        return "(" + ",".join(labels) + ")"

    dist, var = _compute_jc_distances(sequences, labels)
    
    if time.time() > _DEADLINE - 10:
        return "(" + ",".join(labels) + ")"
        
    dist_bionj = _bionj_correction(dist, var, labels)
    
    result = _nj_newick_fast(dist_bionj, labels)
    
    return result
