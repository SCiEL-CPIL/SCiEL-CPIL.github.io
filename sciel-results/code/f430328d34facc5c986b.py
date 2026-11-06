"""Neighbour-joining with vectorized distance computation and deadline guard."""
import time
import math
import numpy as np

_DEADLINE = 260.0
_start_time = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _start_time)

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    enc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    codes = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, ch in enumerate(seq):
            codes[i, j] = enc_map[ch]
    
    diff = (codes[:, None, :] != codes[None, :, :])
    hamming = diff.sum(axis=2).astype(np.float64)
    
    p = hamming / n_sites
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d

def _neighbor_joining(D, labels):
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        a, b, c = labels
        return f"(({a},{b}),{c});"
    
    D = D.astype(np.float64).copy()
    # node_labels: maps internal index -> Newick string or taxon name
    node_labels = list(labels)
    # active: list of internal indices currently in the tree
    active = list(range(n))
    
    next_idx = n
    
    while len(active) > 3:
        m = len(active)
        if _remaining() < 5.0:
            return _star_fallback(node_labels, active)
        
        D_sub = D[np.ix_(active, active)]
        r = D_sub.sum(axis=1)
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_pos = np.unravel_index(np.argmin(Q), Q.shape)
        i_local, j_local = min_pos
        i_idx = active[i_local]
        j_idx = active[j_local]
        
        d_ij = D_sub[i_local, j_local]
        r_i = r[i_local]
        r_j = r[j_local]
        
        # Create new node
        new_idx = next_idx
        next_idx += 1
        node_labels.append(f"({node_labels[i_idx]},{node_labels[j_idx]})")
        
        # Compute distances from new node to remaining nodes
        k_locals = [k for k in range(m) if k not in (i_local, j_local)]
        k_indices = [active[k] for k in k_locals]
        
        # New active list: remaining nodes + new node
        new_active = k_indices + [new_idx]
        new_m = len(new_active)
        
        # Build new D matrix
        D_new = np.zeros((new_m, new_m), dtype=np.float64)
        
        # Old distances among remaining nodes
        for a_pos, a_idx in enumerate(k_indices):
            for b_pos, b_idx in enumerate(k_indices):
                D_new[a_pos, b_pos] = D[a_idx, b_idx]
        
        # Distances from new node to remaining nodes
        for a_pos, k_local in enumerate(k_locals):
            d_new_k = 0.5 * (D_sub[i_local, k_local] + D_sub[j_local, k_local] - d_ij)
            D_new[a_pos, -1] = d_new_k
            D_new[-1, a_pos] = d_new_k
        D_new[-1, -1] = 0.0
        
        D = D_new
        active = new_active
    
    # Final 3 nodes
    a, b, c = [node_labels[x] for x in active]
    return f"(({a},{b}),{c});"

def _star_fallback(node_labels, active):
    labels = [node_labels[x] for x in active]
    if len(labels) <= 1:
        return labels[0] + ";"
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]});"
    inner = ",".join(labels)
    return f"({inner});"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]});"
    
    if _remaining() < 2.0:
        return _star_fallback(list(labels), list(range(n)))
    
    D = _jc_distance_matrix(sequences, labels)
    
    if _remaining() < 2.0:
        return _star_fallback(list(labels), list(range(n)))
    
    result = _neighbor_joining(D, labels)
    return result
