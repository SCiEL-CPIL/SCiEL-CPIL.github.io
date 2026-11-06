import numpy as np
import math

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    idx_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequences[labels[0]])
    seq_array = np.zeros((n, seq_len), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, char in enumerate(seq):
            seq_array[i, j] = idx_map[char]
            
    diff = (seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :])
    p = diff.sum(axis=2) / seq_len
    p_clipped = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_clipped)
    np.fill_diagonal(d, 0)
    return d

def _nj_tree(D, labels):
    n = len(labels)
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    active = list(range(n))
    subtrees = {i: labels[i] for i in range(n)}
    
    dist = {}
    for i in active:
        dist[i] = {}
        for j in active:
            if i != j:
                dist[i][j] = D[i, j]
                
    while len(active) > 3:
        n_curr = len(active)
        r = {i: sum(dist[i][j] for j in active if i != j) for i in active}
        
        min_b = float('inf')
        min_i, min_j = -1, -1
        for i_idx, i in enumerate(active):
            for j_idx in range(i_idx + 1, len(active)):
                j = active[j_idx]
                b = (n_curr - 2) * dist[i][j] - r[i] - r[j]
                if b < min_b:
                    min_b = b
                    min_i, min_j = i, j
                    
        D_ij = dist[min_i][min_j]
        l_i = 0.5 * D_ij + (1.0 / (2.0 * (n_curr - 2.0))) * (r[min_i] - r[min_j])
        l_j = 0.5 * D_ij - (1.0 / (2.0 * (n_curr - 2.0))) * (r[min_i] - r[min_j])
        
        new_subtree = f"({subtrees[min_i]},{subtrees[min_j]})"
        new_id = -1 - len([x for x in active if x < 0])
        
        subtrees[new_id] = new_subtree
        
        active.remove(min_i)
        active.remove(min_j)
        active.append(new_id)
        
        dist[new_id] = {}
        for k in active:
            if k == new_id:
                continue
            d_new_k = 0.5 * (dist[min_i][k] + dist[min_j][k] - D_ij)
            dist[new_id][k] = d_new_k
            dist[k][new_id] = d_new_k
            
        for k in active:
            if k in dist:
                if min_i in dist[k]:
                    del dist[k][min_i]
                if min_j in dist[k]:
                    del dist[k][min_j]
                    
    remaining = [subtrees[i] for i in active]
    return f"({','.join(remaining)})"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        return "();"
        
    D = _jc_distance_matrix(sequences, labels)
    tree = _nj_tree(D, labels)
    return tree
