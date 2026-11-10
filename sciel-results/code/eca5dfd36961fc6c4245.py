"""Reconstruct phylogenetic topology using Neighbor Joining."""
import numpy as np
import time
import math

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_newick(d, labels):
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = d[i, j]
                
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    next_internal_id = -1
    
    while len(active) > 3:
        n_act = len(active)
        
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist.get((i, j), 0.0)
                    
        best_c = float('inf')
        best_i, best_j = -1, -1
        for i in active:
            for j in active:
                if i == j: continue
                c = (n_act - 2) * dist.get((i, j), 0.0) - r[i] - r[j]
                if c < best_c:
                    best_c = c
                    best_i, best_j = i, j
                    
        i, j = best_i, best_j
        d_ij = dist.get((i, j), 0.0)
        
        a = next_internal_id
        next_internal_id -= 1
        
        for k in active:
            if k == i or k == j: continue
            d_ik = dist.get((i, k), 0.0)
            d_jk = dist.get((j, k), 0.0)
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            dist[(a, k)] = d_ak
            dist[(k, a)] = d_ak
            
        trees[a] = "(" + trees[i] + "," + trees[j] + ")"
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    return "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    return _nj_newick(d, labels)
