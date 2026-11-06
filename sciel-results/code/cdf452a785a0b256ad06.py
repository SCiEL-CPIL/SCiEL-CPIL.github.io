"""Phylogenetic topology recovery: NJ with NumPy acceleration."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0

def _jc_correct(p):
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_dist_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.zeros((n, L), dtype=np.uint8)
    for i, l in enumerate(labels):
        for j, c in enumerate(sequences[l]):
            seqs[i, j] = base_map[c]
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            mismatches = np.sum(seqs[i] != seqs[j])
            p = mismatches / L
            dist[i, j] = _jc_correct(p)
            dist[j, i] = dist[i, j]
    return dist

def _nj_build(dist, labels):
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = dist.astype(float).copy()
    
    while len(nodes) > 3:
        m = len(nodes)
        r = np.array([sum(d[nodes[i], nodes[j]] for j in range(m) if i != j) for i in range(m)])
        
        best_q = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i+1, m):
                q = (m - 2) * d[nodes[i], nodes[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        u = nodes[best_i]
        v = nodes[best_j]
        
        joined = f"({trees[u]},{trees[v]})"
        new_node = max(nodes) + 1
        rest = [x for x in nodes if x not in (u, v)]
        
        d_u = d[u, v]
        for x in rest:
            new_d = 0.5 * (d[u, x] + d[v, x] - d_u)
            d[new_node, x] = new_d
            d[x, new_node] = new_d
        
        trees[new_node] = joined
        nodes = rest + [new_node]
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    if time.time() > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    dist = _compute_dist_matrix(sequences, labels)
    newick = _nj_build(dist, labels)
    
    return newick
