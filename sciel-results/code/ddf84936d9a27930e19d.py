"""Phylogenetic topology recovery using quartet-based greedy split assembly."""
import time
import math
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 255.0


def _time_exceeded():
    return time.time() > _DEADLINE


def _quartet_census(sequences, labels):
    """
    For every 4-taxon subset, count sites supporting each of the 3 unrooted topologies.
    Returns a dictionary mapping (i,j,k,l) sorted tuple to (count_12|34, count_13|24, count_14|23)
    """
    n = len(labels)
    if n < 4:
        return {}
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0])
    
    # Convert to integer arrays
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            seq_arrays[i, j] = char_map.get(c, 0)
    
    census = {}
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        sa = seq_arrays[a]
        sb = seq_arrays[b]
        sc = seq_arrays[c]
        sd = seq_arrays[d]
        
        # For each site, determine which topology is supported
        # Topology 1: (a,b)|(c,d) - supported when a==b and c==d and (a!=c)
        # Topology 2: (a,c)|(b,d) - supported when a==c and b==d and (a!=b)
        # Topology 3: (a,d)|(b,c) - supported when a==d and b==c and (a!=b)
        
        count1 = 0  # a,b | c,d
        count2 = 0  # a,c | b,d
        count3 = 0  # a,d | b,c
        
        for pos in range(L):
            va, vb, vc, vd = sa[pos], sb[pos], sc[pos], sd[pos]
            
            # Check topology 1: (a,b)|(c,d)
            if va == vb and vc == vd and va != vc:
                count1 += 1
            # Check topology 2: (a,c)|(b,d)
            if va == vc and vb == vd and va != vb:
                count2 += 1
            # Check topology 3: (a,d)|(b,c)
            if va == vd and vb == vc and va != vb:
                count3 += 1
        
        census[quad] = (count1, count2, count3)
    
    return census


def _greedy_split_assembly(labels, census):
    """
    Greedy split assembly from quartet counts.
    Build splits incrementally, maximizing total quartet support.
    Returns a list of splits, each split is a frozenset of taxa on one side.
    """
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # We'll work with indices 0..n-1
    # Each split is represented as a frozenset of indices
    # We build the tree by adding splits one at a time
    
    # Initialize: no splits, each taxon is its own component
    # We need to track which splits are compatible with each other
    
    # For efficiency, let's use a different approach:
    # Start with all pairs as potential 2-taxon splits
    # Then greedily add splits that maximize compatible quartet support
    
    # Actually, let's use a simpler greedy approach:
    # 1. Compute a "distance" from quartet data
    # 2. Use NJ on that distance
    
    # Better: use the quartet counts to estimate pairwise distances
    # Then run NJ
    
    if n <= 3:
        return []
    
    # Estimate pairwise distances from quartet data
    # For each pair (i,j), the distance is proportional to the number of quartets
    # where i and j are on the same side of the split
    dist = np.zeros((n, n), dtype=np.float64)
    
    for quad, (c1, c2, c3) in census.items():
        a, b, c, d = quad
        total = c1 + c2 + c3
        if total == 0:
            continue
        
        # Quartet (a,b)|(c,d) supports: a,b close; c,d close; a/c, a/d, b/c, b/d far
        # Quartet (a,c)|(b,d) supports: a,c close; b,d close; a/b, a/d, c/b, c/d far
        # Quartet (a,d)|(b,c) supports: a,d close; b,c close; a/b, a/c, d/b, d/c far
        
        # Contribution to distances:
        # For topology 1 (a,b)|(c,d) with weight c1/total:
        #   d(a,b) += 0, d(c,d) += 0, d(a,c) += 1, d(a,d) += 1, d(b,c) += 1, d(b,d) += 1
        # For topology 2 (a,c)|(b,d) with weight c2/total:
        #   d(a,c) += 0, d(b,d) += 0, d(a,b) += 1, d(a,d) += 1, d(c,b) += 1, d(c,d) += 1
        # For topology 3 (a,d)|(b,c) with weight c3/total:
        #   d(a,d) += 0, d(b,c) += 0, d(a,b) += 1, d(a,c) += 1, d(d,b) += 1, d(d,c) += 1
        
        w1 = c1 / total
        w2 = c2 / total
        w3 = c3 / total
        
        # d(a,b): 0 from top1, 1 from top2, 1 from top3
        dist[a, b] += w2 + w3
        dist[b, a] += w2 + w3
        
        # d(c,d): 0 from top1, 1 from top2, 1 from top3
        dist[c, d] += w2 + w3
        dist[d, c] += w2 + w3
        
        # d(a,c): 1 from top1, 0 from top2, 1 from top3
        dist[a, c] += w1 + w3
        dist[c, a] += w1 + w3
        
        # d(b,d): 1 from top1, 0 from top2, 1 from top3
        dist[b, d] += w1 + w3
        dist[d, b] += w1 + w3
        
        # d(a,d): 1 from top1, 1 from top2, 0 from top3
        dist[a, d] += w1 + w2
        dist[d, a] += w1 + w2
        
        # d(b,c): 1 from top1, 1 from top2, 0 from top3
        dist[b, c] += w1 + w2
        dist[c, b] += w1 + w2
    
    # Scale distances to be comparable to JC distances
    # The max possible contribution per quartet is 2 (when two quartets both give 1)
    # So max distance per pair is 2 * (number of quartets containing that pair)
    # For normalization, we can just use the raw values
    
    return dist


def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string (topology only).
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    dist = D.copy()
    names = list(labels)
    
    tree_parts = list(names)
    num_nodes = n
    
    while num_nodes > 3:
        if _time_exceeded():
            break
        
        m = num_nodes
        r = np.sum(dist, axis=1)
        
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * dist[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        
        new_part = "(" + tree_parts[i] + "," + tree_parts[j] + ")"
        
        new_distances = np.zeros(m - 2)
        new_active = []
        for k in range(m):
            if k == i or k == j:
                continue
            new_distances[len(new_active)] = 0.5 * (dist[i, k] + dist[j, k] - dist[i, j])
            new_active.append(k)
        
        new_dist = np.zeros((m - 1, m - 1), dtype=np.float64)
        for idx, k in enumerate(new_active):
            for idx2, k2 in enumerate(new_active):
                new_dist[idx, idx2] = dist[k, k2]
            new_dist[idx, m - 2] = new_distances[idx]
            new_dist[m - 2, idx] = new_distances[idx]
        
        tree_parts = [tree_parts[k] for k in new_active] + [new_part]
        dist = new_dist
        num_nodes = m - 1
    
    if num_nodes == 3:
        result = "(" + ",".join(tree_parts) + ")"
    else:
        result = "(" + ",".join(tree_parts) + ")"
    
    return result + ";"


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            seq_arrays[i, j] = char_map.get(c, 0)
    
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(seq_arrays[i] != seq_arrays[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string over labels t0..t(n-1).
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # For small n, use quartet census directly
    if n >= 4 and n <= 20:
        census = _quartet_census(sequences, labels)
        if census:
            D_quartet = _greedy_split_assembly(labels, census)
            if D_quartet is not None:
                result = _neighbor_joining(D_quartet, labels)
                return result
    
    # Fallback to JC distance + NJ
    D_jc = _jc_distance_matrix(sequences, labels)
    result = _neighbor_joining(D_jc, labels)
    return result
