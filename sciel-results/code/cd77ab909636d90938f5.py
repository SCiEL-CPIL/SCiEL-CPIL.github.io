"""Rewritten program for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 265.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _jc_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix using numpy."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Convert sequences to integer arrays
    seqs = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            seqs[i, j] = seq_map[c]
    
    num_sites = seqs.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    
    # Compute pairwise Hamming distances using broadcasting
    # For memory efficiency with up to 32 taxa and 2000 sites, this is fine
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(seqs[i] != seqs[j])
            p = diffs / num_sites
            if p >= 0.75:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)
            dist[i, j] = d
            dist[j, i] = d
    
    return dist


def _neighbor_joining(dist):
    """
    Neighbor Joining algorithm. Returns a Newick string.
    Uses a clean implementation that tracks the tree structure.
    """
    n = len(dist)
    if n <= 1:
        return ""
    if n == 2:
        return "(0,1);"
    
    # Active nodes: start with 0..n-1
    active = list(range(n))
    # Distance matrix (will be updated as we merge)
    d = dist.copy()
    
    # Tree structure: each node has a list of children
    # Leaves are 0..n-1, internal nodes are n, n+1, ...
    children = {}
    next_node = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums
        r = np.zeros(m)
        for i_idx in range(m):
            ai = active[i_idx]
            for j_idx in range(m):
                if i_idx != j_idx:
                    r[i_idx] += d[ai, active[j_idx]]
        
        # Find best pair using NJ criterion
        best_criterion = float('inf')
        best_i_idx = -1
        best_j_idx = -1
        for i_idx in range(m):
            for j_idx in range(i_idx + 1, m):
                ai = active[i_idx]
                aj = active[j_idx]
                criterion = (m - 2) * d[ai, aj] - r[i_idx] - r[j_idx]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i_idx = i_idx
                    best_j_idx = j_idx
        
        if best_i_idx == -1:
            break
        
        ai = active[best_i_idx]
        aj = active[best_j_idx]
        
        # Create new internal node
        new_node = next_node
        next_node += 1
        
        children[new_node] = [ai, aj]
        
        # Compute distances from new node to all other active nodes
        for k_idx in range(m):
            if k_idx == best_i_idx or k_idx == best_j_idx:
                continue
            ak = active[k_idx]
            d[new_node, ak] = 0.5 * (d[ai, ak] + d[aj, ak] - d[ai, aj])
            d[ak, new_node] = d[new_node, ak]
        
        # Update active list: remove ai, aj, add new_node
        new_active = []
        for k_idx in range(m):
            if k_idx != best_i_idx and k_idx != best_j_idx:
                new_active.append(active[k_idx])
        new_active.append(new_node)
        active = new_active
        
        # Time check
        if _elapsed() > _DEADLINE - 10:
            break
    
    # Now we have 3 active nodes left. Connect them in a star.
    if len(active) == 3:
        a, b, c = active
        # Create a root that connects all three
        root = next_node
        next_node += 1
        children[root] = [a, b, c]
        root_node = root
    elif len(active) == 2:
        a, b = active
        root = next_node
        next_node += 1
        children[root] = [a, b]
        root_node = root
    else:
        root_node = active[0] if active else 0
    
    # Build Newick string
    def _build_newick(node):
        if node < n:
            return str(node)
        if node not in children:
            return str(node)
        parts = [_build_newick(c) for c in children[node]]
        return "(" + ",".join(parts) + ")"
    
    newick = _build_newick(root_node)
    return newick + ";"


def _map_to_labels(newick, labels):
    """Map numeric labels in Newick string to actual taxon labels."""
    # Replace numeric leaf labels with actual labels
    # We need to be careful with multi-digit numbers
    import re
    
    def _replace(match):
        num = int(match.group())
        if 0 <= num < len(labels):
            return labels[num]
        return match.group()
    
    # Replace numbers that are followed by ), ,, or ; or end of string
    # and are preceded by (, ,, or start of string
    # This is tricky, so let's do it differently
    
    # Actually, let's just replace all occurrences of the numbers
    # But we need to be careful not to replace internal node numbers
    # Since internal nodes are >= n, and we only have n leaves,
    # we can safely replace numbers 0..n-1
    
    result = newick
    for i in range(n - 1, -1, -1):  # Replace in reverse order to avoid issues
        result = result.replace(f",{i},", f",{labels[i]},")
        result = result.replace(f"({i},", f"({labels[i]},")
        result = result.replace(f",{i})", f",{labels[i]})")
        result = result.replace(f"({i})", f"({labels[i]})")
    
    return result


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    try:
        # Build distance matrix
        dist = _jc_distance_matrix(sequences, labels)
        
        # Run Neighbor Joining
        newick_numeric = _neighbor_joining(dist)
        
        # Map to actual labels
        newick = _map_to_labels(newick_numeric, labels)
        
        return newick
    except Exception:
        # Fallback to star tree
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
