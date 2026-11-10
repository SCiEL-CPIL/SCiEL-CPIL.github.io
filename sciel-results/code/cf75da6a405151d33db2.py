"""Phylogenetic topology recovery from simulated Jukes-Cantor sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from itertools import combinations


def _jc_corrected(p):
    """Jukes-Cantor distance correction."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using numpy."""
    n = len(labels)
    seq_dict = sequences
    # Convert to numpy arrays for faster comparison
    arrays = [np.array([ord(c) for c in seq_dict[label]], dtype=np.int32) for label in labels]
    length = len(arrays[0])
    
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(arrays[i] != arrays[j])
            p = diff / length
            d = _jc_corrected(p)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_reconstruct(dist, labels):
    """Neighbour-Joining algorithm with numpy-optimized inner loop."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Work with a mutable distance matrix
    D = dist.copy()
    # Track which taxa are still active
    active = list(range(n))
    # Track the Newick subtree for each node
    subtree = {i: labels[i] for i in range(n)}
    # Next available node id
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums
        row_sums = np.array([sum(D[i][j] for j in active if j != i) for i in active], dtype=np.float64)
        
        # Find pair minimizing (m-2)*D[i][j] - row_sums[i] - row_sums[j]
        best_val = float('inf')
        best_i = active[0]
        best_j = active[1]
        
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                val = (m - 2) * D[i][j] - row_sums[idx_i] - row_sums[idx_j]
                if val < best_val:
                    best_val = val
                    best_i = i
                    best_j = j
        
        # Compute branch lengths
        d_ij = D[best_i][best_j]
        r_i = row_sums[active.index(best_i)]
        r_j = row_sums[active.index(best_j)]
        delta = (r_i - r_j) / (2 * (m - 2))
        len_i = 0.5 * d_ij + delta
        len_j = 0.5 * d_ij - delta
        
        # Create new node
        new_node = next_id
        next_id += 1
        subtree[new_node] = f"({subtree[best_i]},{subtree[best_j]})"
        
        # Update distance matrix: D[new][k] = 0.5*(D[i][k] + D[j][k] - D[i][j])
        rest = [x for x in active if x != best_i and x != best_j]
        for k in rest:
            D[new_node][k] = 0.5 * (D[best_i][k] + D[best_j][k] - d_ij)
            D[k][new_node] = D[new_node][k]
        
        # Remove old nodes, add new
        active = rest + [new_node]
    
    # Build final tree from remaining 3 nodes
    if len(active) == 3:
        a, b, c = active
        return f"({subtree[a]},{subtree[b]},{subtree[c]})"
    elif len(active) == 2:
        a, b = active
        return f"({subtree[a]},{subtree[b]})"
    else:
        return subtree[active[0]]


def _bionj_reconstruct(dist, labels):
    """BioNJ-style: NJ with a few iterations of local search (quartet swaps)."""
    n = len(labels)
    if n <= 4:
        return _nj_reconstruct(dist, labels)
    
    # Start with NJ
    tree = _nj_reconstruct(dist, labels)
    
    # Simple local search: try swapping internal edges (quartet rearrangements)
    # This is simplified - we'll do a few rounds of nearest-neighbour interchange
    # For time efficiency, limit iterations
    deadline = time.time() + 20  # 20 second budget per instance for local search
    
    # Parse tree to get topology as splits
    # For simplicity, we'll do a basic QP (quartet pruning) local search
    
    # Actually, for robustness and speed, let's just return NJ result
    # BioNJ improvements are marginal and complex to implement correctly
    return tree


def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from DNA sequences."""
    start_time = time.time()
    deadline = start_time + 260  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return a simple star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    tree = _nj_reconstruct(dist, labels)
    
    # Ensure valid Newick format
    if not tree.endswith(";"):
        tree += ";"
    
    return tree
# EVOLVE-BLOCK-END
