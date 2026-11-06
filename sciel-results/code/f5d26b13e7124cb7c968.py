# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

_DEADLINE = 265.0
_start_wall = time.time()

def _get_elapsed():
    return time.time() - _start_wall

def _check_deadline():
    return _get_elapsed() > _DEADLINE

def _encode_sequences(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    mat = np.empty((n, L), dtype=np.uint8)
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    for i, label in enumerate(labels):
        seq = sequences[label]
        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        mat[i] = lookup[seq_bytes]
    return mat

def _jc_distance_matrix(mat):
    n, L = mat.shape
    diff = (mat[:, None, :] != mat[None, :, :])
    mismatches = diff.sum(axis=2)
    p = mismatches.astype(np.float64) / float(L)
    p_clipped = np.clip(p, 0.0, 0.75 - 1e-9)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_clipped)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D

def _fast_nj(D, labels):
    """
    Fast Neighbor-Joining algorithm.
    Returns Newick string.
    """
    n = D.shape[0]
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        return f"({labels[0]},{labels[1]})"
    
    active = list(range(n))
    names = {i: labels[i] for i in range(n)}
    dist = D.copy()
    next_id = n
    
    while len(active) > 3:
        if _check_deadline():
            break
        
        m = len(active)
        # Compute Q matrix
        r = dist.sum(axis=1)
        Q = (m - 2) * dist - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        
        idx_i = active[i]
        idx_j = active[j]
        
        name_i = names[idx_i]
        name_j = names[idx_j]
        
        d_ij = dist[i, j]
        delta_i = r[i] / (m - 2)
        delta_j = r[j] / (m - 2)
        
        d_i_new = max(0.0, 0.5 * d_ij + delta_i)
        d_j_new = max(0.0, 0.5 * d_ij + delta_j)
        
        new_name = f"({name_i}:{d_i_new:.4f},{name_j}:{d_j_new:.4f})"
        
        # Update distances
        others = [k for k in range(m) if k != i and k != j]
        if others:
            dist[i, others] = 0.5 * (dist[i, others] + dist[j, others] - d_ij)
            dist[others, i] = dist[i, others]
        
        # Remove j
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[j] = False
        dist = dist[np.ix_(keep_mask, keep_mask)]
        
        # Update active and names
        new_active = []
        for k in range(m):
            if k == j:
                continue
            if k == i:
                new_active.append(next_id)
                names[next_id] = new_name
            else:
                new_active.append(active[k])
        
        active = new_active
        next_id += 1
    
    # Final 3 nodes
    if len(active) == 3:
        names_final = [names[id] for id in active]
        d01 = dist[0, 1]
        d02 = dist[0, 2]
        d12 = dist[1, 2]
        
        if d01 <= d02 and d01 <= d12:
            l0 = max(0.0, 0.5 * (d01 + d02 - d12))
            l1 = max(0.0, 0.5 * (d01 + d12 - d02))
            l2 = max(0.0, 0.5 * (d02 + d12 - d01))
            return f"(({names_final[0]}:{l0:.4f},{names_final[1]}:{l1:.4f}),{names_final[2]}:{l2:.4f})"
        elif d02 <= d01 and d02 <= d12:
            l0 = max(0.0, 0.5 * (d02 + d01 - d12))
            l2 = max(0.0, 0.5 * (d02 + d12 - d01))
            l1 = max(0.0, 0.5 * (d01 + d12 - d02))
            return f"(({names_final[0]}:{l0:.4f},{names_final[2]}:{l2:.4f}),{names_final[1]}:{l1:.4f})"
        else:
            l1 = max(0.0, 0.5 * (d12 + d01 - d02))
            l2 = max(0.0, 0.5 * (d12 + d02 - d01))
            l0 = max(0.0, 0.5 * (d01 + d02 - d12))
            return f"(({names_final[1]}:{l1:.4f},{names_final[2]}:{l2:.4f}),{names_final[0]}:{l0:.4f})"
    
    return "(" + ",".join(names[id] for id in active) + ")"

def _quartet_support(D, a, b, c, d):
    """
    Compute quartet support scores for the three possible splits of (a,b,c,d).
    Returns the split with minimum total distance.
    """
    s1 = D[a, c] + D[b, d]  # ab|cd
    s2 = D[a, b] + D[c, d]  # ac|bd
    s3 = D[a, d] + D[b, c]  # ad|bc
    
    scores = [s1, s2, s3]
    min_idx = scores.index(min(scores))
    
    if min_idx == 0:
        return frozenset({a, b})
    elif min_idx == 1:
        return frozenset({a, c})
    else:
        return frozenset({a, d})

def _build_tree_from_quartets(D, labels):
    """
    Build a tree using quartet-based greedy clustering.
    """
    n = D.shape[0]
    if n <= 3:
        if n == 1:
            return f"{labels[0]};"
        if n == 2:
            return f"({labels[0]},{labels[1]})"
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Start with each taxon as its own cluster
    clusters = [set([i]) for i in range(n)]
    cluster_names = [labels[i] for i in range(n)]
    
    while len(clusters) > 2:
        if _check_deadline():
            break
        
        best_pair = None
        best_score = np.inf
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci = clusters[i]
                cj = clusters[j]
                
                # Average distance between clusters
                d_ij = np.mean([D[a, b] for a in ci for b in cj])
                
                # Distance to rest
                rest = set()
                for k in range(len(clusters)):
                    if k != i and k != j:
                        rest.update(clusters[k])
                
                if not rest:
                    continue
                
                d_i_rest = np.mean([D[a, b] for a in ci for b in rest])
                d_j_rest = np.mean([D[a, b] for a in cj for b in rest])
                
                # NJ-like criterion: lower is better
                score = d_ij - (d_i_rest + d_j_rest) / 2.0
                
                if score < best_score:
                    best_score = score
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        i, j = best_pair
        new_cluster = clusters[i] | clusters[j]
        new_name = f"({cluster_names[i]},{cluster_names[j]})"
        
        clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
        cluster_names = [cluster_names[k] for k in range(len(cluster_names)) if k != i and k != j]
        
        clusters.append(new_cluster)
        cluster_names.append(new_name)
    
    if len(cluster_names) == 2:
        return f"({cluster_names[0]},{cluster_names[1]})"
    elif len(cluster_names) == 1:
        return cluster_names[0] + ";"
    else:
        return "(" + ",".join(cluster_names) + ")"

def _generate_unrooted_binary_trees(n):
    """
    Generate all unrooted binary tree topologies for n leaves.
    Returns a list of trees represented as adjacency structures.
    """
    if n <= 2:
        return [list(range(n))]
    if n == 3:
        return [[0, 1, 2]]
    
    # For n >= 4, use recursive construction
    # This is exponential, so only use for small n
    if n > 10:
        return []
    
    trees = []
    
    def gen(k, current_leaves, current_internal):
        """
        Recursively build trees by adding one leaf at a time.
        """
        if k == n:
            trees.append(list(current_leaves) + list(current_internal))
            return
        
        # Try inserting new leaf k into all possible positions
        # This is a simplified approach
        pass
    
    # For simplicity, just return NJ-based tree for n > 3
    return []

def _felsenstein_pruning(tree, mat, labels):
    """
    Compute likelihood using Felsenstein's pruning algorithm.
    tree: list of nodes (leaves and internal)
    mat: encoded sequences (n, L)
    labels: list of leaf labels
    """
    # This is a placeholder for a full ML implementation
    # For now, return a dummy score
    return 0.0

def _ml_small_n(D, labels):
    """
    Maximum likelihood for small n.
    For now, fall back to NJ.
    """
    return _fast_nj(D, labels)

def _quartet_refine(D, labels, initial_newick):
    """
    Refine a tree using quartet scoring.
    For small n, try to improve the topology.
    """
    n = D.shape[0]
    if n < 4:
        return initial_newick
    
    # For small n, we can try to improve by checking quartet consistency
    # This is a simplified approach
    return initial_newick

def _nj_with_quartet_guidance(D, labels):
    """
    Neighbor-Joining with quartet-based refinement.
    """
    n = D.shape[0]
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        return f"({labels[0]},{labels[1]})"
    
    # First, get initial NJ tree
    newick = _fast_nj(D, labels)
    
    # For small n, try to improve with quartet-based methods
    if n <= 12:
        improved = _quartet_refine(D, labels, newick)
        if improved is not None:
            return improved
    
    return newick

def _star(labels):
    """
    Return a star tree (all leaves connected to a central node).
    """
    n = len(labels)
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    mat = _encode_sequences(sequences, labels)
    L = mat.shape[1]
    
    if L == 0:
        return _star(labels)
    
    D = _jc_distance_matrix(mat)
    
    # Use NJ with quartet guidance
    newick = _nj_with_quartet_guidance(D, labels)
    
    return newick

# EVOLVE-BLOCK-END
