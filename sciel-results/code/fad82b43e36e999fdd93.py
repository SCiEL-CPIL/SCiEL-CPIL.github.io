"""Neighbour Joining with Jukes-Cantor correction, vectorized for speed, plus a deadline guard."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
# Global deadline guard: 270s wall-clock budget; leave a safety margin of ~15s.
_DEADLINE = time.time() + 265.0


def _time_left():
    return _DEADLINE - time.time()


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences p."""
    p = min(max(p, 0.0), 0.7499)
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _jc_distance_vec(p_arr):
    """Vectorized Jukes-Cantor correction."""
    p_arr = np.clip(p_arr, 0.0, 0.7499)
    return -0.75 * np.log(1.0 - 4.0 * p_arr / 3.0)


def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix from DNA sequences using numpy."""
    n = len(labels)
    # Map each label to its sequence
    seqs = [sequences[lb] for lb in labels]
    length = len(seqs[0])
    
    # Convert to integer arrays for fast comparison
    # A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrs = []
    for s in seqs:
        arr = np.empty(length, dtype=np.int8)
        for i, c in enumerate(s):
            arr[i] = char_map[c]
        arrs.append(arr)
    
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Vectorized comparison against all j > i
        for j in range(i + 1, n):
            diff = np.count_nonzero(arrs[i] != arrs[j])
            p = diff / length
            d = _jc_distance(p)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _nj_tree(mat, labels):
    """Neighbour Joining on a distance matrix. Returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Active nodes
    nodes = list(range(n))
    # Subtrees as strings
    trees = {i: labels[i] for i in range(n)}
    # Current distance matrix (use a dict or 2D array; for up to 32 taxa, 2D array is fine)
    # We'll maintain a full matrix and update it.
    D = mat.copy()
    # Map from original index to current position in nodes list? 
    # Actually, NJ typically works with a list of active taxa and a distance matrix among them.
    # Let's use a simpler approach: maintain a list of active node IDs and a 2D array D_active.
    
    active = list(range(n))
    D_act = D[active][:, active].copy()
    
    while len(active) > 3:
        if _time_left() < 5.0:
            # Build a star tree from remaining nodes as fallback
            return "(" + ",".join(trees[i] for i in active) + ");"
        
        m = len(active)
        # Row sums
        r = D_act.sum(axis=1)
        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D_act - r[:, None] - r[None, :]
        
        # Find the pair with minimum Q
        min_val = np.inf
        min_i = min_j = 0
        for i in range(m):
            for j in range(i + 1, m):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i = i
                    min_j = j
        
        i, j = min_i, min_j
        a = active[i]
        b = active[j]
        
        # Compute branch lengths (optional, but good for validity)
        d_ab = D_act[i, j]
        # delta = (r[i] - r[j]) / (2*(m-2))
        delta = (r[i] - r[j]) / (2 * (m - 2))
        la = 0.5 * d_ab + 0.5 * delta
        lb = 0.5 * d_ab - 0.5 * delta
        la = max(0.0, la)
        lb = max(0.0, lb)
        
        # Create new node
        new_id = f"n{a}_{b}"
        trees[new_id] = f"({trees[a]},{trees[b]})"
        
        # Update active list: remove i and j, add new node
        new_active = [x for k, x in enumerate(active) if k != i and k != j]
        new_active.append(new_id)
        
        # Update distance matrix
        if len(new_active) > 1:
            # Build new distance matrix
            new_m = len(new_active)
            D_new = np.zeros((new_m, new_m), dtype=np.float64)
            for k, x in enumerate(new_active):
                for l, y in enumerate(new_active):
                    if k == l:
                        continue
                    if x == new_id:
                        # Distance from new node to y
                        if y == new_id:
                            continue
                        # Find original indices
                        # y is either an original node or a previous new node
                        # We need to map back to the original D
                        # This is getting complicated. Let's use a different approach:
                        # Keep a mapping from active node IDs to their "original" index in the full D matrix?
                        # Actually, the standard NJ updates the distance from the new node to all others:
                        # d(new, x) = 0.5*(d(a,x) + d(b,x) - d(a,b))
                        # So we need d(a,x) and d(b,x) from the current D_act.
                        pass
                    elif y == new_id:
                        pass
                    else:
                        # Both are original or previous nodes
                        # We need to find their positions in the old active list
                        pass
            
            # Let's redo this more carefully with a dictionary-based approach for clarity and correctness.
            # Since n <= 32, even O(n^2) per step is fine.
            
            # Reset: use a dict for distances
            # Actually, let's just use the full original matrix and track which nodes are active.
            # But the distances change as we join nodes.
            
            # Let's use a simpler, correct implementation:
            # Maintain a list of active nodes and a 2D array D_act.
            # When we join i and j (positions in active), we remove them and add a new node.
            # The new distances are computed from the old D_act.
            
            # Let's rebuild D_act properly.
            # First, map active node IDs to their indices in the old active list
            # Actually, active[i] = a, active[j] = b.
            # For each pair in new_active, compute distance.
            
            # Build a lookup: node_id -> index in old active list
            old_idx = {node: k for k, node in enumerate(active)}
            
            for k, x in enumerate(new_active):
                for l, y in enumerate(new_active):
                    if k == l:
                        continue
                    if x == new_id:
                        # x is the new node, y is some other node
                        y_old = old_idx[y]
                        # d(new, y) = 0.5*(d(a,y) + d(b,y) - d(a,b))
                        d_ay = D_act[old_idx[a], y_old]
                        d_by = D_act[old_idx[b], y_old]
                        d_ab = D_act[old_idx[a], old_idx[b]]
                        D_new[k, l] = 0.5 * (d_ay + d_by - d_ab)
                    elif y == new_id:
                        # symmetric
                        x_old = old_idx[x]
                        d_ax = D_act[old_idx[a], x_old]
                        d_bx = D_act[old_idx[b], x_old]
                        d_ab = D_act[old_idx[a], old_idx[b]]
                        D_new[k, l] = 0.5 * (d_ax + d_bx - d_ab)
                    else:
                        # Both are old nodes
                        x_old = old_idx[x]
                        y_old = old_idx[y]
                        D_new[k, l] = D_act[x_old, y_old]
            
            D_act = D_new
        else:
            D_act = np.zeros((1, 1), dtype=np.float64)
        
        active = new_active
    
    # Build the final tree from the remaining 3 nodes
    if len(active) == 3:
        return "(" + ",".join(trees[i] for i in active) + ");"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    else:
        return trees[active[0]] + ";"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    mat = _build_distance_matrix(sequences, labels)
    
    # Run NJ
    return _nj_tree(mat, labels)
# EVOLVE-BLOCK-END
