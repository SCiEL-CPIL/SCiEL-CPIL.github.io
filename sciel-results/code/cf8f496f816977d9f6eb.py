import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix from sequences."""
    n = len(labels)
    seq_len = len(next(iter(sequences.values())))
    
    # Convert sequences to integer arrays
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = np.zeros((n, seq_len), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            seq_arrays[i, j] = char_to_int[c]
    
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_arrays[i] != seq_arrays[j])
            p = diff / seq_len
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat

def _nj_build(dist_mat, n):
    """
    Build a binary tree using Neighbor Joining.
    Returns:
        root: node ID of the root
        internal: dict mapping internal node IDs to (left, right, len_left, len_right)
    """
    if n <= 1:
        return 0, {}
    if n == 2:
        internal = {2: (0, 1, 0.0, 0.0)}
        return 2, internal
    if n == 3:
        # For 3 taxa, create a star-like binary tree
        # (0,1) joined with 2
        len_0 = 0.5 * (dist_mat[0][1] + dist_mat[0][2] - dist_mat[1][2])
        len_1 = 0.5 * (dist_mat[0][1] + dist_mat[1][2] - dist_mat[0][2])
        len_2 = 0.5 * (dist_mat[0][2] + dist_mat[1][2] - dist_mat[0][1])
        internal = {
            3: (0, 1, len_0, len_1),
            4: (3, 2, 0.0, len_2)
        }
        return 4, internal
    
    # General case
    active = list(range(n))
    next_id = n
    internal = {}
    
    # Distance matrix for all potential nodes (size 2n)
    dist = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = dist_mat[i][j]
    
    while len(active) > 3:
        k = len(active)
        
        # Row sums
        r = [0.0] * k
        for i, node in enumerate(active):
            for j, other in enumerate(active):
                if i != j:
                    r[i] += dist[node][other]
        
        # Find best pair
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = active[i]
                node_j = active[j]
                q = (k - 2) * dist[node_i][node_j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        # Branch lengths
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (dist[node_a][node_b] + delta)
        len_b = 0.5 * (dist[node_a][node_b] - delta)
        
        new_node = next_id
        next_id += 1
        
        internal[new_node] = (node_a, node_b, len_a, len_b)
        
        # Update distances
        for x in active:
            if x != node_a and x != node_b:
                dist[new_node][x] = 0.5 * (dist[node_a][x] + dist[node_b][x] - dist[node_a][node_b])
                dist[x][new_node] = dist[new_node][x]
        
        # Remove a and b, add new_node
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
    
    # Finish with 3 nodes
    u, v, w = active
    len_u = 0.5 * (dist[u][v] + dist[u][w] - dist[v][w])
    len_v = 0.5 * (dist[u][v] + dist[v][w] - dist[u][w])
    len_w = 0.5 * (dist[u][w] + dist[v][w] - dist[u][v])
    
    nn1 = next_id
    next_id += 1
    internal[nn1] = (u, v, len_u, len_v)
    
    nn2 = next_id
    next_id += 1
    internal[nn2] = (nn1, w, 0.0, len_w)
    
    return nn2, internal

def _get_newick(node, n, labels, internal):
    """Recursively build Newick string."""
    if node < n:
        return labels[node]
    else:
        left, right, len_l, len_r = internal[node]
        return f"({_get_newick(left, n, labels, internal)}:{len_l:.4f},{_get_newick(right, n, labels, internal)}:{len_r:.4f})"

def _get_splits(root, n, internal):
    """
    Get all non-trivial splits from a binary tree.
    Returns a set of frozensets representing splits.
    """
    # First, get all nodes and their subtrees
    # For each internal node, compute the set of leaves in its subtree
    leaf_sets = {}
    
    def compute_leaf_set(node):
        if node < n:
            return frozenset([node])
        left, right, _, _ = internal[node]
        ls = compute_leaf_set(left) | compute_leaf_set(right)
        leaf_sets[node] = ls
        return ls
    
    compute_leaf_set(root)
    
    # Collect non-trivial splits
    splits = set()
    all_leaves = frozenset(range(n))
    
    for node, ls in leaf_sets.items():
        if 0 < len(ls) < n:
            complement = all_leaves - ls
            # Use the smaller side for canonical representation
            if len(ls) < len(complement):
                splits.add(ls)
            elif len(complement) < len(ls):
                splits.add(complement)
            else:
                # Equal size, use lexicographically smaller
                s1 = tuple(sorted(ls))
                s2 = tuple(sorted(complement))
                splits.add(frozenset(s1 if s1 < s2 else s2))
    
    return splits

def _nni_refine(root, n, labels, internal, dist_mat, time_limit, start_time):
    """
    Perform NNI refinement on the tree.
    Returns the improved root and internal dict.
    """
    def get_subtree_leaves(node):
        if node < n:
            return frozenset([node])
        left, right, _, _ = internal[node]
        return get_subtree_leaves(left) | get_subtree_leaves(right)
    
    def get_path_to_root(node):
        """Get list of internal nodes from node up to root."""
        path = []
        current = node
        while current in internal:
            path.append(current)
            # Find parent
            parent = None
            for parent_id, (l, r, _, _) in internal.items():
                if l == current or r == current:
                    parent = parent_id
                    break
            if parent is None:
                break
            current = parent
        return path
    
    def compute_edge_lengths(root_node):
        """Compute edge lengths for all internal edges using least squares on distances."""
        # For simplicity, use the existing lengths from NJ
        # This is a simplified approach
        pass
    
    def nni_score_change(node_a, node_b, node_c, node_d):
        """
        Compute the change in total tree length when performing NNI
        on the quartet a-b-c-d where the current split is (ab|cd)
        and we consider (ac|bd) or (ad|bc).
        """
        # Get distances
        dab = dist_mat[node_a][node_b]
        dac = dist_mat[node_a][node_c]
        dad = dist_mat[node_a][node_d]
        dbc = dist_mat[node_b][node_c]
        dbd = dist_mat[node_b][node_d]
        dcd = dist_mat[node_c][node_d]
        
        # Current split: (ab|cd)
        # New splits: (ac|bd) and (ad|bc)
        
        # For NNI, we need to know which split is currently present
        # This function assumes we're evaluating a specific NNI move
        
        # Simplified: use the 4-point condition
        # If d_ab + d_cd < d_ac + d_bd and d_ab + d_cd < d_ad + d_bc,
        # then (ab|cd) is the best split
        # Otherwise, try the other splits
        
        score_ab_cd = dab + dcd
        score_ac_bd = dac + dbd
        score_ad_bc = dad + dbc
        
        return score_ab_cd, score_ac_bd, score_ad_bc
    
    # Get all internal edges to consider for NNI
    # An NNI move applies to an internal edge
    # For each internal edge, there are 2 possible alternative topologies
    
    improved = False
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        if time.time() - start_time > time_limit:
            break
        
        # Find all internal edges
        # An internal edge connects two internal nodes
        internal_edges = []
        for node_id, (left, right, _, _) in internal.items():
            # Check if left is internal
            if left in internal:
                internal_edges.append((node_id, left))
            # Check if right is internal
            if right in internal:
                internal_edges.append((node_id, right))
        
        best_move = None
        best_gain = 0.0
        
        for parent, child in internal_edges:
            # The edge is between parent and child
            # child has two children: c_left, c_right
            # parent has another child (the one that's not child)
            if child in internal:
                c_left, c_right, _, _ = internal[child]
            else:
                continue  # Only consider edges where child is internal
            
            # Find the other child of parent
            p_left, p_right, _, _ = internal[parent]
            if p_left == child:
                other_child = p_right
            elif p_right == child:
                other_child = p_left
            else:
                continue
            
            # Now we have:
            # parent connected to child and other_child
            # child connected to c_left and c_right
            # The NNI move will rewire:
            # Current: (c_left, c_right) - parent - other_child
            # Option 1: (c_left, other_child) - parent - c_right
            # Option 2: (c_right, other_child) - parent - c_left
            
            # Compute the gain for each option
            # We need to estimate the change in total tree length
            
            # Get leaf sets for scoring
            leaves_cl = get_subtree_leaves(c_left)
            leaves_cr = get_subtree_leaves(c_right)
            leaves_oc = get_subtree_leaves(other_child)
            
            # For simplicity, use a heuristic based on distances
            # Compute average distances between the three groups
            def avg_dist(group1, group2):
                total = 0.0
                count = 0
                for x in group1:
                    for y in group2:
                        if x < n and y < n:
                            total += dist_mat[x][y]
                            count += 1
                return total / count if count > 0 else 0.0
            
            # Current configuration
            d_cl_cr = avg_dist(leaves_cl, leaves_cr)
            d_cl_oc = avg_dist(leaves_cl, leaves_oc)
            d_cr_oc = avg_dist(leaves_cr, leaves_oc)
            
            # Option 1: (c_left, other_child) and c_right
            # New distances to consider
            gain1 = 0.0
            # Option 2: (c_right, other_child) and c_left
            gain2 = 0.0
            
            # This is a simplified heuristic
            # A more rigorous approach would use least squares
            # For now, we'll use the 4-point condition on representative taxa
            
            if leaves_cl and leaves_cr and leaves_oc:
                # Pick representative taxa
                rep_cl = min(leaves_cl)
                rep_cr = min(leaves_cr)
                rep_oc = min(leaves_oc)
                
                # Only if all are leaves
                if rep_cl < n and rep_cr < n and rep_oc < n:
                    d1 = dist_mat[rep_cl][rep_cr]
                    d2 = dist_mat[rep_cl][rep_oc]
                    d3 = dist_mat[rep_cr][rep_oc]
                    
                    # Current: (cl, cr) | oc
                    # Option 1: (cl, oc) | cr
                    # Option 2: (cr, oc) | cl
                    
                    # Use 4-point condition
                    # Current split score: d1
                    # Option 1 score: d2
                    # Option 2 score: d3
                    
                    # The gain is how much better the alternative is
                    current_score = d1
                    opt1_score = d2
                    opt2_score = d3
                    
                    gain1 = current_score - opt1_score
                    gain2 = current_score - opt2_score
            
            if gain1 > best_gain:
                best_gain = gain1
                best_move = ('opt1', parent, child, c_left, c_right, other_child)
            if gain2 > best_gain:
                best_gain = gain2
                best_move = ('opt2', parent, child, c_left, c_right, other_child)
        
        if best_move is None or best_gain <= 0:
            break
        
        # Apply the best move
        move_type, parent, child, c_left, c_right, other_child = best_move
        
        if move_type == 'opt1':
            # (c_left, other_child) and c_right
            # New child node: (c_left, other_child)
            new_child_id = max(internal.keys()) + 1 if internal else n
            
            # Compute new lengths (simplified)
            len_cl = 0.0
            len_oc = 0.0
            len_cr = 0.0
            
            internal[new_child_id] = (c_left, other_child, len_cl, len_oc)
            internal[parent] = (new_child_id, c_right, 0.0, len_cr)
        else:
            # (c_right, other_child) and c_left
            new_child_id = max(internal.keys()) + 1 if internal else n
            
            len_cr = 0.0
            len_oc = 0.0
            len_cl = 0.0
            
            internal[new_child_id] = (c_right, other_child, len_cr, len_oc)
            internal[parent] = (new_child_id, c_left, 0.0, len_cl)
        
        improved = True
    
    return root, internal, improved

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 260.0  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"
    
    # 1. Compute Distance Matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # 2. Build initial tree with NJ
    root, internal = _nj_build(dist_mat, n)
    
    # 3. NNI Refinement
    if time.time() - start_time < time_limit * 0.7:
        root, internal, _ = _nni_refine(root, n, labels, internal, dist_mat, time_limit, start_time)
    
    # 4. Generate Newick
    newick = _get_newick(root, n, labels, internal)
    
    return newick
