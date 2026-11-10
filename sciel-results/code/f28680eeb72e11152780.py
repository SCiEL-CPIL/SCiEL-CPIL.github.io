import math
import numpy as np
import time

# EVOLVE-BLOCK-START
'''
Robust Neighbor Joining (NJ) with NNI Refinement.
Uses NumPy for distance matrix calculations.
No SciPy dependency.
'''

def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix."""
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n = len(labels)
    # Convert sequences to numpy array of ints
    seqs = np.array([[base_map[c] for c in sequences[label]] for label in labels], dtype=np.int8)
    L = seqs.shape[1]
    
    D = np.zeros((n, n), dtype=np.float64)
    # Vectorized pairwise difference calculation
    # To save memory and time for n<=32, we can do a loop or broadcasting.
    # Broadcasting (n, 1, L) vs (1, n, L) creates (n, n, L) which is 32*32*2000 = 2M floats ~ 16MB. Safe.
    diff_matrix = np.sum(seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :], axis=2)
    p = diff_matrix / L
    
    # Handle p >= 0.75 case to avoid log(0) or negative
    # log(1 - 4p/3)
    # If p > 0.75, 1 - 4p/3 < 0.
    # Clip p to max 0.7499
    p_safe = np.clip(p, 0, 0.7499)
    
    D = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(D, 0.0)
    return D

def _neighbor_joining(D, labels):
    """
    Standard Neighbor Joining algorithm.
    Returns a tree structure as a list of nodes for NNI.
    Node format:
    Leaf: (label, length_to_parent)
    Internal: (left_child_idx, right_child_idx, length_to_parent)
    We build the tree backwards or use a recursive structure.
    For NNI, it's easier to have a tree where we can identify internal edges.
    
    Let's represent the tree as a dictionary of nodes.
    node_id -> {
        'type': 'leaf' or 'internal',
        'label': str (if leaf),
        'children': [id1, id2] (if internal),
        'length': float (length of edge to parent)
    }
    """
    n = len(labels)
    if n <= 2:
        # Trivial cases
        if n == 1:
            return [{'type': 'leaf', 'label': labels[0], 'length': 0.0}], 0
        if n == 2:
            d = D[0, 1]
            # Create a root node connecting 0 and 1
            # In unrooted NJ, we just join them.
            # We'll create a root internal node.
            nodes = []
            nodes.append({'type': 'leaf', 'label': labels[0], 'length': d/2})
            nodes.append({'type': 'leaf', 'label': labels[1], 'length': d/2})
            nodes.append({'type': 'internal', 'children': [0, 1], 'length': 0.0})
            return nodes, 2

    # Active nodes
    active = list(range(n))
    # Current distance matrix
    dist = D.copy()
    
    # Node structures
    # We will assign new IDs to internal nodes as we go
    next_id = n
    nodes = []
    # Initialize leaves
    for i in range(n):
        nodes.append({'type': 'leaf', 'label': labels[i], 'length': 0.0})
    
    # We need to track the "current" tree structure for each active node.
    # Initially, active node i corresponds to nodes[i].
    # When we join two nodes, we create a new node.
    
    # Mapping from active index to node ID in `nodes` list
    # active is a list of node IDs currently in the distance matrix.
    # dist is indexed by these node IDs.
    # Note: dist is fixed size n x n? No, we need to expand it or use a dict.
    # Standard NJ implementations often use a fixed matrix and mark inactive nodes.
    # Here, let's use a dictionary for distances to be safe and simple, or a large enough matrix.
    # Max nodes = 2*n - 1.
    max_nodes = 2 * n
    dist_mat = np.zeros((max_nodes, max_nodes))
    # Copy initial distances
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = D[i, j]
    
    active_ids = list(range(n))
    
    while len(active_ids) > 3:
        m = len(active_ids)
        
        # Calculate r_i = sum of distances from i to other active nodes
        r = np.zeros(m)
        for i in range(m):
            idx = active_ids[i]
            # Sum dist_mat[idx, active_ids[j]] for j != i
            # Since dist_mat is symmetric and diagonal is 0, we can sum all and it's fine
            r[i] = np.sum(dist_mat[idx, active_ids])
            
        # Find pair (a, b) minimizing Q_ab
        best_val = float('inf')
        best_i, best_j = -1, -1
        
        for i in range(m):
            for j in range(i + 1, m):
                idx_a = active_ids[i]
                idx_b = active_ids[j]
                q_val = (m - 2) * dist_mat[idx_a, idx_b] - r[i] - r[j]
                if q_val < best_val:
                    best_val = q_val
                    best_i, best_j = i, j
        
        idx_a = active_ids[best_i]
        idx_b = active_ids[best_j]
        
        d_ab = dist_mat[idx_a, idx_b]
        r_a = r[best_i]
        r_b = r[best_j]
        
        # Branch lengths
        d_a_new = 0.5 * d_ab + (r_a - r_b) / (2.0 * (m - 2))
        d_b_new = 0.5 * d_ab - (r_a - r_b) / (2.0 * (m - 2))
        
        if d_a_new < 0: d_a_new = 0
        if d_b_new < 0: d_b_new = 0
        
        # Create new node
        new_idx = next_id
        next_id += 1
        
        # Update distances to new node
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            idx_x = active_ids[k]
            dist_mat[new_idx, idx_x] = 0.5 * (dist_mat[idx_a, idx_x] + dist_mat[idx_b, idx_x] - d_ab)
            dist_mat[idx_x, new_idx] = dist_mat[new_idx, idx_x]
            
        # Remove a and b, add new
        active_ids.remove(idx_a)
        active_ids.remove(idx_b)
        active_ids.append(new_idx)
        
        # Create node structure
        # The children are the nodes corresponding to idx_a and idx_b
        # We need to set their lengths to the new node
        nodes[idx_a]['length'] = d_a_new
        nodes[idx_b]['length'] = d_b_new
        nodes.append({'type': 'internal', 'children': [idx_a, idx_b], 'length': 0.0})
        
    # Final 3 nodes
    a, b, c = active_ids
    d_ab = dist_mat[a, b]
    d_bc = dist_mat[b, c]
    d_ca = dist_mat[c, a]
    
    # Calculate lengths for the final 3-leaf star
    # The internal node connecting them
    # L_a = (d_ab + d_ca - d_bc) / 2
    L_a = max(0, (d_ab + d_ca - d_bc) / 2.0)
    L_b = max(0, (d_ab + d_bc - d_ca) / 2.0)
    L_c = max(0, (d_ca + d_bc - d_ab) / 2.0)
    
    # We need to create a root for these 3?
    # In an unrooted tree, we can just connect them to a new internal node.
    # But standard NJ finishes with 3 leaves. We can create a new internal node 'root'
    # connecting a, b, c? No, that's a 3-way split.
    # To keep binary, we join two, then join the third?
    # Actually, for NNI, we want a binary tree.
    # Let's pick the pair with smallest distance to join first?
    # Or just create a node connecting a and b, then connect c to that node?
    # The distances are already "to the center".
    # Let's create an internal node X connecting a and b.
    # Then connect c to X?
    # This implies a topology.
    # Better: Create a new internal node 'root' (id = next_id)
    # Connect a, b, c to it? No, that's non-binary.
    # Let's join a and b into node Y.
    # Length a-Y = L_a? No.
    # In the 3-leaf case, the "internal" node is the centroid.
    # If we force binary, we have to pick a pair.
    # Let's pick the pair with smallest distance to join.
    if d_ab <= d_bc and d_ab <= d_ca:
        join_a, join_b, join_c = a, b, c
        d_join = d_ab
        d_ac = d_ca
        d_bc = d_bc
    elif d_bc <= d_ab and d_bc <= d_ca:
        join_a, join_b, join_c = b, c, a
        d_join = d_bc
        d_ac = d_ca
        d_bc = d_ab
    else:
        join_a, join_b, join_c = c, a, b
        d_join = d_ca
        d_ac = d_ab
        d_bc = d_bc
        
    # Create internal node Y joining join_a and join_b
    y_idx = next_id
    next_id += 1
    
    # Lengths to Y
    # The distance between join_a and join_b is d_join.
    # The length from join_a to Y is roughly d_join/2?
    # Actually, the "L_a" calculated above is the distance from the leaf to the centroid.
    # If we join a and b, the edge a-Y and b-Y should be such that they are consistent.
    # Let's just use L_a and L_b for the joined pair?
    # If we join a and b, the branch lengths to the new node Y are:
    # len_a = L_a
    # len_b = L_b
    # Then we connect c to Y with length L_c?
    # This forms a 3-way split at Y.
    # To make it binary, we can consider Y as an internal node with children a, b.
    # And then create a root Z connecting Y and c.
    # Length Y-Z = 0?
    # Length c-Z = L_c?
    # This is a valid unrooted binary tree representation.
    
    nodes[join_a]['length'] = L_a
    nodes[join_b]['length'] = L_b
    nodes.append({'type': 'internal', 'children': [join_a, join_b], 'length': 0.0}) # Node Y
    
    nodes[join_c]['length'] = L_c
    z_idx = next_id
    next_id += 1
    nodes.append({'type': 'internal', 'children': [y_idx, join_c], 'length': 0.0}) # Node Z (Root)
    
    return nodes, z_idx

def _nni_refine(nodes, root_id, D, labels, time_limit):
    """
    Perform NNI refinement.
    Minimize the sum of squared errors between tree distances and observed distances.
    """
    start_time = time.time()
    
    # Helper to calculate tree distance between two leaves
    def get_leaf_path(leaf_id, nodes, root_id):
        # Find path from leaf to root
        path = []
        current = leaf_id
        while current != -1:
            path.append(current)
            node = nodes[current]
            if node['type'] == 'leaf':
                break
            # Find parent
            # This is slow. We should precompute parents.
            pass
        return path

    # Precompute parent pointers
    parent = [-1] * len(nodes)
    for i, node in enumerate(nodes):
        if node['type'] == 'internal':
            for child in node['children']:
                parent[child] = i
                
    # Get leaf indices
    leaf_indices = [i for i, node in enumerate(nodes) if node['type'] == 'leaf']
    # Map leaf index to label
    leaf_label_map = {i: nodes[i]['label'] for i in leaf_indices}
    
    # Calculate tree distances
    def calc_tree_distances(nodes, parent, leaf_indices):
        # Distance between leaf i and leaf j is sum of lengths on path
        # We can compute LCA or just sum up.
        # For N=32, O(N^2 * Depth) is fine.
        n_leaves = len(leaf_indices)
        tree_dist = np.zeros((n_leaves, n_leaves))
        
        # Precompute depth and ancestor lists?
        # Simpler: for each pair, find LCA.
        for i in range(n_leaves):
            li = leaf_indices[i]
            # Path from li to root
            path_i = []
            curr = li
            while curr != -1:
                path_i.append(curr)
                curr = parent[curr]
            path_set = set(path_i)
            
            for j in range(i + 1, n_leaves):
                lj = leaf_indices[j]
                # Path from lj to root
                path_j = []
                curr = lj
                while curr != -1:
                    path_j.append(curr)
                    curr = parent[curr]
                
                # Find LCA
                # The LCA is the last common node in path_i and path_j?
                # No, path is leaf->root. LCA is the first common node from root?
                # Actually, LCA is the node where the paths merge.
                # Since path_i is [leaf, ..., root], and path_j is [leaf, ..., root]
                # The LCA is the node in path_i that is also in path_j and is closest to the leaves?
                # No, it's the node with the smallest depth (closest to root) that is common?
                # Wait, leaf is depth 0, root is depth H.
                # Common nodes are the ancestors. The LCA is the one with the smallest depth (closest to root) among common ancestors?
                # No, LCA is the *lowest* common ancestor, so it has the *largest* depth (closest to leaves) among common ancestors.
                # Let's find the first common node when traversing from leaf upwards?
                # path_i: [l, p1, p2, ..., root]
                # path_j: [m, q1, q2, ..., root]
                # Intersection is the ancestors.
                # The LCA is the one that appears first in the intersection when looking from the root down?
                # Or simply, iterate path_i, find the first node that is in path_j.
                # Since both go to root, the first common node encountered from the leaf side is NOT the LCA.
                # Example: A-B-C and D-E-C. Path A: [A, B, C]. Path D: [D, E, C].
                # Common: C. LCA is C.
                # Example: A-B-C-D and A-B-E-D. Path A: [A, B, C, D]. Path E: [E, B, D].
                # Common: B, D. LCA is B.
                # If we iterate path_i [A, B, C, D], first common with path_j_set {E, B, D} is B. Correct.
                
                lca = None
                for node_id in path_i:
                    if node_id in path_set: # Wait, path_set is path_i. We need path_j set.
                        pass
                
                # Let's do it properly
                path_j_set = set(path_j)
                for node_id in path_i:
                    if node_id in path_j_set:
                        lca = node_id
                        break
                        
                if lca is None:
                    continue # Should not happen
                    
                # Distance = depth(li) + depth(lj) - 2 * depth(lca)
                # Depth is number of edges from root? Or sum of lengths?
                # Sum of lengths is better.
                # dist = sum(len(node) for node in path from li to lca) + sum(len(node) for node in path from lj to lca)
                
                # Sum lengths from li to lca
                d1 = 0
                curr = li
                while curr != lca:
                    d1 += nodes[curr]['length']
                    curr = parent[curr]
                    
                d2 = 0
                curr = lj
                while curr != lca:
                    d2 += nodes[curr]['length']
                    curr = parent[curr]
                    
                tree_dist[i, j] = d1 + d2
                tree_dist[j, i] = d1 + d2
                
        return tree_dist

    # Current tree distances
    # Map leaf indices to D indices
    # D is indexed by labels t0..tn
    # leaf_indices are node IDs in `nodes`
    # We need to map node ID -> label -> D index
    label_to_d_idx = {label: i for i, label in enumerate(labels)}
    
    # Construct matrix of observed distances for leaves in leaf_indices order
    n_leaves = len(leaf_indices)
    obs_dist = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        label_i = nodes[leaf_indices[i]]['label']
        idx_i = label_to_d_idx[label_i]
        for j in range(i + 1, n_leaves):
            label_j = nodes[leaf_indices[j]]['label']
            idx_j = label_to_d_idx[label_j]
            obs_dist[i, j] = D[idx_i, idx_j]
            obs_dist[j, i] = D[idx_i, idx_j]
            
    # Calculate score: Sum of squared errors
    def calculate_score(nodes, parent):
        tree_dist = calc_tree_distances(nodes, parent, leaf_indices)
        # Only consider upper triangle
        err = 0
        for i in range(n_leaves):
            for j in range(i + 1, n_leaves):
                diff = tree_dist[i, j] - obs_dist[i, j]
                err += diff * diff
        return err

    current_score = calculate_score(nodes, parent)
    
    # NNI Search
    # Identify internal edges
    # An internal edge connects an internal node to its parent (if parent is internal) or root?
    # In a binary unrooted tree, every internal edge connects two internal nodes.
    # We can iterate over all internal nodes (except root) and consider the edge to their parent.
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        # Iterate over all internal nodes that are not the root
        for u in range(len(nodes)):
            if nodes[u]['type'] != 'internal':
                continue
            if u == root_id:
                continue
            
            p = parent[u]
            if p == -1 or nodes[p]['type'] != 'internal':
                continue
                
            # Edge (p, u) is an internal edge.
            # The two subtrees attached to u are c1, c2.
            # The other child of p is v.
            c1, c2 = nodes[u]['children']
            
            # Find v
            if nodes[p]['children'][0] == u:
                v = nodes[p]['children'][1]
            else:
                v = nodes[p]['children'][0]
                
            # Current topology: p connected to u and v. u connected to c1 and c2.
            # Alternative 1: p connected to c1 and (u connected to c2 and v)? No.
            # NNI moves for edge (p, u):
            # 1. Current: (c1, c2) attached to p via u. v attached to p.
            #    Structure: p -- u -- (c1, c2) and p -- v
            # 2. Swap c1 and v:
            #    New u' connected to c2 and v.
            #    New p' connected to c1 and u'.
            # 3. Swap c2 and v:
            #    New u' connected to c1 and v.
            #    New p' connected to c2 and u'.
            
            # We need to reassign children and update lengths?
            # For topology scoring, we can just swap the children pointers and recalculate score.
            # But lengths matter for the distance calculation.
            # If we swap topologies, the branch lengths change.
            # Simple heuristic: Keep the lengths associated with the "subtrees" (c1, c2, v) as they are,
            # but reassign which edge connects to which.
            # This is an approximation.
            
            # Let's try the swap and see if score improves.
            # Save original state
            orig_u_children = nodes[u]['children'][:]
            orig_p_children = nodes[p]['children'][:]
            
            # Try Swap 1: v and c1
            # New structure:
            # u has children [c2, v]
            # p has children [c1, u] (u is still child of p, but u's children changed)
            # Wait, the edge lengths are attached to the nodes.
            # nodes[u]['length'] is length from p to u.
            # nodes[c1]['length'] is length from u to c1.
            # nodes[c2]['length'] is length from u to c2.
            # nodes[v]['length'] is length from p to v.
            
            # If we swap c1 and v:
            # New edge p-u still exists?
            # In NNI, we cut the edge (p, u) and (u, c1) and (p, v)?
            # No, NNI on edge (p, u) swaps one of u's children with p's other child.
            # The edge (p, u) remains.
            # The children of u become {c2, v}.
            # The children of p become {c1, u}.
            # The lengths:
            # We assume the length of the edge connecting a subtree to a node is preserved.
            # So len(c1) stays same, len(v) stays same, len(c2) stays same, len(u) stays same.
            
            nodes[u]['children'] = [c2, v]
            nodes[p]['children'] = [c1, u]
            
            score1 = calculate_score(nodes, parent)
            
            if score1 < current_score:
                current_score = score1
                improved = True
                # Update lengths? We kept them. This is a heuristic.
            else:
                # Try Swap 2: v and c2
                nodes[u]['children'] = [c1, v]
                nodes[p]['children'] = [c2, u]
                
                score2 = calculate_score(nodes, parent)
                
                if score2 < current_score:
                    current_score = score2
                    improved = True
                else:
                    # Revert
                    nodes[u]['children'] = orig_u_children
                    nodes[p]['children'] = orig_p_children

    return nodes, root_id

def _tree_to_newick(nodes, root_id):
    def rec(u):
        node = nodes[u]
        if node['type'] == 'leaf':
            return node['label']
        else:
            c1, c2 = node['children']
            s1 = rec(c1)
            s2 = rec(c2)
            # Length is to parent
            # In Newick, (child1:len1, child2:len2)
            # The length printed is the length of the branch leading TO the child.
            return f"({s1}:{nodes[c1]['length']:.6f},{s2}:{nodes[c2]['length']:.6f})"
    return rec(root_id) + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 170.0
    
    # 1. Compute Distances
    D = _jc_distance_matrix(sequences, labels)
    
    # 2. NJ Tree
    nodes, root_id = _neighbor_joining(D, labels)
    
    # 3. NNI Refinement
    # Allocate time for NNI
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5: # At least 5 seconds left
        nodes, root_id = _nni_refine(nodes, root_id, D, labels, remaining - 5)
        
    # 4. Return Newick
    newick = _tree_to_newick(nodes, root_id)
    
    return newick

# EVOLVE-BLOCK-END
