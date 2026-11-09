<<<<<<< SEARCH
def _nj_newick(D, labels):
    """
    Neighbor Joining returning Newick string.
    D: n x n distance matrix.
    labels: list of labels.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
    
    # We will maintain a list of "active" nodes.
    # node_struct: id -> representation
    # For leaves: string label
    # For internal nodes: tuple (left_node_id, right_node_id, len_left, len_right)
    # dist: dict (id1, id2) -> distance
    
    active = list(range(n))
    node_struct = {i: labels[i] for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(n) if i != j}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        # Calculate r_i
        r = {}
        for i in active:
            r[i] = sum(dist[(i, j)] for j in active if j != i)
        
        # Find min Q
        min_q = float('inf')
        min_i, min_j = -1, -1
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        d_ij = dist[(min_i, min_j)]
        d_i = 0.5 * (d_ij + (r[min_i] - r[min_j]) / (m - 2))
        d_j = d_ij - d_i
        
        new_node = next_id
        next_id += 1
        
        # Store structure: (left_id, right_id, len_left, len_right)
        node_struct[new_node] = (min_i, min_j, d_i, d_j)
        
        # Update distances to new node
        for k in active:
            if k == min_i or k == min_j:
                continue
            d_new_k = 0.5 * (dist[(min_i, k)] + dist[(min_j, k)] - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
        
        # Remove old nodes, add new
        active.remove(min_i)
        active.remove(min_j)
        active.append(new_node)
        
    # Now active has 3 nodes. Join them.
    # We need to build the final tree.
    # Let's join two of them, then the result with the third.
    # But for the final Newick, we can just output a star or a binary tree.
    # Standard NJ ends with 3 nodes connected to a central node? No, it's a binary tree.
    # Let's join a and b into n1, then n1 and c into n2 (root).
    # But wait, the last step of NJ usually just connects the remaining 3 nodes.
    # If we treat it as binary:
    # Join a and b.
    # d_ab = dist[(a, b)]
    # d_a = 0.5 * (d_ab + dist[(a, c)] - dist[(b, c)])
    # d_b = d_ab - d_a
    # New node n1.
    # Then we have n1 and c.
    # The distance between n1 and c is dist[(n1, c)] which should have been calculated in the previous step if it was not the last one?
    # Wait, if we exit the loop with 3 nodes, we haven't updated distances for the new node yet?
    # No, we update distances inside the loop.
    # So if we have 3 nodes left, they are the result of previous joins.
    # Let's just join two of them manually.
    
    a, b, c = active
    d_ab = dist[(a, b)]
    d_a = 0.5 * (d_ab + dist[(a, c)] - dist[(b, c)])
    d_b = d_ab - d_a
    
    new_node = next_id
    next_id += 1
    node_struct[new_node] = (a, b, d_a, d_b)
    
    # Now we need to connect new_node and c.
    # The distance between new_node and c is dist[(new_node, c)]?
    # It should have been calculated if new_node was created in a previous iteration.
    # But here new_node is created NOW.
    # We need the distance between the clade (a,b) and c.
    # This is not directly in dist.
    # However, we can calculate the branch length from the clade to c.
    # In an unrooted tree, the edge connecting the cherry (a,b) to c has length:
    # len_edge = 0.5 * (dist[(a, c)] + dist[(b, c)] - dist[(a, b)])
    # This is the length of the branch from the internal node (a,b) to c.
    # Let's call it d_c_branch.
    d_c_branch = 0.5 * (dist[(a, c)] + dist[(b, c)] - d_ab)
    
    # Now we have:
    # a: d_a
    # b: d_b
    # c: d_c_branch
    # And they meet at the internal node.
    # The Newick string for this unrooted tree (rooted at the internal node) is:
    # (a:d_a, b:d_b, c:d_c_branch);
    # But we need to handle the case where a or b or c are internal nodes themselves.
    # We need a helper to convert node_id to Newick string.
    
    def to_newick(node_id):
        struct = node_struct[node_id]
        if isinstance(struct, str):
            return struct
        else:
            left_id, right_id, len_left, len_right = struct
            return f"({to_newick(left_id)}:{len_left:.4f},{to_newick(right_id)}:{len_right:.4f})"
    
    # The final tree is not just (a, b, c) if a, b, c are internal.
    # The structure is:
    # Root
    #  |-- a (len d_a)
    #  |-- b (len d_b)
    #  |-- c (len d_c_branch)
    # But a, b, c are subtrees.
    # So the Newick is:
    # (to_newick(a):d_a, to_newick(b):d_b, to_newick(c):d_c_branch);
    # Wait, to_newick(a) returns the subtree for a.
    # If a is a leaf, it returns the label.
    # If a is internal, it returns (left:..., right:...).
    # So the branch length d_a is attached to the subtree a.
    # This is correct.
    
    return f"({to_newick(a)}:{d_a:.4f},{to_newick(b)}:{d_b:.4f},{to_newick(c)}:{d_c_branch:.4f});"
=======
def _nj_newick(D, labels, time_limit=None):
    """
    Neighbor Joining returning Newick string.
    D: n x n distance matrix.
    labels: list of labels.
    time_limit: optional max seconds for NJ computation.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
    
    active = list(range(n))
    node_struct = {i: labels[i] for i in range(n)}
    # Use a 2D array for faster distance lookups
    # We'll maintain a mapping from node_id to index in a growing array
    # But simpler: use dict for dist as before, but with numpy for r computation
    
    dist = {}
    for i in range(n):
        for j in range(i+1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]
    
    next_id = n
    start_t = time.time()
    
    while len(active) > 3:
        if time_limit is not None and (time.time() - start_t) > time_limit:
            # Fallback: star tree
            break
        
        m = len(active)
        # Calculate r_i using vectorized sum where possible
        r = np.zeros(m)
        for idx_i, i in enumerate(active):
            s = 0.0
            for idx_j, j in enumerate(active):
                if i != j:
                    s += dist.get((i, j), 0.0)
            r[idx_i] = s
        
        # Find min Q
        min_q = float('inf')
        min_i_idx, min_j_idx = -1, -1
        for idx_i in range(m):
            for idx_j in range(idx_i + 1, m):
                i = active[idx_i]
                j = active[idx_j]
                q = (m - 2) * dist.get((i, j), 0.0) - r[idx_i] - r[idx_j]
                if q < min_q:
                    min_q = q
                    min_i_idx = idx_i
                    min_j_idx = idx_j
        
        min_i = active[min_i_idx]
        min_j = active[min_j_idx]
        
        d_ij = dist.get((min_i, min_j), 0.0)
        d_i = 0.5 * (d_ij + (r[min_i_idx] - r[min_j_idx]) / (m - 2))
        d_j = d_ij - d_i
        
        new_node = next_id
        next_id += 1
        
        node_struct[new_node] = (min_i, min_j, d_i, d_j)
        
        # Update distances to new node
        for idx_k, k in enumerate(active):
            if k == min_i or k == min_j:
                continue
            d_new_k = 0.5 * (dist.get((min_i, k), 0.0) + dist.get((min_j, k), 0.0) - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
        
        # Remove old nodes, add new
        active.remove(min_i)
        active.remove(min_j)
        active.append(new_node)
        
    # Now active has 3 or more nodes (if we broke early) or exactly 3
    if len(active) == 3:
        a, b, c = active
        d_ab = dist.get((a, b), 0.0)
        d_a = 0.5 * (d_ab + dist.get((a, c), 0.0) - dist.get((b, c), 0.0))
        d_b = d_ab - d_a
        d_c_branch = 0.5 * (dist.get((a, c), 0.0) + dist.get((b, c), 0.0) - d_ab)
        
        def to_newick(node_id):
            struct = node_struct[node_id]
            if isinstance(struct, str):
                return struct
            else:
                left_id, right_id, len_left, len_right = struct
                return f"({to_newick(left_id)}:{len_left:.4f},{to_newick(right_id)}:{len_right:.4f})"
        
        return f"({to_newick(a)}:{d_a:.4f},{to_newick(b)}:{d_b:.4f},{to_newick(c)}:{d_c_branch:.4f});"
    else:
        # Fallback: star tree with all remaining active nodes
        def to_newick(node_id):
            struct = node_struct[node_id]
            if isinstance(struct, str):
                return struct
            else:
                left_id, right_id, len_left, len_right = struct
                return f"({to_newick(left_id)}:{len_left:.4f},{to_newick(right_id)}:{len_right:.4f})"
        
        parts = [f"{to_newick(node)}:0.0" for node in active]
        return "(" + ",".join(parts) + ");"
>>>>>>> REPLACE

