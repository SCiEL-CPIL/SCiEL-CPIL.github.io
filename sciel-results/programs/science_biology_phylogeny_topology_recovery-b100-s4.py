import math
import time

def _jc_distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    start_time = time.time()
    timeout = 270.0
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    # Handle trivial cases
    if n_taxa == 1:
        return f"({labels[0]});"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Compute initial distance matrix
    # Using a dictionary or list of lists. List of lists is faster for small n.
    # Map labels to indices 0..n-1
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = labels
    
    # D is a list of lists
    D = [[0.0] * n_taxa for _ in range(n_taxa)]
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            D[i][j] = d
            D[j][i] = d

    # Active nodes
    active = list(range(n_taxa))
    # Tree representation: for each node, store its "Newick string" or subtree structure
    # Since we just need topology for RF distance, we can store the Newick string directly.
    # But updating Newick strings by concatenation is O(n) per join, which is fine.
    # Alternatively, store a tree structure (left, right, length) and convert at the end.
    # Storing Newick strings is simpler for the final output.
    
    # Let's store the Newick string for each active node.
    # Initially, each leaf is its label.
    node_strings = [label for label in labels]
    
    # We need to update the distance matrix.
    # It's easier to maintain a dictionary of distances between active nodes?
    # Or a list of lists indexed by active node indices?
    # Let's use a dictionary keyed by tuple of active node IDs (original indices or new ones).
    # Since we create new nodes, let's assign new IDs.
    
    # Better approach:
    # Keep a list `nodes` of current node IDs.
    # Keep a dictionary `dist` mapping (id1, id2) -> distance.
    # Keep a dictionary `tree_str` mapping id -> newick string.
    
    # Let's re-initialize with this structure.
    nodes = list(range(n_taxa))
    dist = {}
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            dist[(i, j)] = D[i][j]
            dist[(j, i)] = D[i][j]
            
    tree_str = {i: labels[i] for i in range(n_taxa)}
    next_id = n_taxa

    n = len(nodes)
    
    while n > 3:
        # Check time
        if time.time() - start_time > timeout - 10:
            # Return current best guess (star tree of remaining + already joined?)
            # This shouldn't happen, but safety first.
            # Just return a star tree of all original labels to be safe? 
            # Or the current state.
            # Let's just break and construct a tree from current state.
            pass

        # Compute r_i and v_i for all active nodes
        r = {}
        v = {}
        for i in nodes:
            r_i = 0.0
            for j in nodes:
                if i != j:
                    r_i += dist[(i, j)]
            r[i] = r_i
            
            # Compute variance v_i
            mean_d = r_i / (n - 1)
            sum_sq = 0.0
            for j in nodes:
                if i != j:
                    diff = dist[(i, j)] - mean_d
                    sum_sq += diff * diff
            v[i] = sum_sq / (n - 2)
            
        # Find best pair (i, j) minimizing BIONJ criterion
        best_criterion = float('inf')
        best_i = -1
        best_j = -1
        
        for idx_i in range(len(nodes)):
            i = nodes[idx_i]
            for idx_j in range(idx_i + 1, len(nodes)):
                j = nodes[idx_j]
                
                d_ij = dist[(i, j)]
                
                # Compute v_ij
                # v_ij = 1/(n-3) * sum_{k != i, j} (d_ik - d_jk - (r_i - r_j)/(n-3))^2
                mean_diff = (r[i] - r[j]) / (n - 3)
                sum_sq_diff = 0.0
                for k in nodes:
                    if k != i and k != j:
                        diff_k = dist[(i, k)] - dist[(j, k)] - mean_diff
                        sum_sq_diff += diff_k * diff_k
                v_ij = sum_sq_diff / (n - 3)
                
                criterion = (n - 2) * d_ij - r[i] - r[j] - ((n - 2) / (n - 3)) * (v[i] + v[j] - 2 * v_ij)
                
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j
                    
        i = best_i
        j = best_j
        
        # Join i and j
        new_id = next_id
        next_id += 1
        
        # Update tree string
        # We need to know which one is "left" and "right" to form a valid Newick string.
        # The order doesn't matter for unrooted topology.
        # We can just do (str_i, str_j)
        tree_str[new_id] = f"({tree_str[i]},{tree_str[j]})"
        
        # Update distances
        d_ij = dist[(i, j)]
        for k in nodes:
            if k != i and k != j:
                # Standard NJ distance update
                d_new_k = 0.5 * (dist[(i, k)] + dist[(j, k)] - d_ij)
                dist[(new_id, k)] = d_new_k
                dist[(k, new_id)] = d_new_k
                
        # Remove i and j, add new_id
        nodes.remove(i)
        nodes.remove(j)
        nodes.append(new_id)
        n = len(nodes)
        
    # Connect the last 3 nodes
    # The last 3 nodes form a star.
    # We need to combine them into a single Newick string.
    # Since it's unrooted, (A,B,C) is fine.
    # However, the tree_str for the last 3 nodes are just leaves or subtrees.
    # We need to merge them.
    # A star tree of 3 nodes is just a triple.
    # Newick for a star of 3: (A, B, C)
    # But wait, the previous joins created binary trees.
    # The last 3 nodes are the "roots" of the 3 subtrees.
    # We can just join them as (S1, S2, S3).
    
    # One nuance: if we just return "(" + ",".join(tree_str[x] for x in nodes) + ");", 
    # it creates a polytomy (star) of 3. This is topologically correct for the last step.
    
    return "(" + ",".join(tree_str[x] for x in nodes) + ");"
