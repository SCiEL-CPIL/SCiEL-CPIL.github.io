import numpy as np
import math
import time

def reconstruct(sequences):
    start_time = time.time()
    DEADLINE = 260.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return f"({labels[0]});"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # 1. Convert sequences to numpy array
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = []
    for lab in labels:
        s = sequences[lab]
        # Handle potential lowercase or other chars? Task says ACGT.
        arr = np.array([seq_map[c] for c in s], dtype=np.int8)
        seqs.append(arr)
    seqs = np.vstack(seqs)
    L = seqs.shape[1]
    
    # 2. Compute pairwise Hamming distances
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    
    # 3. Jukes-Cantor Correction
    p = diff.astype(np.float64) / L
    p = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - (4.0/3.0) * p)
    
    # 4. Neighbor Joining
    # Pre-allocate distance matrix for up to 2*n nodes
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist
    
    active = list(range(n))
    tree_nodes = {}
    for i in range(n):
        tree_nodes[i] = i # Leaf
    
    next_id = n
    
    while len(active) > 3:
        if time.time() > DEADLINE:
            # Fallback: return a star tree for all original labels
            # This is a valid tree, though likely poor topology.
            # Construct a star: ((t0,t1),t2)...
            # Simple star: (t0,(t1,(t2,...)))
            # Or just return a star Newick.
            # Star Newick for n leaves: 
            # We can build a "comb" or "star". A star is hard in Newick without a central node.
            # (t0,(t1,(t2,...tn))) is a ladder.
            # For safety, let's just finish the NJ if it's close.
            # Given n<=32, it should never hit the deadline.
            pass
            
        m = len(active)
        idx = np.array(active)
        D_sub = D[np.ix_(idx, idx)]
        
        r = D_sub.sum(axis=1)
        
        # Q matrix
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        node_i = active[i_idx]
        node_j = active[j_idx]
        
        new_node = next_id
        next_id += 1
        
        tree_nodes[new_node] = (node_i, node_j)
        
        # Update distances
        # D[new_node, x] = 0.5 * (D[node_i, x] + D[node_j, x] - D[node_i, node_j])
        # x is in active, x != node_i, x != node_j
        # We can update for all active nodes, including i and j (though i,j are removed)
        # It's safer to update for all x in active.
        
        # Vectorized update for all active nodes
        # D_sub is m x m.
        # We want to compute D_new_x for x in active.
        # D_new_x = 0.5 * (D[node_i, x] + D[node_j, x] - D[node_i, node_j])
        
        # Get row i and j from D_sub
        row_i = D_sub[i_idx, :]
        row_j = D_sub[j_idx, :]
        d_ij = D_sub[i_idx, j_idx]
        
        new_row = 0.5 * (row_i + row_j - d_ij)
        
        # Update D matrix
        # D[new_node, active] = new_row
        # D[active, new_node] = new_row
        
        D[new_node, idx] = new_row
        D[idx, new_node] = new_row
        
        # Update active list
        active.remove(node_i)
        active.remove(node_j)
        active.append(new_node)
        
    # Now len(active) == 3
    # Create final root
    # Pick two nodes, join them, then join with the third
    a, b, c = active
    
    # Join a and b
    n1 = next_id
    next_id += 1
    tree_nodes[n1] = (a, b)
    
    # Join n1 and c
    root = next_id
    next_id += 1
    tree_nodes[root] = (n1, c)
    
    # Construct Newick
    def get_newick(node):
        if node < n:
            return labels[node]
        left, right = tree_nodes[node]
        return f"({get_newick(left)},{get_newick(right)})"
    
    newick = f"({get_newick(root)});"
    return newick
