import numpy as np
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
'''
Vectorized Neighbor Joining (NJ) using Jukes-Cantor distances.
Uses NumPy for efficient pairwise distance calculation and NJ criterion updates.
'''

def reconstruct(sequences):
    start_time = time.time()
    DEADLINE = 270.0  # Safety margin from 270s wall-clock kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # 1. Convert sequences to numpy arrays for fast distance calculation
    # Map A,C,G,T to 0,1,2,3
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Create a matrix of shape (n, length) with integer codes
    # 0:A, 1:C, 2:G, 3:T
    seq_matrix = np.zeros((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        # Vectorized mapping is faster than loop for each char, but for 2000 length, 
        # a lookup table or direct comparison is good.
        # Let's use a simple mapping approach for the whole string
        # Actually, let's just compute distances directly using broadcasting on one-hot or similar?
        # No, let's stick to the integer matrix.
        for j, char in enumerate(seq):
            if char == 'A':
                seq_matrix[i, j] = 0
            elif char == 'C':
                seq_matrix[i, j] = 1
            elif char == 'G':
                seq_matrix[i, j] = 2
            else: # T
                seq_matrix[i, j] = 3

    # 2. Compute Pairwise Distances (Jukes-Cantor)
    # D[i, j] = -0.75 * ln(1 - 4/3 * p_ij)
    # p_ij = fraction of differing sites
    
    # Compute pairwise hamming distances
    # seq_matrix is (n, L). We want (n, n) hamming distances.
    # (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=2)
    # This creates an (n, n, L) boolean array. For n=32, L=2000, this is 32*32*2000 = 2M booleans. Very fast.
    
    # To save memory and time, we can do this in chunks or just directly if n is small enough.
    # n max 32. 32x32x2000 is fine.
    
    diff_matrix = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=2)
    p_matrix = diff_matrix / length
    
    # Handle p >= 0.75 to avoid log(0) or negative
    # If p >= 0.75, the term inside log is <= 0.
    # Theoretically, if p=0.75, dist is inf. If p>0.75, it's invalid.
    # In practice, for random sequences, p ~ 0.75.
    # We cap the distance at a large value to prevent NaNs/Infs.
    p_safe = np.clip(p_matrix, 0, 0.7499)
    
    # D = -0.75 * ln(1 - 4/3 * p)
    log_arg = 1.0 - (4.0 / 3.0) * p_safe
    # Ensure log_arg > 0
    log_arg = np.maximum(log_arg, 1e-10)
    
    D = -0.75 * np.log(log_arg)
    
    # Set diagonal to 0
    np.fill_diagonal(D, 0.0)
    
    # 3. Neighbor Joining Algorithm
    # We maintain a list of active nodes. Initially, these are the taxa.
    # We also maintain a tree structure for each node.
    # Since we are joining pairs, we can represent the tree as a string or a nested structure.
    # For efficiency in the NJ loop, we will keep track of:
    # - active_nodes: list of node indices/ids
    # - node_trees: dict mapping node_id -> Newick string (or partial tree)
    # - D: the current distance matrix for active nodes
    
    # To make indexing easier, let's use a list of active node IDs.
    # Initially, node IDs are 0 to n-1.
    active_nodes = list(range(n))
    node_trees = {i: labels[i] for i in range(n)}
    
    # D is currently the full n x n matrix.
    # We will operate on sub-matrices corresponding to active_nodes.
    
    # Helper to get sub-matrix
    def get_sub_matrix(D_full, nodes):
        indices = np.array(nodes)
        return D_full[indices][:, indices]
    
    # Current D for active nodes
    # We will update D in place or create new sub-matrices.
    # Since n is small (<=32), we can just use a dictionary or 2D array of size (n_active, n_active).
    # But managing indices is tricky.
    # Alternative: Keep the full n x n matrix D_full.
    # When we join a and b, we create a new node c.
    # We need to update distances from c to all other active nodes.
    # The formula for d(c, x) is:
    # d(c, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
    
    # Let's use a list of active node IDs and a separate distance matrix D_active.
    # D_active[i, j] is the distance between active_nodes[i] and active_nodes[j].
    
    # Initialize D_active as a copy of the relevant part of D
    # Since active_nodes is [0..n-1] initially, D_active = D
    
    D_active = D.copy()
    
    # Main NJ Loop
    while len(active_nodes) > 2:
        if time.time() > DEADLINE - 10:
            # Time out, return a star tree or whatever we have
            break
            
        m = len(active_nodes)
        
        if m == 3:
            # Last step, just join the 3 nodes in a trifurcation or binary join
            # For unrooted topology, any binary join of 3 leaves is topologically equivalent to a star if we don't care about branch lengths?
            # No, a binary join of 3 leaves creates a tree with one internal node connecting all 3.
            # e.g. (A, (B, C)) is the same topology as (B, (A, C)) for 3 taxa.
            # So we can just join the last two.
            a_idx = 0
            b_idx = 1
            a_id = active_nodes[a_idx]
            b_id = active_nodes[b_idx]
            
            d_ab = D_active[a_idx, b_idx]
            
            # Construct the new node
            new_id = -1 # Placeholder, we will manage IDs carefully
            # Let's just use a counter for new node IDs
            # Actually, we can just use the index in the list.
            # But we need to store the tree structure.
            
            # Join a and b
            tree_a = node_trees[a_id]
            tree_b = node_trees[b_id]
            new_tree = f"({tree_a},{tree_b})"
            
            # Create new node ID. Let's use a global counter or just append to a list.
            # Since we remove a and b and add new, we can use a new unique ID.
            # Let's use negative indices or a counter.
            # To keep it simple, let's use a list of all node IDs created so far.
            # But we only care about the current active ones.
            # Let's use a counter.
            pass # We need a better way to manage node IDs.

    # Let's restart the NJ loop logic with proper node ID management.
    
    active_nodes = list(range(n))
    node_trees = {i: labels[i] for i in range(n)}
    D_active = D.copy()
    
    next_node_id = n
    
    while len(active_nodes) > 2:
        if time.time() > DEADLINE - 5:
            break
            
        m = len(active_nodes)
        
        # Compute Q matrix (NJ criterion)
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        # where r[i] = sum_k D[i, k]
        
        r = D_active.sum(axis=1)
        
        # Q[i, j] = (m - 2) * D_active[i, j] - r[i] - r[j]
        # We want to minimize Q (or maximize -Q). The standard NJ picks the pair with minimum Q.
        # Wait, Saitou-Nei 1987 says:
        # q(i, j) = (n-2) d(i,j) - r(i) - r(j)
        # Pick the pair (i, j) that minimizes q(i, j).
        
        # Let's compute the Q matrix
        # Q = (m-2)*D_active - r[:, None] - r[None, :]
        Q = (m - 2) * D_active - r[:, np.newaxis] - r[np.newaxis, :]
        
        # We want the pair (i, j) with i < j that minimizes Q[i, j]
        # Set diagonal to inf to avoid picking i=i
        np.fill_diagonal(Q, np.inf)
        
        # Find the index of the minimum value
        min_val, min_idx_flat = np.unravel_index(np.argmin(Q), Q.shape)
        
        # Ensure i < j for consistency, though argmin on full matrix might give (i,j) or (j,i)
        # Since Q is symmetric, it doesn't matter which one we pick, but we need to be careful with indices.
        i, j = min_idx_flat
        if i > j:
            i, j = j, i
            
        a_id = active_nodes[i]
        b_id = active_nodes[j]
        
        # Calculate branch lengths (optional, but good for Newick if required, though task says optional)
        # d(a, b)
        d_ab = D_active[i, j]
        r_a = r[i]
        r_b = r[j]
        
        # Branch length from a to new node c
        # l(a, c) = 0.5 * d_ab + (r_a - r_b) / (2 * (m - 2))
        # l(b, c) = 0.5 * d_ab - (r_a - r_b) / (2 * (m - 2))
        
        # We don't strictly need branch lengths for topology, but let's keep them for valid Newick if we want.
        # The task says "branch lengths optional". So we can omit them to save time/complexity.
        # We will just create a topology.
        
        # Create new node
        c_id = next_node_id
        next_node_id += 1
        
        tree_a = node_trees[a_id]
        tree_b = node_trees[b_id]
        node_trees[c_id] = f"({tree_a},{tree_b})"
        
        # Update active nodes
        # Remove i and j, add c_id
        # We need to be careful with indices.
        
        # New active nodes list
        new_active_nodes = []
        for k in range(m):
            if k == i or k == j:
                continue
            new_active_nodes.append(active_nodes[k])
        
        new_active_nodes.append(c_id)
        
        # Update D_active
        # The new distance matrix will be of size (m-1) x (m-1)
        # The new node c is at the end of new_active_nodes.
        
        # Distances from c to all other existing nodes x:
        # d(c, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        
        # We need to map the old indices to new indices.
        # Let's build the new D matrix.
        
        # Indices in old D_active for the remaining nodes
        remaining_indices = [k for k in range(m) if k != i and k != j]
        
        # d(c, x) for x in remaining
        # d(a, x) is D_active[i, x]
        # d(b, x) is D_active[j, x]
        
        d_c_x = 0.5 * (D_active[i, remaining_indices] + D_active[j, remaining_indices] - d_ab)
        
        # The new D matrix consists of:
        # 1. The sub-matrix of D_active for remaining_indices x remaining_indices
        # 2. The row/col for c, which is d_c_x and 0 for c-c
        
        D_sub = D_active[np.ix_(remaining_indices, remaining_indices)]
        
        # Append c row and col
        # d_c_x is a 1D array of length m-2
        # We need to add a column of d_c_x to D_sub, and a row of d_c_x to D_sub.
        
        # Create new D matrix of size (m-1) x (m-1)
        new_D = np.zeros((m - 1, m - 1))
        
        # Fill the top-left (m-2)x(m-2) block
        new_D[:m-2, :m-2] = D_sub
        
        # Fill the last column (distances from c to others)
        new_D[:m-2, m-1] = d_c_x
        
        # Fill the last row (distances from others to c)
        new_D[m-1, :m-2] = d_c_x
        
        # Last element is 0
        new_D[m-1, m-1] = 0.0
        
        D_active = new_D
        active_nodes = new_active_nodes
        
    # End while loop
    
    # Now we have 2 or 3 active nodes.
    # If 3, we need to join them.
    # If 2, we just join them.
    
    if len(active_nodes) == 3:
        # We have 3 nodes left.
        # We can form a tree by joining two, then joining the result with the third.
        # Topologically, for 3 leaves, any binary tree is the same unrooted topology (a star).
        # But Newick requires a binary tree or we can output a trifurcation?
        # "Reconstruct the unrooted tree topology... every taxon exactly once"
        # Standard Newick is binary. A trifurcation (A,B,C) is not strictly binary.
        # However, (A,(B,C)) is a valid binary representation of the 3-taxon star.
        
        # Let's just join the first two, then join with the third.
        a_id = active_nodes[0]
        b_id = active_nodes[1]
        c_id = active_nodes[2]
        
        # Join a and b
        d_ab = D_active[0, 1]
        new_id_1 = next_node_id
        next_node_id += 1
        node_trees[new_id_1] = f"({node_trees[a_id]},{node_trees[b_id]})"
        
        # Join new_id_1 and c_id
        # Distance d(new_id_1, c_id) is not strictly needed for topology, but let's compute it for consistency if we were doing full NJ.
        # For the last step, we just need the topology.
        # d(new_id_1, c_id) = ?
        # In NJ, the last step is just connecting the remaining nodes.
        # We can just create the tree.
        
        new_id_2 = next_node_id
        next_node_id += 1
        node_trees[new_id_2] = f"({node_trees[new_id_1]},{node_trees[c_id]})"
        
        final_tree = node_trees[new_id_2]
        
    elif len(active_nodes) == 2:
        a_id = active_nodes[0]
        b_id = active_nodes[1]
        final_tree = f"({node_trees[a_id]},{node_trees[b_id]})"
        
    elif len(active_nodes) == 1:
        final_tree = node_trees[active_nodes[0]]
        
    else:
        # Should not happen
        final_tree = ",".join(node_trees[a] for a in active_nodes)
        
    return final_tree + ";"

# EVOLVE-BLOCK-END
