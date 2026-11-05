import numpy as np

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Convert sequences to a binary matrix for fast distance calculation
    # 0/1 encoding for ACGT doesn't matter for Hamming distance, just need mismatch count
    # We can just compare strings directly but vectorized is better.
    # Let's create a matrix of characters, or just compute distances directly.
    
    # Efficiently compute pairwise Hamming distances
    # sequences is a dict {label: string}
    seq_list = [sequences[label] for label in labels]
    n_sites = len(seq_list[0])
    
    # Convert to numpy array of chars for broadcasting
    # shape: (n_taxa, n_sites)
    seq_array = np.array(seq_list)
    
    # Compute pairwise Hamming distances
    # diff[i, j] is 1 if seq[i] != seq[j] at any position, sum over positions
    # We can compute mismatch matrix:
    # seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :]
    # This creates a boolean array of shape (n_taxa, n_taxa, n_sites)
    # Summing over axis=2 gives Hamming distances
    
    # For n_taxa=32, n_sites=2000, this is 32*32*2000 = 2,048,000 operations, very fast.
    mismatch_matrix = np.sum(seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :], axis=2).astype(np.float64)
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    p = mismatch_matrix / n_sites
    
    # Handle p >= 0.75 (log(0) or negative)
    # If p >= 0.75, the distance is effectively infinite. 
    # In practice, for simulated data, this is rare unless branch lengths are huge.
    # We cap the distance to a large value to avoid NaNs.
    p_clipped = np.clip(p, 0, 0.7499)
    
    D = -0.75 * np.log(1 - 4 * p_clipped / 3)
    
    # Set diagonal to 0
    np.fill_diagonal(D, 0)
    
    # Neighbor-Joining Algorithm
    # We maintain a list of active nodes.
    # Initially, active nodes are the taxa 0..n_taxa-1
    # We will build the tree by joining pairs.
    
    # To build the Newick string, we need to keep track of the subtrees.
    # Let's use a list `trees` where trees[i] is the Newick string for node i.
    # Initially, trees[i] = labels[i]
    
    trees = [labels[i] for i in range(n_taxa)]
    active = list(range(n_taxa))
    current_D = D.copy()
    n = n_taxa
    
    while len(active) > 3:
        n = len(active)
        
        # Compute row sums r_i = sum_j D[i,j] for i in active
        # Note: D is the full matrix, but we only care about active nodes.
        # However, it's easier to work with the submatrix or just use the full matrix indices.
        # Let's extract the submatrix for active nodes to keep indices simple.
        # Actually, since we update D in place, we can just use the indices in `active`.
        
        # r_i = sum of D[i, j] for all j in active
        # We can compute this using numpy sum over the active columns.
        # But D is updated. Let's just compute sums over the active set.
        
        # Efficient way:
        # r = D[:, active].sum(axis=1)  # This gives sums for all rows, we only need active rows.
        # But D is n_taxa x n_taxa.
        
        # Let's create a submatrix view for active nodes? No, D is modified.
        # We just need r for active nodes.
        r = np.zeros(n_taxa)
        for i in active:
            r[i] = np.sum(D[i, active])
        
        # Compute Q matrix for active nodes: Q[i,j] = (n-2)D[i,j] - r[i] - r[j]
        # We only need to find the min Q[i,j] for i != j in active.
        
        # Construct Q for active nodes
        # Q_active shape (n, n)
        # D_active = D[np.ix_(active, active)]
        D_active = D[np.ix_(active, active)]
        r_active = r[active]
        
        Q = (n - 2) * D_active - np.outer(r_active, np.ones(n)) - np.outer(np.ones(n), r_active)
        
        # Set diagonal to inf so we don't pick i=j
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair with minimum Q
        min_val = np.min(Q)
        # Get indices in the active submatrix
        idx_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(idx_flat, Q.shape)
        
        # Map back to original labels/indices
        i = active[i_idx]
        j = active[j_idx]
        
        # Join i and j
        # New node name
        new_node_id = n_taxa + len(active) - n # Just a unique identifier, doesn't matter for topology if we track trees
        # Actually, we just need to append to trees and update active list.
        
        # Calculate distances from new node to other active nodes
        # d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
        # This is for k in active, k != i, k != j
        
        # We need to update the D matrix.
        # We'll add a new row/column to D? Or just keep a dict?
        # Since n is small (<=32), we can just manage a list of nodes and a dict or 2D array that grows.
        # But D is fixed size n_taxa x n_taxa. 
        # It's easier to maintain a dynamic distance matrix.
        
        # Let's restart the data structure to be more dynamic.
        # We will maintain:
        # 1. `dist`: a 2D numpy array of size (num_nodes, num_nodes).
        # 2. `nodes`: a list of node IDs.
        # 3. `trees`: a list of Newick strings corresponding to nodes.
        
        # This approach is cleaner.
        # Let's refactor the loop to use a dynamic matrix.
        
        pass # The above logic is getting messy. Let's implement the dynamic version directly.

    # Refactored Dynamic NJ Implementation
    
    # Re-initialize
    trees = [labels[i] for i in range(n_taxa)]
    # dist is n_taxa x n_taxa
    dist = D.copy()
    nodes = list(range(n_taxa))
    
    while len(nodes) > 3:
        n = len(nodes)
        
        # Compute row sums for current nodes
        # dist is (n_taxa, n_taxa), but we only care about rows/cols in `nodes`
        # Let's extract the submatrix
        D_sub = dist[np.ix_(nodes, nodes)]
        
        # r_i = sum of row i in D_sub
        r = np.sum(D_sub, axis=1)
        
        # Q matrix
        Q = (n - 2) * D_sub - np.outer(r, np.ones(n)) - np.outer(np.ones(n), r)
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        min_val = np.min(Q)
        idx_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(idx_flat, Q.shape)
        
        i_node = nodes[i_idx]
        j_node = nodes[j_idx]
        
        # Create new node
        new_node_id = len(nodes) # Use length as ID? No, IDs should be unique.
        # Let's just append to the lists.
        # We need to update the dist matrix to include the new node.
        # Current dist size is (n_taxa, n_taxa).
        # We need to expand it to (n_taxa+1, n_taxa+1).
        
        # This resizing is inefficient if done every step.
        # Alternative: Pre-allocate a large enough matrix? Max nodes is n_taxa-2.
        # Or just use a dictionary for distances? 
        # Given n<=32, even a dictionary or list of lists is fine.
        # But NumPy is faster.
        
        # Let's pre-allocate a matrix of size (2*n_taxa, 2*n_taxa) to be safe, or just resize.
        # Resizing 32x32 to 33x33 is trivial.
        
        new_dist = np.zeros((dist.shape[0] + 1, dist.shape[0] + 1))
        new_dist[:dist.shape[0], :dist.shape[0]] = dist
        
        # Calculate distances from new node to all existing nodes
        # d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
        # k ranges over all current nodes (indices 0 to dist.shape[0]-1)
        
        d_ij = D_sub[i_idx, j_idx]
        
        # Vectorized calculation
        # d_i_k = D_sub[i_idx, :]  (distances from i to all nodes)
        # d_j_k = D_sub[j_idx, :]  (distances from j to all nodes)
        
        d_new_k = 0.5 * (D_sub[i_idx, :] + D_sub[j_idx, :] - d_ij)
        
        # Update new_dist
        new_dist[dist.shape[0], :dist.shape[0]] = d_new_k
        new_dist[:dist.shape[0], dist.shape[0]] = d_new_k
        new_dist[dist.shape[0], dist.shape[0]] = 0.0
        
        dist = new_dist
        
        # Update trees
        new_tree_str = f"({trees[i_node]},{trees[j_node]})"
        trees.append(new_tree_str)
        
        # Update nodes list
        # Remove i_node and j_node, add new_node_id
        nodes = [x for x in nodes if x not in (i_node, j_node)]
        nodes.append(dist.shape[0] - 1)
        
    # Construct final Newick
    # The last remaining 3 nodes form the final unrooted tree.
    # An unrooted tree with 3 nodes is just a star: (A,B,C)
    # But wait, NJ stops when 3 nodes are left.
    # The structure is built up.
    # The `trees` list contains the Newick strings for all nodes.
    # The final tree is composed of the 3 remaining nodes in `nodes`.
    # How to combine 3 nodes into an unrooted tree?
    # In NJ, when 3 nodes remain, the topology is fixed.
    # The three subtrees are connected to a central node (if rooted) or just form a triangle?
    # For an unrooted tree, the three subtrees are connected by three edges meeting at a central point?
    # No, an unrooted tree with 3 leaves is just (A,B,C).
    # If the 3 nodes are subtrees T1, T2, T3, the final Newick is (T1,T2,T3).
    
    final_nodes = nodes
    # Sort them for consistency? Not strictly required for RF distance, but good practice.
    # The order in the Newick string doesn't affect the topology.
    
    final_newick = "(" + ",".join(trees[node] for node in final_nodes) + ");"
    
    return final_newick
