    import numpy as np

    def _jc_distance_vectorized(sequences, labels):
        """
        Compute pairwise JC distance matrix using NumPy.
        """
        n = len(labels)
        # Convert sequences to integer arrays
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # This loop is O(N*L)
        int_seqs = np.array([[base_map[c] for c in sequences[label]] for label in labels])
        
        # Compute pairwise differences
        # int_seqs is (N, L)
        # Broadcasting: (N, 1, L) vs (1, N, L)
        # This creates (N, N, L) boolean array
        diff = int_seqs[:, np.newaxis, :] != int_seqs[np.newaxis, :, :]
        diff_count = np.sum(diff, axis=2)
        
        L = int_seqs.shape[1]
        p = diff_count / L
        
        dist = np.zeros((n, n))
        mask = p < 0.75
        arg = 1.0 - (4.0/3.0) * p
        arg = np.clip(arg, 1e-10, 1.0)
        dist[mask] = -0.75 * np.log(arg[mask])
        dist[~mask] = 3.0
        
        np.fill_diagonal(dist, 0.0)
        return dist

    def _upgma(dist, labels):
        """
        Perform UPGMA clustering.
        Returns Newick string.
        """
        n = len(labels)
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return f"({labels[0]},{labels[1]});"
            
        # dist: (n, n)
        # sizes: (n,)
        # active: list of indices
        # We will modify dist and sizes in place or create new arrays.
        # To avoid complex index mapping, we'll maintain a list of active nodes.
        # And a mapping from original index to current representation.
        
        # Better approach:
        # clusters: list of objects. Each object is either a label or a tuple (left, right, height).
        # But for distance updates, we need an array.
        
        # Let's use the standard array-based UPGMA.
        # current_dist: (m, m) where m is current number of clusters.
        # current_sizes: (m,)
        # cluster_tree: list of (left, right) for each new cluster.
        
        m = n
        current_dist = dist.copy()
        current_sizes = np.ones(m)
        # cluster_tree[i] will be (left_idx, right_idx) where indices are in the ORIGINAL space?
        # No, we need to track the structure.
        
        # Let's store the tree structure as a list of tuples (left, right).
        # Initially, clusters 0..n-1 are leaves.
        # When we merge i and j into k, we store (i, j).
        # The new cluster k replaces i and j in the active set.
        
        # To keep it simple, we'll maintain a list `nodes` where nodes[i] is the "id" of the cluster.
        # Actually, let's just record the merges.
        
        merges = [] # list of (left, right, height)
        
        # active_indices: list of indices in current_dist that are active
        # We will compact the matrix each step? That's O(N^3).
        # For N=32, N^3 = 32768. Very fast.
        
        active = list(range(n))
        
        while len(active) > 1:
            # Find min distance pair in active
            min_d = np.inf
            min_i, min_j = -1, -1
            for idx_i in range(len(active)):
                i = active[idx_i]
                for idx_j in range(idx_i + 1, len(active)):
                    j = active[idx_j]
                    if current_dist[i, j] < min_d:
                        min_d = current_dist[i, j]
                        min_i, min_j = i, j
            
            # Merge i and j
            # New node id: we can use a new index or reuse.
            # Let's reuse one of the indices, say i.
            # But we need to update distances from i to all other active nodes k.
            # d(i, k) = (size(i)*d(i,k) + size(j)*d(j,k)) / (size(i)+size(j))
            
            new_size = current_sizes[i] + current_sizes[j]
            new_height = min_d / 2.0 # UPGMA height
            
            # Update distances from i to all other active nodes
            for idx_k in range(len(active)):
                k = active[idx_k]
                if k == i or k == j:
                    continue
                current_dist[i, k] = (current_sizes[i] * current_dist[i, k] + current_sizes[j] * current_dist[j, k]) / new_size
                current_dist[k, i] = current_dist[i, k]
            
            # Update size
            current_sizes[i] = new_size
            
            # Record merge
            # We need to know what i and j represented.
            # i and j are indices in the original matrix?
            # Yes, we are not compacting the matrix, just marking active/inactive.
            # So i and j are original indices?
            # No, if we merge 0 and 1 into 0, then 0 represents {0,1}.
            # If we later merge 0 and 2, we record (0, 2).
            # But 0 is now a cluster.
            # So we need a mapping from current index to cluster structure.
            
            # Let's use a different strategy:
            # Store the tree as a list of (left, right) where left and right are "node ids".
            # Node ids 0..n-1 are leaves.
            # New nodes get ids n, n+1, ...
            
            # But the distance matrix is indexed by node ids.
            # If we add new nodes, we need to expand the matrix.
            
            # This is getting complicated.
            # Let's stick to the compacting matrix approach.
            
            # Remove j from active
            active.remove(j)
            
            # Record the merge in a way that allows reconstruction.
            # We need to know that node i is now a cluster containing the previous content of i and j.
            # Let's maintain a `node_repr` dictionary.
            # node_repr[i] = labels[i] initially.
            # node_repr[new_node] = (node_repr[i], node_repr[j])
            
            # But we are reusing index i.
            # So we need to update node_repr[i].
            
            # Let's re-initialize with this structure.
            
            break # This logic is flawed in the loop above. Let's restart the UPGMA function.
        
        # Restart _upgma
        pass

    def _upgma(dist, labels):
        n = len(labels)
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return f"({labels[0]},{labels[1]});"
            
        # dist: (n, n)
        # sizes: (n,)
        # active: list of indices
        # node_repr: dict index -> label or (left, right)
        
        current_dist = dist.copy()
        current_sizes = np.ones(n)
        active = list(range(n))
        node_repr = {i: labels[i] for i in range(n)}
        
        while len(active) > 1:
            # Find min distance pair
            min_d = np.inf
            min_i, min_j = -1, -1
            for idx_i in range(len(active)):
                i = active[idx_i]
                for idx_j in range(idx_i + 1, len(active)):
                    j = active[idx_j]
                    if current_dist[i, j] < min_d:
                        min_d = current_dist[i, j]
                        min_i, min_j = i, j
            
            # Merge i and j into i
            new_size = current_sizes[i] + current_sizes[j]
            
            # Update distances from i to others
            for k in active:
                if k == i or k == j:
                    continue
                current_dist[i, k] = (current_sizes[i] * current_dist[i, k] + current_sizes[j] * current_dist[j, k]) / new_size
                current_dist[k, i] = current_dist[i, k]
            
            # Update size
            current_sizes[i] = new_size
            
            # Update node_repr
            node_repr[i] = (node_repr[i], node_repr[j])
            
            # Remove j from active
            active.remove(j)
            
        # Build Newick
        def to_newick(node):
            if isinstance(node, str):
                return node
            else:
                left, right = node
                return f"({to_newick(left)},{to_newick(right)})"
        
        root = active[0]
        newick = to_newick(node_repr[root])
        return newick + ";"

    def reconstruct(sequences):
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        n = len(labels)
        
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return f"({labels[0]},{labels[1]});"
            
        dist = _jc_distance_vectorized(sequences, labels)
        newick = _upgma(dist, labels)
        
        return newick
    