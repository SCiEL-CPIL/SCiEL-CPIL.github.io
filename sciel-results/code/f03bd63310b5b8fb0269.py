<<<<<<< SEARCH
    def fast_local_search(labels, max_rounds=200):
        """
        Fast greedy 1-flip local search.
        Key insight: gain[v] = 2 * (num neighbors with same label as v) - degree[v]
        We maintain same_count incrementally for speed.
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count for each vertex
        # same_count[v] = number of neighbors of v with same label as v
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
            
            if best_gain <= 0:
                break
            
            # Flip vertex best_v
            old_label = labels[best_v]
            labels[best_v] = 1 - old_label
            current_cut += best_gain
            
            # Update gains for neighbors of best_v
            # When we flip v, the same_count for each neighbor u changes:
            # If u and v had same label, after flip they differ -> same_count[u] decreases by 1
            # If u and v had different labels, after flip they same -> same_count[u] increases by 1
            # So: same_count[u] += (1 if labels[u] == new_label else -1) for each neighbor u
            # new_label = 1 - old_label
            # For neighbor u: if labels[u] == new_label, same_count[u] += 1; else same_count[u] -= 1
            # This is equivalent to: same_count[u] += (2 * labels[u] - 1) * (2 * new_label - 1) ... 
            # Simpler: for each neighbor u of v, same_count[u] changes by +1 if labels[u] == new_label else -1
            # = (2*labels[u] - 1) * (2*new_label - 1) ... no
            # Let's just compute: delta_same[u] = 1 if labels[u] == new_label else -1
            # = 2*labels[u] - 1 if new_label == 1, or -(2*labels[u] - 1) if new_label == 0
            # = (2*labels[u] - 1) * (2*new_label - 1)
            # Actually: if new_label == 1: delta = 1 if labels[u]==1 else -1 = 2*labels[u] - 1
            # if new_label == 0: delta = 1 if labels[u]==0 else -1 = -(2*labels[u] - 1)
            # So delta = (2*labels[u] - 1) * (2*new_label - 1)
            
            new_label = 1 - old_label
            # Get neighbors of best_v
            neighbors = np.where(adj[best_v] == 1)[0]
            if len(neighbors) > 0:
                # For each neighbor, compute delta to same_count
                # delta_same[u] = 1 if labels[u] == new_label else -1
                # = 2*labels[u] - 1 if new_label == 1, else -(2*labels[u] - 1)
                # = (2*labels[u] - 1) * (2*new_label - 1)
                sign = 2 * new_label - 1  # 1 if new_label=1, -1 if new_label=0
                delta_same = (2 * labels[neighbors] - 1) * sign
                same_count[neighbors] += delta_same
                gains[neighbors] += 2 * delta_same
            
            # Also update gain for best_v itself (it's now flipped)
            # same_count[best_v] should be recomputed or updated
            # After flip, same_count[best_v] = degree[best_v] - old_same_count[best_v]
            # Because neighbors that were same are now different, and vice versa
            same_count[best_v] = degree[best_v] - same_count[best_v]
            gains[best_v] = 2 * same_count[best_v] - degree[best_v]
        
        return labels, current_cut
=======
    def fast_local_search(labels, max_rounds=200):
        """
        Batch-update 1-flip local search.
        Each round, identify all vertices with positive gain and flip them simultaneously.
        This converges faster than sequential greedy, allowing more total improvements.
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count for each vertex
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Find all vertices with positive gain
            flip_mask = gains > 0
            if not np.any(flip_mask):
                break
            
            flip_indices = np.where(flip_mask)[0]
            
            # Compute total gain before applying
            total_gain = gains[flip_indices].sum()
            current_cut += total_gain
            
            # Apply all flips simultaneously
            labels[flip_indices] = 1 - labels[flip_indices]
            
            # Update same_count and gains for all affected vertices
            # For each flipped vertex v, its neighbors' same_count changes
            # We need to handle interactions between flipped vertices carefully
            
            # Strategy: recompute same_count for all vertices that are neighbors of any flipped vertex
            # plus the flipped vertices themselves
            
            # Get all vertices that need updating: flipped vertices + their neighbors
            affected = set(flip_indices.tolist())
            for v in flip_indices:
                neighbors = np.where(adj[v] == 1)[0]
                affected.update(neighbors.tolist())
            
            affected_arr = np.array(list(affected), dtype=np.int32)
            
            # Recompute same_count for affected vertices
            # same_count[u] = sum of adj[u,v] where labels[u] == labels[v]
            # We can compute this efficiently for the subset
            for u in affected_arr:
                same_count[u] = (adj[u] * (labels == labels[u])).sum()
                gains[u] = 2 * same_count[u] - degree[u]
        
        return labels, current_cut
>>>>>>> REPLACE

