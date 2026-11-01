    for restart in range(200):
        if time.time() - start_time > timeout:
            break
            
        random.seed(restart)
        
        # Initialize uncovered
        uncovered = np.ones(n_t_subsets, dtype=bool)
        block_indices_found = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Calculate gain for all blocks
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            # Find the block with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Select from Top-K blocks
            TOP_K = 10
            if n_k_subsets > TOP_K:
                top_k_indices = np.argpartition(gains, -TOP_K)[-TOP_K:]
            else:
                top_k_indices = np.arange(n_k_subsets)
                
            # Pick one randomly from these top K
            chosen_idx = top_k_indices[random.randint(0, len(top_k_indices)-1)]
            
            block_indices_found.append(chosen_idx)
            
            # Update uncovered
            uncovered[cover_matrix[chosen_idx]] = False
            
        # Post-processing: Remove redundant blocks
        # We have a list of block indices.
        # We want to remove any block whose coverage is fully contained in the coverage of the remaining blocks.
        # Let's use a coverage count array.
        
        if block_indices_found:
            # Initialize counts
            trip_counts = np.zeros(n_t_subsets, dtype=np.int32)
            
            # Add all blocks
            for idx in block_indices_found:
                trip_counts[cover_matrix[idx]] += 1
                
            # Try to remove blocks in reverse order (last added are often redundant)
            # We need to be careful: removing a block might make another block redundant, 
            # but it won't make a previously non-redundant block redundant.
            # So a single pass in reverse is sufficient for "greedy removal".
            
            new_block_indices = []
            for idx in reversed(block_indices_found):
                # Check if this block is redundant
                # It is redundant if for all its triples, the count is > 1
                # (meaning at least one other block covers it)
                block_triples = cover_matrix[idx]
                if np.all(trip_counts[block_triples] > 1):
                    # Redundant, remove it
                    trip_counts[block_triples] -= 1
                else:
                    # Not redundant, keep it
                    new_block_indices.append(idx)
                    
            block_indices_found = new_block_indices
            
            current_count = len(block_indices_found)
            if current_count < best_count:
                best_count = current_count
                # Convert indices back to blocks for the return value
                best_blocks = [all_k_subsets[idx] for idx in block_indices_found]
