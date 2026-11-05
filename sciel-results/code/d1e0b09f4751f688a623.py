<<<<<<< SEARCH
    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p
=======
    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try to find a lower-rank decomposition using numpy-based optimization
    # Strategy: use the block decomposition as a starting point and try to
    # reduce rank by finding linear dependencies among the rank-1 terms
    def try_reduce_rank_via_dependencies(current_decomp, current_rank):
        """Try to reduce rank by finding linear dependencies among columns."""
        nonlocal best_result, best_rank
        
        U_np = np.array(current_decomp["U"], dtype=float)
        V_np = np.array(current_decomp["V"], dtype=float)
        W_np = np.array(current_decomp["W"], dtype=float)
        
        # For each column, check if it can be expressed as a linear combination
        # of other columns while maintaining the tensor decomposition
        
        # This is a complex optimization problem. Instead, let's try a simpler approach:
        # Try to merge pairs of columns that have similar patterns
        
        # Actually, let's try a different approach: use randomized rounding
        # to find a sparse representation with fewer non-zero columns
        
        # For now, this is a placeholder for more sophisticated methods
        return current_decomp, current_rank
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Now try to improve by merging Strassen terms across blocks
    # The idea: if two blocks share structure, we might be able to share some rank-1 terms
    # This is a heuristic approach - try to find a lower rank decomposition by
    # allowing some entries to be shared across blocks
    
    # Try a randomized search for better decompositions
    def try_random_improvement(max_iters=5000):
        """Try to find a lower rank decomposition via random search."""
        nonlocal best_result, best_rank
        
        # Start from the best block decomposition
        # and try to reduce rank by merging similar terms
        
        # For each pair of blocks, check if we can merge some terms
        # This is complex, so we'll use a simpler heuristic:
        # Try to find a decomposition with rank = best_rank - 1, best_rank - 2, etc.
        
        for target_rank in range(best_rank - 1, max(30, best_rank - 10), -1):
            if time.time() > deadline:
                return
            
            # Try to construct a decomposition of the target rank
            # by perturbing the current best decomposition
            
            # Simple approach: try to zero out some entries and adjust others
            # This is a very rough heuristic
            
            # Actually, let's try a different approach:
            # Use the fact that the tensor is structured and try to find
            # a lower rank decomposition by exploiting the structure
            
            # For now, just return the best we have
            pass
        
        return

    # Call the random improvement (may not find anything better)
    try_random_improvement()
=======
    # Try to improve by using a more sophisticated search
    # Key insight: the current block approach gives 84, but we can do better
    # by exploiting the non-square structure more cleverly
    
    def try_optimized_block_search():
        """Try more sophisticated block decompositions."""
        nonlocal best_result, best_rank
        
        # For n=3, m=5, p=6, let's try to find the optimal partition
        # that minimizes the total rank
        
        # The rank of a block (si, sj, sk) is:
        # - 7 if si=sj=sk=2 (Strassen)
        # - si*sj*sk otherwise (schoolbook)
        
        # We want to partition i into groups, j into groups, k into groups
        # such that the sum of block ranks is minimized
        
        # Let's be more systematic: try all possible partitions
        # A partition of {0,...,n-1} is a set of subsets that partition the set
        
        def get_all_partitions(elements):
            """Generate all possible partitions of a set of elements."""
            if not elements:
                yield []
                return
            
            first = elements[0]
            rest = elements[1:]
            
            for partition in get_all_partitions(rest):
                # Add first to each existing group
                for i, group in enumerate(partition):
                    new_partition = partition[:i] + [group + [first]] + partition[i+1:]
                    yield new_partition
                
                # Start a new group with first
                yield partition + [[first]]
        
        # For n=3, m=5, p=6, the number of partitions is manageable
        # n=3: 5 partitions
        # m=5: 52 partitions  
        # p=6: 203 partitions
        
        # But we can be smarter: only consider partitions where groups have size 1 or 2
        # (since Strassen only helps for 2x2x2 blocks)
        
        def get_binary_partitions(elements):
            """Generate partitions where each group has size 1 or 2."""
            if not elements:
                yield []
                return
            
            first = elements[0]
            rest = elements[1:]
            
            # Option 1: first is alone
            for partition in get_binary_partitions(rest):
                yield [[first]] + partition
            
            # Option 2: first is paired with another element
            for i in range(len(rest)):
                paired = [first, rest[i]]
                remaining = rest[:i] + rest[i+1:]
                for partition in get_binary_partitions(remaining):
                    yield [paired] + partition
        
        # Try all binary partitions
        i_partitions = list(get_binary_partitions(range(n)))
        j_partitions = list(get_binary_partitions(range(m)))
        k_partitions = list(get_binary_partitions(range(p)))
        
        print(f"Trying {len(i_partitions)} x {len(j_partitions)} x {len(k_partitions)} = {len(i_partitions)*len(j_partitions)*len(k_partitions)} combinations")
        
        for i_part in i_partitions:
            for j_part in j_partitions:
                for k_part in k_partitions:
                    if time.time() > deadline:
                        return
                    
                    # Calculate rank for this partition
                    total_rank = 0
                    for ig in i_part:
                        for jg in j_part:
                            for kg in k_part:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    total_rank += 7
                                else:
                                    total_rank += si * sj * sk
                    
                    if total_rank >= best_rank:
                        continue
                    
                    # Build the decomposition
                    result = build_decomposition(i_part, j_part, k_part)
                    
                    # Verify
                    if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                        best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                        best_rank = total_rank
                        print(f"Found better decomposition with rank {best_rank}")
        
        return
    
    # Call the optimized search
    try_optimized_block_search()
>>>>>>> REPLACE

