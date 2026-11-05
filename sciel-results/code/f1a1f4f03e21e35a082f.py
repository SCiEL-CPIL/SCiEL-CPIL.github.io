<<<<<<< SEARCH
    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_triplet_ids(b1)
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = block_triplet_ids(b2)
                
                critical_points = set()
                
                # Check b1's triplets
                for tid in ids_b1:
                    c = count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                        
                # Check b2's triplets (only those not in b1)
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    if c == 1:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both blocks are redundant together
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    # We can form a block containing all critical points
                    new_block = list(critical_points)
                    # Fill with random points
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Verify validity
                    # The condition for validity is that ALL previously covered triplets remain covered.
                    # We checked that the "critical" ones (those with count dropping to 0) are in new_block.
                    # Triplets with count > critical threshold remain covered by other blocks.
                    # So this swap is valid.
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False
=======
    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_triplet_ids(b1)
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = block_triplet_ids(b2)
                
                critical_points = set()
                
                # Check b1's triplets
                for tid in ids_b1:
                    c = count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                        
                # Check b2's triplets (only those not in b1)
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    if c == 1:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both blocks are redundant together
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    # We can form a block containing all critical points
                    new_block = list(critical_points)
                    # Fill with random points
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        """Try to replace one block with another that covers all its critical triplets."""
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_triplet_ids(b1)
            
            # Find critical triplets (those only covered by b1)
            critical_ids = set()
            for tid in ids_b1:
                if count[tid] == 1:
                    critical_ids.add(tid)
            
            if not critical_ids:
                # Block is redundant, remove it
                update_counts([b1], -1)
                return [b for idx, b in enumerate(blocks) if idx != i], True
            
            # Try to find another block that covers all critical_ids
            for j in range(n):
                if j == i:
                    continue
                b2 = blocks[j]
                ids_b2 = block_triplet_ids(b2)
                
                if critical_ids.issubset(ids_b2):
                    # b2 covers all critical triplets of b1
                    # Check: after removing b1, are all triplets still covered?
                    # Critical ones are covered by b2. Non-critical ones had count >= 2, 
                    # so after removing b1 they still have count >= 1.
                    # But we need to check: does b2 cover any triplet that was ONLY covered by b1?
                    # That's exactly the critical_ids check.
                    # Also need to verify that b2's own triplets remain covered.
                    # b2 stays, so its triplets are fine.
                    update_counts([b1], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i], True
                    
        return blocks, False
>>>>>>> REPLACE

