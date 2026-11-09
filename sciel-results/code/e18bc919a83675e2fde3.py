<<<<<<< SEARCH
    def local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        in_A = set(A)
        in_B = set(B)
        all_elements = set(range(1, n + 1))
        
        # We will try to improve the solution.
        # Priority: 1. Add new elements. 2. Swap/Move to allow adding.
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1. Try to add any unused element
            unused = [x for x in range(1, n + 1) if x not in in_A and x not in in_B]
            random.shuffle(unused)
            
            for x in unused:
                if time.time() > deadline - 1:
                    break
                    
                # Try add to A
                if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    sums_A.add(x + x)
                    in_A.add(x)
                    improved = True
                    break # Restart loop with new state
                    
                # Try add to B
                if len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    sums_B.add(x + x)
                    in_B.add(x)
                    improved = True
                    break # Restart loop with new state
            
            if improved:
                continue
                
            # 2. If no direct addition, try swaps or moves to free up space
            # Try Swaps
            # To make this efficient, we don't test all pairs every time.
            # We pick a random subset or just iterate if time permits.
            
            # Let's try a limited number of random swaps/moves
            attempts = 0
            while attempts < 20 and time.time() < deadline - 1:
                attempts += 1
                
                # Random Swap
                if random.random() < 0.5 and len(A) > 0 and len(B) > 0:
                    a_idx = random.randint(0, len(A) - 1)
                    b_idx = random.randint(0, len(B) - 1)
                    a_val = A[a_idx]
                    b_val = B[b_idx]
                    
                    # Remove a from A, b from B
                    A_temp = A[:a_idx] + A[a_idx+1:]
                    B_temp = B[:b_idx] + B[b_idx+1:]
                    
                    # Recalculate sums/diffs for temp sets? 
                    # That's O(N^2). Too slow for many swaps.
                    # We need incremental removal logic or just accept the cost 
                    # if N is small (64). 64^2 = 4096. 20 swaps = 80k ops. 
                    # In Python, this might be slow. 
                    # Let's optimize: Only do swaps if we are stuck and have time.
                    
                    # For now, let's stick to a simpler local search: 
                    # Just try to add elements. If stuck, do a few random restarts.
                    # The greedy + polish is usually enough for this size.
                    
                    # Let's implement a "k-opt" style: remove k elements, try to re-add.
                    # Too complex for the time limit.
                    
                    break # Skip complex swaps for now, rely on multiple restarts
                    
                # Random Move (A to B or B to A)
                else:
                    if len(A) > 0 and len(B) < max_size:
                        a_idx = random.randint(0, len(A) - 1)
                        a_val = A[a_idx]
                        A_temp = A[:a_idx] + A[a_idx+1:]
                        
                        # Check if a_val can go to B
                        # We need diffs of A_temp and sums of B.
                        # Recalculating is expensive. 
                        # Given the time constraint and Python speed, 
                        # multiple fast restarts are better than one slow deep search.
                        
                        break

        return A, B, sums_A, sums_B, diffs_A, diffs_B
=======
    def remove_from_A(x, A, sums_A, diffs_A):
        """Remove x from A and update sums/diffs incrementally."""
        idx = A.index(x)
        A.pop(idx)
        # Remove sums involving x
        for a in A:
            sums_A.discard(x + a)
        sums_A.discard(x + x)
        # Remove diffs involving x
        for a in A:
            diffs_A.discard(x - a)
            diffs_A.discard(a - x)
    
    def remove_from_B(x, B, sums_B, diffs_B):
        """Remove x from B and update sums/diffs incrementally."""
        idx = B.index(x)
        B.pop(idx)
        for b in B:
            sums_B.discard(x + b)
        sums_B.discard(x + x)
        for b in B:
            diffs_B.discard(x - b)
            diffs_B.discard(b - x)
    
    def local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        in_A = set(A)
        in_B = set(B)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add any unused element
            unused = [x for x in range(1, n + 1) if x not in in_A and x not in in_B]
            random.shuffle(unused)
            
            for x in unused:
                if time.time() > deadline - 2:
                    break
                    
                # Try add to A
                if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    sums_A.add(x + x)
                    in_A.add(x)
                    improved = True
                    break
                    
                # Try add to B
                if len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    sums_B.add(x + x)
                    in_B.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. If stuck, try removing one element to unblock
            # Remove a random element from the larger set and try to re-add
            if len(A) > len(B) and len(A) > 1:
                idx = random.randint(0, len(A) - 1)
                x = A[idx]
                remove_from_A(x, A, sums_A, diffs_A)
                in_A.discard(x)
                improved = True
            elif len(B) > len(A) and len(B) > 1:
                idx = random.randint(0, len(B) - 1)
                x = B[idx]
                remove_from_B(x, B, sums_B, diffs_B)
                in_B.discard(x)
                improved = True
            elif len(A) > 1:
                idx = random.randint(0, len(A) - 1)
                x = A[idx]
                remove_from_A(x, A, sums_A, diffs_A)
                in_A.discard(x)
                improved = True
            elif len(B) > 1:
                idx = random.randint(0, len(B) - 1)
                x = B[idx]
                remove_from_B(x, B, sums_B, diffs_B)
                in_B.discard(x)
                improved = True

        return A, B, sums_A, sums_B, diffs_A, diffs_B
>>>>>>> REPLACE

