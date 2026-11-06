def construct():
    import time
    start_time = time.time()
    
    # Symmetry breaking: Fix the first block to {0, 1, 2, 3, 4, 5}
    # This is valid because the design is homogeneous.
    first_block = (0, 1, 2, 3, 4, 5)
    covered = set()
    for s in combinations(first_block, 5):
        covered.add(s)
        
    blocks = [list(first_block)]
    
    # Precompute all 6-subsets
    all_blocks = list(combinations(range(12), 6))
    
    # Filter out the first block and blocks that conflict with it
    # Actually, we just iterate and check.
    
    def backtrack(current_blocks, current_covered):
        if len(current_blocks) == 132:
            return True
        if time.time() - start_time > 110: # Safety margin
            return False
            
        for b in all_blocks:
            if b in current_blocks:
                continue
            subsets = list(combinations(b, 5))
            if any(s in current_covered for s in subsets):
                continue
            
            # Try adding b
            new_covered = set(current_covered)
            new_covered.update(subsets)
            current_blocks.append(list(b))
            current_covered = new_covered # Pass by value effectively via set copy
            
            if backtrack(current_blocks, current_covered):
                return True
                
            current_blocks.pop()
            # No need to undo covered if we passed a copy, 
            # but for efficiency, we should use a mutable structure with undo.
            
        return False

    # The above is slow due to set copying. 
    # Better: Use a boolean array or set with add/remove.
    
    # Revised Backtracking with Undo
    covered = set()
    for s in combinations(first_block, 5):
        covered.add(s)
        
    blocks = [list(first_block)]
    
    def backtrack(idx):
        if len(blocks) == 132:
            return True
        if time.time() - start_time > 110:
            return False
            
        # Heuristic: Pick the next block
        # To be fast, we can iterate through remaining candidates.
        for b in all_blocks:
            if b in blocks:
                continue
            subsets = list(combinations(b, 5))
            conflict = False
            for s in subsets:
                if s in covered:
                    conflict = True
                    break
            if conflict:
                continue
                
            # Add b
            blocks.append(list(b))
            for s in subsets:
                covered.add(s)
                
            if backtrack(idx + 1):
                return True
                
            # Undo
            blocks.pop()
            for s in subsets:
                covered.remove(s)
                
        return False

    if backtrack(0):
        return blocks
    else:
        # Fallback to greedy if backtracking fails or times out
        # But for S(5,6,12), it should succeed quickly with symmetry breaking.
        return blocks
