import math

def construct():
    max_val = 256
    
    # We will try to find the longest sequence.
    # Max possible length is bounded by log2(256) * something?
    # Let's just search for lengths 12 down to 1?
    # Or incrementally.
    
    def is_valid_sidon(gaps):
        # gaps: list of ints
        n = len(gaps)
        prefixes = [0] * (n + 1)
        for i in range(n):
            prefixes[i+1] = prefixes[i] + gaps[i]
        
        diffs = set()
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                d = prefixes[j] - prefixes[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Backtracking search
    def search(gaps, last_gap, last_prefix, diffs):
        # gaps: current list
        # last_gap: last element in gaps
        # last_prefix: sum of gaps
        # diffs: set of existing differences
        
        # Try to add next gap
        # We want to maximize length.
        # This function returns a valid extension or None?
        # Better: Use it to find one solution of a specific length.
        
        pass

    # Let's use a beam search with a better state representation and heuristic.
    # State: (last_gap, last_prefix, diffs_set, seq_tuple)
    # To speed up, we can limit the beam width and use a greedy "least restrictive" choice.
    
    # Actually, let's try a simple greedy with backtracking for length 10.
    # If it finds one, return it.
    # If not, return the best found (length 9).
    
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256] # Fallback
    
    def find_sequence(target_len):
        # Backtracking
        def backtrack(gaps, last_gap, last_prefix, diffs):
            if len(gaps) == target_len:
                return gaps[:]
            
            # Heuristic: try smaller gaps first? Or larger?
            # Smaller gaps keep prefix sum low, allowing more room?
            # But Sidon property requires spread.
            # Let's try all valid gaps.
            
            start = last_gap + 1
            if start > max_val:
                return None
            
            for x in range(start, max_val + 1):
                new_prefix = last_prefix + x
                
                # Check collisions
                valid = True
                new_diffs = set()
                # New diffs are new_prefix - P_i for all i
                # We need P_i.
                # We can store prefixes in the recursion stack?
                # Or compute from gaps.
                
                # To avoid computing all P_i, we can store the list of prefixes.
                # But that's heavy.
                
                # Let's store prefixes in the state.
                # backtrack(gaps, prefixes, diffs)
                
                pass
        
        # Refined Backtrack
        def bt(gaps, prefixes, diffs):
            if len(gaps) == target_len:
                return gaps[:]
            
            last_gap = gaps[-1] if gaps else 0
            last_prefix = prefixes[-1]
            
            start = last_gap + 1
            if start > max_val:
                return None
            
            # Try candidates
            # Optimization: If we have many candidates, we might want to order them.
            # Smaller gaps are generally safer for keeping prefix sum low?
            # But we need to fill the "gaps" in the difference set.
            
            for x in range(start, max_val + 1):
                new_prefix = last_prefix + x
                new_diffs = set()
                valid = True
                
                # Check new differences: new_prefix - p for p in prefixes
                for p in prefixes:
                    d = new_prefix - p
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Recurse
                res = bt(gaps + [x], prefixes + [new_prefix], diffs | new_diffs)
                if res:
                    return res
            
            return None

        return bt([], [0], set())

    # Try lengths from high to low
    # Max possible length?
    # Sum of 12 gaps min 1..12 = 78. Max 256..245.
    # Let's try 10, 11, 12.
    
    for L in range(12, 8, -1):
        seq = find_sequence(L)
        if seq:
            # Verify
            if is_valid_sidon(seq):
                return seq
    
    return best_seq
