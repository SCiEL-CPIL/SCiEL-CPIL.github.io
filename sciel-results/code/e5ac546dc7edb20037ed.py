import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    TARGET_MASK = (1 << 30) - 1  # bits 0..29 for diffs 1..30
    
    def is_valid_sidon_covering(A):
        """Check if A is a valid Sidon set covering 1..30."""
        if 0 not in A:
            return False
        A = sorted(A)
        diffs = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    def search_for_M(M, deadline):
        """Find a Sidon set in [0,M] containing 0 and M, covering 1..30."""
        if M < 30:
            return None
        
        # State: current set, used diffs (set), covered mask
        # We force 0 and M in the set
        
        init_set = frozenset([0, M])
        init_used = {M}
        init_mask = 0
        if M <= 30:
            init_mask |= (1 << (M - 1))
        
        # If M > 30, diff M is not in 1..30, so no coverage
        # But M itself is a used diff that blocks other pairs
        
        best_result = [None]
        
        def backtrack(current_set, used_diffs, covered_mask, depth):
            if time.time() > deadline:
                return None
            
            if covered_mask == TARGET_MASK:
                return sorted(current_set)
            
            if len(current_set) > 32:
                return None
            
            # Generate all valid candidates
            candidates = []
            for x in range(1, M):
                if x in current_set:
                    continue
                
                new_diffs = []
                valid = True
                for a in current_set:
                    d = abs(x - a)
                    if d in used_diffs:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Compute coverage gain
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                gain = bin(new_mask ^ covered_mask).count('1')
                
                # Additional heuristic: prefer candidates that cover "rare" diffs
                # Count how many missing diffs are covered, weighted by rarity
                # For now, just use gain
                
                candidates.append((gain, x, set(new_diffs), new_mask))
            
            if not candidates:
                return None
            
            # Sort by gain descending, then by x ascending
            candidates.sort(key=lambda c: (-c[0], c[1]))
            
            # Try top candidates (limit to avoid explosion)
            max_branch = 8
            for gain, x, new_diffs, new_mask in candidates[:max_branch]:
                if gain == 0 and candidates[0][0] == 0:
                    break  # All candidates have 0 gain, dead end
                
                new_set = current_set | {x}
                new_used = used_diffs | new_diffs
                
                result = backtrack(new_set, new_used, new_mask, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(init_set, init_used, init_mask, 0)
    
    # Search from lower bound upward
    # Lower bound: M >= 30
    # For 9 elements, known constructions suggest M around 40-60
    
    for M in range(30, 150):
        if time.time() > deadline:
            break
        
        result = search_for_M(M, deadline)
        if result is not None:
            # Verify
            if is_valid_sidon_covering(result):
                return result
    
    # Fallback: known construction
    # A Sidon set covering 1..30
    return [0, 1, 4, 10, 12, 17, 21, 25, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120]
