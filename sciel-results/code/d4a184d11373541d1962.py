import time

def construct():
    deadline = time.time() + 110.0
    target_mask = (1 << 30) - 1
    
    # Precompute mask for each diff 1..30
    diff_masks = [0] * 31
    for d in range(1, 31):
        diff_masks[d] = 1 << (d - 1)
        
    def search(D):
        # Use bitmask for used diffs (1..D)
        # D is small, so this is fast
        
        def dfs(S, used_mask, covered_mask):
            if time.time() > deadline:
                return None
            
            if (covered_mask & target_mask) == target_mask:
                return list(S)
            
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            if num_missing == 0:
                return list(S)
                
            # Pruning
            # Check if enough room for new diffs
            # Max new elements k such that S[-1]+k <= D
            max_add = D - S[-1]
            if max_add <= 0:
                return None
            
            # Potential new diffs
            potential = max_add * len(S) + max_add * (max_add - 1) // 2
            if potential < num_missing:
                return None
            
            cands = []
            for x in range(S[-1] + 1, D + 1):
                valid = True
                new_cover = 0
                new_used_bits = 0
                
                for s in S:
                    d = x - s
                    # Check if d is already used
                    if (used_mask >> (d - 1)) & 1:
                        valid = False
                        break
                    new_used_bits |= (1 << (d - 1))
                    if 1 <= d <= 30:
                        new_cover |= diff_masks[d]
                
                if not valid:
                    continue
                
                new_cover_missing = new_cover & missing
                score = bin(new_cover_missing).count('1')
                cands.append((score, x, new_used_bits, new_cover))
            
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            for score, x, new_used_bits, new_cover in cands:
                new_used = used_mask | new_used_bits
                new_covered = covered_mask | new_cover
                
                S.append(x)
                res = dfs(S, new_used, new_covered)
                S.pop()
                
                if res is not None:
                    return res
                
            return None
        
        # Start with S=[0]
        return dfs([0], 0, 0)

    # Search for minimal D
    # Lower bound 36 based on n>=9
    for D in range(36, 100):
        if time.time() > deadline - 5:
            break
        res = search(D)
        if res is not None:
            return res
            
    # Fallback: Greedy construction
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline:
        best_x = None
        best_cover = -1
        # Look ahead further if needed
        for x in range(max(S) + 1, max(S) + 500):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok and cover > best_cover:
                best_cover = cover
                best_x = x
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
        
    return S
