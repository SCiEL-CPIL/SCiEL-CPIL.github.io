import time

def construct():
    deadline = time.time() + 140.0
    TARGET = (1 << 30) - 1  # bits 0..29 represent diffs 1..30

    # Precompute: for each d in 1..30, its bit
    # We'll track covered diffs as a bitmask

    def popcount(x):
        return bin(x).count('1')

    def search(D):
        """Find Sidon set in [0,D] containing 0 covering all diffs 1..30.
        Returns the set as list, or None."""
        # We build the set incrementally. S[0] = 0 always.
        # used_diffs: bitmask of all positive differences that have appeared
        # covered: bitmask of which of 1..30 are covered (subset of used_diffs)
        
        best = [None]
        
        def dfs(S, used, covered):
            if (covered & TARGET) == TARGET:
                if best[0] is None:
                    best[0] = list(S)
                return True
            
            n = len(S)
            if n >= 32:
                return False
            
            cur_max = S[-1]
            if cur_max >= D:
                return False
            
            # Pruning: can we cover all remaining diffs?
            missing = TARGET & ~covered
            nmiss = popcount(missing)
            # Remaining elements we can add: at most min(D - cur_max, 32 - n)
            # But more tightly: each new element x > cur_max creates diffs x-s for s in S
            # The number of new diffs from adding k more elements is at most
            # k*n + k*(k-1)/2 (diffs with existing + diffs among new)
            # But also bounded by available range
            max_rem = min(D - cur_max, 32 - n)
            # Max possible new diffs: each of the max_rem new elements pairs with
            # all n existing + other new elements
            max_new_diffs = max_rem * n + max_rem * (max_rem - 1) // 2
            if max_new_diffs < nmiss:
                return False
            
            # Additional pruning: the largest missing diff must be achievable
            # max missing diff
            max_miss = 0
            m = missing
            while m:
                max_miss = max(max_miss, m.bit_length())
                m &= m - 1
            # The max possible new diff is D - 0 = D (from D - 0)
            # Actually max new diff from any x in (cur_max, D] with any s in S is x - 0 <= D
            # But more tightly: x - min(S) where min(S)=0, so max new diff = D
            if max_miss > D:
                return False
            
            # Generate candidates
            cands = []
            for x in range(cur_max + 1, D + 1):
                # Check Sidon: x - s must not be in used for any s in S
                ok = True
                new_cov = 0
                for s in S:
                    d = x - s
                    if d > 0:
                        bit = 1 << (d - 1) if d <= 30 else 0
                        # Check if this diff is already used
                        # used is a bitmask for diffs 1..30, but we also need to track
                        # all diffs (not just 1..30) for Sidon property
                        # Hmm, this is the problem: we need to track ALL diffs, not just 1..30
                        pass
                
                # We need a different approach for tracking all diffs
                pass
            
            # Let me rethink: use a set for all used diffs, but bitmask for covered 1..30
            pass
        
        # Actually, let me use a cleaner approach with a set for all diffs
        # but optimize the inner loop
        
        def dfs2(S, used_set, covered):
            if (covered & TARGET) == TARGET:
                if best[0] is None:
                    best[0] = list(S)
                return True
            
            n = len(S)
            if n >= 32:
                return False
            
            cur_max = S[-1]
            if cur_max >= D:
                return False
            
            missing = TARGET & ~covered
            nmiss = popcount(missing)
            max_rem = min(D - cur_max, 32 - n)
            max_new_diffs = max_rem * n + max_rem * (max_rem - 1) // 2
            if max_new_diffs < nmiss:
                return False
            
            # Generate candidates with coverage scores
            cands = []
            for x in range(cur_max + 1, D + 1):
                # Check Sidon property
                ok = True
                new_cov = 0
                for s in S:
                    d = x - s
                    if d in used_set:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_cov |= (1 << (d - 1))
                if not ok:
                    continue
                new_cov &= ~covered
                cnt = popcount(new_cov)
                cands.append((cnt, x, new_cov))
            
            # Sort by coverage desc, then x asc
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates
            if len(cands) > 20:
                cands = cands[:20]
            
            for cnt, x, new_cov in cands:
                if cnt == 0:
                    # Still need to add elements for future coverage
                    pass
                new_used = used_set | {x - s for s in S}
                S.append(x)
                if dfs2(S, new_used, covered | new_cov):
                    S.pop()
                    return True
                S.pop()
            
            return False
        
        dfs2([0], set(), 0)
        return best[0]
    
    # Try increasing diameters
    for D in range(30, 80):
        if time.time() > deadline - 5:
            break
        res = search(D)
        if res is not None:
            return res
    
    # Fallback: greedy
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline - 2:
        best_x = None
        best_cover = -1
        for x in range(max(S) + 1, max(S) + 200):
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
