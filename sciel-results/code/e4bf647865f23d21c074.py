import time

def construct():
    deadline = time.time() + 115.0
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    max_size = 32
    
    def dfs(S, used_diffs, covered_mask, D):
        if time.time() > deadline:
            return None
        
        if (covered_mask & full_mask) == full_mask:
            return list(S)
            
        missing = full_mask & ~covered_mask
        if missing == 0:
            return list(S)
            
        if len(S) >= max_size:
            return None
            
        # Pruning: Check if we can cover all missing diffs with remaining slots
        remaining_slots = max_size - len(S)
        
        # Capacity check
        max_new_diffs = remaining_slots * len(S) + remaining_slots * (remaining_slots - 1) // 2
        if bin(missing).count('1') > max_new_diffs:
            return None
            
        # Stronger Pruning: Check feasibility of each missing difference
        # A missing difference d must be formed by:
        # 1. x - s = d where s in S and x is a new element (x <= D)
        # 2. x_i - x_j = d where x_i, x_j are new elements
        #
        # If remaining_slots == 0, handled above.
        # If remaining_slots == 1:
        #   We can only add one new element x.
        #   d must be formed by x - s = d for some s in S.
        #   This requires s + d <= D for some s in S.
        #   If no such s exists, prune.
        #
        # If remaining_slots >= 2:
        #   d can be formed by new-new.
        #   However, if d is very large, it might be hard?
        #   Actually, if d > D, it's impossible. But d <= 30 and D >= 30 usually.
        #   If remaining_slots == 1, the new-new case is impossible.
        
        if remaining_slots == 1:
            # Only one new element x can be added.
            # x must be in (S[-1], D].
            # For each missing d, there must exist s in S such that s + d <= D.
            # And specifically, x = s + d must be a valid candidate.
            # Actually, x is a single value. It must cover ALL missing diffs?
            # No, x covers some diffs. The other diffs must already be covered?
            # Wait, if remaining_slots == 1, we add x.
            # The set of diffs becomes used_diffs U {x - s for s in S}.
            # All missing diffs MUST be in this new set.
            # So, for each missing d, d must be equal to x - s for some s in S.
            # This implies x = s + d.
            # So x is determined by d and s.
            # If we pick x, it must satisfy x - s = d for ALL missing d?
            # No, x - s_i = d_i and x - s_j = d_j.
            # This implies s_i + d_i = s_j + d_j = x.
            # So, for a single new element x to cover a set of diffs M,
            # there must be a consistent x such that for all d in M, x-d in S.
            
            # Let's check if there is ANY x that covers all missing diffs.
            # If not, prune.
            
            # Candidate x must be > S[-1] and <= D.
            # For each missing d, x must be s + d for some s in S.
            # So x must be in {s + d | s in S, d in missing}.
            # Let's find intersection of possible x values?
            # Or just iterate possible x?
            # Possible x are limited to range (S[-1], D].
            
            found_x = False
            for x in range(S[-1] + 1, D + 1):
                # Check if x covers all missing diffs
                covers_all = True
                for d in range(1, 31):
                    if missing & (1 << (d - 1)):
                        # Check if x - s == d for some s in S
                        if (x - d) not in S:
                            covers_all = False
                            break
                        # Also check Sidon constraint: x - s must not be in used_diffs
                        # But if it covers missing, it's not in used_diffs (since missing is disjoint from covered).
                        # Wait, used_diffs contains ALL diffs from S.
                        # missing are diffs NOT in used_diffs.
                        # So if x-d is in S, then d is a new diff.
                        # We must ensure no OTHER diff x-s' is in used_diffs.
                        ok_sidon = True
                        for s in S:
                            dd = x - s
                            if dd in used_diffs:
                                ok_sidon = False
                                break
                        if not ok_sidon:
                            covers_all = False
                            break
                if covers_all:
                    found_x = True
                    break
            if not found_x:
                return None

        candidates = []
        
        # Optimization: Use a set for used_diffs
        for x in range(S[-1] + 1, D + 1):
            ok = True
            new_mask = 0
            for s in S:
                d = x - s
                if d in used_diffs:
                    ok = False
                    break
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            if not ok:
                continue
            
            new_missing_covered = new_mask & missing
            coverage_count = bin(new_missing_covered).count('1')
            
            # Prioritize high coverage.
            candidates.append((coverage_count, x))
            
        if not candidates:
            return None
            
        # Sort by coverage (desc), then x (asc)
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        # Limit candidates to avoid explosion
        limit = 1000 if D < 100 else 50
        for _, x in candidates[:limit]:
            new_used = set(used_diffs)
            new_mask = covered_mask
            for s in S:
                d = x - s
                new_used.add(d)
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
                    
            S.append(x)
            res = dfs(S, new_used, new_mask, D)
            if res is not None:
                return res
            S.pop()
            
        return None

    best_res = None
    # Search for minimum D
    # Start from 44 based on findings
    for D in range(44, 200):
        if time.time() > deadline - 5:
            break
        res = dfs([0], set(), 0, D)
        if res is not None:
            return res
            
    # Fallback: Return a valid set if found in previous steps or a large one
    # If we reach here, we didn't find a small D.
    # Generate a valid set using a known construction or greedy.
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and len(S) < max_size:
        best_x = None
        best_score = -1
        # Search up to 1000
        for x in range(max(S) + 1, 1000):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok:
                score = cover * 1000 - x
                if score > best_score:
                    best_score = score
                    best_x = x
            if time.time() > deadline - 2:
                break
        if best_x is None:
            # Try any valid x
            for x in range(max(S) + 1, 1000):
                ok = True
                for s in S:
                    if x - s in used:
                        ok = False
                        break
                if ok:
                    best_x = x
                    break
            if time.time() > deadline - 2:
                break
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
        if time.time() > deadline - 2:
            break
            
    if target:
        pass
        
    return S
