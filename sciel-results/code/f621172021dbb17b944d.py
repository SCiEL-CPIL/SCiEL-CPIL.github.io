import sys
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    target_mask = (1 << 30) - 1  # Bits 0-29 set, representing diffs 1-30
    
    def check_time():
        return time.time() > deadline

    def solve(max_val):
        # DFS to find a Sidon set in [0, max_val] covering all diffs 1..30
        # Returns the list if found, else None
        
        # We try sizes k from 8 to 12 (min size for 30 diffs is 8, but likely 9 or 10)
        for k in range(8, 13):
            if check_time():
                return None
            
            # State: current_set (list), all_used_diffs (set), missing_mask (int), depth
            # We always start with 0
            initial_set = [0]
            initial_used = set()
            initial_missing = target_mask
            
            def dfs(current_set, all_used_diffs, missing_mask, depth):
                if check_time():
                    return None
                
                if missing_mask == 0:
                    return list(current_set)
                
                if depth == k:
                    return None
                
                last_val = current_set[-1]
                
                # Pruning 1: Max new diffs
                remaining_marks = k - depth
                max_new_diffs = remaining_marks * (remaining_marks - 1) // 2 + remaining_marks * depth
                # Count bits in missing_mask
                # Python 3.8+ has int.bit_count()
                missing_count = missing_mask.bit_count()
                if missing_count > max_new_diffs:
                    return None
                
                # Pruning 2: Coverage feasibility
                # For each missing diff d, there must be a way to form it.
                # It can be formed by (new, old) or (new, new).
                # If (new, old): x - y = d => x = y + d.
                # x must be in (last_val, max_val].
                # So for each d in missing, exists y in current_set such that last_val < y + d <= max_val.
                # If no such y for some d, it MUST be formed by (new, new).
                # This is hard to check strictly, but we can do a simple check:
                # If d is missing, and for all y in current_set, y+d > max_val, 
                # then d cannot be formed by (new, old).
                # It must be formed by z_i - z_j = d with z_i, z_j new.
                # This requires at least 2 new marks.
                # We can just generate candidates and let DFS handle it, but this prune helps.
                
                candidates = set()
                for y in current_set:
                    for d in range(1, 31):
                        if (missing_mask >> (d - 1)) & 1:
                            x = y + d
                            if last_val < x <= max_val:
                                candidates.add(x)
                
                # Also, we might need candidates that don't cover missing diffs immediately?
                # No, if we don't cover a missing diff, we waste a mark.
                # But what if we need a "bridge" mark?
                # In Sidon sets, every mark adds diffs.
                # If a mark doesn't cover a missing diff, it adds "useless" diffs (for coverage) 
                # but might enable future marks to cover missing diffs?
                # No, future marks cover diffs with ALL previous marks.
                # If a mark x doesn't cover any missing diff, it means for all y in S, x-y is not in missing.
                # Then any future mark z will have diffs z-x and z-y.
                # If z-x covers a missing diff d, then z-x=d.
                # This is possible.
                # However, restricting to candidates that cover at least one missing diff is a standard heuristic 
                # that works well for "covering" problems.
                # Let's stick to it for speed. If it fails, we might miss a solution, 
                # but given the small size, it's likely fine.
                
                # Sort candidates to try smaller ones first (minimize diameter)
                for x in sorted(candidates):
                    # Check Sidon property
                    valid = True
                    new_diffs = []
                    for y in current_set:
                        diff = x - y
                        if diff in all_used_diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                    
                    if not valid:
                        continue
                    
                    # Update state
                    new_used_diffs = set(all_used_diffs)
                    new_used_diffs.update(new_diffs)
                    
                    new_missing_mask = missing_mask
                    for diff in new_diffs:
                        if 1 <= diff <= 30:
                            new_missing_mask &= ~(1 << (diff - 1))
                    
                    res = dfs(current_set + [x], new_used_diffs, new_missing_mask, depth + 1)
                    if res:
                        return res
                
                return None
            
            res = dfs(initial_set, initial_used, initial_missing, 1)
            if res:
                return res
        
        return None

    # Iterate max_val starting from 30
    for m in range(30, 100):
        if check_time():
            break
        sol = solve(m)
        if sol:
            return sol
    
    # Fallback: Return a known valid set if search fails (should not happen for m=100)
    # Constructing a valid set greedily as fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

if __name__ == "__main__":
    res = construct()
    print(res)
