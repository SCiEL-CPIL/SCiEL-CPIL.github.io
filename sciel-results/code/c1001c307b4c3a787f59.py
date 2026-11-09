import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    target_diffs = list(range(1, 31))
    full_mask = (1 << 30) - 1
    
    # We know we need at least 9 elements.
    # We iterate D from a lower bound.
    # Lower bound for D:
    # If we have k elements, max diffs is k(k-1)/2.
    # We need 30 diffs. k=9 gives 36.
    # The diameter D must be at least the largest difference 30.
    # But Sidon constraints spread them out.
    # Start D from 30.
    
    best_set = None
    
    def solve_for_D(D):
        # State: current_set (tuple), used_diffs (frozenset), covered_mask (int), last_elem (int)
        # We use recursion with backtracking
        
        def dfs(current_set, used_diffs, covered_mask, last_elem):
            if time.time() > deadline:
                return None
            
            if covered_mask == full_mask:
                return list(current_set)
            
            # Pruning: Check if it's possible to cover remaining diffs
            missing_count = 30 - bin(covered_mask).count('1')
            
            # Max elements we can add
            # We are limited by D - last_elem
            # And by the total size limit 32
            max_elements_left = min(D - last_elem, 32 - len(current_set))
            
            if max_elements_left <= 0:
                return None
            
            # Max new differences we can generate
            # If we add m elements, new diffs = m * len(current_set) + m*(m-1)/2
            # This is an upper bound on how many *distinct* new diffs we can cover.
            # Actually, we need to cover 'missing_count' specific diffs.
            # If max possible new diffs < missing_count, prune.
            max_new_diffs = max_elements_left * len(current_set) + max_elements_left * (max_elements_left - 1) // 2
            if max_new_diffs < missing_count:
                return None
            
            # Generate candidates
            # Candidates must be > last_elem and <= D
            # And must cover at least one missing diff
            candidates = set()
            
            # For each missing diff d, and each element s in current_set
            # x = s + d is a candidate
            for d in range(1, 31):
                if not (covered_mask & (1 << (d - 1))):
                    for s in current_set:
                        x = s + d
                        if last_elem < x <= D:
                            candidates.add(x)
            
            # If no candidates, return None
            if not candidates:
                return None
            
            # Sort candidates by how many missing diffs they cover (greedy heuristic)
            # This helps find a solution faster
            def score_candidate(x):
                count = 0
                for d in range(1, 31):
                    if not (covered_mask & (1 << (d - 1))):
                        # Check if x - s == d for some s in current_set
                        # Equivalent to: is d in {x - s for s in current_set}?
                        # We can just check if x - d is in current_set
                        if x - d in current_set:
                            count += 1
                return count
            
            sorted_candidates = sorted(candidates, key=score_candidate, reverse=True)
            
            for x in sorted_candidates:
                # Check validity: no duplicate differences
                valid = True
                new_diffs = []
                for s in current_set:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_used_diffs = used_diffs.union(new_diffs)
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                new_set = current_set + (x,)
                
                res = dfs(new_set, new_used_diffs, new_mask, x)
                if res is not None:
                    return res
            
            return None
        
        return dfs((0,), frozenset(), 0, 0)

    # Try D from 30 upwards
    # A 9-mark Golomb ruler covering 1..30 likely has D around 50-60.
    # Start from 30 to be safe, but it will fail fast.
    for D in range(30, 150):
        if time.time() > deadline:
            break
        
        result = solve_for_D(D)
        if result is not None:
            return result
    
    # Fallback: Greedy construction
    S = [0]
    covered = set()
    while not covered.issuperset(set(range(1, 31))) and len(S) <= 32:
        missing = set(range(1, 31)) - covered
        found = False
        for d in sorted(missing):
            for s in S:
                c = s + d
                if c in S:
                    continue
                new_S = S + [c]
                new_S.sort()
                # Check Sidon
                diffs = set(y - x for i, x in enumerate(new_S) for y in new_S[i+1:])
                if len(diffs) == len(new_S) * (len(new_S) - 1) // 2:
                    S = new_S
                    covered = diffs
                    found = True
                    break
            if found:
                break
        if not found:
            # Add a large number to avoid collision
            c = 2 * max(S) + 3
            while True:
                new_S = S + [c]
                new_S.sort()
                diffs = set(y - x for i, x in enumerate(new_S) for y in new_S[i+1:])
                if len(diffs) == len(new_S) * (len(new_S) - 1) // 2:
                    S = new_S
                    covered = diffs
                    break
                c += 1
    return S

if __name__ == "__main__":
    print(construct())
