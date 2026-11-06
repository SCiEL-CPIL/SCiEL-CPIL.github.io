# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    N = 4096
    
    def check_addition(S, x, sums_set):
        """Check if adding x to S preserves unique triple sums. Returns (valid, new_sums_list)."""
        new_sums = []
        # (x, x, x)
        s3 = x * 3
        new_sums.append(s3)
        # (x, x, s) for each s in S
        for s in S:
            new_sums.append(2 * x + s)
        # (x, s, t) for s <= t in S
        for i in range(len(S)):
            si = S[i]
            for j in range(i, len(S)):
                sj = S[j]
                new_sums.append(x + si + sj)
        
        # Check all new sums against existing sums_set and each other
        for s in new_sums:
            if s in sums_set:
                return False, None
        if len(new_sums) != len(set(new_sums)):
            return False, None
        return True, new_sums
    
    def greedy_search(ordering):
        """Greedy search with given ordering of candidates."""
        S = []
        sums_set = set()
        for x in ordering:
            if time.time() > deadline:
                break
            valid, new_sums = check_addition(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S
    
    def greedy_with_conflict_score(ordering):
        """Greedy that scores candidates by how many conflicts they'd cause, picks least-conflicting."""
        S = []
        sums_set = set()
        candidates = list(ordering)
        
        while candidates:
            if time.time() > deadline:
                break
            best_x = None
            best_conflicts = float('inf')
            best_new_sums = None
            
            # Sample a subset of candidates for large sets to save time
            sample_size = min(len(candidates), 200)
            if len(candidates) > sample_size:
                sample = random.sample(candidates, sample_size)
            else:
                sample = candidates
            
            for x in sample:
                # Quick conflict count
                conflicts = 0
                new_sums = []
                new_sums.append(3 * x)
                for s in S:
                    new_sums.append(2 * x + s)
                for i in range(len(S)):
                    si = S[i]
                    for j in range(i, len(S)):
                        sj = S[j]
                        new_sums.append(x + si + sj)
                
                for s in new_sums:
                    if s in sums_set:
                        conflicts += 1
                        if conflicts >= best_conflicts:
                            break
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_x = x
                    best_new_sums = new_sums
                    if conflicts == 0:
                        break
            
            if best_x is not None and best_conflicts == 0:
                S.append(best_x)
                sums_set.update(best_new_sums)
                candidates.remove(best_x)
            else:
                break
        return S
    
    def try_improve(S):
        """Try to improve set S by adding elements, then single swaps, then double swaps."""
        if time.time() > deadline:
            return S
        S = sorted(S)
        sums_set = set()
        for a, b, c in combinations_with_replacement(S, 3):
            sums_set.add(a + b + c)
        
        # Phase 1: Try adding elements
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                valid, new_sums = check_addition(S, x, sums_set)
                if valid:
                    S.append(x)
                    S.sort()
                    sums_set.update(new_sums)
                    S_set.add(x)
                    improved = True
                    break
        
        # Phase 2: Try single-element swaps (remove one, add one)
        if time.time() > deadline - 5:
            return S
        for _ in range(3):  # Up to 3 rounds of single swaps
            if time.time() > deadline - 2:
                break
            found_swap = False
            for i in range(len(S)):
                if time.time() > deadline - 2:
                    break
                S_no_i = S[:i] + S[i+1:]
                sums_set_no_i = set()
                for a, b, c in combinations_with_replacement(S_no_i, 3):
                    sums_set_no_i.add(a + b + c)
                S_set_no_i = set(S_no_i)
                
                for x in range(1, N + 1):
                    if x in S_set_no_i:
                        continue
                    valid, new_sums = check_addition(S_no_i, x, sums_set_no_i)
                    if valid:
                        S = S_no_i + [x]
                        S.sort()
                        sums_set = sums_set_no_i | set(new_sums)
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap:
                break
        
        # Phase 3: Try double swaps (remove one, add two) - expensive, only if time remains
        if time.time() > deadline - 5:
            return S
        for i in range(len(S)):
            if time.time() > deadline - 2:
                break
            S_no_i = S[:i] + S[i+1:]
            sums_set_no_i = set()
            for a, b, c in combinations_with_replacement(S_no_i, 3):
                sums_set_no_i.add(a + b + c)
            S_set_no_i = set(S_no_i)
            
            found_double = False
            for x1 in range(1, N + 1):
                if time.time() > deadline - 2:
                    break
                if x1 in S_set_no_i:
                    continue
                valid1, new_sums1 = check_addition(S_no_i, x1, sums_set_no_i)
                if not valid1:
                    continue
                S_tmp = S_no_i + [x1]
                S_tmp.sort()
                sums_set_tmp = sums_set_no_i | set(new_sums1)
                S_set_tmp = S_set_no_i | {x1}
                
                for x2 in range(x1 + 1, N + 1):
                    if time.time() > deadline - 2:
                        break
                    if x2 in S_set_tmp:
                        continue
                    valid2, new_sums2 = check_addition(S_tmp, x2, sums_set_tmp)
                    if valid2:
                        S = S_tmp + [x2]
                        S.sort()
                        sums_set = sums_set_tmp | set(new_sums2)
                        found_double = True
                        break
                if found_double:
                    break
            if found_double:
                break
        
        return S
    
    best_set = []
    
    # Strategy 1: Ascending greedy
    S1 = greedy_search(range(1, N + 1))
    best_set = max(best_set, S1, key=len)
    
    # Strategy 2: Descending greedy
    if time.time() < deadline:
        S2 = greedy_search(range(N, 0, -1))
        best_set = max(best_set, S2, key=len)
    
    # Strategy 3: Conflict-scored greedy
    if time.time() < deadline:
        random.seed(42)
        S3 = greedy_with_conflict_score(range(1, N + 1))
        best_set = max(best_set, S3, key=len)
    
    # Strategy 4: Multiple random orderings
    if time.time() < deadline:
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 12345 + 67890)
            order = list(range(1, N + 1))
            random.shuffle(order)
            S_rand = greedy_search(order)
            if len(S_rand) > len(best_set):
                best_set = S_rand
    
    # Local improvement on best set
    if time.time() < deadline - 5 and len(best_set) > 0:
        best_set = try_improve(best_set)
    
    # Try more random starts with improvement if time permits
    if time.time() < deadline - 10:
        for attempt in range(40):
            if time.time() > deadline - 5:
                break
            random.seed(attempt * 98765 + 43210)
            order = list(range(1, N + 1))
            random.shuffle(order)
            S_rand = greedy_search(order)
            if len(S_rand) > len(best_set):
                best_set = S_rand
            elif len(S_rand) >= len(best_set) and time.time() < deadline - 10:
                S_improved = try_improve(S_rand)
                if len(S_improved) > len(best_set):
                    best_set = S_improved
    
    return best_set
# EVOLVE-BLOCK-END
