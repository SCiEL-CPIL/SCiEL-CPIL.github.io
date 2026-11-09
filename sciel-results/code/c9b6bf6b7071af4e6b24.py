import time
import random
import sys

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 85  # Keep 5s margin for evaluator overhead (90s total)
    
    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            sums = sums | {s + x for s in sums}
        return sums
    
    def check_divisibility(sums_set):
        """Check if any two distinct sums in sums_set divide each other."""
        if len(sums_set) <= 1:
            return True
        sorted_sums = sorted(sums_set)
        max_val = sorted_sums[-1]
        # For each sum u, check if any multiple 2u, 3u, ... exists in the set
        for u in sorted_sums:
            if u == 0:
                continue
            for mult in range(2, max_val // u + 1):
                if u * mult in sums_set:
                    return False
        return True

    def check_divisibility_fast(new_sums, existing_sums):
        """
        Fast check: verify that adding new_sums to existing_sums doesn't create
        divisibility violations. Only checks cross-divisibility between new and existing,
        and within new_sums (since existing_sums is already valid).
        """
        # Check within new_sums
        if len(new_sums) > 1:
            sorted_new = sorted(new_sums)
            max_new = sorted_new[-1]
            for u in sorted_new:
                if u == 0:
                    continue
                for mult in range(2, max_new // u + 1):
                    if u * mult in new_sums:
                        return False
        # Check cross-divisibility: for each u in new_sums, check if u*k in existing_sums
        for u in new_sums:
            if u == 0:
                continue
            max_check = max(existing_sums) if existing_sums else 0
            for mult in range(2, max_check // u + 1):
                if u * mult in existing_sums:
                    return False
            # Also check: for each v in existing_sums, does v divide some new sum?
            # i.e., does v*k exist in new_sums for k>=2?
            for v in existing_sums:
                if v == 0:
                    continue
                max_new = max(new_sums) if new_sums else 0
                for mult in range(2, max_new // v + 1):
                    if v * mult in new_sums:
                        return False
        return True
    
    def can_add(x, current_sums):
        """
        Check if we can add x to a set with given subset sums.
        current_sums: set of all subset sums including 0
        Returns (bool, new_sums) if valid.
        """
        new_sums = {x + s for s in current_sums}
        
        # Check distinctness: new_sums should not overlap with current_sums
        if new_sums & current_sums:
            return False, None
        
        # Fast divisibility check
        if not check_divisibility_fast(new_sums - {0}, current_sums - {0}):
            return False, None
        
        combined = current_sums | new_sums
        return True, combined
    
    def greedy_build(candidate_order, start_set=None, start_sums=None):
        """Greedy build using a given order of candidates, optionally starting from an existing set."""
        if start_set is not None:
            S = list(start_set)
            current_sums = set(start_sums)
        else:
            S = []
            current_sums = {0}
        
        existing = set(S)
        for x in candidate_order:
            if time.time() - start_time > time_limit:
                break
            if x in existing:
                continue
            valid, new_sums = can_add(x, current_sums)
            if valid:
                S.append(x)
                existing.add(x)
                current_sums = new_sums
        return S, current_sums
    
    def local_search(S, sums_set):
        """Try to improve the set by removing one element and adding a different one."""
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_copy = list(S)
            sums_copy = set(sums_set)
            
            for remove_idx in range(len(S_copy)):
                if time.time() - start_time > time_limit:
                    break
                
                x_removed = S_copy[remove_idx]
                S_no = S_copy[:remove_idx] + S_copy[remove_idx+1:]
                sums_no = compute_sums(S_no)
                in_set = set(S_no)
                
                # Try adding each candidate not in S_no, prefer larger numbers
                for x in range(256, 0, -1):
                    if x in in_set:
                        continue
                    valid, new_sums = can_add(x, sums_no)
                    if valid:
                        S_new = S_no + [x]
                        if len(S_new) > len(S_copy):
                            S_copy = S_new
                            sums_copy = new_sums
                            improved = True
                            break
                if improved:
                    break
        
        return S_copy, sums_copy
    
    # Strategy: Multiple random restarts with different heuristics
    best_set = []
    best_size = 0
    best_sums = {0}
    
    random.seed(12345)
    
    num_restarts = 60
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart < 10:
            # Start with larger numbers first
            candidates = list(range(256, 0, -1))
            random.shuffle(candidates[:100])
            random.shuffle(candidates[100:])
        elif restart < 20:
            # Random order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif restart < 30:
            # Weighted toward larger numbers
            candidates = list(range(1, 257))
            candidates.sort(key=lambda x: -x + random.gauss(0, 50))
        elif restart < 45:
            # Start from best_set and try to add more
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            S, sums = greedy_build(candidates, best_set, best_sums)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
                best_sums = sums
            continue
        else:
            # Mixed: large primes first, then random
            primes = [p for p in range(129, 257) if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
            candidates = primes + [x for x in range(1, 129) if x not in primes]
            random.shuffle(candidates[20:])
        
        S, sums = greedy_build(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_sums = sums
    
    # Local search: try to improve best_set
    if best_size >= 2:
        S_ls, sums_ls = local_search(best_set, best_sums)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = S_ls[:]
            best_sums = sums_ls
    
    # Additional local search rounds with perturbation
    for round_num in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb: remove 1-2 random elements and try to refill
        if len(best_set) >= 3:
            S_pert = list(best_set)
            random.shuffle(S_pert)
            num_remove = min(2, len(S_pert) - 1)
            S_pert = S_pert[num_remove:]
            sums_pert = compute_sums(S_pert)
            
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            S_new, sums_new = greedy_build(candidates, S_pert, sums_pert)
            
            if len(S_new) > best_size:
                best_size = len(S_new)
                best_set = S_new[:]
                best_sums = sums_new
    
    # Final local search
    if best_size >= 2:
        S_ls2, sums_ls2 = local_search(best_set, best_sums)
        if len(S_ls2) > best_size:
            best_size = len(S_ls2)
            best_set = S_ls2[:]
            best_sums = sums_ls2
    
    # One more round of greedy extension from best
    if time.time() - start_time < time_limit - 5:
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        S_ext, sums_ext = greedy_build(candidates, best_set, best_sums)
        if len(S_ext) > best_size:
            best_size = len(S_ext)
            best_set = S_ext[:]
            best_sums = sums_ext
        S_ls3, sums_ls3 = local_search(best_set, best_sums)
        if len(S_ls3) > best_size:
            best_size = len(S_ls3)
            best_set = S_ls3[:]
            best_sums = sums_ls3
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty = final_sums - {0}
        if len(non_empty) != (1 << len(best_set)) - 1:
            best_set = []
        elif not check_divisibility(non_empty):
            best_set = []
    
    # Ensure we return a valid set
    if not best_set or len(best_set) < 1:
        return [256]
    
    # Convert to pure Python ints
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(res)
