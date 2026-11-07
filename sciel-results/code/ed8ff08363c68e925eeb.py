import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s kill

    # Precompute divisors for all possible sums (up to 30*256 = 7680)
    MAX_SUM = 7680
    divisors = [[] for _ in range(MAX_SUM + 1)]
    for d in range(1, MAX_SUM + 1):
        for m in range(d, MAX_SUM + 1, d):
            divisors[m].append(d)

    def is_valid_set(S):
        """Check if set S satisfies all constraints."""
        if len(S) != len(set(S)):
            return False
        # Compute all subset sums
        sums = set()
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            # Check for duplicate sums
            if sums & new_sums:
                return False
            sums |= new_sums
        
        # Check divisibility: no two distinct sums divide each other
        sum_list = sorted(sums)
        for i, a in enumerate(sum_list):
            for b in sum_list[i+1:]:
                if b % a == 0:
                    return False
        return True

    def try_add(sums_set, x):
        """Try to add x to a set with current subset sums. Returns (valid, new_sums) or (False, None)."""
        new_sums = set()
        new_sums.add(x)
        for s in sums_set:
            new_sums.add(s + x)
        
        # Check for duplicate sums
        if sums_set & new_sums:
            return False, None
        
        all_sums = sums_set | new_sums
        max_sum = max(all_sums)
        
        for s in new_sums:
            # Check multiples of s in all_sums
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in all_sums:
                    return False, None
                multiple += s
            
            # Check divisors of s in all_sums (excluding s itself)
            for d in divisors[s]:
                if d == s:
                    continue
                if d in all_sums:
                    return False, None
                # Also check if s divides d (i.e., d is a multiple of s)
                # But we already check multiples above for s in new_sums.
                # Here d is a divisor of s, so s = k*d for some k>=2.
                # We need to check if d is in all_sums. Done above.
                # We also need to check if s is in all_sums as a multiple of d.
                # That's covered by the "multiples of s" check? No.
                # If d divides s, then s is a multiple of d. We need to ensure no other sum t in all_sums 
                # is such that t divides s or s divides t.
                # The loop above checks: for each new sum s, does any multiple of s exist in all_sums?
                # And does any divisor of s exist in all_sums?
                # This covers both directions for the new sums.
                # Since old sums were already valid among themselves, we only need to check interactions
                # between new sums and all sums (old + new).
                # The check "for s in new_sums: check multiples of s in all_sums" covers s | t for t in all_sums.
                # The check "for s in new_sums: check divisors of s in all_sums" covers t | s for t in all_sums.
                # This is sufficient.
        
        return True, all_sums

    def greedy_search(order, current_sums=None, current_set=None):
        if current_sums is None:
            current_sums = set()
        if current_set is None:
            current_set = []
        
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums = try_add(current_sums, x)
            if valid:
                current_set.append(x)
                current_sums = new_sums
        return current_set

    def get_sums(S):
        sums = set()
        for x in S:
            new_sums = {x + s for s in sums}
            new_sums.add(x)
            sums |= new_sums
        return sums

    candidates = list(range(1, 257))
    best_set = []
    best_size = 0

    # Strategy 1: Greedy with various orderings
    # Descending
    s1 = greedy_search(sorted(candidates, reverse=True))
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1

    # Ascending
    s2 = greedy_search(sorted(candidates))
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2

    # Primes first
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    s3 = greedy_search(primes + composites)
    if len(s3) > best_size:
        best_size = len(s3)
        best_set = s3

    # Randomized restarts
    for i in range(500):
        if time.time() > deadline - 5.0:
            break
        random_order = random.sample(candidates, len(candidates))
        s = greedy_search(random_order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local search: try to improve by swapping or adding
    if best_set and time.time() < deadline - 10.0:
        current_best = list(best_set)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            used = set(current_best)
            
            # Try to add a new element
            for x in sorted(candidates, reverse=True):
                if x in used:
                    continue
                trial_set = current_best + [x]
                trial_sums = get_sums(trial_set)
                # Check divisibility
                valid = True
                sum_list = sorted(trial_sums)
                for i, a in enumerate(sum_list):
                    for b in sum_list[i+1:]:
                        if b % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current_best = trial_set
                    improved = True
                    used.add(x)
                    break
            
            # Try swaps if no addition was made
            if not improved:
                for i in range(len(current_best)):
                    if time.time() > deadline - 5.0:
                        break
                    trial_set = current_best[:i] + current_best[i+1:]
                    trial_sums = get_sums(trial_set)
                    
                    for x in sorted(candidates, reverse=True):
                        if x in used:
                            continue
                        valid, new_sums = try_add(trial_sums, x)
                        if valid:
                            # Check if this swap allows adding more later? 
                            # For now, just accept if it's valid and same size.
                            # To make progress, we need a better heuristic.
                            # Let's just do it and hope for the best.
                            current_best = trial_set + [x]
                            improved = True
                            break
                    if improved:
                        break

    # Final validation
    if not best_set:
        return [11, 13, 17]
    
    # Ensure types
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(res)
