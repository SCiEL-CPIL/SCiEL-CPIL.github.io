"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 50.0  # Stay well under 60s
    
    # Heuristic: Small numbers are bad because they divide many things.
    # We want numbers that are "large" relative to each other to ensure unique sums (superincreasing),
    # but also avoid divisibility.
    # A superincreasing sequence a1 < a2 < ... < an satisfies a_i > sum(a_1..a_{i-1}).
    # This guarantees unique subset sums.
    # We just need to check divisibility among the resulting subset sums.
    
    # Let's use a backtracking search.
    # We will try to build a superincreasing sequence.
    # To maximize size, we should try to keep elements as small as possible?
    # No, if elements are small, the subset sums are small, and small sums divide large sums more easily?
    # Actually, if we have a large sum S, it is less likely to be divisible by a small sum s if S is not a multiple.
    # But if s is very small (like 1 or 2), it divides many things.
    # So we should avoid small numbers.
    
    # Strategy:
    # 1. Generate a list of candidate numbers.
    # 2. Sort them in a way that prioritizes "safe" numbers.
    # 3. Backtrack to find the longest valid superincreasing sequence.
    
    # Since the sequence must be superincreasing, the order is fixed by value.
    # If we pick a set, we sort it. Let the sorted set be x1 < x2 < ... < xk.
    # Condition: x_i > sum(x_1..x_{i-1}) for all i > 1.
    
    # We can iterate through numbers 1..256 and decide to include or not.
    # But we must maintain the superincreasing property.
    # Let current_sum be the sum of selected elements so far.
    # The next element x must be > current_sum.
    
    # This drastically reduces the search space.
    
    best_set = []
    best_size = 0
    
    def is_valid_divisibility(sums):
        # Check if any two distinct sums divide each other
        sum_list = sorted(sums)
        n = len(sum_list)
        for i in range(n):
            si = sum_list[i]
            for j in range(i+1, n):
                sj = sum_list[j]
                if sj % si == 0:
                    return False
        return True
    
    def backtrack(current_elements, current_sums, current_sum, next_min):
        nonlocal best_set, best_size
        
        if len(current_elements) > best_size:
            best_set = list(current_elements)
            best_size = len(current_elements)
            
        if time.time() - start_time > time_limit:
            return
            
        # Try adding each candidate > current_sum
        # Candidates must be <= 256
        # Also, to be superincreasing, candidate > current_sum
        # And candidate must not create divisibility issues with current_sums
        
        # We can iterate candidates from next_min to 256
        # But we should try candidates that are "likely" to work.
        # Larger candidates are "safer" for uniqueness (already guaranteed by superincreasing)
        # but might create new sums that are multiples of old sums.
        
        # Let's try candidates in a randomized order or descending order?
        # Descending might find large sets faster? Or ascending to allow more elements?
        # If we pick small elements, current_sum grows slowly, allowing more elements.
        # But small elements create small sums that divide many things.
        # So there's a trade-off.
        
        # Let's try a mix: prioritize candidates that are primes? Primes have fewer divisors.
        # But the sums themselves can be composite.
        
        # Let's just iterate candidates from next_min to 256.
        # To speed up, we can skip candidates that are multiples of any existing sum?
        # If candidate c is a multiple of some sum s, then c is a new sum.
        # If s < c, then s divides c. Invalid.
        # So c must not be a multiple of any existing sum.
        
        # Also, for any existing sum s, s+c must not be a multiple of any other sum t (including s).
        # If s+c is a multiple of s, then c is a multiple of s. (Already checked)
        # If s+c is a multiple of t (t != s), then (s+c) % t == 0.
        
        candidates = []
        for c in range(next_min, 257):
            # Check if c is multiple of any existing sum
            ok = True
            for s in current_sums:
                if c % s == 0:
                    ok = False
                    break
            if not ok:
                continue
                
            # Check if s+c is multiple of any existing sum
            # Note: s+c > s, so we only check if s+c is divisible by t where t is in current_sums
            # t can be s or other sums.
            # If t == s, (s+c)%s == c%s. We already checked c%s != 0.
            # So we only need to check t != s?
            # Wait, if t=s, (s+c)%s = c%s. If c%s != 0, then s does not divide s+c.
            # So we just need to check other sums t.
            
            new_sums_candidate = set(current_sums)
            for s in current_sums:
                ns = s + c
                new_sums_candidate.add(ns)
            new_sums_candidate.add(c)
            
            # Check divisibility among new_sums_candidate
            # This is expensive. Let's do a quick check first.
            # Quick check:
            # 1. c is not multiple of any s in current_sums. (Done)
            # 2. For each s in current_sums, s+c is not multiple of any t in current_sums.
            #    If t == s, (s+c)%s = c%s != 0.
            #    If t != s, check (s+c)%t == 0.
            
            valid = True
            for s in current_sums:
                ns = s + c
                for t in current_sums:
                    if t == s:
                        continue
                    if ns % t == 0:
                        valid = False
                        break
                if not valid:
                    break
                # Also check if ns is multiple of c?
                # If ns % c == 0, then c divides ns.
                # ns = s+c. (s+c)%c = s%c.
                # So we need s%c != 0.
                if s % c == 0:
                    valid = False
                    break
            
            if valid:
                candidates.append(c)
                
        # Shuffle candidates to avoid bias?
        # Or sort by some heuristic?
        # Let's try descending order first (larger elements might be safer for future steps? No, smaller allows more steps).
        # Let's try ascending order to maximize count.
        # But small numbers are risky.
        # Let's try a randomized order for diversity in backtracking.
        
        random.shuffle(candidates)
        
        for c in candidates:
            # Compute new sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + c)
            new_sums.add(c)
            
            # Full check
            if is_valid_divisibility(new_sums):
                backtrack(current_elements + [c], new_sums, current_sum + c, c + 1)
                # Prune: if we found a set of size 10, we can stop?
                # Max possible size is limited by sum.
                # 1+2+4+...+2^k = 2^(k+1)-1 <= 256*30? No, elements <= 256.
                # Superincreasing: x1 >= 1, x2 > 1, x3 > 1+x2, ...
                # Min sum for k elements: 1, 2, 4, 8, 16, 32, 64, 128, 256 (too big? 256 is allowed).
                # 1,2,4,8,16,32,64,128 sum = 255. Next must be > 255, so 256.
                # Set: {1,2,4,8,16,32,64,128,256}. Size 9.
                # But 1 divides everything. So 1 cannot be in set with others.
                # So min elements are larger.
                # If we skip 1, start with 2? 2 divides 4, 8...
                # So we need to avoid small divisors.
                
            if time.time() - start_time > time_limit:
                return

    # Start with empty set
    # We can start with different initial elements to explore different branches
    # But backtracking will handle it.
    # To speed up, we can try starting with a few specific "good" elements.
    # Or just call backtrack once.
    
    backtrack([], set(), 0, 1)
    
    # If backtracking didn't find anything (or failed), fallback to greedy
    if not best_set:
        # Greedy fallback
        selected = []
        current_sums = set()
        current_sum = 0
        for c in range(1, 257):
            if c <= current_sum:
                continue
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + c)
            new_sums.add(c)
            if is_valid_divisibility(new_sums):
                selected.append(c)
                current_sums = new_sums
                current_sum += c
        best_set = selected
        
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
