import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute smallest prime factor (SPF) for numbers up to 128
    # This helps in ordering candidates: numbers with larger smallest prime factors
    # are "safer" because they introduce new prime factors less frequently,
    # reducing the chance of product collisions.
    max_val = 128
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Sort candidates in decreasing order of their smallest prime factor.
    # If SPF is equal, sort by value (ascending).
    # This prioritizes primes (highest SPF) and prime powers.
    # 2 has SPF 2, 3 has SPF 3, ..., 127 has SPF 127.
    # 4 has SPF 2, 8 has SPF 2, etc.
    # So primes will be tried first, then prime powers, then composites.
    candidates = list(range(2, max_val + 1))
    candidates.sort(key=lambda x: (-spf[x], x))
    
    best_seq = []
    best_len = 0
    # We can pre-fill best_seq with a quick greedy result to have a baseline
    # Greedy by the sorted candidates
    temp_seq = []
    temp_prods = set()
    for x in candidates:
        if not temp_seq:
            temp_seq.append(x)
            temp_prods.add(x)
            continue
        # Check if x can be appended to temp_seq
        n = len(temp_seq)
        prod = 1
        ok = True
        new_p = []
        for i in range(n - 1, -1, -1):
            prod *= temp_seq[i]
            p = prod * x
            new_p.append(p)
            if p in temp_prods:
                ok = False
                break
        new_p.append(x)
        if x in temp_prods:
            ok = False
        if len(new_p) != len(set(new_p)):
            ok = False
        if ok:
            temp_seq.append(x)
            temp_prods.update(new_p)
    
    if len(temp_seq) > best_len:
        best_len = len(temp_seq)
        best_seq = temp_seq[:]
        
    # Backtracking search
    # State: current sequence, set of all block products
    # Pruning:
    # 1. If len(seq) + (128 - max(seq)) <= best_len, prune.
    #    (Max possible length is current len + remaining numbers in [max(seq)+1, 128])
    # 2. Time check.
    
    def backtrack(seq, prods):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
            
        current_len = len(seq)
        if current_len > best_len:
            best_len = current_len
            best_seq = seq[:]
            
        if not seq:
            last_val = 1
        else:
            last_val = seq[-1]
            
        # Pruning bound
        # Max possible length if we pick all remaining numbers
        max_possible = current_len + (max_val - last_val)
        if max_possible <= best_len:
            return
            
        # Try adding candidates
        # We must pick a candidate > last_val
        # To be efficient, we iterate through candidates that are > last_val
        # However, the sorted candidate list is by SPF, not by value.
        # We need to find candidates > last_val.
        # Since the space is small (128), we can just iterate through all candidates 
        # and check value constraint, or pre-filter.
        
        # Optimization: Only consider candidates that are > last_val.
        # Since we want to maximize length, and we are in a "branch and bound",
        # the order of trying candidates matters.
        # Trying "safer" candidates (high SPF) first is good for finding valid long sequences.
        
        for x in candidates:
            if time.time() > deadline:
                return
            if x <= last_val:
                continue
                
            # Check if x can be added
            n = len(seq)
            prod = 1
            ok = True
            new_p = []
            for i in range(n - 1, -1, -1):
                prod *= seq[i]
                p = prod * x
                new_p.append(p)
                if p in prods:
                    ok = False
                    break
            new_p.append(x)
            if x in prods:
                ok = False
            if len(new_p) != len(set(new_p)):
                ok = False
                
            if ok:
                new_seq = seq + [x]
                new_prods = prods.union(new_p)
                backtrack(new_seq, new_prods)
                
                # Optional: If we found a solution that saturates the bound, we could stop
                # But we don't know the true max.
                
    # Start backtracking
    # To improve performance, we can start with the greedy result as a "seed" for the best,
    # but the backtrack function will explore from empty.
    # Actually, starting from empty explores 2^127 space.
    # We should use the greedy result to set a high bar for best_len.
    # The pruning `max_possible <= best_len` will cut off many branches early.
    
    # However, backtracking from empty with the sorted candidates (primes first)
    # might be slow if it tries to build a sequence of primes and gets stuck.
    # Let's try to initialize the backtrack with the greedy sequence?
    # No, standard backtrack is from empty.
    # But we can also try to extend the greedy solution further?
    # The greedy solution is already "maximal" in the sense that no single number can be appended.
    # But maybe a different order allows a longer sequence.
    
    # Let's run the backtrack.
    # To speed up, we can limit the depth or use a heuristic.
    # Given the small N=128, and the strong pruning, it should be manageable.
    
    backtrack([], set())
    
    return best_seq

# Quick test to ensure it doesn't hang or crash
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
