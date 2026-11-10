import time

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    n = 5040
    target = 4979
    divs = _get_divisors(n)
    
    # Greedy fallback
    remainder = target
    greedy_chosen = []
    for d in divs:
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
    if remainder != 0:
        return greedy_chosen
        
    best_solution = greedy_chosen
    best_k = len(greedy_chosen)
    
    # Check k=1
    if target in divs:
        return [target]
        
    # Check k=2
    div_set = set(divs)
    found_2 = False
    for d in divs:
        if target - d in div_set and target - d != d:
            return [d, target - d]
            
    # Check k=3
    # Iterate d1, check if target-d1 is sum of 2 distinct divisors not d1
    # Precompute 2-sums? 
    # Number of divs is 60. 2-sums is ~1800.
    # We can just iterate pairs for each d1? 60 * 1800 = 108000. Fast.
    
    # But we need distinctness.
    # Let's use a set of (d1, d2) sums? No, just check if (target - d1) can be formed by 2 divs from divs \ {d1}.
    
    # Efficient k=3 check:
    # Create a set of all divisors.
    # For each d1 in divs:
    #   rem = target - d1
    #   Check if rem is sum of 2 distinct divisors from divs, neither equal to d1.
    #   To check 2-sums, iterate d2 in divs:
    #      if d2 == d1: continue
    #      if d2 > rem: continue (if sorted desc, break? No, divs is desc, so if d2 > rem, subsequent are smaller, so we can't break, we must continue until d2 < rem? No, if d2 > rem, d2 + d3 > rem since d3 > 0. So if d2 > rem, we can break? 
    #      Wait, divs is sorted descending.
    #      If d2 > rem, then d2 + d3 > rem. So we can break the inner loop?
    #      No, d3 must be positive. So yes, if d2 > rem, no pair with d2 can work.
    #      But we are looking for d2 + d3 = rem.
    #      So if d2 > rem, break.
    #      Also d2 must be < rem (since d3 > 0).
    #      And d3 = rem - d2.
    #      Check if d3 in div_set, d3 != d2, d3 != d1.
    
    for d1 in divs:
        if time.time() > deadline: break
        rem = target - d1
        if rem <= 0: continue
        # d2 must be < rem
        for d2 in divs:
            if d2 >= rem: continue # d2 + d3 = rem, d3 > 0 => d2 < rem
            if d2 == d1: continue
            d3 = rem - d2
            if d3 in div_set and d3 != d2 and d3 != d1:
                return [d1, d2, d3]
                
    # Check k=4
    # MITM: 2 sums.
    # Generate all pairs (i, j) with i < j.
    # Store sum -> list of pairs.
    # But we need distinctness with the other pair.
    
    pair_sums = {}
    for i in range(len(divs)):
        if time.time() > deadline: break
        for j in range(i+1, len(divs)):
            s = divs[i] + divs[j]
            if s > target: continue
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
            
    # Now for each pair (k, l), check if target - (divs[k]+divs[l]) is in pair_sums
    # And the pairs are disjoint.
    for k in range(len(divs)):
        if time.time() > deadline: break
        for l in range(k+1, len(divs)):
            s1 = divs[k] + divs[l]
            rem = target - s1
            if rem < 0: continue
            if rem in pair_sums:
                for (i, j) in pair_sums[rem]:
                    # Check disjointness
                    if i != k and i != l and j != k and j != l:
                        return [divs[k], divs[l], divs[i], divs[j]]
                        
    return best_solution
