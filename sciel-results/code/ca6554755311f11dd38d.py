import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # We want to maximize L such that 1..L are all covered
    # Key insight: This is a covering problem. We need to find residue classes
    # such that every integer in 1..L is covered, with sum(1/p) <= 1.
    
    # The Erdős-Ruzsa result suggests that with primes up to x and C=1,
    # we can cover up to roughly x * log(x) or similar.
    # For x=1000, we want to get L >= 999.
    
    # Strategy: Use a greedy approach that focuses on extending the prefix 1..L.
    # At each step, find the (p, r) that covers the most uncovered numbers
    # in the range 1..(current_max + lookahead).
    
    # But we need to be smarter. Let's think about it differently:
    # We want to cover 1, 2, 3, ..., L.
    # For each prime p, choosing residue r means we cover r, r+p, r+2p, ...
    
    # A key observation: if we want to cover 1..L, and we use prime p with residue r,
    # then we cover all numbers congruent to r mod p in 1..L.
    # The number of such numbers is approximately L/p.
    
    # The budget constraint is sum(1/p) <= 1.
    # To maximize L, we want to use small primes (which have large 1/p) 
    # but that uses budget quickly. Alternatively, use many medium primes.
    
    # Let's try a different approach: binary search on L, and for each L,
    # check if there exists a set of (p,r) with sum(1/p) <= 1 that covers 1..L.
    # This is a set cover / hitting set problem, which is NP-hard in general.
    
    # For practical purposes, let's use a greedy heuristic that is very aggressive
    # about covering the prefix.
    
    # Approach: 
    # 1. Start with target L = 1000 (or higher).
    # 2. Use greedy: at each step, pick the (p,r) that covers the most 
    #    uncovered numbers in 1..L, weighted by 1/(1/p) = p (i.e., gain per unit budget).
    # 3. If we can't cover all of 1..L, reduce L and retry.
    
    # Actually, let's just do a direct greedy that tries to maximize the prefix.
    
    # For efficiency, we'll limit the range we consider.
    # Let's aim for L around 1000-2000 initially.
    
    MAX_SEARCH = 5000  # We'll try to cover up to this
    
    def greedy_cover(target_L, primes_list, budget_frac):
        """Greedy cover of 1..target_L with budget constraint."""
        covered = bytearray(target_L + 1)
        used_primes = set()
        classes = []
        budget = Fraction(budget_frac)
        
        # Sort primes by 1/p descending
        sorted_primes = sorted(primes_list, key=lambda p: -1/p)
        
        for _ in range(len(sorted_primes)):
            if time.time() - start_time > DEADLINE:
                break
            
            best_gain = 0
            best_p = None
            best_r = None
            best_cost = None
            
            for p in sorted_primes:
                if p in used_primes:
                    continue
                
                cost = Fraction(1, p)
                if budget - cost < 0:
                    continue
                
                # Find best residue for this prime
                best_r_for_p = -1
                best_count_for_p = 0
                
                for r in range(p):
                    count = 0
                    n = r if r >= 1 else r + p
                    while n <= target_L:
                        if not covered[n]:
                            count += 1
                        n += p
                    
                    if count > best_count_for_p:
                        best_count_for_p = count
                        best_r_for_p = r
                
                if best_r_for_p >= 0 and best_count_for_p > best_gain:
                    best_gain = best_count_for_p
                    best_p = p
                    best_r = best_r_for_p
                    best_cost = cost
            
            if best_p is None or best_gain == 0:
                break
            
            p = best_p
            r = best_r
            used_primes.add(p)
            classes.append([p, r])
            budget -= best_cost
            
            # Update covered
            n = r if r >= 1 else r + p
            while n <= target_L:
                covered[n] = 1
                n += p
        
        return classes, covered
    
    # Try to cover progressively larger prefixes
    # Start with a reasonable target and increase
    
    best_classes = []
    best_L = 0
    
    # First, try to cover 1..1000
    for target_L in [1000, 1500, 2000, 3000, 4000, 5000]:
        if time.time() - start_time > DEADLINE:
            break
        
        classes, covered = greedy_cover(target_L, primes, Fraction(1, 1))
        
        # Check how much of the prefix is actually covered
        L = 0
        for i in range(1, target_L + 1):
            if covered[i]:
                L = i
            else:
                break
        
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Now try to improve with local search: try swapping primes
    # For each prime in the solution, try replacing it with another prime
    # and see if we can extend the prefix further
    
    if best_classes:
        # Local search improvement
        for _ in range(50):  # limit iterations
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            for i in range(len(best_classes)):
                p_old = best_classes[i][0]
                r_old = best_classes[i][1]
                
                # Try removing this class and replacing with another
                for p_new in primes:
                    if p_new == p_old:
                        continue
                    if any(c[0] == p_new for c in best_classes):
                        continue
                    
                    # Check budget
                    old_cost = Fraction(1, p_old)
                    new_cost = Fraction(1, p_new)
                    total_cost = sum(Fraction(1, c[0]) for c in best_classes)
                    if total_cost - old_cost + new_cost > Fraction(1, 1):
                        continue
                    
                    # Try all residues for p_new
                    # Rebuild coverage without p_old
                    temp_classes = [c for j, c in enumerate(best_classes) if j != i]
                    
                    # Find best residue for p_new
                    target = best_L + 100  # look ahead
                    covered_temp = bytearray(target + 1)
                    for c in temp_classes:
                        p, r = c
                        n = r if r >= 1 else r + p
                        while n <= target:
                            covered_temp[n] = 1
                            n += p
                    
                    best_r_new = -1
                    best_gain_new = 0
                    for r in range(p_new):
                        gain = 0
                        n = r if r >= 1 else r + p_new
                        while n <= target:
                            if not covered_temp[n]:
                                gain += 1
                            n += p_new
                        if gain > best_gain_new:
                            best_gain_new = gain
                            best_r_new = r
                    
                    if best_r_new >= 0:
                        # Check if this improves the prefix
                        new_classes = temp_classes + [[p_new, best_r_new]]
                        # Rebuild coverage
                        covered_check = bytearray(target + 1)
                        for c in new_classes:
                            p, r = c
                            n = r if r >= 1 else r + p
                            while n <= target:
                                covered_check[n] = 1
                                n += p
                        
                        L_new = 0
                        for i in range(1, target + 1):
                            if covered_check[i]:
                                L_new = i
                            else:
                                break
                        
                        if L_new > best_L:
                            best_L = L_new
                            best_classes = new_classes
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
    
    # Final verification and return
    if best_classes:
        # Verify the solution
        total_cost = sum(Fraction(1, c[0]) for c in best_classes)
        if total_cost <= Fraction(1, 1):
            # Verify coverage
            max_check = max(best_L, 1000)
            covered_final = bytearray(max_check + 1)
            for p, r in best_classes:
                n = r if r >= 1 else r + p
                while n <= max_check:
                    covered_final[n] = 1
                    n += p
            
            L_verified = 0
            for i in range(1, max_check + 1):
                if covered_final[i]:
                    L_verified = i
                else:
                    break
            
            return best_classes
        else:
            # Budget exceeded, return empty or fallback
            pass
    
    # Fallback: return a simple solution
    # Use small primes with residues that cover 1..something
    primes_small = [p for p in primes if p <= 100]
    classes = []
    budget = Fraction(1, 1)
    for p in primes_small:
        cost = Fraction(1, p)
        if budget - cost < 0:
            break
        # Choose residue to cover 1 mod p
        r = 1 % p
        classes.append([p, r])
        budget -= cost
    
    return classes
