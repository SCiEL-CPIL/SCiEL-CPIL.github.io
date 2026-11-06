import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for the 120s limit (or 150s if that's the hard kill)
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Strategy: Select primes to maximize coverage density while respecting budget sum(1/p) <= 1.
    # Smaller primes are better for coverage.
    # The sum of reciprocals of primes 13..997 is approx 0.94.
    # We can afford to include 11 (cost 1/11 ~ 0.09) by dropping a few large primes or just checking if it fits.
    # Let's start with primes >= 13, which is a safe baseline that fits the budget.
    # Sum_{p>=13} 1/p <= Sum_{p>=11} 1/p - 1/11. 
    # Sum_{p<1000} 1/p is approx 2.0. 
    # Sum_{p=2,3,5,7,11} 1/p = 0.5 + 0.333 + 0.2 + 0.142 + 0.090 = 1.265.
    # So sum_{p>=13} 1/p = 2.0 - 1.265 = 0.735? No, Mertens theorem says sum_{p<=x} 1/p = log log x + M.
    # log log 1000 + 0.261 = log(6.9) + 0.261 = 1.93 + 0.261 = 2.19.
    # Let's just compute the exact cost for primes >= 13.
    
    # A better set might include 11, 7, 5, 3 if we can drop enough large primes.
    # However, dropping many large primes reduces the "resolution" of the covering system.
    # The inspiration (score 2.67) used primes >= 13.
    # Let's stick to primes >= 13 for stability and focus on optimizing residues.
    # We can try to add 11 if there is budget.
    
    selected_primes = [p for p in primes if p >= 13]
    cost = sum(Fraction(1, p) for p in selected_primes)
    
    # Try to add 11
    if cost + Fraction(1, 11) <= Fraction(1):
        selected_primes.append(11)
        cost += Fraction(1, 11)
        selected_primes.sort()
        
    # Try to add 7?
    if cost + Fraction(1, 7) <= Fraction(1):
        selected_primes.append(7)
        cost += Fraction(1, 7)
        selected_primes.sort()
        
    # Try to add 5?
    if cost + Fraction(1, 5) <= Fraction(1):
        selected_primes.append(5)
        cost += Fraction(1, 5)
        selected_primes.sort()

    # Verify budget
    final_cost = sum(Fraction(1, p) for p in selected_primes)
    assert final_cost <= Fraction(1), f"Budget exceeded: {final_cost}"
    
    N = len(selected_primes)
    MAX_L = 10000 # We want to find L as large as possible, capped at 10^5 by problem, but 10000 is a good search horizon
    
    # Data structure for coverage
    # counts[n] = number of primes covering n
    counts = [0] * (MAX_L + 2)
    
    def apply_change(i, old_r, new_r):
        p = selected_primes[i]
        # Decrement old
        # Note: residue 0 means multiples of p (p, 2p, ...). 
        # The problem says 0 <= r < p.
        # If r=0, the class is {k*p | k in Z}.
        # The integers 1..L are covered if n % p == r.
        # So if r=0, n is covered if n % p == 0.
        
        n = old_r
        if n == 0:
            n = p # First positive multiple
        while n <= MAX_L:
            counts[n] -= 1
            n += p
        
        n = new_r
        if n == 0:
            n = p
        while n <= MAX_L:
            counts[n] += 1
            n += p

    def calc_L():
        l = 0
        while l < MAX_L and counts[l+1] > 0:
            l += 1
        return l

    # Initialize residues
    # Random initialization
    residues = [random.randint(0, p - 1) for p in selected_primes]
    
    # Initialize counts
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        n = r
        if n == 0:
            n = p
        while n <= MAX_L:
            counts[n] += 1
            n += p

    L = calc_L()
    best_L = L
    best_residues = residues[:]

    # Helper to run greedy extension from current state
    # This is the "repair" mechanism. It tries to fix the first hole (L+1) by moving a prime's residue.
    def greedy_extend(current_L):
        extended = True
        L_local = current_L
        # Limit iterations to avoid infinite loops if stuck
        max_iters = 10000
        iters = 0
        
        while extended and time.time() < deadline - 5 and iters < max_iters:
            extended = False
            iters += 1
            # Shuffle to avoid bias
            order = list(range(N))
            random.shuffle(order)
            
            # We want to cover L_local + 1.
            # We can change the residue of any prime p_i such that (L_local + 1) % p_i is a valid residue.
            # Actually, we can change residue of ANY prime to cover L_local + 1, provided it doesn't break coverage of 1..L_local.
            
            target_n = L_local + 1
            
            for i in order:
                if time.time() > deadline - 5:
                    break
                p = selected_primes[i]
                r_old = residues[i]
                # To cover target_n with prime p, we need r_new such that target_n % p == r_new % p.
                # Since 0 <= r < p, r_new is uniquely determined as target_n % p.
                r_new = target_n % p
                
                if r_old == r_new:
                    continue

                # Check safety: does changing r_old to r_new break any n <= L_local?
                # We need counts[n] > 1 for all n covered by old_r and n <= L_local.
                # Also, we must ensure that the new residue doesn't "uncover" something that was only covered by this prime.
                # Wait, changing r_old to r_new removes coverage from old_r's class and adds to new_r's class.
                # So we must ensure that for all n in old_r's class (n <= L_local), counts[n] > 1.
                
                safe = True
                n = r_old
                if n == 0:
                    n = p
                while n <= L_local:
                    if counts[n] == 1:
                        safe = False
                        break
                    n += p

                if safe:
                    apply_change(i, r_old, r_new)
                    residues[i] = r_new
                    
                    # Extend L
                    while L_local < MAX_L and counts[L_local+1] > 0:
                        L_local += 1
                    
                    extended = True
                    break # Restart search with new L
        
        return L_local

    # Main Search Loop: Iterated Local Search
    # We keep the best state found so far.
    # If stuck, we perturb the current state and try to repair.
    
    while time.time() < deadline - 10:
        # Try to extend from current L
        new_L = greedy_extend(L)
        
        if new_L > L:
            L = new_L
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        else:
            # Stuck, need to kick
            # Save state
            save_residues = residues[:]
            save_counts = counts[:]
            save_L = L
            
            # Perform a kick: change K random residues
            K = random.randint(1, 3)
            for _ in range(K):
                i_kick = random.randint(0, N - 1)
                p_kick = selected_primes[i_kick]
                old_r_kick = residues[i_kick]
                new_r_kick = random.randint(0, p_kick - 1)
                
                if old_r_kick != new_r_kick:
                    apply_change(i_kick, old_r_kick, new_r_kick)
                    residues[i_kick] = new_r_kick
            
            # Calculate new L after kick
            L_temp = calc_L()
            
            # Try to repair/extend from L_temp
            L_repaired = greedy_extend(L_temp)
            
            # Decision: Accept or Revert?
            if L_repaired > best_L:
                L = L_repaired
                best_L = L
                best_residues = residues[:]
            elif L_repaired >= L:
                # We are at least as good as before the kick
                L = L_repaired
            else:
                # Revert
                residues[:] = save_residues
                counts[:] = save_counts
                L = save_L

    # Final Check
    # Ensure the result is valid
    final_cost = sum(Fraction(1, p) for p in selected_primes)
    if final_cost > Fraction(1, 1):
        # Fallback: This is a hard error, but we should return something valid.
        # Since we checked at the start, this shouldn't happen.
        pass

    # Construct result
    return [[selected_primes[i], best_residues[i]] for i in range(N)]
