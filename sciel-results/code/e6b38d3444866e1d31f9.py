"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _sieve_primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target L to cover. The task says L >= 999 is the goal. 
    # We aim for slightly higher to be safe, but 999 is the threshold.
    # Let's target L_target = 1200 to allow some slack, but verify coverage for 1..L.
    # Actually, the score is the largest L such that 1..L is covered.
    # If we cover 1..1200, score is at least 1200.
    # The budget is sum(1/p) <= 1.
    
    L_target = 1500  # Try to cover up to 1500, but we only need 1..999 to beat 1.0 significantly? 
    # Wait, the seed score is 1.0. The seed covers roughly 1..168 (primes 13..311).
    # Actually, let's check the seed's L.
    # Primes from 13 to 1000.
    # The seed uses p_i to cover i.
    # p_0 = 13 covers 1. p_1 = 17 covers 2. ...
    # This is a very inefficient use of budget.
    # 1/13 + 1/17 + ...
    
    # Strategy: Greedy Set Cover on the set of integers {1, ..., L_target}.
    # Universe U = {1, ..., L_target}
    # Candidates: All (p, r) where p is prime <= 1000, 0 <= r < p.
    # Cost: 1/p.
    # We want to select a set of classes such that sum(1/p) <= 1 and they cover U.
    # Greedy: At each step, pick the class (p, r) that covers the most *uncovered* elements in U,
    # such that adding it doesn't exceed the budget.
    
    # To make this feasible, we need to be smart.
    # Number of candidates: sum_{p<=1000} p ~ 76127.
    # L_target = 1500.
    # Uncovered set changes dynamically.
    
    # Optimization: 
    # 1. Precompute for each prime p, the residues r that are "useful".
    #    A residue r is useful if it covers at least one number in 1..L_target that is not yet covered.
    #    Actually, for a fixed p, the residue r that covers n is n % p.
    #    So for a fixed p, the set of covered numbers is {r, r+p, r+2p, ...}.
    
    # Let's try a simpler greedy:
    # Maintain `covered` as a boolean array of size L_target+1.
    # Maintain `budget_remaining` as Fraction.
    # In each iteration:
    #   For each prime p:
    #     Find the residue r that maximizes the number of *currently uncovered* numbers in 1..L_target 
    #     that are congruent to r mod p.
    #     Let this max count be c_p.
    #     If c_p > 0 and budget_remaining >= 1/p:
    #        Consider adding (p, r).
    #   Select the (p, r) that maximizes c_p (or maybe c_p / (1/p) ? No, just c_p is standard greedy).
    #   Actually, maximizing coverage per unit cost might be better? 
    #   But costs are 1/p. Small p has high cost. Large p has low cost but covers fewer points in range.
    #   For p > L_target, each residue covers at most 1 point. Cost 1/p.
    #   For p < L_target, each residue covers ~L_target/p points.
    
    # Let's stick to standard greedy: max new coverage.
    # Tie-breaker: lower cost (larger p) is better for budget preservation?
    # Or higher coverage is strictly better.
    
    primes = _sieve_primes(1000)
    # Sort primes? We iterate all primes each step.
    
    covered = [False] * (L_target + 1)
    selected_classes = []
    budget_used = Fraction(0)
    
    # Precompute ranges for each prime?
    # For a given p and r, the numbers covered are r, r+p, ... <= L_target.
    # Note: r is in 0..p-1. But we care about numbers 1..L_target.
    # If r==0, covered numbers are p, 2p, ...
    # If r>0, covered numbers are r, r+p, ...
    
    # To speed up "find best r for p":
    # We can maintain a count of uncovered numbers for each residue class of each prime.
    # Let counts[p][r] = number of uncovered n in 1..L_target such that n % p == r.
    # Initially, counts[p][r] = count of n in 1..L_target with n%p==r.
    # When we add a class (p0, r0), we mark those numbers as covered.
    # Then for ALL primes p, we must decrement counts[p][n%p] for each newly covered n.
    # This is O(L_target * pi(1000)) per step.
    # pi(1000) = 168.
    # L_target = 1500.
    # Steps: How many classes do we need?
    # Budget 1. Average cost?
    # If we use small primes, we cover many.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + ...
    # Sum of reciprocals of primes diverges slowly.
    # Sum up to 1000 is ~ 2.2.
    # So we can afford a subset.
    # How many steps? Maybe 10-20?
    # 20 * 1500 * 168 = 5,040,000 operations. This is fast in Python (seconds).
    
    # Initialize counts
    # counts is a dict: p -> list of size p
    counts = {}
    for p in primes:
        cnt = [0] * p
        # Count n in 1..L_target
        # n = r, r+p, ...
        # For r in 0..p-1:
        #   if r == 0: n = p, 2p, ...
        #   else: n = r, r+p, ...
        #   Number of terms: floor((L_target - r) / p) if r > 0 else floor(L_target / p)
        #   Wait, if r==0, first term is p. So terms are p, 2p, ..., k*p <= L_target.
        #   Count is floor(L_target / p).
        #   If r > 0, first term is r. Terms r, r+p, ...
        #   Count is floor((L_target - r) / p) + 1 if L_target >= r else 0.
        
        for r in range(p):
            if r == 0:
                cnt[r] = L_target // p
            else:
                if L_target >= r:
                    cnt[r] = (L_target - r) // p + 1
                else:
                    cnt[r] = 0
        counts[p] = cnt
        
    best_score_so_far = 0
    best_classes_so_far = []
    
    while time.time() < deadline:
        best_p = None
        best_r = None
        best_gain = 0
        
        # Find the (p, r) with max counts[p][r]
        for p in primes:
            # If budget is insufficient for this p, skip?
            # We need to check if budget_used + 1/p <= 1.
            if budget_used + Fraction(1, p) > 1:
                continue
            
            # Find max in counts[p]
            # counts[p] is a list.
            max_gain = max(counts[p])
            if max_gain > best_gain:
                best_gain = max_gain
                best_p = p
                best_r = counts[p].index(max_gain)
                
        if best_p is None or best_gain == 0:
            break
            
        # Select (best_p, best_r)
        selected_classes.append([best_p, best_r])
        budget_used += Fraction(1, best_p)
        
        # Update covered and counts
        p0 = best_p
        r0 = best_r
        
        # Identify newly covered numbers
        # Numbers n in 1..L_target such that n % p0 == r0 and not covered[n]
        # Note: if r0 == 0, n is multiple of p0.
        # if r0 > 0, n = r0, r0+p0, ...
        
        start_n = r0 if r0 > 0 else p0
        # Wait, if r0 == 0, the numbers are p0, 2p0, ...
        # If r0 > 0, the numbers are r0, r0+p0, ...
        # But we must ensure n >= 1.
        # If r0 == 0, start is p0 (which is >= 2).
        # If r0 > 0, start is r0 (which is >= 1).
        
        newly_covered = []
        for n in range(start_n, L_target + 1, p0):
            if not covered[n]:
                covered[n] = True
                newly_covered.append(n)
                
        # Update counts for all primes
        for p in primes:
            if p == p0:
                # We just used this residue.
                # The count for this residue should drop to 0?
                # No, other numbers in this residue class might have been covered by OTHER primes previously.
                # But we just covered ALL remaining uncovered numbers in this residue class.
                # So counts[p0][r0] should become 0.
                # However, we also need to update counts for OTHER residues of p0?
                # No, we only covered numbers with n % p0 == r0.
                # So counts[p0][r0] becomes 0.
                # Other counts[p0][r] for r != r0 are unchanged by this step?
                # Yes, because we didn't cover any n with n % p0 != r0.
                # BUT, we must check if any of the `newly_covered` numbers affect other primes.
                # The `newly_covered` list contains numbers n.
                # For each n, for each prime p, n % p is some residue r.
                # We must decrement counts[p][r] by 1.
                
                # We handle p0 specially? No, just process all p uniformly.
                # But for p0, we know we covered all of them.
                # Let's just iterate `newly_covered` and update all `counts`.
                pass # Handled in the loop below
                
        for n in newly_covered:
            for p in primes:
                r = n % p
                if counts[p][r] > 0:
                    counts[p][r] -= 1
                    
        # Check current coverage length
        # Find largest L such that 1..L is covered
        current_L = 0
        for i in range(1, L_target + 1):
            if covered[i]:
                current_L = i
            else:
                break
                
        if current_L > best_score_so_far:
            best_score_so_far = current_L
            best_classes_so_far = [c[:] for c in selected_classes]
            
        if current_L == L_target:
            break
            
    # Return the best classes found
    # Verify budget one last time
    total_budget = sum(Fraction(1, p) for p, r in best_classes_so_far)
    if total_budget > 1:
        # Should not happen, but safety
        return []
        
    return best_classes_so_far
# EVOLVE-BLOCK-END
