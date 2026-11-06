"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _sieve(1000)
    N = 100000
    
    # We will use a count array to track coverage.
    # cover[i] = number of primes covering i.
    # We need cover[i] > 0 for all i in 1..L.
    
    # Pre-compute masks for efficiency? 
    # For a prime p, the mask for residue r is 1s at i where i%p==r.
    # Creating these on the fly is O(N/p). 
    # Since we have at most ~10-20 primes, and we do many local search steps,
    # it's better to pre-compute the indices for the *selected* primes only, 
    # or just use vectorized operations on the full range if possible.
    # Actually, for local search, we change one residue at a time.
    # Let's stick to the index array approach but optimize the selection.
    
    # Dynamic Prime Selection Strategy:
    # 1. Start with a core set that is known to be good: {2, 3, 7}. 
    #    Sum: 1/2 + 1/3 + 1/7 = 0.5 + 0.333 + 0.142 = 0.975.
    #    This leaves very little budget. 
    #    Alternative: {2, 5, 7, 11} -> 0.5 + 0.2 + 0.142 + 0.09 = 0.932.
    #    Alternative: {2, 5, 7, 11, 13} -> 0.932 + 0.076 = 1.008 (Too much).
    #    So {2, 5, 7, 11} is a strong candidate with 4 primes.
    #    {2, 3, 7} has 3 primes.
    #    More primes generally allow finer gap filling.
    
    # Let's try to construct a set greedily by "efficiency" = coverage_gain / cost.
    # But coverage gain depends on current state.
    
    # Strategy:
    # 1. Generate a few candidate prime sets.
    # 2. For each set, run a fast directed local search.
    
    candidate_sets = []
    
    # Set 1: {2, 3, 7}
    candidate_sets.append([2, 3, 7])
    
    # Set 2: {2, 5, 7, 11}
    candidate_sets.append([2, 5, 7, 11])
    
    # Set 3: {2, 5, 7, 11, 13} is invalid (>1). 
    # What about {3, 5, 7, 11, 13, 17}?
    # 1/3+1/5+1/7+1/11+1/13+1/17 = 0.333+0.2+0.142+0.09+0.076+0.058 = 0.90.
    # This has 6 primes! More flexibility.
    candidate_sets.append([3, 5, 7, 11, 13, 17])
    
    # Set 4: {2, 3, 5} is invalid.
    # Set 5: {2, 3, 11, 13}?
    # 0.5 + 0.333 + 0.09 + 0.076 = 0.999.
    candidate_sets.append([2, 3, 11, 13])
    
    # Set 6: {2, 5, 11, 13, 17}?
    # 0.5 + 0.2 + 0.09 + 0.076 + 0.058 = 0.924.
    candidate_sets.append([2, 5, 11, 13, 17])
    
    # Set 7: {3, 5, 7, 11, 13, 17, 19}?
    # 0.90 + 0.052 = 0.952.
    candidate_sets.append([3, 5, 7, 11, 13, 17, 19])
    
    best_L_global = 0
    best_pairs_global = []
    
    for ps in candidate_sets:
        # Check budget
        budget_sum = sum(Fraction(1, p) for p in ps)
        if budget_sum > 1:
            continue
            
        # Pre-compute indices for this specific set of primes
        # To save memory and time, we only store indices for the primes in this set
        indices = {}
        for p in ps:
            indices[p] = []
            for r in range(p):
                start = r if r > 0 else p
                if start <= N:
                    indices[p].append(np.arange(start, N + 1, p))
                else:
                    indices[p].append(np.array([], dtype=int))
        
        # Initialize coverage
        cover_count = np.zeros(N + 1, dtype=np.uint8)
        
        # Greedy Initialization of Residues
        # We want to cover 1..L as much as possible.
        # Start with L=0.
        # For each prime, pick the residue that maximizes the new L.
        
        assigned = {}
        current_L = 0
        
        # Order primes: Smaller primes cover more numbers, so place them first?
        # Or place larger primes first to fill specific gaps?
        # Let's try placing in increasing order.
        
        for p in sorted(ps):
            best_r = 0
            best_L_for_p = 0
            
            # Try all residues
            for r in range(p):
                idx = indices[p][r]
                if len(idx) == 0:
                    continue
                # Temporarily add
                cover_count[idx] += 1
                
                # Calculate new L
                # We can optimize this: if current_L is 0, just find first 0.
                # If current_L > 0, we only care if we extended it or if we broke it.
                # But for init, let's just compute full L.
                # Optimization: We know 1..current_L was covered. 
                # If we add a prime, 1..current_L remains covered (unless we remove, which we don't in init).
                # So new_L >= current_L.
                # We just need to find the first gap after current_L.
                
                # Check if current_L is still valid (it is, because we only added coverage).
                # Find first i > current_L such that cover_count[i] == 0.
                # If no such i up to N, new_L = N.
                
                # To make this fast, we can check segments.
                # But N=10^5 is small. 
                # Let's use a helper that finds the first zero in cover_count[start:]
                
                # Actually, we can just do a linear scan from current_L+1.
                # Worst case O(N) per residue. 
                # Sum over p: O(N * sum(p)). 
                # For p=19, 19*10^5 = 1.9M ops. 
                # For 7 primes, ~10M ops. This is fine in Python if vectorized? 
                # No, linear scan in Python is slow. 
                # Let's use numpy: np.flatnonzero(cover_count[current_L+1:] == 0)
                
                # But cover_count is uint8.
                
                # Efficient check:
                # We know 1..current_L are > 0.
                # We look for the first 0 in current_L+1 .. N.
                # If none, L=N.
                
                # To avoid slicing overhead, we can maintain a "next_gap" pointer?
                # No, adding coverage can fill gaps, so next_gap can move forward.
                # But it can never move backward.
                # So we can keep a pointer `ptr` that only increases.
                
                # Let's implement a fast L calculator using the pointer.
                
                # However, during the "try residue" phase, we are modifying cover_count.
                # We can't easily use a persistent pointer if we revert.
                # But in init, we don't revert, we commit.
                
                # Let's just use the numpy method for init simplicity, it's fast enough.
                
                zeros = np.flatnonzero(cover_count[current_L + 1:] == 0)
                if len(zeros) == 0:
                    new_L = N
                else:
                    new_L = current_L + 1 + zeros[0]
                
                if new_L > best_L_for_p:
                    best_L_for_p = new_L
                    best_r = r
                    
                # Revert
                cover_count[idx] -= 1
            
            # Commit best
            assigned[p] = best_r
            idx = indices[p][best_r]
            if len(idx) > 0:
                cover_count[idx] += 1
            current_L = best_L_for_p
            
        # Now we have a good starting point.
        if current_L > best_L_global:
            best_L_global = current_L
            best_pairs_global = [(p, assigned[p]) for p in ps]
            
        # Directed Local Search
        # Goal: Extend L.
        # Current state: cover_count, assigned residues, current_L.
        
        # We maintain a list of active primes.
        active_primes = list(ps)
        
        # Helper to get L from cover_count starting from a known valid prefix
        # We assume 1..current_L are covered.
        
        while time.time() < deadline - 5:
            # If we reached N, we can't do better.
            if current_L >= N:
                break
                
            # The gap is at current_L + 1.
            target = current_L + 1
            
            # Try to find a move that covers `target`.
            # A move consists of changing the residue of one prime p.
            # For each prime p, we need to find a residue r such that:
            # 1. target % p == r (so target is covered by p)
            # 2. Changing p's residue from old_r to r does not uncover any i in 1..current_L.
            #    This is true if for all i in 1..current_L, cover_count[i] > 1 
            #    OR i is not in the old residue class of p.
            #    Actually, if i is in old_r class, cover_count[i] decreases by 1.
            #    So we need cover_count[i] > 1 for all i in 1..current_L such that i % p == old_r.
            
            found_move = False
            new_L = current_L
            
            # To speed up, we can pre-compute for each prime p, the set of residues that cover `target`.
            # There is exactly one such residue: r_target = target % p.
            
            # So for each p, we only need to test ONE candidate residue: r_target.
            # We check if switching p to r_target is valid (doesn't break prefix).
            
            # Pre-compute "critical" indices for each prime?
            # For prime p, the indices in 1..current_L that are in residue old_r are:
            # old_r, old_r+p, ... up to current_L.
            # We need to check if any of these have cover_count == 1.
            
            # We can maintain for each prime p, a count of how many i in 1..current_L 
            # are covered ONLY by p? 
            # That's expensive to update.
            
            # Alternative: Just check the specific indices.
            # Number of such indices is current_L / p.
            # For p=2, 50000 checks. 
            # For 7 primes, ~100k checks per iteration.
            # This is fast in Python if we use numpy? 
            # We can use the indices array!
            
            # For prime p, old_r = assigned[p].
            # The indices in 1..current_L covered by p in old_r are:
            # indices[p][old_r] intersect [1, current_L].
            # We can get this via binary search or slicing.
            
            # Let's just iterate through the candidates.
            
            # Optimization: Sort primes by size? Smaller primes have more indices to check.
            # Larger primes have fewer indices. 
            # It's likely that changing a large prime is "safer" (fewer indices to check for criticality).
            # But large primes are less likely to be the "bottleneck" for coverage?
            # Actually, to cover `target`, any prime can do it.
            
            # Let's try primes in random order or increasing order.
            
            # To make this faster, we can pre-compute a "safety" mask?
            # Let's just do the check.
            
            for p in active_primes:
                old_r = assigned[p]
                new_r = target % p
                
                if new_r == old_r:
                    # Already covers target. But we are at a gap, so it shouldn't?
                    # Wait, if p covers target, and target is a gap, then cover_count[target] == 0.
                    # If p covers target, then cover_count[target] should be > 0.
                    # Contradiction. So new_r != old_r is guaranteed if target is a gap?
                    # No, if p covers target, then target is NOT a gap.
                    # So if target is a gap, NO prime covers it.
                    # So for all p, target % p != assigned[p].
                    # So new_r != old_r always.
                    
                    # Wait, is it possible that p covers target but cover_count[target] is 0?
                    # No, cover_count is the sum of all primes.
                    # So if target is a gap, no prime covers it.
                    # So we must change some prime to cover it.
                    
                    pass
                
                # Check if switching p from old_r to new_r is valid.
                # We need to ensure that for all i in 1..current_L, 
                # if i % p == old_r, then cover_count[i] > 1.
                
                # Get indices of i in 1..current_L such that i % p == old_r.
                # indices[p][old_r] contains all such i up to N.
                # We need the subset <= current_L.
                
                # Since indices are sorted, we can find the end.
                idx_arr = indices[p][old_r]
                if len(idx_arr) == 0:
                    # p doesn't cover anything in 1..N? Unlikely for small p.
                    is_valid = True
                else:
                    # Find how many are <= current_L
                    # Use binary search
                    # np.searchsorted(idx_arr, current_L, side='right')
                    end_idx = np.searchsorted(idx_arr, current_L, side='right')
                    if end_idx == 0:
                        is_valid = True
                    else:
                        # Check cover_count for these indices
                        # We need min(cover_count[idx_arr[:end_idx]]) > 1
                        # If min is 1, then switching p would leave some i with count 0.
                        # So we need min > 1.
                        # Actually, we need min >= 2.
                        
                        # Vectorized min
                        min_count = np.min(cover_count[idx_arr[:end_idx]])
                        is_valid = (min_count > 1)
                
                if is_valid:
                    # Apply the move
                    # Remove old coverage
                    idx_old = indices[p][old_r]
                    if len(idx_old) > 0:
                        cover_count[idx_old] -= 1
                        
                    # Add new coverage
                    idx_new = indices[p][new_r]
                    if len(idx_new) > 0:
                        cover_count[idx_new] += 1
                        
                    # Update assigned
                    assigned[p] = new_r
                    
                    # Update current_L
                    # We know 1..current_L is still covered.
                    # We know target (current_L+1) is now covered.
                    # We need to find the new gap.
                    
                    # Check forward from target
                    # We can use the same numpy trick
                    zeros = np.flatnonzero(cover_count[target:] == 0)
                    if len(zeros) == 0:
                        new_L = N
                    else:
                        new_L = target + zeros[0] - 1
                        
                    current_L = new_L
                    found_move = True
                    break # Restart the loop with new L
            
            if not found_move:
                # No single move can extend L.
                # This means we are stuck in a local optimum where no single prime change helps.
                # We might need to make a "sacrificial" move that temporarily reduces L 
                # or keeps it same but changes the configuration to allow future moves.
                # Or, we might need to change the prime set.
                
                # For now, let's try a random perturbation to escape.
                # Pick a random prime and change its residue to a random one.
                # Accept if L doesn't decrease too much? 
                # Or just accept if L stays same.
                
                # Let's try to find a move that keeps L same but changes the "frontier".
                # This is hard.
                
                # Alternative: Try a different prime set.
                break
        
        # End while for this prime set
        
    return best_pairs_global

# EVOLVE-BLOCK-END
