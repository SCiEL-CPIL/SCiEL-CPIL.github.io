import time
import random
import math
from fractions import Fraction

def construct():
    t0 = time.time()
    # Safety margin: The evaluator kills at 150s. 
    # We should return well before that to avoid timeouts in the harness.
    # 120s is a good target for the inner loop, leaving 30s for overhead/safety.
    deadline = t0 + 115.0 

    # Generate primes up to 1000
    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    
    # Select primes with budget sum(1/p) <= 1
    # Primes >= 13 have sum(1/p) approx 0.94
    selected_primes = [p for p in all_primes if p >= 13]
    N = len(selected_primes)
    
    # Verify budget exactly
    cost = sum(Fraction(1, p) for p in selected_primes)
    if cost > Fraction(1, 1):
        # This shouldn't happen for primes >= 13, but just in case
        # Remove smallest primes (largest 1/p) until fits
        # Note: Removing small primes reduces coverage capacity significantly.
        # But for primes >= 13, it is safe.
        while cost > Fraction(1, 1) and selected_primes:
            # Remove the smallest prime (largest reciprocal)
            min_idx = 0
            min_p = selected_primes[0]
            for i, p in enumerate(selected_primes):
                if p < min_p:
                    min_p = p
                    min_idx = i
            cost -= Fraction(1, selected_primes[min_idx])
            selected_primes.pop(min_idx)
            N -= 1

    # Max L to track. 
    # With 160 primes, we can cover much more than 1000.
    # Score 2.65 corresponds to L ~ 2650.
    # Let's track up to 10000 to be safe for jumps.
    MAX_L = 10000
    
    # Data structure for coverage
    # counts[n] = number of primes covering n
    counts = [0] * (MAX_L + 2)
    
    def apply_change(i, old_r, new_r):
        p = selected_primes[i]
        # Decrement old
        n = old_r if old_r != 0 else p
        while n <= MAX_L:
            counts[n] -= 1
            n += p
        # Increment new
        n = new_r if new_r != 0 else p
        while n <= MAX_L:
            counts[n] += 1
            n += p

    def calc_L():
        l = 0
        # counts[0] is not used for coverage of 1..L, but we check counts[l+1]
        while l < MAX_L and counts[l+1] > 0:
            l += 1
        return l

    # Initialize residues
    # Strategy: Diagonal start
    residues = [0] * N
    for i in range(N):
        residues[i] = (i + 1) % selected_primes[i]

    # Initialize counts
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        n = r if r != 0 else p
        while n <= MAX_L:
            counts[n] += 1
            n += p

    L = calc_L()
    best_L = L
    best_residues = residues[:]

    # Helper to run greedy extension from current state
    # Returns the new L and updates residues/counts in place
    def greedy_extend(current_L):
        extended = True
        L_local = current_L
        while extended and time.time() < deadline:
            extended = False
            # Shuffle to avoid bias
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline:
                    break
                p = selected_primes[i]
                r_old = residues[i]
                target_r = (L_local + 1) % p

                if r_old == target_r:
                    continue

                # Check safety: does changing r_old break any n <= L_local?
                # We need counts[n] > 1 for all n covered by old_r and n <= L_local
                safe = True
                n = r_old if r_old != 0 else p
                while n <= L_local:
                    if counts[n] == 1:
                        safe = False
                        break
                    n += p

                if safe:
                    apply_change(i, r_old, target_r)
                    residues[i] = target_r
                    
                    # Extend L
                    while L_local < MAX_L and counts[L_local+1] > 0:
                        L_local += 1
                    
                    extended = True
                    break # Restart search with new L
        
        return L_local

    # Main Search Loop: Iterated Local Search
    # We keep the best state found so far.
    # If stuck, we perturb the current state (or best state?) and try again.
    # Perturbing the *current* state is standard ILS.
    # Perturbing the *best* state can help if we are far from best.
    # Let's perturb the current state.
    
    # To avoid wasting time, we limit the number of "stuck" iterations per kick
    stuck_counter = 0
    
    while time.time() < deadline:
        # Try to extend from current L
        new_L = greedy_extend(L)
        
        if new_L > L:
            L = new_L
            stuck_counter = 0
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        else:
            # Stuck
            stuck_counter += 1
            
            # If we are stuck, we need to kick.
            # How often to kick? 
            # If we just failed to extend, we are stuck.
            # Let's kick immediately.
            
            # Save state to allow revert if kick is bad
            save_residues = residues[:]
            save_counts = counts[:]
            save_L = L
            
            # Perform a kick: change K random residues
            # K=1 is a single step. K=2 or 3 helps escape deeper minima.
            # Let's use K=2 or 3 randomly.
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
            # If L_repaired > best_L, definitely accept.
            # If L_repaired >= L (current L before kick), accept (we didn't lose progress).
            # If L_repaired < L, revert.
            
            if L_repaired > best_L:
                L = L_repaired
                best_L = L
                best_residues = residues[:]
                stuck_counter = 0
            elif L_repaired >= L:
                # We are at least as good as before the kick
                L = L_repaired
                # If we improved over best_L, update
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
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
