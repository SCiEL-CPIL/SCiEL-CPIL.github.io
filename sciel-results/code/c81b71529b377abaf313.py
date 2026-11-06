"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _covered_length(covered_arr):
    """Return the largest L such that all 1..L are in covered_arr."""
    L = 0
    for i in range(1, len(covered_arr)):
        if covered_arr[i]:
            L = i
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    primes = _sieve(1000)
    MAX_L = 10**5
    covered_arr = bytearray(MAX_L + 1)
    
    selected = []
    budget_used = Fraction(0, 1)
    inv_primes = [Fraction(1, p) for p in primes]
    
    # Greedy set cover: at each step, pick (p, r) that maximizes new coverage per budget
    while True:
        if time.time() > deadline:
            break
        
        best_efficiency = -1
        best_p_idx = -1
        best_r = -1
        best_new_count = 0
        
        for p_idx, p in enumerate(primes):
            if budget_used + inv_primes[p_idx] > 1:
                continue
            
            # For each residue, count new coverage
            best_local_count = -1
            best_local_r = -1
            
            for r in range(p):
                start = r if r > 0 else p
                count = 0
                for n in range(start, MAX_L + 1, p):
                    if not covered_arr[n]:
                        count += 1
                
                if count > best_local_count:
                    best_local_count = count
                    best_local_r = r
            
            # Efficiency = new_count / (1/p) = new_count * p
            if best_local_count > 0:
                efficiency = best_local_count * p
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_p_idx = p_idx
                    best_r = best_local_r
                    best_new_count = best_local_count
        
        if best_p_idx == -1 or best_new_count == 0:
            break
        
        # Commit
        p = primes[best_p_idx]
        selected.append((p, best_r))
        budget_used += inv_primes[best_p_idx]
        for n in range(best_r if best_r > 0 else p, MAX_L + 1, p):
            covered_arr[n] = 1
        
        if _covered_length(covered_arr) >= 999:
            break
    
    # Local search: try to improve by adjusting residues
    if time.time() < deadline - 10.0:
        for it in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(selected)):
                p, r = selected[i]
                if p > 300:  # skip large primes for speed
                    continue
                best_r = r
                best_score = _covered_length(covered_arr)
                
                for new_r in range(p):
                    if new_r == r:
                        continue
                    # Simulate change
                    # Remove coverage from old residue (only if not covered by others)
                    for n in range(r if r > 0 else p, MAX_L + 1, p):
                        if covered_arr[n]:
                            # Check if covered by other selected primes
                            covered_by_other = False
                            for j, (pj, rj) in enumerate(selected):
                                if j != i and n % pj == rj % pj:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                covered_arr[n] = 0
                    # Add coverage from new residue
                    for n in range(new_r if new_r > 0 else p, MAX_L + 1, p):
                        if not covered_arr[n]:
                            covered_arr[n] = 1
                    
                    new_score = _covered_length(covered_arr)
                    
                    # Revert
                    for n in range(new_r if new_r > 0 else p, MAX_L + 1, p):
                        if covered_arr[n]:
                            covered_by_other = False
                            for j, (pj, rj) in enumerate(selected):
                                if j != i and n % pj == rj % pj:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                covered_arr[n] = 0
                    for n in range(r if r > 0 else p, MAX_L + 1, p):
                        if not covered_arr[n]:
                            covered_arr[n] = 1
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_r = new_r
                        improved = True
                        # Commit this change
                        for n in range(new_r if new_r > 0 else p, MAX_L + 1, p):
                            if not covered_arr[n]:
                                covered_arr[n] = 1
                        for n in range(r if r > 0 else p, MAX_L + 1, p):
                            if covered_arr[n]:
                                covered_by_other = False
                                for j, (pj, rj) in enumerate(selected):
                                    if j != i and n % pj == rj % pj:
                                        covered_by_other = True
                                        break
                                if not covered_by_other:
                                    covered_arr[n] = 0
                        selected[i] = (p, new_r)
                        break  # move to next prime
                
                if _covered_length(covered_arr) >= 999:
                    break
            
            if not improved:
                break
    
    return [[p, r] for p, r in selected]
# EVOLVE-BLOCK-END
