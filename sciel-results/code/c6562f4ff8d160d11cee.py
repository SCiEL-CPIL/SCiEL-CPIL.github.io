import time
from fractions import Fraction
import sys
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    prime_set = set(primes)

    def recip_sum(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s

    # Fast coverage computation using numpy-style approach with bytearray
    def compute_coverage(residues, prime_list, n):
        """Compute coverage array for given residues and primes up to n."""
        covered = bytearray(n + 1)
        for i, p in enumerate(prime_list):
            r = residues[i]
            start = r if r > 0 else p
            for y in range(start, n + 1, p):
                covered[y] = 1
        return covered

    def count_covered(covered, n):
        """Count how many of 1..n are covered."""
        return sum(covered[i] for i in range(1, n + 1))

    # Simulated Annealing with incremental updates and gap-focused moves
    def simulated_annealing_cover(L, prime_list, time_limit, rng):
        """
        Use simulated annealing to find residue assignments for prime_list
        that cover 1..L. Returns (success, assignment) or (False, None).
        """
        n = L
        k = len(prime_list)
        
        # Initialize with residues that maximize initial coverage
        # Greedy initialization: for each prime, pick residue that covers most new numbers
        residues = [0] * k
        covered = bytearray(n + 1)
        
        # Sort primes by size (larger first for better coverage density)
        order = list(range(k))
        order.sort(key=lambda i: -prime_list[i])
        
        for idx in order:
            p = prime_list[idx]
            best_r = 0
            best_gain = -1
            for r in range(p):
                gain = 0
                start = r if r > 0 else p
                for y in range(start, n + 1, p):
                    if not covered[y]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[idx] = best_r
            start = best_r if best_r > 0 else p
            for y in range(start, n + 1, p):
                covered[y] = 1
        
        current_score = count_covered(covered, n)
        best_score = current_score
        best_residues = residues[:]
        
        # Temperature schedule
        T = 5.0
        T_min = 0.01
        cooling = 0.997
        
        moves_made = 0
        no_improve_count = 0
        max_no_improve = 3000
        
        while time.time() < time_limit and T > T_min:
            # Find smallest uncovered number
            first_uncovered = -1
            for x in range(1, n + 1):
                if not covered[x]:
                    first_uncovered = x
                    break
            
            if first_uncovered == -1:
                return (True, {prime_list[i]: best_residues[i] for i in range(k)})
            
            # Choose which prime to modify
            # Bias toward primes that don't cover first_uncovered
            if rng.random() < 0.6:
                candidates = [i for i in range(k) if first_uncovered % prime_list[i] != residues[i]]
                if candidates:
                    idx = candidates[rng.randint(0, len(candidates) - 1)]
                else:
                    idx = rng.randint(0, k - 1)
            else:
                idx = rng.randint(0, k - 1)
            
            p = prime_list[idx]
            old_r = residues[idx]
            new_r = rng.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Compute delta in score incrementally
            delta = 0
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p
            
            # Positions that become uncovered (covered by old but not new)
            for y in range(old_start, n + 1, p):
                if covered[y] and y % p != new_r:
                    # Check if covered by any other prime
                    still_covered = False
                    for j in range(k):
                        if j == idx:
                            continue
                        if y % prime_list[j] == residues[j]:
                            still_covered = True
                            break
                    if not still_covered:
                        delta -= 1
            
            # Positions that become covered (not covered by old but covered by new)
            for y in range(new_start, n + 1, p):
                if not covered[y] and y % p == new_r:
                    delta += 1
            
            # Accept or reject
            accept = False
            if delta >= 0:
                accept = True
            else:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    accept = True
            
            if accept:
                # Update covered array
                for y in range(old_start, n + 1, p):
                    if covered[y] and y % p != new_r:
                        still_covered = False
                        for j in range(k):
                            if j == idx:
                                continue
                            if y % prime_list[j] == residues[j]:
                                still_covered = True
                                break
                        if not still_covered:
                            covered[y] = 0
                
                for y in range(new_start, n + 1, p):
                    covered[y] = 1
                
                residues[idx] = new_r
                current_score += delta
                moves_made += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Restart with best solution
                    residues = best_residues[:]
                    covered = compute_coverage(residues, prime_list, n)
                    current_score = count_covered(covered, n)
                    no_improve_count = 0
                    T = max(T, 1.0)
            
            T *= cooling
        
        if best_score >= n:
            return (True, {prime_list[i]: best_residues[i] for i in range(k)})
        
        return (False, None)

    # Residue refinement: try to improve coverage by adjusting residues
    def refine_residues(residues, prime_list, n, time_limit, rng):
        """Try to improve coverage by local search on residues."""
        k = len(prime_list)
        covered = compute_coverage(residues, prime_list, n)
        current_score = count_covered(covered, n)
        best_score = current_score
        best_residues = residues[:]
        
        T = 2.0
        T_min = 0.05
        cooling = 0.998
        
        while time.time() < time_limit and T > T_min:
            # Find a gap
            gaps = []
            for x in range(1, n + 1):
                if not covered[x]:
                    gaps.append(x)
            
            if not gaps:
                return best_residues
            
            # Pick a random gap
            target = gaps[rng.randint(0, len(gaps) - 1)]
            
            # Try to cover it by changing a prime's residue
            improved = False
            for idx in range(k):
                p = prime_list[idx]
                old_r = residues[idx]
                new_r = target % p
                
                if new_r == old_r:
                    continue
                
                # Compute delta
                delta = 0
                old_start = old_r if old_r > 0 else p
                new_start = new_r if new_r > 0 else p
                
                for y in range(old_start, n + 1, p):
                    if covered[y] and y % p != new_r:
                        still_covered = False
                        for j in range(k):
                            if j == idx:
                                continue
                            if y % prime_list[j] == residues[j]:
                                still_covered = True
                                break
                        if not still_covered:
                            delta -= 1
                
                for y in range(new_start, n + 1, p):
                    if not covered[y] and y % p == new_r:
                        delta += 1
                
                if delta >= 0:
                    # Accept
                    for y in range(old_start, n + 1, p):
                        if covered[y] and y % p != new_r:
                            still_covered = False
                            for j in range(k):
                                if j == idx:
                                    continue
                                if y % prime_list[j] == residues[j]:
                                    still_covered = True
                                    break
                            if not still_covered:
                                covered[y] = 0
                    
                    for y in range(new_start, n + 1, p):
                        covered[y] = 1
                    
                    residues[idx] = new_r
                    current_score += delta
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues[:]
                    break
                else:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        for y in range(old_start, n + 1, p):
                            if covered[y] and y % p != new_r:
                                still_covered = False
                                for j in range(k):
                                    if j == idx:
                                        continue
                                    if y % prime_list[j] == residues[j]:
                                        still_covered = True
                                        break
                                if not still_covered:
                                    covered[y] = 0
                        
                        for y in range(new_start, n + 1, p):
                            covered[y] = 1
                        
                        residues[idx] = new_r
                        current_score += delta
                        improved = True
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                        break
            
            if not improved:
                T *= cooling
        
        return best_residues

    # Candidate prime sets - focus on sets with good coverage potential
    candidate_sets = []
    
    # Set 1: Small primes with good density
    candidate_sets.append([3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
    candidate_sets.append([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    candidate_sets.append([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    
    # Set 2: Include 2 for even coverage
    candidate_sets.append([2, 5, 7, 13, 17, 19, 23, 29, 31])
    candidate_sets.append([2, 5, 7, 13, 17, 19, 23, 29, 31, 37])
    
    # Set 3: Medium primes
    candidate_sets.append([5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    
    # Set 4: Primes >= 13
    primes_ge13 = [p for p in primes if p >= 13]
    s = 0
    selected = []
    for p in primes_ge13:
        if s + 1.0/p <= 0.99:
            selected.append(p)
            s += 1.0/p
        else:
            break
    candidate_sets.append(selected)
    
    # Set 5: Mixed
    candidate_sets.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    
    # Deduplicate and validate
    valid_sets = []
    seen = set()
    for ps in candidate_sets:
        if len(ps) == 0:
            continue
        key = tuple(sorted(ps))
        if key in seen:
            continue
        seen.add(key)
        if recip_sum(ps) <= 1:
            valid_sets.append(ps)
    
    best_L_overall = 0
    best_assignment_overall = None
    best_primes_overall = None
    
    rng = random.Random(42)
    
    # Try each prime set with increasing L
    for prime_set in valid_sets:
        if time.time() > deadline - 15:
            break
        
        # Try increasing L values
        test_Ls = [400, 600, 800, 900, 950, 999, 1000]
        
        for L in test_Ls:
            if time.time() > deadline - 10:
                break
            if L <= best_L_overall:
                continue
            
            time_limit = deadline - 10
            success, assign = simulated_annealing_cover(L, prime_set, time_limit, rng)
            
            if success and assign is not None:
                # Verify
                total = Fraction(0, 1)
                for p in prime_set:
                    total += Fraction(1, p)
                if total > 1:
                    continue
                
                covered_check = [False] * (L + 1)
                for p, r in assign.items():
                    if p not in prime_set:
                        continue
                    for y in range(1, L + 1):
                        if y % p == r:
                            covered_check[y] = True
                
                if all(covered_check[i] for i in range(1, L + 1)):
                    if L > best_L_overall:
                        best_L_overall = L
                        best_assignment_overall = dict(assign)
                        best_primes_overall = prime_set[:]
                        break  # This prime set works for this L, try next set
        
        # If we found a solution, try to extend it with refinement
        if best_assignment_overall is not None and best_primes_overall is not None:
            if time.time() < deadline - 5:
                # Try to extend coverage beyond current best
                current_L = best_L_overall
                extended = False
                for target_L in [current_L + 10, current_L + 50, current_L + 100, 999, 1000]:
                    if target_L <= best_L_overall:
                        continue
                    if time.time() > deadline - 5:
                        break
                    
                    # Use refinement on the existing assignment
                    residues = [best_assignment_overall[p] for p in best_primes_overall]
                    refined = refine_residues(residues, best_primes_overall, target_L, deadline - 5, rng)
                    
                    # Check if refined covers target_L
                    covered_check = [False] * (target_L + 1)
                    for i, p in enumerate(best_primes_overall):
                        r = refined[i]
                        for y in range(1, target_L + 1):
                            if y % p == r:
                                covered_check[y] = True
                    
                    if all(covered_check[i] for i in range(1, target_L + 1)):
                        best_L_overall = target_L
                        best_assignment_overall = {best_primes_overall[i]: refined[i] for i in range(len(best_primes_overall))}
                        extended = True
                        break
    
    # If we found a good solution, return it
    if best_assignment_overall is not None and best_primes_overall is not None:
        result = []
        for p in best_primes_overall:
            if p in best_assignment_overall:
                result.append([p, best_assignment_overall[p]])
        
        total = Fraction(0, 1)
        for p in best_primes_overall:
            total += Fraction(1, p)
        if total <= 1:
            return result
    
    # Fallback: simple greedy construction
    fallback_primes = [p for p in primes if p >= 13]
    s = 0
    selected = []
    for p in fallback_primes:
        if s + 1.0/p <= 0.99:
            selected.append(p)
            s += 1.0/p
        else:
            break
    
    result = []
    for i, p in enumerate(selected):
        result.append([p, (i + 1) % p])
    
    return result
