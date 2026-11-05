# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction
import numpy as np

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 that maximize sum 1/(512 - a).
    
    Strategy:
    1. Build conflict graph using prime factor sets.
    2. Greedy construction starting from high-value candidates.
    3. Local search with multiple move types: add, remove, swap, merge, split.
    4. Multiple restarts with different random seeds and perturbations.
    5. Final exact rational polish to squeeze out last bits of improvement.
    
    Key insight: High scores come from numbers close to 512 (small denominators).
    We want to select as many high-value numbers as possible while maintaining
    pairwise coprimality.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Precompute sorted candidates by score descending
    candidates_by_score = sorted(range(511, 0, -1), key=lambda x: scores[x], reverse=True)
    
    # Build conflict graph: conflict[a] contains all b != a such that gcd(a,b) > 1
    # Use bitmasks for efficiency
    # There are 97 primes below 512, so we can use Python's arbitrary precision ints
    prime_to_idx = {}
    prime_list = [p for p in range(2, 512) if is_prime[p]]
    for i, p in enumerate(prime_list):
        prime_to_idx[p] = i
    
    # Convert prime factor sets to bitmasks
    factor_mask = [0] * 512
    for a in range(1, 512):
        mask = 0
        for p in prime_factors[a]:
            mask |= (1 << prime_to_idx[p])
        factor_mask[a] = mask
    
    # Conflict check: two numbers conflict if their masks share any bit
    def conflicts(a, b):
        return (factor_mask[a] & factor_mask[b]) != 0
    
    # Get score of a set
    def get_score(s):
        return sum(scores[a] for a in s)
    
    # Check if set is valid (pairwise coprime)
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if conflicts(s[i], s[j]):
                    return False
        return True
    
    # Find best number to add to a set given used primes bitmask
    def find_best_add(used_mask, current_set, exclude=None):
        if exclude is None:
            exclude = set()
        for a in candidates_by_score:
            if a in current_set or a in exclude:
                continue
            if (factor_mask[a] & used_mask) == 0:
                return a
        return None
    
    # Compute used primes mask from a set
    def compute_used_mask(s):
        mask = 0
        for a in s:
            mask |= factor_mask[a]
        return mask
    
    def hill_climb(initial_set, max_iters=3000, time_budget=None):
        """
        Hill climbing with multiple move types.
        """
        if time_budget is None:
            time_budget = 10.0
        
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        used_mask = compute_used_mask(current)
        
        hc_start = time.time()
        
        for it in range(max_iters):
            # Time checks
            if time.time() > deadline - 5:
                break
            if time.time() - hc_start > time_budget:
                break
            
            improved = False
            
            # Move 1: Try adding any a not in current that is coprime
            best_add = find_best_add(used_mask, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                used_mask |= factor_mask[best_add]
                current_score += scores[best_add]
                improved = True
                continue
            
            # Move 2: Try 1-for-1 swap
            for i in range(len(current)):
                if time.time() - hc_start > time_budget:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                rem_mask = compute_used_mask(remaining)
                
                best_add = find_best_add(rem_mask, rem_set, exclude={removed})
                if best_add is not None:
                    new_score = rem_score + scores[best_add]
                    if new_score > current_score + 1e-15:
                        current = remaining + [best_add]
                        current_set = rem_set | {best_add}
                        used_mask = compute_used_mask(current)
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Move 3: Try 1-for-2 (remove one, add two)
            for i in range(len(current)):
                if time.time() - hc_start > time_budget:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                rem_mask = compute_used_mask(remaining)
                
                # Get valid candidates
                valid_cands = []
                for a in candidates_by_score:
                    if a in rem_set or a == removed:
                        continue
                    if (factor_mask[a] & rem_mask) != 0:
                        continue
                    valid_cands.append(a)
                    if len(valid_cands) > 60:
                        break
                
                best_pair = None
                best_pair_score = 0
                for idx_a in range(len(valid_cands)):
                    a = valid_cands[idx_a]
                    for idx_b in range(idx_a + 1, len(valid_cands)):
                        b = valid_cands[idx_b]
                        if (factor_mask[a] & factor_mask[b]) != 0:
                            continue
                        s = rem_score + scores[a] + scores[b]
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (a, b)
                
                if best_pair is not None and best_pair_score > current_score + 1e-15:
                    current = remaining + list(best_pair)
                    current_set = rem_set | set(best_pair)
                    used_mask = compute_used_mask(current)
                    current_score = best_pair_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Move 4: Try 2-for-1 (remove two, add one) - MERGE
            if len(current) >= 2:
                for i in range(len(current)):
                    if time.time() - hc_start > time_budget:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() - hc_start > time_budget:
                            break
                        removed1 = current[i]
                        removed2 = current[j]
                        remaining = [x for k, x in enumerate(current) if k != i and k != j]
                        rem_set = set(remaining)
                        rem_score = current_score - scores[removed1] - scores[removed2]
                        rem_mask = compute_used_mask(remaining)
                        
                        best_new_score = 0
                        best_add_candidate = None
                        for a in candidates_by_score:
                            if a in rem_set or a == removed1 or a == removed2:
                                continue
                            if (factor_mask[a] & rem_mask) != 0:
                                continue
                            new_score = rem_score + scores[a]
                            if new_score > best_new_score:
                                best_new_score = new_score
                                best_add_candidate = a
                        
                        if best_add_candidate is not None and best_new_score > current_score + 1e-15:
                            current = remaining + [best_add_candidate]
                            current_set = rem_set | {best_add_candidate}
                            used_mask = compute_used_mask(current)
                            current_score = best_new_score
                            improved = True
                            break
                    if improved:
                        break
            
            if improved:
                continue
            
            # Move 5: Try 2-for-2 (remove two, add two)
            if len(current) >= 2:
                for i in range(len(current)):
                    if time.time() - hc_start > time_budget:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() - hc_start > time_budget:
                            break
                        removed1 = current[i]
                        removed2 = current[j]
                        remaining = [x for k, x in enumerate(current) if k != i and k != j]
                        rem_set = set(remaining)
                        rem_score = current_score - scores[removed1] - scores[removed2]
                        rem_mask = compute_used_mask(remaining)
                        
                        # Find best pair to add
                        valid_cands = []
                        for a in candidates_by_score:
                            if a in rem_set or a == removed1 or a == removed2:
                                continue
                            if (factor_mask[a] & rem_mask) != 0:
                                continue
                            valid_cands.append(a)
                            if len(valid_cands) > 50:
                                break
                        
                        best_pair = None
                        best_pair_score = 0
                        for idx_a in range(len(valid_cands)):
                            a = valid_cands[idx_a]
                            for idx_b in range(idx_a + 1, len(valid_cands)):
                                b = valid_cands[idx_b]
                                if (factor_mask[a] & factor_mask[b]) != 0:
                                    continue
                                s = rem_score + scores[a] + scores[b]
                                if s > best_pair_score:
                                    best_pair_score = s
                                    best_pair = (a, b)
                        
                        if best_pair is not None and best_pair_score > current_score + 1e-15:
                            current = remaining + list(best_pair)
                            current_set = rem_set | set(best_pair)
                            used_mask = compute_used_mask(current)
                            current_score = best_pair_score
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current, current_score
    
    # === Main algorithm ===
    
    # Initial solution: Start with all primes (they're all pairwise coprime)
    initial_set = [p for p in range(2, 512) if is_prime[p]]
    
    best_set = initial_set[:]
    best_score = get_score(best_set)
    
    # Phase 1: Hill climb from initial
    hc_result, hc_score = hill_climb(best_set, max_iters=2000, time_budget=20.0)
    if hc_score > best_score + 1e-15:
        best_set = hc_result
        best_score = hc_score
    
    # Phase 2: Multiple restarts with perturbation
    rng = random.Random(12345)
    
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Create perturbed version of best_set
        perturbed = best_set[:]
        
        # Randomly remove 1-3 elements
        k = rng.randint(1, min(3, len(perturbed)))
        for _ in range(k):
            if len(perturbed) > 1:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        
        # Try hill climbing from perturbed state
        hc_result, hc_score = hill_climb(perturbed, max_iters=1000, time_budget=8.0)
        
        if hc_score > best_score + 1e-15:
            best_set = hc_result
            best_score = hc_score
    
    # Phase 3: Greedy construction from scratch with different orderings
    for seed in range(5):
        if time.time() > deadline - 25:
            break
        
        # Create shuffled candidate order
        shuffled = candidates_by_score[:]
        random.Random(seed).shuffle(shuffled)
        
        # Greedy construction
        current = []
        current_set = set()
        used_mask = 0
        current_score = 0.0
        
        for a in shuffled:
            if (factor_mask[a] & used_mask) == 0:
                current.append(a)
                current_set.add(a)
                used_mask |= factor_mask[a]
                current_score += scores[a]
        
        # Polish
        hc_result, hc_score = hill_climb(current, max_iters=1500, time_budget=10.0)
        if hc_score > best_score + 1e-15:
            best_set = hc_result
            best_score = hc_score
    
    # Phase 4: Final exact rational polish
    if time.time() < deadline - 10:
        exact_scores = [Fraction(1, 512 - a) for a in range(512)]
        
        def exact_score(s):
            return sum(exact_scores[a] for a in s)
        
        current = list(best_set)
        current_exact = exact_score(current)
        current_set = set(current)
        used_mask = compute_used_mask(current)
        
        for it in range(500):
            if time.time() > deadline - 2:
                break
            
            improved = False
            
            # Try adding
            best_add = None
            for a in candidates_by_score:
                if a in current_set:
                    continue
                if (factor_mask[a] & used_mask) != 0:
                    continue
                new_exact = current_exact + exact_scores[a]
                if new_exact > current_exact:
                    best_add = a
                    break
            
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                used_mask |= factor_mask[best_add]
                current_exact += exact_scores[best_add]
                improved = True
            
            if not improved:
                # Try 2-for-1 merge
                for i in range(len(current)):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline - 2:
                            break
                        removed1 = current[i]
                        removed2 = current[j]
                        remaining = [x for k, x in enumerate(current) if k != i and k != j]
                        rem_set = set(remaining)
                        rem_exact = current_exact - exact_scores[removed1] - exact_scores[removed2]
                        rem_mask = compute_used_mask(remaining)
                        
                        best_new_exact = rem_exact
                        best_add_candidate = None
                        for a in candidates_by_score:
                            if a in rem_set or a == removed1 or a == removed2:
                                continue
                            if (factor_mask[a] & rem_mask) != 0:
                                continue
                            new_exact = rem_exact + exact_scores[a]
                            if new_exact > best_new_exact:
                                best_new_exact = new_exact
                                best_add_candidate = a
                        
                        if best_add_candidate is not None and best_new_exact > current_exact:
                            current = remaining + [best_add_candidate]
                            current_set = rem_set | {best_add_candidate}
                            used_mask = compute_used_mask(current)
                            current_exact = best_new_exact
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        if current_exact > exact_score(best_set):
            best_set = current
            best_score = float(current_exact)
    
    return best_set
# EVOLVE-BLOCK-END
