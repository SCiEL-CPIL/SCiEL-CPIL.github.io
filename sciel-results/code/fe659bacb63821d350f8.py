"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 maximizing sum 1/(512 - a).
    Uses greedy initialization followed by multi-swap hill climbing.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 512
    
    # Precompute weights: weight of vertex a is 1/(512 - a)
    weights = [0.0] * N  # index 0 unused, 1..511
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Precompute prime factors for each number 1..511
    pfactors = [set() for _ in range(N)]
    for a in range(2, N):
        pfactors[a] = _prime_factors(a)
    
    def is_coprime(a, b):
        return len(pfactors[a] & pfactors[b]) == 0
    
    def can_add(candidate, current_set):
        """Check if candidate can be added to current_set (all pairwise coprime)."""
        for x in current_set:
            if not is_coprime(candidate, x):
                return False
        return True
    
    def compute_score(current_set):
        return sum(weights[a] for a in current_set)
    
    # Step 1: Greedy initialization
    # Process vertices in order of decreasing weight (i.e., increasing a)
    greedy_set = []
    available = list(range(1, N))  # candidates 1..511
    
    # Sort by weight descending (larger a = smaller denominator = larger weight)
    for a in sorted(available, key=lambda x: -weights[x]):
        if time.time() > deadline:
            break
        if can_add(a, greedy_set):
            greedy_set.append(a)
    
    best_set = list(greedy_set)
    best_score = compute_score(best_set)
    
    # Step 2: Multi-swap hill climbing
    # Try removing k elements and adding m elements to improve the score
    # Focus on swaps that replace low-weight elements with higher-weight ones
    
    # Pre-sort all candidates by weight descending for quick access
    all_candidates = sorted(range(1, N), key=lambda x: -weights[x])
    
    # For efficiency, precompute which candidates conflict with which
    # conflict_mask[a] = set of b such that gcd(a,b) > 1
    conflict_mask = [set() for _ in range(N)]
    for a in range(2, N):
        for b in range(a + 1, N):
            if not is_coprime(a, b):
                conflict_mask[a].add(b)
                conflict_mask[b].add(a)
    
    def can_add_fast(candidate, current_set):
        """Fast check using precomputed conflict masks."""
        for x in current_set:
            if candidate in conflict_mask[x]:
                return False
        return True
    
    # Hill climbing with single and multi-swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single additions: add a candidate if it improves score
        for a in all_candidates:
            if time.time() > deadline:
                break
            if a not in best_set:
                if can_add_fast(a, best_set):
                    new_score = best_score + weights[a]
                    if new_score > best_score + 1e-12:
                        best_set.append(a)
                        best_score = new_score
                        improved = True
        
        # Try single removals: remove an element if it allows adding better ones
        # (handled implicitly by multi-swap)
        
        # Try 1-for-1 swaps
        for i, remove_a in enumerate(list(best_set)):
            if time.time() > deadline:
                break
            for add_b in all_candidates:
                if time.time() > deadline:
                    break
                if add_b == remove_a or add_b in best_set:
                    continue
                # Check if add_b is coprime to all other elements in best_set
                # (excluding remove_a)
                valid = True
                for x in best_set:
                    if x != remove_a and add_b in conflict_mask[x]:
                        valid = False
                        break
                if valid:
                    delta = weights[add_b] - weights[remove_a]
                    if delta > 1e-12:
                        best_set[i] = add_b
                        best_score += delta
                        improved = True
                        break  # restart swap search after improvement
        
        # Try 2-for-1 swaps: remove 2 elements, add 1
        if time.time() < deadline - 5:
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_set)):
                    if time.time() > deadline:
                        break
                    remove_a = best_set[i]
                    remove_b = best_set[j]
                    for add_c in all_candidates:
                        if time.time() > deadline:
                            break
                        if add_c == remove_a or add_c == remove_b or add_c in best_set:
                            continue
                        # Check if add_c is coprime to all other elements
                        valid = True
                        for x in best_set:
                            if x != remove_a and x != remove_b and add_c in conflict_mask[x]:
                                valid = False
                                break
                        if valid:
                            delta = weights[add_c] - weights[remove_a] - weights[remove_b]
                            if delta > 1e-12:
                                # Perform the swap
                                best_set[i] = add_c
                                best_set.pop(j)
                                best_score += delta
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        # Try 1-for-2 swaps: remove 1 element, add 2
        if time.time() < deadline - 5:
            for i, remove_a in enumerate(list(best_set)):
                if time.time() > deadline:
                    break
                # Find pairs of candidates that are both coprime to all other elements
                # and coprime to each other
                candidates = [b for b in all_candidates if b not in best_set and b != remove_a]
                # Filter: b must be coprime to all elements in best_set except remove_a
                valid_candidates = []
                for b in candidates:
                    ok = True
                    for x in best_set:
                        if x != remove_a and b in conflict_mask[x]:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(b)
                
                # Try pairs from valid_candidates
                for idx1 in range(len(valid_candidates)):
                    if time.time() > deadline:
                        break
                    for idx2 in range(idx1 + 1, len(valid_candidates)):
                        if time.time() > deadline:
                            break
                        b1 = valid_candidates[idx1]
                        b2 = valid_candidates[idx2]
                        if b1 in conflict_mask[b2]:
                            continue
                        delta = weights[b1] + weights[b2] - weights[remove_a]
                        if delta > 1e-12:
                            best_set[i] = b1
                            best_set.append(b2)
                            best_score += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Final validation: ensure pairwise coprimality
    final_set = list(best_set)
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            if not is_coprime(final_set[i], final_set[j]):
                # This shouldn't happen, but if it does, fall back to greedy
                return greedy_set
    
    return final_set

# EVOLVE-BLOCK-END
