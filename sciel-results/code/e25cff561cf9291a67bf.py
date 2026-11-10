import time
import random
from math import gcd, isqrt
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 140s to be safe.
    deadline = start_time + 140.0
    
    N = 512
    
    # 1. Precomputation
    # values[a] = 1 / (512 - a) as Fraction
    # prime_factors[a] = list of prime factors of a
    values = [Fraction(0)] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N
    
    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)
        
    # Identify primes
    primes = [a for a in range(1, N) if is_prime_arr[a]]
    
    # 2. Initialization
    # Start with all primes. They are pairwise coprime.
    selected = set(primes)
    
    # Mapping: prime -> set of selected elements that have this prime as a factor
    # This allows O(|factors(c)|) conflict detection for candidate c
    prime_to_selected = {}
    for b in selected:
        for p in prime_factors[b]:
            if p not in prime_to_selected:
                prime_to_selected[p] = set()
            prime_to_selected[p].add(b)
            
    current_score = sum(values[a] for a in selected)
    
    # Helper to apply a move and update state
    def apply_move(add_candidates, remove_elements):
        nonlocal current_score
        # Remove elements
        for b in remove_elements:
            if b not in selected: continue
            for p in prime_factors[b]:
                if p in prime_to_selected:
                    prime_to_selected[p].discard(b)
                    if not prime_to_selected[p]:
                        del prime_to_selected[p]
            selected.remove(b)
            current_score -= values[b]
        
        # Add candidates
        for c in add_candidates:
            if c in selected: continue
            selected.add(c)
            for p in prime_factors[c]:
                if p not in prime_to_selected:
                    prime_to_selected[p] = set()
                prime_to_selected[p].add(c)
            current_score += values[c]
            
    def get_conflict_set(candidates):
        """Get the set of selected elements that conflict with any of the candidates."""
        conflict_set = set()
        for c in candidates:
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
        return conflict_set

    # 3. Phase 1: 1-for-k First-Improvement Search
    # Scan candidates. If adding c and removing conflicts improves score, do it.
    # Repeat until no improvement.
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        # Iterate over all non-selected candidates
        # To make it faster, we can just iterate 1..511
        for c in range(1, N):
            if c in selected:
                continue
            
            conflict_set = get_conflict_set([c])
            conflict_cost = sum(values[b] for b in conflict_set)
            gain = values[c] - conflict_cost
            
            if gain > 0:
                apply_move([c], list(conflict_set))
                improved = True
                # Restart scan after an improvement to ensure we don't miss cascading improvements
                # But to save time, we could just break and restart the while loop
                break

    # 4. Phase 2: 2-for-k Best-Improvement Search
    # Try adding pairs of high-value composites.
    if time.time() < deadline - 15.0:
        # Sort composites by value descending
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:60] # Limit to top 60 to keep O(K^2) manageable
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    
                    conflict_set = get_conflict_set([c1, c2])
                    conflict_cost = sum(values[b] for b in conflict_set)
                    gain = values[c1] + values[c2] - conflict_cost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(conflict_set)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                improved = True

    # 5. Phase 3: 3-for-k Best-Improvement Search
    # Try adding triples of high-value composites.
    if time.time() < deadline - 10.0:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:40] # Smaller set for O(K^3)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    for k in range(j + 1, len(top_composites)):
                        c3 = top_composites[k]
                        if c3 in selected:
                            continue
                        if gcd(c1, c3) != 1 or gcd(c2, c3) != 1:
                            continue
                        
                        conflict_set = get_conflict_set([c1, c2, c3])
                        conflict_cost = sum(values[b] for b in conflict_set)
                        gain = values[c1] + values[c2] + values[c3] - conflict_cost
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_add = [c1, c2, c3]
                            best_remove = list(conflict_set)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                improved = True

    # 6. Phase 4: Final Polish (1-for-k again)
    # Ensure we haven't left any easy 1-for-k gains on the table after the swaps
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for c in range(1, N):
                if c in selected:
                    continue
                conflict_set = get_conflict_set([c])
                conflict_cost = sum(values[b] for b in conflict_set)
                gain = values[c] - conflict_cost
                if gain > 0:
                    apply_move([c], list(conflict_set))
                    improved = True
                    break

    return list(selected)
