"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt, gcd
import time

def construct():
    N = 256
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each number 1..N, compute its set of prime factors
    prime_factors = [set() for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            prime_factors[multiple].add(p)
    
    # Check if two numbers are coprime
    def are_coprime(a, b):
        return len(prime_factors[a] & prime_factors[b]) == 0
    
    # Build conflict graph: nodes are 1..N, edge if NOT coprime
    # We want maximum weight independent set where weight = value
    
    # Branch and bound MWIS
    # Nodes sorted by value descending for better bounding
    nodes = list(range(N, 0, -1))  # 256, 255, ..., 1
    
    # Precompute for each node, which other nodes it conflicts with
    # Use bitmasks for efficiency
    # Since N=256, we can use Python ints as bitmasks
    
    # conflict_mask[i] is a bitmask of nodes that conflict with node i
    # Index nodes 0..255 corresponding to values 1..256
    # node index i corresponds to value i+1
    conflict_mask = [0] * N
    for i in range(N):
        val_i = i + 1
        for j in range(N):
            if i == j:
                continue
            val_j = j + 1
            if not are_coprime(val_i, val_j):
                conflict_mask[i] |= (1 << j)
    
    # Also precompute: for each node, the set of non-conflicting nodes (candidates)
    # candidates_mask[i] = all nodes j where j > i (to avoid duplicates) and not conflicting
    # Actually, let's use a different approach: branch on inclusion/exclusion
    
    # Upper bound: sum of all remaining node values that could still be added
    # Given current selected set S, a node can be added if it's coprime to all in S
    
    # Better approach: use recursive branch and bound with pruning
    # State: (index, current_selection_mask, current_sum)
    # But 256 nodes is too many for naive recursion
    
    # Alternative: exploit structure. Each prime can appear in at most one selected number.
    # Group numbers by their "prime signature" - actually, each number has a set of primes.
    # Two numbers conflict iff they share at least one prime.
    
    # Key insight: we can think of this as selecting at most one number per "prime group"
    # but numbers can have multiple primes, so it's more complex.
    
    # Let's use a smarter approach: 
    # 1. Start with the prime-power baseline (one number per prime)
    # 2. Use local search to improve
    
    # Prime power baseline
    baseline = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        baseline.append(power)
    
    # Verify
    for i in range(len(baseline)):
        for j in range(i + 1, len(baseline)):
            if gcd(baseline[i], baseline[j]) != 1:
                return [1] + [int(x) for x in primes]
    
    best_sum = sum(baseline)
    best_set = set(baseline)
    
    # Local search with multiple strategies
    # Strategy 1: Try to replace each element with a larger coprime number
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        for elem in sorted(best_set, key=lambda x: -x):  # try to replace larger elements first? No, try all
            others = best_set - {elem}
            # Find best replacement
            best_replacement = None
            for candidate in range(N, elem, -1):  # only larger candidates
                if candidate in best_set:
                    continue
                if all(are_coprime(candidate, x) for x in others):
                    best_replacement = candidate
                    break
            if best_replacement is not None:
                best_set.remove(elem)
                best_set.add(best_replacement)
                best_sum += (best_replacement - elem)
                improved = True
        if not improved:
            break
    
    # Strategy 2: Try to add numbers
    for candidate in range(N, 0, -1):
        if time.time() > deadline:
            break
        if candidate in best_set:
            continue
        if all(are_coprime(candidate, x) for x in best_set):
            best_set.add(candidate)
            best_sum += candidate
    
    # Strategy 3: Simulated annealing-like swaps
    import random
    random.seed(42)
    current_set = set(best_set)
    current_sum = best_sum
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        # Pick a random element to potentially remove
        if not current_set:
            break
        elem = random.choice(list(current_set))
        others = current_set - {elem}
        
        # Try to find a better replacement
        best_gain = 0
        best_candidate = None
        for candidate in range(N, 0, -1):
            if candidate in current_set:
                continue
            if candidate <= elem:
                continue
            if all(are_coprime(candidate, x) for x in others):
                gain = candidate - elem
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
                    break  # first larger coprime is good enough
        
        if best_candidate is not None:
            current_set.remove(elem)
            current_set.add(best_candidate)
            current_sum += best_gain
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
    
    # Strategy 4: Try removing pairs and replacing with better combinations
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(best_set) < 2:
            break
        # Pick two random elements
        elems = list(best_set)
        if len(elems) < 2:
            break
        e1, e2 = random.sample(elems, 2)
        others = best_set - {e1, e2}
        
        # Try to find two numbers that replace e1,e2 with higher total
        best_pair_sum = e1 + e2
        best_pair = None
        # Try all pairs (expensive, so limit)
        candidates = [c for c in range(N, 0, -1) if c not in best_set]
        for i_idx, c1 in enumerate(candidates[:100]):  # limit to top 100
            if c1 in others:
                continue
            if all(are_coprime(c1, x) for x in others):
                for c2 in candidates[i_idx+1:i_idx+101]:
                    if c2 in others or c2 == c1:
                        continue
                    if all(are_coprime(c2, x) for x in others) and are_coprime(c1, c2):
                        pair_sum = c1 + c2
                        if pair_sum > best_pair_sum:
                            best_pair_sum = pair_sum
                            best_pair = (c1, c2)
                            break
            if best_pair:
                break
        if best_pair:
            best_set.remove(e1)
            best_set.remove(e2)
            best_set.add(best_pair[0])
            best_set.add(best_pair[1])
            best_sum += (best_pair[0] + best_pair[1]) - (e1 + e2)
    
    # Final verification
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            if gcd(list(best_set)[i], list(best_set)[j]) != 1:
                # Fallback to prime powers
                result = [1]
                for p in primes:
                    power = p
                    while power * p <= N:
                        power *= p
                    result.append(power)
                return [int(x) for x in result]
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
