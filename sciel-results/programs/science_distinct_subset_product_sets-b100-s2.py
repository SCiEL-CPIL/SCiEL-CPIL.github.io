"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

P = {'n': 40, 'maximum_size': 18}

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}
DIM = len(PRIMES)

def get_vector(n):
    """Get the prime exponent vector for n."""
    vec = [0] * DIM
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute vectors for all candidates 2..40
CANDIDATES = list(range(2, P['n'] + 1))
VECTORS = [get_vector(x) for x in CANDIDATES]
# Map vector to candidate index for potential reverse lookup if needed, 
# but we work with indices of CANDIDATES.

def is_independent_mitm(selected_indices, new_idx):
    """
    Check if the vector at new_idx is in the {-1, 0, 1}-span of the vectors at selected_indices.
    If yes, adding it creates a dependency (collision).
    If no, it is independent.
    """
    v_new = VECTORS[new_idx]
    if not selected_indices:
        return True # Always independent if set is empty
        
    k = len(selected_indices)
    # Split selected indices into two halves
    mid = k // 2
    left_indices = selected_indices[:mid]
    right_indices = selected_indices[mid:]
    
    # Generate all {-1, 0, 1} sums for left half
    left_sums = set()
    # Optimization: if left half is small, just iterate
    # 3^mid possibilities
    # Use recursive generation or product
    # product([-1, 0, 1], repeat=len(left_indices))
    
    def gen_sums(indices):
        if not indices:
            yield (0,) * DIM
            return
        first = indices[0]
        rest = indices[1:]
        v_first = VECTORS[first]
        for rest_sum in gen_sums(rest):
            # 0 * v_first
            yield tuple(r for r in rest_sum)
            # 1 * v_first
            yield tuple(r + v_first[i] for i, r in enumerate(rest_sum))
            # -1 * v_first
            yield tuple(r - v_first[i] for i, r in enumerate(rest_sum))
            
    # This recursive generator might be slow in Python for 3^9.
    # Let's use iterative set expansion which is faster in Python sets.
    
    def expand_sums(indices):
        sums = {tuple([0]*DIM)}
        for idx in indices:
            v = VECTORS[idx]
            new_sums = set()
            for s in sums:
                # s
                new_sums.add(s)
                # s + v
                new_sums.add(tuple(s[i] + v[i] for i in range(DIM)))
                # s - v
                new_sums.add(tuple(s[i] - v[i] for i in range(DIM)))
            sums = new_sums
        return sums

    left_sums = expand_sums(left_indices)
    
    # For each sum in right half, check if v_new - sum_right is in left_sums
    # We can iterate over right sums and check membership
    
    # To save time, we can also check if v_new is 0 (impossible for n>1)
    
    for r_sum in expand_sums(right_indices):
        # We need left_sum such that left_sum + r_sum = v_new
        # => left_sum = v_new - r_sum
        target = tuple(v_new[i] - r_sum[i] for i in range(DIM))
        if target in left_sums:
            # Found a dependency
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial best set: Primes
    primes = [i for i, x in enumerate(CANDIDATES) if all(x % d for d in range(2, int(x**0.5)+1))]
    best_set = set(primes)
    best_size = len(best_set)
    
    # Local Search with MITM check
    # We will use a randomized restart approach with greedy addition and swaps.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Start from a random subset of primes or just primes
        # To escape local optima, we can start from a random valid set of size ~10
        # But starting from primes is a good baseline.
        # Let's try starting from a random permutation of primes and greedily adding.
        
        S = set(primes)
        
        # Greedy Add
        added = True
        while added and time.time() < deadline:
            added = False
            candidates = [c for c in range(len(CANDIDATES)) if c not in S]
            random.shuffle(candidates)
            for c in candidates:
                if is_independent_mitm(list(S), c):
                    S.add(c)
                    added = True
                    break
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size == P['maximum_size']:
                break
                
        # Swap Phase
        if len(S) < P['maximum_size'] and time.time() < deadline - 5:
            for _ in range(50): # Reduced swap attempts per restart due to cost of MITM
                if time.time() > deadline:
                    break
                if not S:
                    break
                to_remove = random.choice(list(S))
                S.remove(to_remove)
                
                candidates = [c for c in range(len(CANDIDATES)) if c not in S]
                if not candidates:
                    S.add(to_remove)
                    break
                to_add = random.choice(candidates)
                
                # Check if swap is valid
                # We need to check if S \ {to_remove} + {to_add} is independent.
                # This is equivalent to checking if to_add is independent from S \ {to_remove}.
                # But wait, S \ {to_remove} is independent. We just need to check if to_add 
                # is in the span of S \ {to_remove}.
                
                if is_independent_mitm(list(S), to_add):
                    S.add(to_add)
                    # Try to add more
                    added = True
                    while added and time.time() < deadline:
                        added = False
                        candidates = [c for c in range(len(CANDIDATES)) if c not in S]
                        random.shuffle(candidates)
                        for c in candidates:
                            if is_independent_mitm(list(S), c):
                                S.add(c)
                                added = True
                                break
                    if len(S) > best_size:
                        best_set = S
                        best_size = len(S)
                        if best_size == P['maximum_size']:
                            break
                else:
                    S.add(to_remove) # Restore
                    
        if best_size == P['maximum_size']:
            break

    # Convert indices back to values
    return [CANDIDATES[i] for i in best_set]

# EVOLVE-BLOCK-END
