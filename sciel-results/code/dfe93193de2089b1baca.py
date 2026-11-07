"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute factorizations for all numbers 2..40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _factorize(n):
    exponents = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return tuple(exponents)

# Precompute factorization vectors for each number 2..40
_FACTORS = {}
for i in range(2, 41):
    _FACTORS[i] = _factorize(i)

def _product_distinct(nums):
    """Check if all 2^k subset products are distinct using prime exponent vectors."""
    k = len(nums)
    if k == 0:
        return True
    
    # Get factorization vectors for each number
    factors = [_FACTORS[x] for x in nums]
    
    # Use DP: for each mask, the exponent vector is the sum of factors of set bits
    # We store exponent vectors as tuples for hashing
    num_subsets = 1 << k
    seen = set()
    seen.add((0,) * _NUM_PRIMES)  # empty subset
    
    # DP table: exp_vec[mask] = tuple of exponents
    # Use incremental computation
    exp_vecs = [(0,) * _NUM_PRIMES] * num_subsets
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        # Add factors[idx] to exp_vecs[prev_mask]
        prev = exp_vecs[prev_mask]
        f = factors[idx]
        new_vec = tuple(prev[i] + f[i] for i in range(_NUM_PRIMES))
        exp_vecs[mask] = new_vec
        
        if new_vec in seen:
            return False
        seen.add(new_vec)
    
    return True

def _product_distinct_fast(nums, current_exp_vecs=None, current_seen=None):
    """Faster check that can reuse precomputed state."""
    k = len(nums)
    if k == 0:
        return True, set([(0,)*_NUM_PRIMES]), []
    
    factors = [_FACTORS[x] for x in nums]
    num_subsets = 1 << k
    seen = set()
    seen.add((0,) * _NUM_PRIMES)
    
    exp_vecs = [(0,) * _NUM_PRIMES] * num_subsets
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        prev = exp_vecs[prev_mask]
        f = factors[idx]
        new_vec = tuple(prev[i] + f[i] for i in range(_NUM_PRIMES))
        exp_vecs[mask] = new_vec
        
        if new_vec in seen:
            return False, seen, exp_vecs
        seen.add(new_vec)
    
    return True, seen, exp_vecs

def _try_greedy(order=None, time_limit=5.0):
    """Greedy construction with given order of candidates."""
    start_time = time.time()
    if order is None:
        order = list(range(2, 41))
    
    current = []
    for candidate in order:
        if time.time() - start_time > time_limit:
            break
        if len(current) >= P['maximum_size']:
            break
        new_set = current + [candidate]
        if _product_distinct(new_set):
            current = new_set
    
    return current

def _local_search(nums, time_limit=5.0):
    """Local search: try swaps and additions to improve set size."""
    start_time = time.time()
    current = list(nums)
    
    if not _product_distinct(current):
        return None
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Try adding a new element
        for candidate in range(2, 41):
            if time.time() - start_time > time_limit:
                return current
            if candidate not in current and len(current) < P['maximum_size']:
                new_set = current + [candidate]
                if _product_distinct(new_set):
                    current = new_set
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try single swap: remove one, add one
        for i in range(len(current)):
            if time.time() - start_time > time_limit:
                return current
            for candidate in range(2, 41):
                if candidate == current[i] or candidate in current:
                    continue
                new_set = [x for j, x in enumerate(current) if j != i]
                new_set.append(candidate)
                if _product_distinct(new_set):
                    current = new_set
                    improved = True
                    break
            if improved:
                break
    
    return current

def _simulate_annealing(start_set, time_limit=10.0):
    """Simulated annealing to escape local optima."""
    start_time = time.time()
    current = list(start_set)
    best = list(current)
    
    if not _product_distinct(current):
        return None
    
    best_size = len(current)
    temperature = 50.0
    cooling = 0.95
    
    while time.time() - start_time < time_limit and temperature > 1.0:
        # Generate a random neighbor
        neighbor = list(current)
        move_type = random.random()
        
        if move_type < 0.4 and len(neighbor) < P['maximum_size']:
            # Add a random element
            candidates = [x for x in range(2, 41) if x not in neighbor]
            if candidates:
                neighbor.append(random.choice(candidates))
        elif move_type < 0.7:
            # Swap: remove one, add one
            if neighbor:
                rem_idx = random.randint(0, len(neighbor) - 1)
                removed = neighbor.pop(rem_idx)
                candidates = [x for x in range(2, 41) if x not in neighbor and x != removed]
                if candidates:
                    neighbor.append(random.choice(candidates))
                else:
                    neighbor.append(removed)
        elif move_type < 0.85:
            # Remove one
            if neighbor:
                neighbor.pop(random.randint(0, len(neighbor) - 1))
        else:
            # Two-swap: remove two, add two
            if len(neighbor) >= 2:
                idxs = random.sample(range(len(neighbor)), 2)
                removed = [neighbor[i] for i in sorted(idxs, reverse=True)]
                for i in sorted(idxs, reverse=True):
                    neighbor.pop(i)
                candidates = [x for x in range(2, 41) if x not in neighbor]
                if len(candidates) >= 2:
                    adds = random.sample(candidates, 2)
                    neighbor.extend(adds)
                elif len(candidates) == 1:
                    neighbor.append(candidates[0])
        
        # Evaluate neighbor
        if not _product_distinct(neighbor):
            continue
        
        neighbor_size = len(neighbor)
        current_size = len(current)
        
        # Acceptance criterion
        if neighbor_size > current_size:
            current = neighbor
            if neighbor_size > best_size:
                best = list(neighbor)
                best_size = neighbor_size
        else:
            # Accept worse with some probability
            delta = current_size - neighbor_size
            prob = math.exp(-delta / temperature) if temperature > 0 else 0
            if random.random() < prob:
                current = neighbor
        
        temperature *= cooling
    
    return best

def construct():
    """Construct a set of distinct integers in 2..40 with distinct subset products."""
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_size = 0
    
    # Phase 1: Try multiple greedy orderings
    # Natural order
    greedy1 = _try_greedy(list(range(2, 41)), time_limit=3.0)
    if greedy1 and len(greedy1) > best_size:
        best_set = greedy1
        best_size = len(greedy1)
    
    # Reverse order
    greedy2 = _try_greedy(list(range(40, 1, -1)), time_limit=3.0)
    if greedy2 and len(greedy2) > best_size:
        best_set = greedy2
        best_size = len(greedy2)
    
    # Prime-heavy order
    primes_order = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                    4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21,
                    22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35,
                    36, 38, 39, 40]
    greedy3 = _try_greedy(primes_order, time_limit=3.0)
    if greedy3 and len(greedy3) > best_size:
        best_set = greedy3
        best_size = len(greedy3)
    
    # Random orderings
    for trial in range(10):
        if time.time() > deadline:
            break
        order = list(range(2, 41))
        random.shuffle(order)
        greedy = _try_greedy(order, time_limit=2.0)
        if greedy and len(greedy) > best_size:
            best_set = greedy
            best_size = len(greedy)
    
    # Phase 2: Local search from various starting points
    starting_points = []
    if best_set:
        starting_points.append(best_set[:])
    
    # Add some random valid starting points
    for _ in range(15):
        if time.time() > deadline:
            break
        size = random.randint(3, min(12, P['maximum_size']))
        candidate = random.sample(range(2, 41), size)
        if _product_distinct(candidate):
            starting_points.append(candidate)
    
    # Run local search from each starting point
    for i, start in enumerate(starting_points):
        if time.time() > deadline - 2.0:
            break
        remaining = deadline - time.time() - 2.0
        if remaining < 0.5:
            break
        per_time = max(0.5, min(remaining / max(1, len(starting_points) - i), 4.0))
        
        result = _local_search(start, time_limit=per_time)
        if result and len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Phase 3: Simulated annealing from best and other starts
    if time.time() < deadline - 5.0:
        sa_starts = [best_set] if best_set else []
        # Add a few more starts
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            size = random.randint(5, min(12, P['maximum_size']))
            candidate = random.sample(range(2, 41), size)
            if _product_distinct(candidate):
                sa_starts.append(candidate)
        
        for start in sa_starts:
            if time.time() > deadline - 2.0:
                break
            remaining = deadline - time.time() - 2.0
            if remaining < 1.0:
                break
            per_time = max(1.0, min(remaining / max(1, len(sa_starts) + 1 - sa_starts.index(start)), 8.0))
            
            result = _simulate_annealing(start, time_limit=per_time)
            if result and len(result) > best_size:
                best_size = len(result)
                best_set = result
    
    # Phase 4: If we haven't reached maximum, try more aggressive search
    if best_size < P['maximum_size'] and time.time() < deadline - 3.0:
        # Try to extend the best set more aggressively
        if best_set:
            extended = _local_search(best_set, time_limit=deadline - time.time() - 2.0)
            if extended and len(extended) > best_size:
                best_size = len(extended)
                best_set = extended
    
    # Final safety check
    if best_set is None:
        best_set = [2]
        best_size = 1
    elif not _product_distinct(best_set):
        best_set = [2]
        best_size = 1
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
