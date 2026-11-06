import time
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _get_paleyn_orientation(n):
    """Constructs a Paley tournament orientation for prime n where n % 4 == 3."""
    if n < 2:
        return None
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if not is_prime or n % 4 != 3:
        return None
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _orientation_to_in_bits(n, orientation):
    """Convert orientation list to in_bits array (bitmask of who points to each vertex)."""
    in_bits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                # i -> j
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                in_bits[i] |= (1 << j)
            idx += 1
    return in_bits

def _in_bits_to_orientation(n, in_bits):
    """Convert in_bits array to orientation list."""
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if (in_bits[j] >> i) & 1:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return orientation

def _validate_tournament(n, in_bits):
    """Check if tournament is valid (every 4-set has a common dominator)."""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    if (common & ~mask) == 0:
                        return False
    return True

def _greedy_prune(n, in_bits, deadline):
    """
    Greedy pruning: iteratively remove a vertex that is not needed
    as the unique dominator for any 4-set.
    """
    active = set(range(n))
    cur_in_bits = in_bits[:]
    
    while time.time() < deadline - 3:
        if len(active) <= 5:
            break
        
        active_list = list(active)
        found_removal = False
        
        # Try to find a removable vertex
        for v in active_list:
            if time.time() > deadline - 3:
                break
            
            others = [x for x in active if x != v]
            if len(others) < 3:
                continue
            
            # Check if v can be removed: for every 3-subset T of others,
            # the 4-set {v} ∪ T must have a dominator in active \ ({v} ∪ T)
            removable = True
            for x, y, z in combinations(others, 3):
                mask4 = (1 << v) | (1 << x) | (1 << y) | (1 << z)
                common = cur_in_bits[v] & cur_in_bits[x] & cur_in_bits[y] & cur_in_bits[z]
                dominators = common & ~mask4
                # Check if any dominator is in active \ {v, x, y, z}
                found = False
                for w in active:
                    if w != v and w != x and w != y and w != z:
                        if (dominators >> w) & 1:
                            found = True
                            break
                if not found:
                    removable = False
                    break
            
            if removable:
                active.discard(v)
                found_removal = True
                break
        
        if not found_removal:
            break
    
    # Rebuild in_bits for the pruned tournament
    if len(active) < n:
        active = sorted(active)
        old_to_new = {x: i for i, x in enumerate(active)}
        new_in_bits = [0] * len(active)
        for old_i in active:
            new_i = old_to_new[old_i]
            for old_j in active:
                if old_i == old_j:
                    continue
                if (cur_in_bits[old_i] >> old_j) & 1:
                    new_in_bits[new_i] |= (1 << old_to_new[old_j])
        return len(active), new_in_bits
    return n, in_bits

def _local_search(n, in_bits, time_limit, seed=42):
    """
    Simulated annealing local search to minimize the number of bad 4-sets.
    Returns orientation if valid found, else None.
    """
    if n > 60:
        return None
    
    start_time = time.time()
    rng = random.Random(seed)
    cur_in_bits = in_bits[:]
    
    # Precompute all 4-subsets
    tuples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    tuples.append((a, b, c, d))
    
    num_tuples = len(tuples)
    if num_tuples == 0:
        return _in_bits_to_orientation(n, cur_in_bits)
    
    # Compute initial bad count using numpy for speed
    states = np.zeros(num_tuples, dtype=np.int32)
    for idx_t, (a, b, c, d) in enumerate(tuples):
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        common = cur_in_bits[a] & cur_in_bits[b] & cur_in_bits[c] & cur_in_bits[d]
        states[idx_t] = 1 if (common & ~mask) == 0 else 0
    bad = int(np.sum(states))
    
    if bad == 0:
        return _in_bits_to_orientation(n, cur_in_bits)
    
    # Map vertex to tuple indices
    vertex_to_tuples = [[] for _ in range(n)]
    for i, t in enumerate(tuples):
        for v in t:
            vertex_to_tuples[v].append(i)
    
    # Precompute masks as Python int list (safe for large n)
    masks = [(1 << a) | (1 << b) | (1 << c) | (1 << d) for a, b, c, d in tuples]
    
    T = 2.0
    T_min = 0.001
    cooling = 0.9998
    best_bad = bad
    best_in_bits = cur_in_bits[:]
    
    while time.time() - start_time < time_limit:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Get affected tuple indices
        aff_set = set(vertex_to_tuples[i]) | set(vertex_to_tuples[j])
        aff_list = list(aff_set)
        
        delta = 0
        changed = []
        
        # Determine current direction
        if (cur_in_bits[j] >> i) & 1:
            # Currently i -> j, flipping to j -> i
            new_in_i = cur_in_bits[i] | (1 << j)
            new_in_j = cur_in_bits[j] & ~(1 << i)
        else:
            # Currently j -> i, flipping to i -> j
            new_in_i = cur_in_bits[i] & ~(1 << j)
            new_in_j = cur_in_bits[j] | (1 << i)
        
        for idx_t in aff_list:
            a, b, c, d = tuples[idx_t]
            mask = masks[idx_t]
            
            ni_a = new_in_i if a == i else (new_in_j if a == j else cur_in_bits[a])
            ni_b = new_in_i if b == i else (new_in_j if b == j else cur_in_bits[b])
            ni_c = new_in_i if c == i else (new_in_j if c == j else cur_in_bits[c])
            ni_d = new_in_i if d == i else (new_in_j if d == j else cur_in_bits[d])
            
            common = ni_a & ni_b & ni_c & ni_d
            new_state = 1 if (common & ~mask) == 0 else 0
            old_state = states[idx_t]
            
            if old_state != new_state:
                delta += (new_state - old_state)
                changed.append((idx_t, new_state))
        
        # Acceptance criterion
        if delta < 0 or (T > T_min and rng.random() < np.exp(-max(delta, 0) / T)):
            if (cur_in_bits[j] >> i) & 1:
                cur_in_bits[i] |= (1 << j)
                cur_in_bits[j] &= ~(1 << i)
            else:
                cur_in_bits[i] &= ~(1 << j)
                cur_in_bits[j] |= (1 << i)
            
            for idx_t, state in changed:
                states[idx_t] = state
            bad += delta
            
            if bad < best_bad:
                best_bad = bad
                best_in_bits = cur_in_bits[:]
            
            if bad == 0:
                return _in_bits_to_orientation(n, cur_in_bits)
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Return best found
    if best_bad == 0:
        return _in_bits_to_orientation(n, best_in_bits)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_n = 91
    best_orientation = None
    
    # Step 1: Get Paley(67) as starting point
    paley_orient = _get_paleyn_orientation(67)
    if paley_orient is None:
        n = 90
        return {'n': n, 'orientation': [0] * (n * (n - 1) // 2)}
    
    paley_in_bits = _orientation_to_in_bits(67, paley_orient)
    
    # Verify Paley(67) is valid
    if _validate_tournament(67, paley_in_bits):
        best_n = 67
        best_orientation = paley_orient
    
    # Step 2: Greedy pruning on Paley(67)
    pruned_n, pruned_in_bits = _greedy_prune(67, paley_in_bits, deadline - 30)
    
    if pruned_n < best_n:
        if _validate_tournament(pruned_n, pruned_in_bits):
            best_n = pruned_n
            best_orientation = _in_bits_to_orientation(pruned_n, pruned_in_bits)
    
    # Step 3: Try local search for smaller n values
    # Search from min(best_n - 1, 50) down to 31
    search_start = min(best_n - 1, 50) if best_n < 91 else 50
    search_end = 31
    
    for n in range(search_start, search_end - 1, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - 10 - time.time()
        if remaining < 3:
            break
        
        # Allocate time based on how many n values are left
        n_values_left = search_start - n + 1
        time_per_n = min(20.0, remaining / max(1, n_values_left))
        
        # Try multiple seeds with different initialization strategies
        found = False
        seeds = [42, 123, 777, 999, 555, 111, 222, 333, 444, 666]
        
        for seed_idx, seed in enumerate(seeds):
            if time.time() > deadline - 10:
                break
            
            time_left = min(time_per_n / max(1, len(seeds) - seed_idx), deadline - time.time() - 10)
            if time_left < 0.5:
                break
            
            # Different initialization strategies
            if seed_idx % 3 == 0:
                # Random initialization
                rng = random.Random(seed)
                in_bits = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.random() < 0.5:
                            in_bits[j] |= (1 << i)
                        else:
                            in_bits[i] |= (1 << j)
            elif seed_idx % 3 == 1:
                # Regular tournament-like initialization
                in_bits = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        # Use a deterministic but non-trivial pattern
                        if (i * j + i + j) % 2 == 0:
                            in_bits[j] |= (1 << i)
                        else:
                            in_bits[i] |= (1 << j)
            else:
                # Use pruned result if available, scaled down
                if pruned_n >= n and pruned_n <= 60:
                    in_bits = [0] * n
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (pruned_in_bits[i] >> j) & 1:
                                in_bits[j] |= (1 << i)
                            else:
                                in_bits[i] |= (1 << j)
                else:
                    rng = random.Random(seed + 1000)
                    in_bits = [0] * n
                    for i in range(n):
                        for j in range(i + 1, n):
                            if rng.random() < 0.5:
                                in_bits[j] |= (1 << i)
                            else:
                                in_bits[i] |= (1 << j)
            
            orientation = _local_search(n, in_bits, time_left, seed=seed)
            if orientation is not None:
                best_n = n
                best_orientation = orientation
                found = True
                break
        
        if found:
            continue  # Try smaller n
    
    # Fallback to Paley(67) if nothing better
    if best_orientation is None or best_n > 67:
        best_n = 67
        best_orientation = paley_orient
    
    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
