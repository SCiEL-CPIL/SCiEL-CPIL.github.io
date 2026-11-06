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

def _count_bad_sets(n, in_bits):
    """Count 4-sets with no common dominator outside the set."""
    bad = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    if (common & ~mask) == 0:
                        bad += 1
    return bad

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

def _greedy_prune(n, in_bits, time_limit, deadline):
    """
    Greedily prune vertices from a valid tournament.
    Try removing each vertex; if the result is still valid, keep the removal.
    Returns (pruned_n, pruned_in_bits) or (n, in_bits) if no pruning possible.
    """
    # Work with a list of active vertices
    active = list(range(n))
    cur_in_bits = in_bits[:]
    cur_n = n
    
    while time.time() < deadline - 5:
        removed_any = False
        for v in list(active):
            if time.time() > deadline - 5:
                    break
            # Try removing vertex v
            new_active = [x for x in active if x != v]
            if len(new_active) < 5:
                continue
            
            # Check if removing v keeps validity
            # For each 4-set in new_active, check common dominator in new_active
            valid = True
            # Build index mapping
            new_idx = {x: i for i, x in enumerate(new_active)}
            
            for combo in combinations(new_active, 4):
                mask = 0
                for x in combo:
                    mask |= (1 << x)
                common = cur_in_bits[combo[0]] & cur_in_bits[combo[1]] & cur_in_bits[combo[2]] & cur_in_bits[combo[3]]
                # Common dominator must be in new_active and not in the 4-set
                common_in_new = common & ~mask
                # Check if any dominator is in new_active
                has_dominator = False
                for x in new_active:
                    if (common_in_new >> x) & 1:
                        has_dominator = True
                        break
                if not has_dominator:
                    valid = False
                    break
            
            if valid:
                # Remove v
                active = new_active
                removed_any = True
                cur_n = len(active)
        
        if not removed_any:
            break
    
    # Rebuild in_bits for the pruned tournament
    if len(active) < n:
        # Map old indices to new indices
        old_to_new = {x: i for i, x in enumerate(active)}
        new_in_bits = [0] * len(active)
        for old_i in active:
            new_i = old_to_new[old_i]
            for old_j in active:
                if old_i == old_j:
                    continue
                # Check if old_j -> old_i in original
                if (cur_in_bits[old_i] >> old_j) & 1:
                    new_in_bits[new_i] |= (1 << old_to_new[old_j])
    
    return len(active), new_in_bits if len(active) < n else in_bits

def _local_search(n, in_bits, time_limit, seed=42):
    """
    Local search to minimize the number of bad 4-sets.
    Returns (orientation, bad_count) if bad_count == 0, else None.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Work with in_bits
    cur_in_bits = in_bits[:]
    
    # Precompute edge index
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
    
    # Precompute 4-subsets and their states
    tuples = []
    states = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    tuples.append((a, b, c, d))
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = cur_in_bits[a] & cur_in_bits[b] & cur_in_bits[c] & cur_in_bits[d]
                    states.append(1 if (common & ~mask) == 0 else 0)
    
    bad = sum(states)
    if bad == 0:
        return _in_bits_to_orientation(n, cur_in_bits)
    
    vertex_to_tuples = [[] for _ in range(n)]
    for i, t in enumerate(tuples):
        for v in t:
            vertex_to_tuples[v].append(i)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        aff_idx = set(vertex_to_tuples[i])
        aff_idx.update(vertex_to_tuples[j])
        
        delta = 0
        changed_states = []
        
        if (cur_in_bits[j] >> i) & 1:
            new_in_i = cur_in_bits[i] | (1 << j)
            new_in_j = cur_in_bits[j] & ~(1 << i)
            flip_to_j_to_i = True
        else:
            new_in_i = cur_in_bits[i] & ~(1 << j)
            new_in_j = cur_in_bits[j] | (1 << i)
            flip_to_j_to_i = False
        
        for idx_t in aff_idx:
            a, b, c, d = tuples[idx_t]
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            ni_a = new_in_i if a == i else (new_in_j if a == j else cur_in_bits[a])
            ni_b = new_in_i if b == i else (new_in_j if b == j else cur_in_bits[b])
            ni_c = new_in_i if c == i else (new_in_j if c == j else cur_in_bits[c])
            ni_d = new_in_i if d == i else (new_in_j if d == j else cur_in_bits[d])
            
            common = ni_a & ni_b & ni_c & ni_d
            is_bad_now = 1 if (common & ~mask) == 0 else 0
            
            old_state = states[idx_t]
            if old_state != is_bad_now:
                delta += (is_bad_now - old_state)
                changed_states.append((idx_t, is_bad_now))
        
        if delta < 0 or (T > T_min and rng.random() < np.exp(-max(delta, 0) / T)):
            if flip_to_j_to_i:
                cur_in_bits[i] |= (1 << j)
                cur_in_bits[j] &= ~(1 << i)
            else:
                cur_in_bits[i] &= ~(1 << j)
                cur_in_bits[j] |= (1 << i)
            
            for idx_t, state in changed_states:
                states[idx_t] = state
            bad += delta
            
            if bad == 0:
                return _in_bits_to_orientation(n, cur_in_bits)
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    best_n = 91
    best_orientation = None
    
    # Step 1: Get Paley(67) as starting point
    paley_orient = _get_paleyn_orientation(67)
    if paley_orient is None:
        # Fallback
        n = 90
        return {'n': n, 'orientation': [0] * (n * (n - 1) // 2)}
    
    paley_in_bits = _orientation_to_in_bits(67, paley_orient)
    
    # Step 2: Greedy pruning
    pruned_n, pruned_in_bits = _greedy_prune(67, paley_in_bits, 30.0, deadline)
    
    if pruned_n < 67:
        # Verify and use pruned result
        if _validate_tournament(pruned_n, pruned_in_bits):
            best_n = pruned_n
            best_orientation = _in_bits_to_orientation(pruned_n, pruned_in_bits)
    
    # Step 3: Try local search on smaller n values
    # Try n from best_n-1 down to 31
    search_start = best_n - 1 if best_n < 91 else 66
    
    for n in range(search_start, 30, -1):
        if time.time() > deadline - 15:
            break
        
        # Time allocation: more time for larger n (closer to current best)
        remaining = deadline - 15 - time.time()
        attempts_left = n - 30
        time_alloc = max(2.0, min(8.0, remaining / max(1, attempts_left)))
        
        # Try a few seeds
        for seed in [42, 123, 777]:
            if time.time() > deadline - 15:
                break
            
            # Initialize with random tournament
            rng = random.Random(seed)
            in_bits = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        in_bits[j] |= (1 << i)
                    else:
                        in_bits[i] |= (1 << j)
            
            orientation = _local_search(n, in_bits, time_alloc / 3, seed=seed)
            if orientation is not None:
                best_n = n
                best_orientation = orientation
                break  # Found valid for this n, try smaller
        
        # If we found a valid tournament for this n, continue to smaller n
    
    # Fallback to Paley(67) if nothing better
    if best_orientation is None or best_n > 67:
        best_n = 67
        best_orientation = paley_orient
    
    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
