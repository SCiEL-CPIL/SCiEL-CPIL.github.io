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

def _fast_prune(n, in_bits, deadline):
    """
    Fast greedy pruning: only check 4-sets that include the removed vertex.
    A vertex v can be removed if every 4-set containing v has a dominator
    that is NOT v and NOT in the 4-set.
    Returns (pruned_n, pruned_in_bits).
    """
    active = list(range(n))
    cur_in_bits = in_bits[:]
    
    while time.time() < deadline - 3:
        removed_any = False
        for v in list(active):
            if time.time() > deadline - 3:
                break
            if len(active) <= 5:
                break
            
            # Check: for every 4-set S containing v, there exists a vertex
            # in (active \ S) that dominates all of S.
            # Equivalently: for every 3-set T subset of (active \ {v}),
            # the set {v} ∪ T must have a dominator in active \ ({v} ∪ T).
            # That dominator can be any vertex in active \ ({v} ∪ T).
            
            valid = True
            others = [x for x in active if x != v]
            
            for a in range(len(others)):
                for b in range(a + 1, len(others)):
                    for c in range(b + 1, len(others)):
                        x, y, z = others[a], others[b], others[c]
                        mask4 = (1 << v) | (1 << x) | (1 << y) | (1 << z)
                        common = cur_in_bits[v] & cur_in_bits[x] & cur_in_bits[y] & cur_in_bits[z]
                        # Need a dominator in active \ mask4
                        # i.e., common has a bit set that is in active but not in mask4
                        dominators = common & ~mask4
                        # Check if any dominator is in active
                        found = False
                        for w in active:
                            if (dominators >> w) & 1:
                                found = True
                                break
                        if not found:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                active = [x for x in active if x != v]
                removed_any = True
        
        if not removed_any:
            break
    
    # Rebuild in_bits for the pruned tournament
    if len(active) < n:
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

def _local_search_small(n, in_bits, time_limit, seed=42):
    """
    Local search for small n (n <= 50) to find valid tournament.
    Uses precomputed 4-subsets.
    Returns orientation if valid found, else None.
    """
    if n > 55:
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
    
    # Compute initial bad count
    states = [0] * num_tuples
    for idx_t, (a, b, c, d) in enumerate(tuples):
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        common = cur_in_bits[a] & cur_in_bits[b] & cur_in_bits[c] & cur_in_bits[d]
        states[idx_t] = 1 if (common & ~mask) == 0 else 0
    bad = sum(states)
    
    if bad == 0:
        return _in_bits_to_orientation(n, cur_in_bits)
    
    # Map vertex to tuple indices
    vertex_to_tuples = [[] for _ in range(n)]
    for i, t in enumerate(tuples):
        for v in t:
            vertex_to_tuples[v].append(i)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Get affected tuples
        aff_list = []
        for idx_t in vertex_to_tuples[i]:
            aff_list.append(idx_t)
        for idx_t in vertex_to_tuples[j]:
            if idx_t not in aff_list:
                aff_list.append(idx_t)
        
        delta = 0
        changed = []
        
        # Determine current direction
        # in_bits[j] has bit i set if i -> j
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
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
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
        
        if delta < 0 or (T > T_min and rng.random() < np.exp(-max(delta, 0) / T)):
            if (cur_in_bits[j] >> i) & 1:
                # i -> j, flip to j -> i
                cur_in_bits[i] |= (1 << j)
                cur_in_bits[j] &= ~(1 << i)
            else:
                # j -> i, flip to i -> j
                cur_in_bits[i] &= ~(1 << j)
                cur_in_bits[j] |= (1 << i)
            
            for idx_t, state in changed:
                states[idx_t] = state
            bad += delta
            
            if bad == 0:
                return _in_bits_to_orientation(n, cur_in_bits)
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    return None

def _prune_and_repair(n_orig, in_bits_orig, target_n, deadline):
    """
    Strategy: prune the original tournament down, then try to repair
    by flipping edges to fix any broken 4-sets.
    Returns orientation if a valid tournament of size <= target_n is found.
    """
    # First, try to prune to target_n
    pruned_n, pruned_bits = _fast_prune(n_orig, in_bits_orig, deadline - 10)
    
    if pruned_n <= target_n:
        if _validate_tournament(pruned_n, pruned_bits):
            return _in_bits_to_orientation(pruned_n, pruned_bits)
    
    # If pruning didn't get us there, try local search on the pruned result
    if pruned_n <= 55 and pruned_n >= target_n:
        orient = _local_search_small(pruned_n, pruned_bits, min(20.0, deadline - time.time() - 5), seed=42)
        if orient is not None:
            return orient
    
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
    
    # Step 2: Fast greedy pruning
    pruned_n, pruned_in_bits = _fast_prune(67, paley_in_bits, deadline - 20)
    
    if pruned_n < best_n:
        if _validate_tournament(pruned_n, pruned_in_bits):
            best_n = pruned_n
            best_orientation = _in_bits_to_orientation(pruned_n, pruned_in_bits)
    
    # Step 3: Try to find even smaller valid tournaments
    # Try local search starting from various initializations
    search_start = best_n - 1 if best_n < 91 else 66
    
    for n in range(search_start, 30, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - 10 - time.time()
        if remaining < 3:
            break
        
        # Time per n value
        time_per_n = min(15.0, remaining / max(1, search_start - n + 1))
        
        # Try multiple seeds
        found = False
        for seed in [42, 123, 777, 999, 555]:
            if time.time() > deadline - 10:
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
            
            time_left = min(time_per_n / 5, deadline - time.time() - 10)
            if time_left < 0.5:
                break
            
            orientation = _local_search_small(n, in_bits, time_left, seed=seed)
            if orientation is not None:
                best_n = n
                best_orientation = orientation
                found = True
                break
        
        if found:
            continue  # Try smaller n
        else:
            # If we couldn't find valid for this n, likely can't for smaller
            # But keep trying with less time
            pass
    
    # Fallback to Paley(67) if nothing better
    if best_orientation is None or best_n > 67:
        best_n = 67
        best_orientation = paley_orient
    
    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
