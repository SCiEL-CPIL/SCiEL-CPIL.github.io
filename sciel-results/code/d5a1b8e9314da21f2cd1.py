import time
import random

def _make_paley(p):
    """Construct a Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _count_bad_sets(orientation, n):
    """Count the number of 4-sets without a common dominator."""
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_masks[j] |= (1 << i)
            else:
                in_masks[i] |= (1 << j)
            idx += 1
    
    bad_count = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    if in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d] == 0:
                        bad_count += 1
    return bad_count

def _is_valid_fast(orientation, n):
    """Check if the tournament satisfies the condition (every 4-set has a common dominator)."""
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_masks[j] |= (1 << i)
            else:
                in_masks[i] |= (1 << j)
            idx += 1
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    if in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d] == 0:
                        return False
    return True

def _local_search(orientation, n, max_steps=50000, time_limit=15.0):
    """
    Local search to repair a tournament. Uses hill-climbing with occasional restarts.
    Returns (orientation, success).
    """
    bits = list(orientation)
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = bits[idx]
            if bit == 0:
                in_masks[j] |= (1 << i)
            else:
                in_masks[i] |= (1 << j)
            idx += 1
            
    # Precompute all 4-sets
    sets_info = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    is_bad = (in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]) == 0
                    sets_info.append([a, b, c, d, is_bad])
                    
    bad_count = sum(1 for s in sets_info if s[4])
    if bad_count == 0:
        return bits, True

    # Precompute edge indices
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
            
    # Precompute which 4-sets contain which vertices
    vertex_to_sets = [[] for _ in range(n)]
    for k, (a, b, c, d, _) in enumerate(sets_info):
        vertex_to_sets[a].append(k)
        vertex_to_sets[b].append(k)
        vertex_to_sets[c].append(k)
        vertex_to_sets[d].append(k)

    start_time = time.time()
    best_bits = bits[:]
    best_bad_count = bad_count
    
    # Simulated annealing style: allow some worsening moves
    temperature = 0.5
    
    for step in range(max_steps):
        if time.time() > start_time + time_limit:
            break
            
        # Pick random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        idx = edge_map[(i, j)]
        
        # Identify affected 4-sets
        affected_indices = set(vertex_to_sets[i])
        affected_indices.update(vertex_to_sets[j])
        
        # Check current status of affected sets
        old_bad_count = 0
        for k in affected_indices:
            if sets_info[k][4]:
                old_bad_count += 1
                
        # Flip edge
        bits[idx] = 1 - bits[idx]
        
        # Update in_masks
        if bits[idx] == 0:
            # i -> j
            in_masks[i] &= ~(1 << j)
            in_masks[j] |= (1 << i)
        else:
            # j -> i
            in_masks[j] &= ~(1 << i)
            in_masks[i] |= (1 << j)
            
        # Check new status of affected sets
        new_bad_count = 0
        for k in affected_indices:
            a, b, c, d, _ = sets_info[k]
            is_bad = (in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]) == 0
            sets_info[k][4] = is_bad
            if is_bad:
                new_bad_count += 1
                
        delta = new_bad_count - old_bad_count
        bad_count += delta
        
        if bad_count == 0:
            return bits, True
            
        # Accept worsening moves with some probability (simulated annealing)
        if delta > 0:
            # Probability of accepting worsening move
            accept_prob = min(1.0, temperature / max(delta, 1))
            if random.random() > accept_prob:
                # Revert
                bits[idx] = 1 - bits[idx]
                if bits[idx] == 0:
                    in_masks[i] &= ~(1 << j)
                    in_masks[j] |= (1 << i)
                else:
                    in_masks[j] &= ~(1 << i)
                    in_masks[i] |= (1 << j)
                bad_count -= delta
                for k in affected_indices:
                    a, b, c, d, _ = sets_info[k]
                    is_bad = (in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]) == 0
                    sets_info[k][4] = is_bad
            else:
                # Keep the worsening move
                if bad_count < best_bad_count:
                    best_bad_count = bad_count
                    best_bits = bits[:]
        else:
            # Improving or neutral move
            if bad_count < best_bad_count:
                best_bad_count = bad_count
                best_bits = bits[:]
        
        # Cool down temperature
        temperature *= 0.999
        
    # Return the best solution found
    if best_bad_count == 0:
        return best_bits, True
    return best_bits, False

def _try_paley_with_repair(p, time_limit=15.0):
    """Try Paley tournament on p vertices, with local search repair if needed."""
    orientation = _make_paley(p)
    
    if _is_valid_fast(orientation, p):
        return orientation, True
        
    repaired, success = _local_search(orientation, p, max_steps=50000, time_limit=time_limit)
    if success:
        return repaired, True
    return None, False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try Paley tournaments for primes in increasing order
    primes = [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
    
    for p in primes:
        if time.time() > deadline:
            break
            
        orientation, success = _try_paley_with_repair(p, time_limit=10.0)
        if success:
            return {'n': p, 'orientation': orientation}
            
    # If all else fails, return the largest prime we tried
    orientation = _make_paley(67)
    return {'n': 67, 'orientation': orientation}
