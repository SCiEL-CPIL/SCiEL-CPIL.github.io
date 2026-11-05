import time
import random
import numpy as np
from itertools import combinations

def _build_in_neigh(n, orientation):
    """Build in-neighbor bitsets from orientation list."""
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[i] |= (1 << j)
            idx += 1
    return in_neigh

def _build_edge_map(n):
    """Precompute edge index -> (i, j) mapping."""
    edge_map = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map.append((i, j))
            idx += 1
    return edge_map

def _check_property_fast(n, in_neigh):
    """Check if every 4 vertices have a common dominator using bitsets."""
    for a in range(n - 3):
        ia = in_neigh[a]
        for b in range(a + 1, n - 2):
            ia_b = ia & in_neigh[b]
            for c in range(b + 1, n - 1):
                ia_b_c = ia_b & in_neigh[c]
                for d in range(c + 1, n):
                    common = ia_b_c & in_neigh[d]
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    if (common & ~mask) == 0:
                        return False
    return True

def _check_property(n, orientation):
    in_neigh = _build_in_neigh(n, orientation)
    return _check_property_fast(n, in_neigh)

def _paley_orientation(p):
    """Return orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    n = p
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': n, 'orientation': orientation}

def _try_paley_and_check(p, deadline):
    """Try Paley tournament on p vertices and check if valid."""
    if time.time() > deadline:
        return None
    result = _paley_orientation(p)
    n = result['n']
    in_neigh = _build_in_neigh(n, result['orientation'])
    if _check_property_fast(n, in_neigh):
        return result
    return None

def _greedy_constructive(deadline, target_n, seed=None):
    """
    Greedy constructive algorithm: start with a small valid core and add vertices
    one at a time, choosing orientations to maintain the 4-domination property.
    
    The key insight: when adding vertex v to a tournament on {0,...,v-1}, we need
    every 4-set that includes v to have a common dominator. Since v is new, any 4-set
    containing v must be dominated by one of the existing vertices (v cannot dominate
    a set that includes itself).
    
    For a 4-set {v, a, b, c}, we need some existing vertex d (d != v,a,b,c) such that
    d -> v, d -> a, d -> b, d -> c.
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with a small valid core. A tournament on 5 vertices where one vertex
    # dominates all others works for 4-sets within the core.
    # Actually, let's start with n=5 and a "king" structure: vertex 0 beats all others.
    # But we need to be more careful.
    
    # Let's start with a random small tournament and grow it.
    # We'll maintain the in_neigh bitsets incrementally.
    
    # Start with 5 vertices
    n_start = 5
    in_neigh = [0] * n_start
    # Create a random initial tournament on n_start vertices
    for i in range(n_start):
        for j in range(i + 1, n_start):
            if random.random() < 0.5:
                # i -> j
                in_neigh[j] |= (1 << i)
            else:
                # j -> i
                in_neigh[i] |= (1 << j)
    
    # Verify the core is valid
    if not _check_property_fast(n_start, in_neigh):
        # Try a few random cores
        for attempt in range(100):
            in_neigh = [0] * n_start
            for i in range(n_start):
                for j in range(i + 1, n_start):
                    if random.random() < 0.5:
                        in_neigh[j] |= (1 << i)
                    else:
                        in_neigh[i] |= (1 << j)
            if _check_property_fast(n_start, in_neigh):
                break
        else:
            # If no valid core found, use a star: vertex 0 beats all
            in_neigh = [0] * n_start
            for j in range(1, n_start):
                in_neigh[j] |= (1 << 0)
            # Also need to orient edges among 1..4
            for i in range(1, n_start):
                for j in range(i + 1, n_start):
                    if random.random() < 0.5:
                        in_neigh[j] |= (1 << i)
                    else:
                        in_neigh[i] |= (1 << j)
    
    current_n = n_start
    
    # Add vertices one by one
    while current_n < target_n:
        if time.time() > deadline:
            break
        
        v = current_n  # New vertex index
        in_neigh.append(0)  # Initialize in_neigh for new vertex
        
        # Determine orientation from each existing vertex i to v
        # For each 4-set {v, a, b, c} (where a,b,c < v), we need a dominator.
        # The dominator must be some d < v, d != a,b,c, such that d->v, d->a, d->b, d->c.
        
        # Strategy: For each existing vertex i, decide whether i->v or v->i.
        # If i->v, then i can potentially dominate 4-sets {v, a, b, c} where i->a, i->b, i->c.
        # If v->i, then i cannot dominate any 4-set containing v (since i doesn't beat v).
        
        # We want to choose orientations to ensure all 4-sets {v, a, b, c} are dominated.
        # A 4-set {v, a, b, c} is dominated if there exists d < v, d != a,b,c, with
        # d->v, d->a, d->b, d->c.
        
        # Let's try a greedy approach: for each vertex i, compute how many 4-sets {v, a, b, c}
        # would be dominated if i->v (i.e., i beats a, b, c and i->v).
        # Then choose orientations to maximize coverage.
        
        # First, compute for each i < v, the set of 4-tuples (a,b,c) such that i->a, i->b, i->c.
        # If we set i->v, then all such 4-sets {v,a,b,c} are dominated by i.
        
        # But we also need to consider that multiple vertices can dominate the same 4-set.
        
        # Simplified greedy: try all 2^v orientations? No, v can be up to 67.
        
        # Alternative: randomize and check.
        # Let's try a heuristic: set i->v if i has high out-degree to existing vertices.
        
        # Actually, let's use a different approach: 
        # 1. Randomly orient edges from v to existing vertices.
        # 2. Check if all 4-sets {v, a, b, c} are dominated.
        # 3. If not, flip some edges and try again.
        
        # For efficiency, let's precompute which 4-sets {v, a, b, c} need to be dominated.
        # There are C(v, 3) such sets.
        
        # Let's try multiple random orientations and pick the best one.
        best_orientation = None
        best_violations = float('inf')
        
        num_trials = min(50, max(5, 200 // current_n))
        
        for trial in range(num_trials):
            # Randomly orient edges from v to existing vertices
            candidate_in_neigh = in_neigh[:] + [0]
            
            # Track which existing vertices beat v
            beats_v = 0  # Bitset of vertices i < v such that i->v
            
            for i in range(current_n):
                if random.random() < 0.5:
                    # i -> v
                    beats_v |= (1 << i)
                    candidate_in_neigh[v] |= (1 << i)
                else:
                    # v -> i
                    candidate_in_neigh[i] |= (1 << v)
            
            # Now check which 4-sets {v, a, b, c} are dominated.
            # A 4-set {v, a, b, c} is dominated if there exists d < v, d != a,b,c,
            # such that d->v, d->a, d->b, d->c.
            
            # For each d that beats v, compute the set of 3-tuples (a,b,c) that d dominates.
            # d dominates (a,b,c) if d->a, d->b, d->c, i.e., a,b,c are in in_neigh[d].
            
            # We need every 3-subset of {0,...,v-1} to be dominated by at least one d that beats v.
            
            # Let's count violations: 3-subsets not dominated by any d in beats_v.
            
            # For each d in beats_v, the 3-subsets dominated by d are those where all three
            # vertices are in in_neigh[d].
            
            # This is complex. Let's just check directly for a sample of 4-sets.
            
            # Actually, let's check all 4-sets containing v.
            violations = 0
            for a in range(current_n - 2):
                for b in range(a + 1, current_n - 1):
                    for c in range(b + 1, current_n):
                        # Check if {v, a, b, c} has a common dominator
                        # Common dominator d must satisfy: d->v, d->a, d->b, d->c
                        # d->v means d is in beats_v
                        # d->a, d->b, d->c means d is in in_neigh[a] & in_neigh[b] & in_neigh[c]
                        
                        common = beats_v & candidate_in_neigh[a] & candidate_in_neigh[b] & candidate_in_neigh[c]
                        # d must not be a, b, or c (but d->a implies d != a, etc., so this is automatic)
                        if common == 0:
                            violations += 1
            
            if violations < best_violations:
                best_violations = violations
                best_orientation = candidate_in_neigh[:]
            
            if best_violations == 0:
                break
        
        if best_violations == 0:
            in_neigh = best_orientation
            current_n += 1
        else:
            # Can't add vertex v without violations. Try a different random seed for the core?
            # Or accept that we can't reach target_n with this approach.
            break
    
    # Convert in_neigh back to orientation
    orientation = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            if (in_neigh[j] >> i) & 1:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    
    return {'n': current_n, 'orientation': orientation}

def _local_search(n, time_limit, edge_map):
    """Local search for a valid tournament on n vertices with incremental violation tracking."""
    deadline = time.time() + time_limit
    
    # Initialize random tournament
    orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    in_neigh = _build_in_neigh(n, orientation)
    
    def count_violations(in_neigh, n):
        count = 0
        for a in range(n - 3):
            ia = in_neigh[a]
            for b in range(a + 1, n - 2):
                ia_b = ia & in_neigh[b]
                if ia_b == 0:
                    for c in range(b + 1, n - 1):
                        for d in range(c + 1, n):
                            count += 1
                    continue
                for c in range(b + 1, n - 1):
                    ia_b_c = ia_b & in_neigh[c]
                    if ia_b_c == 0:
                        for d in range(c + 1, n):
                            count += 1
                        continue
                    for d in range(c + 1, n):
                        common = ia_b_c & in_neigh[d]
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        if (common & ~mask) == 0:
                            count += 1
        return count
    
    # Count initial violations
    violations = count_violations(in_neigh, n)
    if violations == 0:
        return {'n': n, 'orientation': orientation}
    
    best_violations = violations
    best_orientation = orientation[:]
    
    step = 0
    initial_temp = 1.0
    final_temp = 0.01
    total_steps_est = max(1, int(time_limit * 1000))
    
    while time.time() < deadline:
        step += 1
        
        edge_idx = random.randint(0, len(orientation) - 1)
        i, j = edge_map[edge_idx]
        
        orientation[edge_idx] = 1 - orientation[edge_idx]
        
        old_bit = 1 - orientation[edge_idx]
        if old_bit == 0:
            in_neigh[j] &= ~(1 << i)
            in_neigh[i] |= (1 << j)
        else:
            in_neigh[i] &= ~(1 << j)
            in_neigh[j] |= (1 << i)
        
        check_interval = max(1, step // 100)
        if step % max(1, check_interval) == 0 or step <= 100:
            violations = count_violations(in_neigh, n)
            if violations < best_violations:
                best_violations = violations
                best_orientation = orientation[:]
                if violations == 0:
                    return {'n': n, 'orientation': orientation}
            else:
                if step < total_steps_est * 0.8:
                    temp = initial_temp * (final_temp / initial_temp) ** (step / max(1, total_steps_est * 0.8))
                    delta = violations - best_violations
                    if delta > 0 and random.random() < np.exp(-delta / max(temp, 1e-10)):
                        pass
                    else:
                        orientation[edge_idx] = 1 - orientation[edge_idx]
                        if old_bit == 0:
                            in_neigh[j] |= (1 << i)
                            in_neigh[i] &= ~(1 << j)
                        else:
                            in_neigh[i] |= (1 << j)
                            in_neigh[j] &= ~(1 << i)
                else:
                    orientation[edge_idx] = 1 - orientation[edge_idx]
                    if old_bit == 0:
                        in_neigh[j] |= (1 << i)
                        in_neigh[i] &= ~(1 << j)
                    else:
                        in_neigh[i] |= (1 << j)
                        in_neigh[j] &= ~(1 << i)
    
    if best_violations == 0:
        return {'n': n, 'orientation': best_orientation}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_result = None
    best_n = 91
    
    # Try greedy constructive for various target sizes
    for target_n in [60, 55, 50, 45, 40, 35, 31]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        time_for_this = min(20.0, remaining / 2)
        if time_for_this < 2:
            break
        
        result = _greedy_constructive(deadline, target_n)
        if result is not None and result['n'] < best_n:
            best_result = result
            best_n = result['n']
        
        # If greedy found a valid tournament, try to improve with local search
        if best_result is not None and best_result['n'] < 67:
            # Try local search to find even smaller n
            pass
    
    # Try Paley tournaments
    primes_to_try = [31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in primes_to_try:
        if time.time() > deadline:
            break
        if p >= best_n:
            continue
            
        result = _try_paley_and_check(p, deadline)
        if result is not None:
            best_result = result
            best_n = p
            break
    
    # If we found a valid result, try local search for smaller n
    if best_result is not None and best_n < 67:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            for target_n in range(best_n - 1, max(4, best_n - 8), -1):
                if time.time() > deadline - 5:
                    break
                time_for_this = min(15.0, (deadline - time.time()) / 3)
                if time_for_this < 1:
                    break
                edge_map = _build_edge_map(target_n)
                result = _local_search(target_n, time_for_this, edge_map)
                if result is not None:
                    best_result = result
                    best_n = target_n
                    continue
                else:
                    break
    
    # Fallback: return Paley 67 if nothing better found
    if best_result is None or best_n > 67:
        if time.time() < deadline:
            result = _try_paley_and_check(67, deadline)
            if result is not None:
                return result
        return _paley_orientation(67)
    
    return best_result
