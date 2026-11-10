import time
import random
import sys

# EVOLVE-BLOCK-START

def _score_fast(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _score_with_bound(A, current_best):
    """Return max(|A+A|, |A*A|) but stop early if either set exceeds current_best."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            ss.add(s)
            if len(ss) >= current_best:
                return current_best
            p = ai * A[j]
            ps.add(p)
            if len(ps) >= current_best:
                return current_best
    return max(len(ss), len(ps))

def _exponent_to_value(e):
    """Convert exponent tuple (x,y,z) to 2^x * 3^y * 5^z."""
    x, y, z = e
    return (1 << x) * (3 ** y) * (5 ** z)

def _value_to_exponent(val):
    """Convert value to (x,y,z) if it's 5-smooth, else None."""
    x = 0
    while val % 2 == 0:
        val //= 2
        x += 1
    y = 0
    while val % 3 == 0:
        val //= 3
        y += 1
    z = 0
    while val % 5 == 0:
        val //= 5
        z += 1
    if val != 1:
        return None
    return (x, y, z)

def _generate_lattice(max_x=14, max_y=9, max_z=6):
    """Generate all 5-smooth numbers up to 10000 as exponent tuples."""
    lattice = []
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            for z in range(max_z + 1):
                val = (1 << x) * (3 ** y) * (5 ** z)
                if val <= 10000:
                    lattice.append(((x, y, z), val))
    lattice.sort(key=lambda t: t[1])
    return lattice

def _exponent_distance(e1, e2):
    """L1 distance in exponent space."""
    return abs(e1[0] - e2[0]) + abs(e1[1] - e2[1]) + abs(e1[2] - e2[2])

def _exponent_product_polytope_volume(exponents):
    """
    Estimate the 'volume' of the product polytope in exponent space.
    A*A corresponds to all sums e_i + e_j. The number of distinct sums
    is bounded by the volume of the Minkowski sum of the exponent set.
    We use the bounding box volume as a proxy.
    """
    if not exponents:
        return 0
    xs = [e[0] for e in exponents]
    ys = [e[1] for e in exponents]
    zs = [e[2] for e in exponents]
    # Minkowski sum of a set with itself has bounding box = 2 * original
    vol = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) * (max(zs) - min(zs) + 1)
    return vol

def _local_search_exponent(A, best_score, lattice, deadline):
    """
    Local search in exponent space.
    A is a list of 20 values. We represent them as exponent tuples.
    Moves: change one exponent coordinate by +/-1, or swap with a nearby lattice point.
    """
    n = len(A)
    A = list(A)
    
    # Convert to exponents
    exps = []
    for v in A:
        e = _value_to_exponent(v)
        if e is None:
            # Not smooth, use (0,0,0) as fallback
            exps.append((0, 0, 0))
        else:
            exps.append(e)
    
    A_set = set(A)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single-coordinate moves in exponent space
        for i in range(n):
            if time.time() > deadline:
                break
            old_exp = exps[i]
            old_val = A[i]
            A_set.discard(old_val)
            
            # Try +/-1 on each coordinate
            for coord in range(3):
                for delta in [1, -1]:
                    new_exp = list(old_exp)
                    new_exp[coord] += delta
                    if new_exp[coord] < 0:
                        continue
                    new_val = _exponent_to_value(tuple(new_exp))
                    if new_val > 10000 or new_val == old_val or new_val in A_set:
                        continue
                    
                    A[i] = new_val
                    sc = _score_with_bound(A, best_score)
                    if sc < best_score:
                        best_score = sc
                        exps[i] = tuple(new_exp)
                        improved = True
                        A_set.add(new_val)
                        A.sort()
                        break
                    else:
                        A[i] = old_val
                if improved:
                    break
            if improved:
                break
        
        # Try swapping with nearby lattice points (by value proximity)
        if not improved:
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = A[i]
                A_set.discard(old_val)
                
                # Find nearby smooth numbers in value space
                # Look in range [old_val/2, old_val*2]
                lo = max(1, old_val // 2)
                hi = min(10000, old_val * 2)
                
                candidates = [(abs(lv - old_val), lv) for le, lv in lattice if lo <= lv <= hi]
                candidates.sort()
                
                for _, new_val in candidates[:20]:  # Try up to 20 nearest
                    if new_val == old_val or new_val in A_set:
                        continue
                    
                    A[i] = new_val
                    sc = _score_with_bound(A, best_score)
                    if sc < best_score:
                        best_score = sc
                        new_exp = _value_to_exponent(new_val)
                        if new_exp:
                            exps[i] = new_exp
                        improved = True
                        A_set.add(new_val)
                        A.sort()
                        break
                    else:
                        A[i] = old_val
                
                A_set.add(old_val)
                if improved:
                    break
        
        # Random perturbation in exponent space
        if not improved:
            found = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                old_val = A[i]
                old_exp = exps[i]
                A_set.discard(old_val)
                
                # Random small move in exponent space
                new_exp = [max(0, old_exp[c] + random.randint(-2, 2)) for c in range(3)]
                new_val = _exponent_to_value(tuple(new_exp))
                if new_val > 10000 or new_val == old_val or new_val in A_set:
                    A_set.add(old_val)
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    exps[i] = tuple(new_exp)
                    A_set.add(new_val)
                    A.sort()
                    found = True
                    break
                else:
                    A[i] = old_val
                    A_set.add(old_val)
            
            if not found:
                break
    
    return A, best_score

def _local_search_value(A, best_score, pool, deadline):
    """Standard value-space local search."""
    n = len(A)
    A = list(A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old_val = A[i]
            A_set = set(A)
            A_set.discard(old_val)
            
            for new_val in pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    A.sort()
                    break
                A[i] = old_val
            
            if improved:
                break
        
        if not improved:
            found = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                old_val = A[i]
                new_val = random.choice(pool)
                if new_val == old_val or new_val in A:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    found = True
                    A.sort()
                    break
                A[i] = old_val
            
            if not found:
                break
    
    return A, best_score

def _generate_clustered_lattice_set(lattice, n=20, seed=42):
    """
    Generate a set of n smooth numbers that are clustered in exponent space.
    Start from a random point and grow by taking nearest neighbors in exponent space.
    """
    rng = random.Random(seed)
    
    # Pick a random starting point in the middle of the lattice
    mid_x = 3
    mid_y = 2
    mid_z = 1
    start_val = _exponent_to_value((mid_x, mid_y, mid_z))
    
    if start_val > 10000:
        start_val = 16  # 2^4
        mid_x, mid_y, mid_z = 4, 0, 0
    
    # Greedy: start from center, add nearest unclaimed smooth number
    selected = [start_val]
    selected_set = {start_val}
    
    current_center = (mid_x, mid_y, mid_z)
    
    for _ in range(n - 1):
        # Find the unselected smooth number closest to current center in exponent space
        best_dist = float('inf')
        best_val = None
        best_exp = None
        
        for exp, val in lattice:
            if val in selected_set:
                continue
            dist = _exponent_distance(exp, current_center)
            if dist < best_dist:
                best_dist = dist
                best_val = val
                best_exp = exp
        
        if best_val is None:
            break
        
        selected.append(best_val)
        selected_set.add(best_val)
        current_center = best_exp
    
    selected.sort()
    return selected

def _generate_multiplicative_grid(lattice, n=20):
    """
    Generate a set from a 2D slice of the 3D lattice.
    Fix one coordinate and take a grid in the other two.
    This gives a structured product set.
    """
    # Try different fixed coordinates
    best_set = None
    best_score = float('inf')
    
    for fix_coord in range(3):
        for fix_val in range(0, 8):
            # Take all lattice points with this fixed coordinate
            points = []
            for exp, val in lattice:
                if exp[fix_coord] == fix_val:
                    points.append((exp, val))
            
            if len(points) < n:
                continue
            
            # Sort by value and take first n
            points.sort(key=lambda t: t[1])
            candidate = [val for _, val in points[:n]]
            
            if len(candidate) != n:
                continue
            
            sc = _score_fast(candidate)
            if sc < best_score:
                best_score = sc
                best_set = candidate
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: evaluator kills at 90s
    
    n = 20
    maximum = 10000
    
    # Generate the full lattice of 5-smooth numbers
    lattice = _generate_lattice()
    smooth_list = [val for _, val in lattice]
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Try structured lattice sets
    # 1a: Multiplicative grid (2D slice)
    grid_set, grid_score = _generate_multiplicative_grid(lattice, n)
    if grid_set and grid_score < best_score:
        best_score = grid_score
        best_A = grid_set
    
    # 1b: Clustered lattice sets with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        if time.time() > deadline:
            break
        clustered = _generate_clustered_lattice_set(lattice, n, seed)
        if len(clustered) == n:
            sc = _score_with_bound(clustered, best_score)
            if sc < best_score:
                best_score = sc
                best_A = clustered
    
    # 1c: First 20 smooth numbers
    if len(smooth_list) >= n:
        candidate = smooth_list[:n]
        sc = _score_with_bound(candidate, best_score)
        if sc < best_score:
            best_score = sc
            best_A = candidate
    
    if best_A is None:
        best_A = list(range(1, n + 1))
        best_score = _score_fast(best_A)
    
    # Phase 2: Exponent-space local search
    max_val = max(best_A)
    # Pool for value-space search: smooth numbers near current set
    pool = [s for s in smooth_list if s <= max_val * 3]
    pool.sort()
    
    best_A, best_score = _local_search_exponent(best_A, best_score, lattice, deadline)
    
    # Phase 3: Value-space local search for final polish
    pool = [s for s in smooth_list if s <= max(best_A) * 3]
    pool.sort()
    best_A, best_score = _local_search_value(best_A, best_score, pool, deadline)
    
    # Phase 4: Try a few more random restarts in exponent space
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Random clustered set
        seed = random.randint(0, 100000)
        clustered = _generate_clustered_lattice_set(lattice, n, seed)
        if len(clustered) != n:
            continue
        
        sc = _score_with_bound(clustered, best_score)
        if sc < best_score:
            best_score = sc
            best_A = clustered
        
        # Exponent-space local search from this candidate
        best_A, best_score = _local_search_exponent(best_A, best_score, lattice, deadline)
        
        # Value-space polish
        pool = [s for s in smooth_list if s <= max(best_A) * 3]
        pool.sort()
        best_A, best_score = _local_search_value(best_A, best_score, pool, deadline)
    
    # Final validation
    if len(best_A) != n or len(set(best_A)) != n or any(x <= 0 or x > maximum for x in best_A):
        best_A = list(range(1, n + 1))
    
    return best_A

# EVOLVE-BLOCK-END
