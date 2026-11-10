"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute_data(all_points):
    """
    Precompute:
    - dists[i][j]: squared distance between point i and point j
    - For each point p and each existing set S, we need fast can_add checks.
    
    Key insight: For a point p and set S, p can be added iff:
    (1) All distances d(p, s) for s in S are distinct
    (2) For all pairs (a, b) in S: d(p,a) != d(a,b) and d(p,b) != d(a,b)
    
    We precompute dists and use a hybrid approach.
    """
    n = len(all_points)
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = all_points[i]
        for j in range(i + 1, n):
            xj, yj = all_points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dists[i][j] = d
            dists[j][i] = d
    return dists


def _can_add(p_idx, chosen, dists):
    """Check if p_idx can be added to chosen without forming isosceles triangle."""
    if not chosen:
        return True
    p = p_idx
    # Check 1: all distances from p to members of chosen must be distinct
    seen = set()
    for a in chosen:
        d = dists[p][a]
        if d in seen:
            return False
        seen.add(d)
    # Check 2: for each pair (a,b) in chosen, d(p,a) != d(a,b) and d(p,b) != d(a,b)
    k = len(chosen)
    for i in range(k):
        a = chosen[i]
        dp_a = dists[p][a]
        for j in range(i + 1, k):
            b = chosen[j]
            d_ab = dists[a][b]
            if dp_a == d_ab or dists[p][b] == d_ab:
                return False
    return True


def _is_maximal(chosen, all_indices, dists):
    """Check if no point can be added."""
    chosen_set = set(chosen)
    for p in all_indices:
        if p not in chosen_set:
            if _can_add(p, chosen, dists):
                return False
    return True


def _block_degree(p, dists, all_indices):
    """
    Compute how many other points p 'blocks' — i.e., how many points q
    cannot coexist with p in any valid no-isosceles set.
    A point q is blocked by p if {p, q} would force a third point into
    an isosceles configuration... this is complex.
    
    Simpler proxy: count how many points q have the property that
    there exists some r such that {p, q, r} forms an isosceles triangle.
    This gives a measure of how 'constraining' p is.
    """
    count = 0
    for q in all_indices:
        if q == p:
            continue
        # Check if there exists r such that {p,q,r} is isosceles
        # d(p,q) = d(p,r) or d(p,q) = d(q,r) or d(p,r) = d(q,r)
        d_pq = dists[p][q]
        blocked = False
        for r in all_indices:
            if r == p or r == q:
                continue
            if d_pq == dists[p][r] or d_pq == dists[q][r] or dists[p][r] == dists[q][r]:
                blocked = True
                break
        if blocked:
            count += 1
    return count


def _greedy_with_heuristic(order, dists, deadline):
    """
    Greedy construction: add points in the given order.
    Returns the maximal set.
    """
    chosen = []
    for p_idx in order:
        if time.time() > deadline:
            return None
        if _can_add(p_idx, chosen, dists):
            chosen.append(p_idx)
    return chosen


def _iterative_improve(chosen, all_indices, dists, deadline):
    """
    Iterative improvement:
    1. Try removing each point; if set remains maximal, keep removal.
    2. After removals, try to see if we can add any point back that was 
       previously blocked (this shouldn't happen if maximal, but after removals
       it might).
    
    Key structural difference: we do multiple passes of removal, and after
    each successful removal, we immediately check if the set is still maximal.
    If not, we try to add back the most 'constraining' blocked point.
    """
    chosen = list(chosen)
    chosen_set = set(chosen)
    
    # Phase 1: Aggressive removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(chosen)):
            if time.time() > deadline:
                return chosen
            candidate = chosen[:i] + chosen[i+1:]
            if _is_maximal(candidate, all_indices, dists):
                chosen = candidate
                chosen_set = set(chosen)
                changed = True
                break  # Restart scan after modification
    
    # Phase 2: Try single-point swaps to reduce size
    # Remove point i, add point j (where j was previously blocked)
    for i in range(len(chosen)):
        if time.time() > deadline:
            break
        removed = chosen[i]
        base = chosen[:i] + chosen[i+1:]
        # Find points that can be added to base
        for j in all_indices:
            if j in chosen_set:
                continue
            if _can_add(j, base, dists):
                new_set = base + [j]
                if _is_maximal(new_set, all_indices, dists):
                    chosen = new_set
                    chosen_set = set(chosen)
                    break
        if time.time() > deadline:
            break
    
    return chosen


def _smart_order(all_indices, dists, rng, mode):
    """
    Generate a point ordering using different heuristics.
    mode: 0=random, 1=by block degree (high first), 2=by block degree (low first),
          3=spatial clustering, 4=mixed
    """
    n = len(all_indices)
    if mode == 0:
        order = all_indices[:]
        rng.shuffle(order)
        return order
    elif mode == 1:
        # High block degree first (more constraining points early)
        degrees = [(p, _block_degree(p, dists, all_indices)) for p in all_indices]
        degrees.sort(key=lambda x: -x[1])
        order = [p for p, _ in degrees]
        # Add some randomness
        for _ in range(min(10, n)):
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            order[i], order[j] = order[j], order[i]
        return order
    elif mode == 2:
        # Low block degree first (less constraining points early)
        degrees = [(p, _block_degree(p, dists, all_indices)) for p in all_indices]
        degrees.sort(key=lambda x: x[1])
        order = [p for p, _ in degrees]
        for _ in range(min(10, n)):
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            order[i], order[j] = order[j], order[i]
        return order
    elif mode == 3:
        # Spatial: group by quadrant, shuffle within
        quadrants = {0: [], 1: [], 2: [], 3: []}
        mid = len(all_indices) ** 0.5
        for p in all_indices:
            # p is an index; decode to (x,y)
            x = p // 16
            y = p % 16
            qx = 0 if x < 8 else 1
            qy = 0 if y < 8 else 1
            q = qx * 2 + qy
            quadrants[q].append(p)
        order = []
        rng.shuffle(quadrants[0])
        rng.shuffle(quadrants[1])
        rng.shuffle(quadrants[2])
        rng.shuffle(quadrants[3])
        order.extend(quadrants[0])
        order.extend(quadrants[1])
        order.extend(quadrants[2])
        order.extend(quadrants[3])
        return order
    else:
        # Mixed: interleave random and degree-based
        rand_order = all_indices[:]
        rng.shuffle(rand_order)
        degrees = [(p, _block_degree(p, dists, all_indices)) for p in all_indices]
        degrees.sort(key=lambda x: -x[1])
        deg_order = [p for p, _ in degrees]
        order = []
        for i in range(len(rand_order)):
            if i % 2 == 0:
                order.append(rand_order[i])
            else:
                order.append(deg_order[i])
        return order


def construct():
    deadline = time.time() + 280
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    all_point_indices = list(range(n_points))
    
    dists = _precompute_data(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Precompute block degrees once (expensive but done once)
    # Actually, _block_degree is O(n^3) which is 16^3 * 16 = too slow for all points
    # Let's use a cheaper proxy: just use random and spatial orders
    
    # Strategy: multiple different ordering heuristics, each with greedy + improve
    
    # Trial 0: row-major
    result = _greedy_with_heuristic(all_point_indices, dists, deadline)
    if result is not None:
        result = _iterative_improve(result, all_point_indices, dists, deadline)
        best = [all_points[i] for i in result]
        best_size = len(best)
    
    # Various randomized and heuristic orderings
    modes = [0, 3, 0, 3, 0, 3, 1, 2, 4, 0]
    trial = 0
    for mode in modes:
        if time.time() > deadline:
            break
        order = _smart_order(all_point_indices, dists, rng, mode)
        result = _greedy_with_heuristic(order, dists, deadline)
        if result is None:
            continue
        result = _iterative_improve(result, all_point_indices, dists, deadline)
        if len(result) < best_size:
            best = [all_points[i] for i in result]
            best_size = len(result)
            if best_size <= 12:
                break
    
    # More random trials if time remains
    while time.time() < deadline:
        order = all_point_indices[:]
        rng.shuffle(order)
        result = _greedy_with_heuristic(order, dists, deadline)
        if result is None:
            break
        result = _iterative_improve(result, all_point_indices, dists, deadline)
        if len(result) < best_size:
            best = [all_points[i] for i in result]
            best_size = len(result)
            if best_size <= 12:
                break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
