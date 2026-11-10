"""Optimized search for minimum maximal no-isosceles set on N=16 grid."""
import time
import random
import itertools
from collections import defaultdict

N = 16
DEADLINE = 320  # seconds, with 10s safety margin before 330s kill

# Precompute all squared distances between pairs of grid points
# and group by distance for fast isosceles detection
_ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
_POINT_INDEX = {p: i for i, p in enumerate(_ALL_POINTS)}
_NUM_POINTS = len(_ALL_POINTS)

# For each point, store list of (distance, other_point_index)
# Actually, for isosceles check we need: given point p and two existing points a, b,
# check if dist(p,a)==dist(p,b) or dist(p,a)==dist(a,b) or dist(p,b)==dist(a,b)

# Precompute squared distances from each point to all others
# dist_from[i][j] = squared distance between _ALL_POINTS[i] and _ALL_POINTS[j]
dist_from = [[0] * _NUM_POINTS for _ in range(_NUM_POINTS)]
for i, (xi, yi) in enumerate(_ALL_POINTS):
    for j, (xj, yj) in enumerate(_ALL_POINTS):
        dx = xi - xj
        dy = yi - yj
        dist_from[i][j] = dx * dx + dy * dy

# For faster isosceles check with a set of chosen points:
# When we add point p, we need to verify that for every pair (a, b) in chosen,
# the triangle (p, a, b) is not isosceles.
# This is O(k^2) per insertion where k = |chosen|.

# Alternative: maintain for each pair of chosen points, the set of "forbidden" points
# A point p is forbidden by pair (a,b) if triangle (p,a,b) is isosceles.
# We can precompute for each pair (a,b) the set of points that form an isosceles triangle with them.

# Precompute forbidden sets: for each pair (i, j), which points k make triangle (i,j,k) isosceles?
# This is O(N^6) precomputation which is 16^6 = 16M... too much.

# Better approach: maintain a set of "blocked" points. When we add point p to chosen,
# for each existing point a in chosen, and each existing point b in chosen (a != b),
# the new point p creates constraints. But actually, the constraint is on FUTURE points.
# 
# Let me think differently. A set S is "no-isosceles" if no three points in S form an isosceles triangle.
# A set S is "maximal" if no grid point can be added to S while maintaining the no-isosceles property.
#
# To check if point p can be added to S: for every pair (a,b) in S, triangle (p,a,b) must not be isosceles.
# Also, p must not coincide with any point in S (but we're working with distinct grid points).
#
# Efficient check: For a candidate point p and current set S of size k, we need to check k*(k-1)/2 pairs.
# Each check is O(1) with precomputed distances.
#
# For random restarts, we want to be fast. Let's use a different strategy:
# Instead of checking all pairs, maintain for each point p a counter of how many "isosceles conflicts" 
# it has with the current set. But this is complex.
#
# Simpler: just do the O(k^2) check per candidate, but be smart about which candidates to try.

def _can_add(p_idx, chosen_indices):
    """Check if point p_idx can be added to the set of chosen_indices without creating an isosceles triangle."""
    k = len(chosen_indices)
    if k < 2:
        return True
    for i in range(k):
        a = chosen_indices[i]
        da = dist_from[p_idx][a]
        for j in range(i + 1, k):
            b = chosen_indices[j]
            db = dist_from[p_idx][b]
            ab = dist_from[a][b]
            if da == db or da == ab or db == ab:
                return False
    return True


def _construct_greedy_randomized(start_seed):
    """Greedy construction with randomized point ordering."""
    rng = random.Random(start_seed)
    order = list(range(_NUM_POINTS))
    rng.shuffle(order)
    
    chosen = []
    for idx in order:
        if _can_add(idx, chosen):
            chosen.append(idx)
    
    return chosen


def _is_maximal(chosen_indices):
    """Check if the set is maximal (no point can be added)."""
    chosen_set = set(chosen_indices)
    for idx in range(_NUM_POINTS):
        if idx in chosen_set:
            continue
        if _can_add(idx, chosen_indices):
            return False
    return True


def _try_improve(chosen_indices, rng, max_iterations=5000):
    """Try to improve the set by removing points and rebuilding."""
    best = list(chosen_indices)
    best_size = len(best)
    
    for _ in range(max_iterations):
        # Try removing a random point and see if we can rebuild a smaller maximal set
        if len(best) <= 1:
            break
        idx_to_remove = rng.choice(best)
        # Remove it
        remaining = [x for x in best if x != idx_to_remove]
        
        # Try to complete to a maximal set greedily from remaining
        # But we need to ensure maximality
        # Let's try: start from remaining, add points greedily in random order
        # until maximal
        rng2 = random.Random(rng.randint(0, 2**31))
        order = list(range(_NUM_POINTS))
        rng2.shuffle(order)
        
        new_set = list(remaining)
        new_set_set = set(remaining)
        
        for idx in order:
            if idx in new_set_set:
                continue
            if _can_add(idx, new_set):
                new_set.append(idx)
                new_set_set.add(idx)
        
        # Check if maximal
        if _is_maximal(new_set):
            if len(new_set) < best_size:
                best = new_set
                best_size = len(new_set)
    
    return best


def construct():
    start_time = time.time()
    
    # First, get a baseline from a few randomized greedy constructions
    best_set = None
    best_size = _NUM_POINTS + 1
    
    # Try several random orderings
    for seed in range(50):
        if time.time() - start_time > DEADLINE - 30:
            break
        s = _construct_greedy_randomized(seed)
        if _is_maximal(s) and len(s) < best_size:
            best_size = len(s)
            best_set = s
    
    if best_set is None:
        # Fallback: simple greedy
        best_set = _construct_greedy_randomized(0)
        best_size = len(best_set)
    
    # Now try to improve with local search
    rng = random.Random(42)
    while time.time() - start_time < DEADLINE - 10:
        improved = _try_improve(best_set, rng, max_iterations=2000)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
        # Occasionally try a completely new random start
        if rng.random() < 0.1:
            new_s = _construct_greedy_randomized(rng.randint(0, 100000))
            if _is_maximal(new_s) and len(new_s) < best_size:
                best_size = len(new_s)
                best_set = new_s
    
    # Convert back to (x, y) tuples
    return [_ALL_POINTS[i] for i in best_set]
