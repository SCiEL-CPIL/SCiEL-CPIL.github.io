import itertools
import random
import time
from collections import defaultdict

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
dist = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        dist[i][j] = d
        dist[j][i] = d

# Precompute for each point, which other points form an isosceles triangle with it
# conflict_with[i] = set of points j such that {i, j} would create an isosceles triangle
# Actually, we need: for a set S, adding point p is valid if for all pairs (a,b) in S,
# p does not form an isosceles triangle with a and b.
# 
# Better: precompute for each point p, and each pair (a,b), whether p,a,b form isosceles.
# Then can_add(p, S) checks if any pair in S conflicts with p.

# Precompute: for each point p, which pairs (a,b) with a<b would form isosceles with p
# isosceles_with[p] = set of (a,b) pairs such that {p,a,b} is isosceles
isosceles_with = [set() for _ in range(NUM_POINTS)]
for p in range(NUM_POINTS):
    for a in range(p+1, NUM_POINTS):
        for b in range(a+1, NUM_POINTS):
            if b == p:
                continue
            pa = dist[p][a]
            pb = dist[p][b]
            ab = dist[a][b]
            if pa == pb or pa == ab or pb == ab:
                isosceles_with[p].add((a, b))

def get_pair_set(set_indices):
    """Get the set of pairs in the given set."""
    pairs = set()
    m = len(set_indices)
    for i in range(m):
        for j in range(i+1, m):
            a, b = set_indices[i], set_indices[j]
            if a > b:
                a, b = b, a
            pairs.add((a, b))
    return pairs

def can_add_fast(p, set_indices, pair_set=None):
    """Check if point p can be added to the set without creating isosceles triangles."""
    if pair_set is None:
        pair_set = get_pair_set(set_indices)
    # Check if any pair in the set conflicts with p
    for pair in pair_set:
        if pair in isosceles_with[p]:
            return False
    return True

def is_maximal_fast(set_indices, pair_set=None):
    """Check if the set is maximal (no point can be added)."""
    if pair_set is None:
        pair_set = get_pair_set(set_indices)
    set_set = set(set_indices)
    for p in range(NUM_POINTS):
        if p not in set_set:
            if can_add_fast(p, set_indices, pair_set):
                return False
    return True

def backtracking_search(max_size, deadline, start_time):
    """
    Search for a maximal no-isosceles set of size <= max_size.
    Uses backtracking with forward checking.
    """
    best_set = None
    best_size = max_size + 1
    
    def backtrack(current_set, current_pairs, remaining_points, depth, max_depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 5.0:
            return False
        
        # If we've reached the target size and the set is maximal, we found a solution
        if len(current_set) == max_size:
            if is_maximal_fast(current_set, current_pairs):
                best_set = list(current_set)
                best_size = max_size
                return True
            return False
        
        # Pruning: if we can't possibly reach a maximal set of this size
        # (heuristic: check if there are enough remaining points)
        if depth >= max_depth:
            return False
        
        # Try adding each remaining point
        for p in remaining_points:
            if can_add_fast(p, current_set, current_pairs):
                # Add p
                new_set = current_set + [p]
                new_pairs = set(current_pairs)
                for q in current_set:
                    a, b = p, q
                    if a > b:
                        a, b = b, a
                    new_pairs.add((a, b))
                
                # Forward checking: which points can still be added?
                new_remaining = []
                for r in remaining_points:
                    if r == p:
                        continue
                    if can_add_fast(r, new_set, new_pairs):
                        new_remaining.append(r)
                
                # Recurse
                if backtrack(new_set, new_pairs, new_remaining, depth + 1, max_depth):
                    return True
                
                # Backtrack
                # (no cleanup needed since we use new lists)
        
        return False
    
    # Start with empty set
    initial_remaining = list(range(NUM_POINTS))
    backtrack([], set(), initial_remaining, 0, max_size)
    
    return best_set

def local_search_v2(set_indices, deadline):
    """
    Improved local search:
    1. Try removing points (1-moves)
    2. Try swapping points (2-moves) with better heuristics
    3. Use restarts to escape local minima
    """
    current = list(set_indices)
    current_size = len(current)
    best_set = list(current)
    best_size = current_size
    
    while time.time() < deadline - 10.0:
        improved = False
        current_pairs = get_pair_set(current)
        
        # 1-moves: Try removing a point
        for i in range(len(current)):
            p = current[i]
            new_set = current[:i] + current[i+1:]
            if is_maximal_fast(new_set):
                current = new_set
                current_size = len(current)
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current)
                break
        
        if improved:
            continue
        
        # 2-moves: Try swapping a point
        if len(current) > 0:
            current_set = set(current)
            candidates_out = [p for p in range(NUM_POINTS) if p not in current_set]
            
            # Try all swaps (not just random)
            found_swap = False
            for p_in in current:
                reduced_set = [x for x in current if x != p_in]
                reduced_pairs = get_pair_set(reduced_set)
                
                for p_out in candidates_out:
                    if can_add_fast(p_out, reduced_set, reduced_pairs):
                        new_set = reduced_set + [p_out]
                        if is_maximal_fast(new_set):
                            current = new_set
                            found_swap = True
                            break
                
                if found_swap:
                    break
            
            if not found_swap:
                # No improvement possible, break
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Try backtracking search for small maximal sets
    # Start with a reasonable upper bound and try to find smaller sets
    # First, get an initial solution via greedy
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    
    # Greedy construction
    greedy_set = []
    greedy_pairs = set()
    for p in order:
        if can_add_fast(p, greedy_set, greedy_pairs):
            greedy_set.append(p)
            for q in greedy_set[:-1]:
                a, b = p, q
                if a > b:
                    a, b = b, a
                greedy_pairs.add((a, b))
    
    if greedy_set:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Phase 2: Local search to improve
    if best_set is not None:
        improved_set = local_search_v2(best_set, deadline - 20.0)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Phase 3: Try backtracking search for even smaller sets
    # Try sizes from best_size - 1 down to some minimum
    min_possible = 3  # Minimum for a non-trivial set
    
    target_size = best_size - 1
    while target_size >= min_possible and time.time() < deadline - 10.0:
        found = backtracking_search(target_size, deadline, start_time)
        if found is not None:
            best_size = target_size
            best_set = found
            break
        target_size -= 1
    
    if best_set is None:
        # Fallback
        best_set = greedy_set
    
    return [POINTS[i] for i in best_set]
