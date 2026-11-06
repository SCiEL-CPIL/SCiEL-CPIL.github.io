"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import itertools

N = 16

# EVOLVE-BLOCK-START
DEADLINE_MARGIN = 10.0

def construct():
    start_time = time.time()
    deadline = start_time + 300.0 - DEADLINE_MARGIN
    
    # Precompute all points as (x, y) tuples
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    
    # Index mapping: (x, y) -> index
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Precompute squared distances between all pairs of points
    # dist[i, j] = squared distance between point i and point j
    dist = np.zeros((n_points, n_points), dtype=np.int64)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dx = all_points[i][0] - all_points[j][0]
            dy = all_points[i][1] - all_points[j][1]
            d = dx * dx + dy * dy
            dist[i, j] = d
            dist[j, i] = d
    
    # Precompute all minimal isosceles triples (hyperedges)
    # A triple (i, j, k) is isosceles if any two squared distances are equal
    # We store for each point, the set of hyperedges (as frozensets of indices) it belongs to
    # And for each hyperedge, the set of points it contains
    
    # For efficiency, we'll store hyperedges as sorted tuples of 3 indices
    # And build:
    #   point_to_hyperedges: dict idx -> list of hyperedge indices
    #   hyperedge_points: list of sets of point indices
    
    # First, find all isosceles triples
    # For each pair (i, j), find all k such that (i, j, k) is isosceles
    # Isosceles conditions:
    #   d(i,j) == d(i,k)  [vertex at i]
    #   d(i,j) == d(j,k)  [vertex at j]
    #   d(i,k) == d(j,k)  [vertex at k]
    
    # To avoid duplicates, we can iterate over all triples but that's O(n^3) = 16^6/6 ~ 8M, might be ok
    # But let's be smarter.
    
    # For each vertex v, and each distance d, find all points u such that d(v,u) = d
    # Then any pair (u1, u2) from that set forms an isosceles triangle (v, u1, u2) with vertex at v.
    
    # Also need to handle the other two cases.
    
    # Let's just enumerate all triples for N=16 since 256 choose 3 = ~2.7M, which is feasible
    hyperedges = []  # list of frozensets of 3 point indices
    hyperedge_set = set()
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dij = dist[i, j]
            for k in range(j + 1, n_points):
                dik = dist[i, k]
                djk = dist[j, k]
                if dij == dik or dij == djk or dik == djk:
                    he = frozenset([i, j, k])
                    if he not in hyperedge_set:
                        hyperedge_set.add(he)
                        hyperedges.append(he)
    
    n_hyperedges = len(hyperedges)
    
    # Build point_to_hyperedges: for each point, list of hyperedge indices containing it
    point_to_hyperedges = [[] for _ in range(n_points)]
    for h_idx, he in enumerate(hyperedges):
        for p in he:
            point_to_hyperedges[p].append(h_idx)
    
    # hyperedge_points: for each hyperedge, the set of points (as a list)
    hyperedge_points = [list(he) for he in hyperedges]
    
    # Now, the problem is: find a minimal maximal no-isosceles set.
    # A set S is no-isosceles if it contains no hyperedge entirely.
    # S is maximal if adding any point p not in S creates at least one isosceles triple.
    #   i.e., for every p not in S, there exists a hyperedge H such that H \ {p} ⊆ S.
    #   Equivalently, for every p not in S, there exist two points a, b in S such that (a, b, p) is isosceles.
    
    # This is a hitting set / set cover type problem with maximality constraint.
    
    # Strategy: Greedy construction with local search.
    # 1. Start with a random greedy set.
    # 2. Try to remove points while maintaining maximality.
    # 3. Try to swap points to reduce size.
    
    def is_no_isosceles(chosen_indices):
        """Check if chosen_indices contains no isosceles triple."""
        chosen_set = set(chosen_indices)
        for he in hyperedges:
            if all(p in chosen_set for p in he):
                return False
        return True
    
    def is_maximal(chosen_indices):
        """Check if chosen_indices is maximal: adding any point creates an isosceles triple."""
        chosen_set = set(chosen_indices)
        for p in range(n_points):
            if p in chosen_set:
                continue
            # Check if adding p creates an isosceles triple
            # i.e., exists a, b in chosen_set such that (a, b, p) is isosceles
            # This is equivalent to: exists hyperedge H containing p such that H \ {p} ⊆ chosen_set
            found = False
            for h_idx in point_to_hyperedges[p]:
                he = hyperedges[h_idx]
                others = [q for q in he if q != p]
                if all(q in chosen_set for q in others):
                    found = True
                    break
            if not found:
                return False
        return True
    
    def greedy_construction(order):
        """Greedy: add points in order if they don't create an isosceles triple."""
        chosen = []
        chosen_set = set()
        for p in order:
            # Check if adding p creates an isosceles triple with existing chosen
            # i.e., exists a, b in chosen such that (a, b, p) is isosceles
            # Use precomputed: for each hyperedge containing p, check if other two are in chosen
            can_add = True
            for h_idx in point_to_hyperedges[p]:
                he = hyperedges[h_idx]
                others = [q for q in he if q != p]
                if all(q in chosen_set for q in others):
                    can_add = False
                    break
            if can_add:
                chosen.append(p)
                chosen_set.add(p)
        return chosen
    
    def try_remove_point(chosen, p_idx):
        """Try to remove point p_idx from chosen. Returns new chosen if removal keeps no-isosceles and maximal."""
        if p_idx not in chosen:
            return None
        new_chosen = [p for p in chosen if p != p_idx]
        if not is_no_isosceles(new_chosen):
            return None
        if not is_maximal(new_chosen):
            return None
        return new_chosen
    
    def local_search(chosen):
        """Try to reduce the set by removing points one at a time."""
        changed = True
        while changed:
            changed = False
            for p in list(chosen):
                new_chosen = try_remove_point(chosen, p)
                if new_chosen is not None:
                    chosen = new_chosen
                    changed = True
        return chosen
    
    best_set = None
    best_size = n_points + 1
    
    # Random restarts
    np.random.seed(42)
    
    while time.time() < deadline:
        order = list(range(n_points))
        np.random.shuffle(order)
        
        chosen = greedy_construction(order)
        
        # Local search to try to remove points
        chosen = local_search(chosen)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen
            
            # If we get very small, we can be more aggressive
            if best_size <= 8:
                # Try more local search or different strategies
                pass
    
    if best_set is None:
        # Fallback: deterministic greedy
        chosen = greedy_construction(list(range(n_points)))
        chosen = local_search(chosen)
        best_set = chosen
    
    # Convert indices back to (x, y) tuples
    return [all_points[i] for i in best_set]
# EVOLVE-BLOCK-END
