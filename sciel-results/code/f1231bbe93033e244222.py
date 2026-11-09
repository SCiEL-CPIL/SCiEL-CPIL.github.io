"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
DEADLINE = 300  # safety margin before 330s kill

# EVOLVE-BLOCK-START
import numpy as np

def _precompute_isosceles_conflicts():
    """
    Precompute for each point p, the set of pairs (a,b) such that {p,a,b} is isosceles.
    We represent this as: for each p, a list of (a, b) pairs that conflict with p.
    Also build a 2D array conflict_mask[p][q] = set of r such that {p,q,r} is isosceles? 
    Actually, let's think differently.
    
    We want to find a set S such that no triple in S is isosceles.
    This is a 3-uniform hypergraph independent set problem.
    
    For branch and bound, we need efficient propagation.
    Let's precompute: for each pair (a,b), the set of points c such that {a,b,c} is isosceles.
    Then if we have a,b in our set, we cannot add any c in that set.
    
    Also for maximality: a set S is maximal if for every point p not in S, there exist a,b in S such that {p,a,b} is isosceles.
    """
    points = [(x, y) for x in range(N) for y in range(N)]
    n_pts = len(points)
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # For each pair (i, j), compute the set of k such that {i,j,k} is isosceles
    # isosceles(i,j,k) means: d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k)
    
    # Let's precompute all squared distances
    # dist[i][j] = squared distance between point i and point j
    coords = np.array(points)
    # Compute pairwise squared distances
    # diff[i,j,k] = coords[i,k] - coords[j,k]
    # This is O(n^2) memory which is 256^2 = 65536, fine
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # For each pair (i,j), find all k such that dist[i,j]==dist[i,k] or dist[i,j]==dist[j,k] or dist[i,k]==dist[j,k]
    # Let's build conflict_sets[i][j] = bitmask of k that form isosceles with i,j
    # n_pts = 256, so we can use Python ints as bitmasks
    
    conflict_sets = [[0] * n_pts for _ in range(n_pts)]
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dij = dist[i, j]
            mask = 0
            # Case 1: dist[i,j] == dist[i,k] => k on circle centered at i with radius dij
            # Case 2: dist[i,j] == dist[j,k] => k on circle centered at j with radius dij
            # Case 3: dist[i,k] == dist[j,k] => k on perpendicular bisector
            
            # For efficiency, let's just iterate k for now, but that's O(n^3)
            # n=256, n^3 = 16 million, might be okay in numpy? Let's try vectorized.
            
            # Actually, let's do it more carefully.
            # For fixed i,j:
            # k such that dist[i,k] == dij: these are points at distance dij from i
            # k such that dist[j,k] == dij: these are points at distance dij from j
            # k such that dist[i,k] == dist[j,k]: these are points equidistant from i and j
            
            mask_i = np.where(dist[i, :] == dij)[0]
            mask_j = np.where(dist[j, :] == dij)[0]
            mask_eq = np.where(dist[i, :] == dist[j, :])[0]
            
            all_k = np.union1d(np.union1d(mask_i, mask_j), mask_eq)
            # Exclude i and j themselves? The triple {i,j,k} with k=i or k=j is degenerate.
            # But in our problem, we consider distinct points. So k != i and k != j.
            all_k = all_k[(all_k != i) & (all_k != j)]
            
            for k in all_k:
                mask |= (1 << k)
            
            conflict_sets[i][j] = mask
            conflict_sets[j][i] = mask
    
    return points, conflict_sets, dist

def _is_maximal(chosen, conflict_sets, n_pts):
    """Check if chosen set is maximal: no point can be added without creating an isosceles triple."""
    chosen_set = set(chosen)
    for p in range(n_pts):
        if p in chosen_set:
            continue
        # Can p be added? p can be added iff there is no pair (a,b) in chosen such that {p,a,b} is isosceles
        # i.e., for all pairs (a,b) in chosen, p is NOT in conflict_sets[a][b]
        can_add = True
        chosen_list = list(chosen_set)
        for idx_a in range(len(chosen_list)):
            a = chosen_list[idx_a]
            for idx_b in range(idx_a + 1, len(chosen_list)):
                b = chosen_list[idx_b]
                if conflict_sets[a][b] & (1 << p):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def construct():
    start_time = time.time()
    points, conflict_sets, dist = _precompute_isosceles_conflicts()
    n_pts = len(points)
    
    # We want to find a minimal maximal no-isosceles set.
    # Let's use a greedy approach with local search.
    # First, create a maximal set via greedy.
    # Then try to remove points while maintaining maximality? No, removing points makes it non-maximal.
    # Actually, we want to MINIMIZE the size of a maximal set.
    # A maximal set cannot have points removed (it would no longer be maximal... wait).
    # Actually, if S is maximal, removing a point from S gives S' which is not necessarily maximal, 
    # but S' is still a no-isosceles set. The question is whether S' is maximal.
    # If S' is not maximal, we can add points to make it maximal again, potentially with fewer total points?
    # No, that doesn't make sense.
    
    # Let me re-read the task: "maximal no-isosceles set (no further grid point can be added)"
    # So S is maximal if S is no-isosceles AND for every p not in S, S ∪ {p} is NOT no-isosceles.
    # We want to minimize |S| over all maximal no-isosceles sets.
    
    # Strategy: 
    # 1. Generate a maximal no-isosceles set S.
    # 2. Try to find a smaller maximal no-isosceles set.
    #    One approach: take S, try removing points one by one. After removing a point, 
    #    the set may no longer be maximal. To restore maximality, we might need to add points.
    #    But that could increase size.
    
    # Alternative: Use a different construction.
    # Let's try a greedy algorithm that picks points in an order that leads to smaller maximal sets.
    # Or use simulated annealing / local search on the space of maximal sets.
    
    # For now, let's do a greedy construction and then try to optimize.
    
    # Greedy: pick points in some order, adding if it doesn't create an isosceles triple.
    # The order matters. Let's try different orderings.
    
    best_set = None
    best_size = n_pts + 1
    
    # Try row-major order
    order = list(range(n_pts))
    chosen = []
    chosen_set = set()
    for p in order:
        can_add = True
        for a, b in combinations(chosen, 2):
            if conflict_sets[a][b] & (1 << p):
                can_add = False
                break
        if can_add:
            chosen.append(p)
            chosen_set.add(p)
    
    if _is_maximal(chosen, conflict_sets, n_pts):
        best_set = chosen[:]
        best_size = len(chosen)
    
    # Try to improve by local search
    # Idea: try swapping points or using a different greedy order.
    
    # Let's try a randomized greedy with multiple restarts
    random.seed(42)
    for restart in range(100):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        order = list(range(n_pts))
        random.shuffle(order)
        chosen = []
        chosen_set = set()
        for p in order:
            can_add = True
            for a, b in combinations(chosen, 2):
                if conflict_sets[a][b] & (1 << p):
                    can_add = False
                    break
            if can_add:
                chosen.append(p)
                chosen_set.add(p)
        
        if _is_maximal(chosen, conflict_sets, n_pts):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_set = chosen[:]
    
    # Try local search: try to replace a point in the set with another point
    # to see if we can get a smaller maximal set
    if best_set is not None:
        # Try to shrink the set
        # For each point in best_set, try removing it and see if we can make it maximal again
        # with fewer total points? This is complex.
        
        # Alternative: try a different approach - start from scratch with a biased greedy
        # that prefers points that "block" more other points.
        
        for restart in range(200):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Greedy with heuristic: pick the point that, when added, blocks the most other points
            remaining = set(range(n_pts))
            chosen = []
            chosen_set = set()
            
            while remaining:
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                best_p = None
                best_block_count = -1
                
                for p in remaining:
                    # Check if p can be added
                    can_add = True
                    for a, b in combinations(chosen, 2):
                        if conflict_sets[a][b] & (1 << p):
                            can_add = False
                            break
                    if not can_add:
                        continue
                    
                    # Count how many points in remaining would be blocked by adding p
                    # A point q in remaining is blocked if there exist a,b in chosen ∪ {p} 
                    # such that {q,a,b} is isosceles.
                    # This is expensive to compute exactly. Let's approximate.
                    # For now, just use a simple heuristic: pick p that has the most conflicts with remaining points
                    block_count = 0
                    for q in remaining:
                        if q == p:
                            continue
                        # Check if p and q conflict with any existing pair in chosen
                        # This is getting complex. Let's just use a simple score.
                        pass
                    
                    # Use a simpler heuristic: just pick randomly from addable points
                    # or pick the one with highest "conflict degree"
                    if best_p is None:
                        best_p = p
                        best_block_count = 0
                    else:
                        # Random tie-breaking
                        if random.random() < 0.5:
                            best_p = p
                
                if best_p is None:
                    break
                
                chosen.append(best_p)
                chosen_set.add(best_p)
                remaining.remove(best_p)
            
            if _is_maximal(chosen, conflict_sets, n_pts):
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best_set = chosen[:]
    
    if best_set is None:
        # Fallback: return the first greedy result
        order = list(range(n_pts))
        chosen = []
        for p in order:
            can_add = True
            for a, b in combinations(chosen, 2):
                if conflict_sets[a][b] & (1 << p):
                    can_add = False
                    break
            if can_add:
                chosen.append(p)
        best_set = chosen
    
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
