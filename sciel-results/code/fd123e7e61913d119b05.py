"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Precompute coordinates and squared distances
# Points are indexed by 0 to N*N-1
# point index i corresponds to (i // N, i % N)
num_points = N * N
coords = np.array([(i // N, i % N) for i in range(num_points)], dtype=np.int32)

# dist[i][j] is squared distance between point i and point j
# Compute distances using broadcasting
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sum(diff**2, axis=2).astype(np.int32)

# Precompute conflict graph
# conflict[i] is a numpy boolean array of length num_points, where conflict[i][j] is True
# if points i and j form an isosceles triangle with any third point?
# No, the constraint is "no isosceles triangle".
# A set S is valid if no three points in S form an isosceles triangle.
# This is a 3-uniform hypergraph independent set problem.
# However, we can reformulate this.
# If we have a set S, it is valid if for every pair (a, b) in S,
# there is no c in S such that (a, b, c) is isosceles.
# Actually, the condition "no isosceles triangle" means that for any three distinct points a, b, c in S,
# the distances da, db, dc are all distinct (non-degenerate) or the triangle is degenerate?
# The task says "no-isosceles set". Usually this means no three points form an isosceles triangle.
# Degenerate triangles (collinear) are often considered isosceles if two sides are equal (0, 0, d).
# Let's stick to the definition: d_ab == d_ac or d_ab == d_bc or d_ac == d_bc.

# To speed up checks, we can precompute for each point p, which other points q
# are "conflicting" with p in the sense that if p and q are in the set,
# we must exclude all r such that (p, q, r) is isosceles.
# This is still complex.
# Let's use the bitset approach mentioned in the inspiration.
# Represent the set S as a bitset (numpy array of bools or int64).
# For a candidate point p, it can be added to S if for all pairs (a, b) in S, (p, a, b) is not isosceles.
# Equivalently, p is NOT in the "forbidden set" F(S) where F(S) = union over all pairs (a, b) in S of {c | (a, b, c) is isosceles}.
# Computing F(S) from scratch is expensive.
#
# Alternative:
# When adding p, check against all existing pairs in S.
# If S is small, this is fine.
#
# Let's optimize the `can_add` check using numpy.
# Given p and a list of points in S (as indices), we can compute distances from p to all points in S.
# Then we need to check if any two distances from p to points in S are equal?
# No. The condition is: for any a, b in S, dist[p][a] != dist[p][b] AND dist[p][a] != dist[a][b] AND dist[p][b] != dist[a][b].
# This is pairwise among the three.
#
# Let's stick to the iterative check but optimize with numpy where possible.
# Since N=16, num_points=256.
#
# Better strategy from inspiration: Branch-and-bound or better local search.
# The current program does random greedy + local search (remove & refill).
# The inspiration mentions "Branch-and-bound with bitset conflict propagation".
#
# Let's implement a more efficient local search:
# 1. Start with a random maximal set.
# 2. Try to remove a point p.
# 3. Check if the remaining set is still maximal? No, removing a point makes it non-maximal (potentially).
# 4. We want to find a smaller maximal set.
# 5. If we remove p, we might be able to add other points? No, we want to MINIMIZE size.
# 6. So we just want to remove p if the resulting set is still MAXIMAL.
# 7. Wait, if we remove p, the set is no longer maximal if some q could have been added before but was blocked by p.
# 8. Actually, if S is maximal, and we remove p, S\{p} might be non-maximal.
# 9. If S\{p} is still maximal, then we have found a smaller maximal set.
# 10. So the local search step is: Try to remove p. If S\{p} is maximal, keep it.
# 11. How to check if S\{p} is maximal?
# 12. Check if any q not in S\{p} can be added to S\{p}.
# 13. If yes, S\{p} is not maximal.
# 14. If no, S\{p} is maximal.
#
# This check is expensive: O(|S|^2) for each q, and there are O(N^2) q's.
# Total O(N^4) per removal. N=16, N^4 = 65536. 256 removals * 65536 = 16M ops. Feasible.
#
# We can optimize the "can be added" check.
#
# Let's refine the local search:
# - Maintain S as a set of indices.
# - For each p in S (in random order):
#   - Let S' = S \ {p}.
#   - Check if S' is maximal.
#     - For each q not in S':
#       - If can_add(q, S'):
#         - S' is not maximal. Break.
#   - If S' is maximal:
#     - S = S'.
#     - Continue local search (restart or continue?).
#
# This is a "delete" local search.
#
# To speed up `can_add(q, S')`:
# We need to check if q forms an isosceles triangle with any pair in S'.
#
# Let's implement this.

def is_isosceles(i, j, k):
    d1 = dist[i, j]
    d2 = dist[i, k]
    d3 = dist[j, k]
    return d1 == d2 or d1 == d3 or d2 == d3

def can_add(p, S_set):
    """Check if point p can be added to set S_set without creating an isosceles triangle."""
    if len(S_set) < 2:
        return True
        
    s_list = list(S_set)
    # Optimization: If S_set is large, this is slow.
    # We can precompute for each p, the set of pairs (a, b) that conflict with p.
    # But pairs change.
    
    # Vectorized check?
    # For all a in S_set:
    #   For all b in S_set, b != a:
    #     Check isosceles(p, a, b)
    
    # Let's try to use numpy for the inner loop if S_set is represented as an array.
    # But S_set is a set. Converting to list is O(n).
    
    for i in range(len(s_list)):
        a = s_list[i]
        for j in range(i + 1, len(s_list)):
            b = s_list[j]
            if is_isosceles(p, a, b):
                return False
    return True

def is_maximal(S_set):
    """Check if S_set is maximal."""
    s_list = list(S_set)
    # For each q not in S_set, check if it can be added.
    # If any can be added, not maximal.
    for q in range(num_points):
        if q in S_set:
            continue
        if can_add(q, S_set):
            return False
    return True

def construct():
    start_time = time.time()
    time_limit = 295  # 5s margin
    
    best_size = num_points
    best_set = None
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # 1. Generate a random maximal set
        order = list(range(num_points))
        random.shuffle(order)
        
        current_set = set()
        
        # Greedy add
        for p in order:
            if can_add(p, current_set):
                current_set.add(p)
        
        # 2. Ensure Maximal (Greedy already does a single pass, but let's be sure)
        # The single pass in greedy is not guaranteed to be maximal?
        # As reasoned before, if we iterate through ALL points, and add if possible,
        # and adding only makes it harder for others, then after one pass,
        # no point *later* in the order can be added.
        # But what about points *earlier*?
        # If p_early was rejected, it was invalid. Adding more points keeps it invalid.
        # So one pass IS maximal.
        # Wait, the order is `order`.
        # If p is at index 5 in `order`, and it's rejected, it's invalid wrt current_set at that time.
        # Later points are added. They might block p? No, p is already invalid.
        # They might unblock p? No, adding points only adds constraints.
        # So yes, one pass is maximal.
        
        # 3. Local Search: Delete
        improved = True
        max_local_steps = 50
        step = 0
        while improved and step < max_local_steps and time.time() - start_time < time_limit:
            improved = False
            step += 1
            
            # Try removing each point in the set
            # We do this in a random order to break symmetry
            removal_candidates = list(current_set)
            random.shuffle(removal_candidates)
            
            for p_remove in removal_candidates:
                if time.time() - start_time > time_limit:
                    break
                
                # Tentatively remove p_remove
                new_set = set(current_set)
                new_set.remove(p_remove)
                
                # Check if new_set is maximal
                # This is the expensive part.
                if is_maximal(new_set):
                    current_set = new_set
                    improved = True
                    # Restart local search with the new set
                    break
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # Optional: If we reach a very small size, we might want to polish.
            # But let's see if we can do more restarts.
    
    if best_set is None:
        return []
    
    return [tuple(c) for c in coords[list(best_set)]]
# EVOLVE-BLOCK-END
