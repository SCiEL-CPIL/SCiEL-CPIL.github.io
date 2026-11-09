"""Program for no-isosceles set task using randomized greedy with bitset tracking."""
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    
    Strategy:
    1. Use a greedy algorithm that adds points one by one if they don't form an isosceles triangle with existing points.
    2. Use bitsets to efficiently track squared distances for O(1) conflict checks.
    3. Improve the greedy result by using multiple random orderings (restarts) and keeping the largest set found.
    """
    N_local = N
    points_all = [(x, y) for x in range(N_local) for y in range(N_local)]
    
    def greedy_with_order(order):
        """
        Perform greedy construction using the given order of candidate points.
        Returns the list of points in the constructed set.
        """
        points = []
        # dist_bitsets[i] is a bitset where bit j is 1 if there is a point at squared distance j from points[i]
        dist_bitsets = []
        
        for (px, py) in order:
            valid = True
            new_dists = []
            
            # Check against all existing points
            for i, (ax, ay) in enumerate(points):
                dx = px - ax
                dy = py - ay
                d2 = dx * dx + dy * dy
                new_dists.append(d2)
                
                # If this distance already exists from point i to some other point, 
                # then adding (px,py) would create an isosceles triangle with apex at (ax,ay).
                # Specifically, if d(P, A) == d(A, B) for some B in S, then A is the apex.
                # dist_bitsets[i] contains all d(A, B) for B in S.
                if (dist_bitsets[i] >> d2) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if all distances from P to existing points are distinct.
            # If d(P, A) == d(P, B) for A != B, then P is the apex of an isosceles triangle.
            # We can check this by seeing if the set of new_dists has size equal to len(new_dists).
            # To be faster, we can check during the loop or use a set.
            # Since we break early on conflict, we only get here if no apex-at-existing conflicts.
            # Now check apex-at-P conflict.
            if len(new_dists) != len(set(new_dists)):
                continue
            
            # Valid, add to set
            points.append((px, py))
            
            # Build bitset for new point
            new_bitset = 0
            for d2 in new_dists:
                new_bitset |= (1 << d2)
            dist_bitsets.append(new_bitset)
            
            # Update bitsets for existing points
            for i, d2 in enumerate(new_dists):
                dist_bitsets[i] |= (1 << d2)
                
        return points

    best_set = []
    
    # 1. Try fixed heuristics
    # Lexicographic
    order_lex = sorted(points_all, key=lambda p: (p[0], p[1]))
    res = greedy_with_order(order_lex)
    if len(res) > len(best_set):
        best_set = res
        
    # Reverse Lexicographic
    order_rev = sorted(points_all, key=lambda p: (-p[0], -p[1]))
    res = greedy_with_order(order_rev)
    if len(res) > len(best_set):
        best_set = res
        
    # Diagonal
    order_diag = sorted(points_all, key=lambda p: (p[0] + p[1], p[0]))
    res = greedy_with_order(order_diag)
    if len(res) > len(best_set):
        best_set = res

    # 2. Try random restarts
    # Number of restarts can be tuned. 50-100 is usually safe within time limits.
    num_restarts = 100
    for _ in range(num_restarts):
        random_order = points_all.copy()
        random.shuffle(random_order)
        res = greedy_with_order(random_order)
        if len(res) > len(best_set):
            best_set = res
            
    return best_set
# EVOLVE-BLOCK-END
