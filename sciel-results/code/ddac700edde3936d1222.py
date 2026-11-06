import time
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 30
    # Initial 6x5 grid
    points = [(x, y) for x in range(6) for y in range(5)]
    point_set = set(points)
    
    indices = list(range(n))
    all_pairs = list(itertools.combinations(indices, 2))
    
    # Precompute pairs not involving i
    pairs_not_involving = []
    for i in range(n):
        pairs_not_involving.append([
            (j, k) for (j, k) in all_pairs if j != i and k != i
        ])

    # Precompute constants for area calculation
    # Area(i, j, k) = abs( (xi-xj)*(yk-yj) - (yi-yj)*(xk-xj) )
    # Let dx_jk = xk - xj, dy_jk = yk - yj
    # Term = (xi-xj)*dy_jk - (yi-yj)*dx_jk
    #      = xi*dy_jk - xj*dy_jk - yi*dx_jk + yj*dx_jk
    #      = xi*dy_jk - yi*dx_jk - (xj*dy_jk - yj*dx_jk)
    # Let A_jk = dy_jk, B_jk = -dx_jk, C_jk = -(xj*dy_jk - yj*dx_jk)
    # Area = abs( xi*A_jk + yi*B_jk + C_jk )
    
    # We need to update these if points j or k change.
    # But in our local search, only point i changes.
    # So j and k are fixed.
    # So A_jk, B_jk, C_jk are constant for the duration of the move of i.
    
    def compute_constants(points):
        """Compute A, B, C for all pairs (j, k)."""
        consts = {}
        for j, k in all_pairs:
            xj, yj = points[j]
            xk, yk = points[k]
            dx = xk - xj
            dy = yk - yj
            A = dy
            B = -dx
            C = -(xj * dy - yj * dx)
            consts[(j, k)] = (A, B, C)
        return consts

    def get_area(i, j, k, points, consts):
        xi, yi = points[i]
        A, B, C = consts[(j, k)]
        return abs(xi * A + yi * B + C)

    def init_counts(points, consts):
        counts = Counter()
        for i, j, k in itertools.combinations(range(n), 3):
            a = get_area(i, j, k, points, consts)
            if a > 0:
                counts[a] += 1
        return counts

    consts = compute_constants(points)
    counts = init_counts(points, consts)
    best_score = max(counts.values()) if counts else 0
    best_points = points[:]

    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            current_point = points[i]
            xi, yi = current_point
            
            best_move = None
            best_move_score = best_score
            
            # Try neighbors in a window
            # Use a slightly larger window to escape local minima
            window = 8 
            for dx in range(-window, window + 1):
                for dy in range(-window, window + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_point = (xi + dx, yi + dy)
                    if new_point in point_set:
                        continue
                    if not (-30 <= new_point[0] <= 30 and -30 <= new_point[1] <= 30):
                        continue
                    
                    # Update counts
                    # We need to subtract areas involving i with old point
                    # and add areas involving i with new point
                    
                    # Optimization: We can compute the delta for each pair
                    # But we need the new max.
                    # Let's just update the Counter.
                    
                    # Save old counts? No, update in place.
                    
                    # To speed up, we can calculate the new area for each pair
                    # and update the counter.
                    
                    # Note: consts are valid because j, k don't change.
                    
                    for j, k in pairs_not_involving[i]:
                        old_a = get_area(i, j, k, points, consts)
                        if old_a > 0:
                            counts[old_a] -= 1
                            
                        # Temporarily set points[i] to new_point?
                        # Or just calculate directly.
                        # get_area uses points[i]. 
                        # We can pass xi_new, yi_new directly to a helper.
                        
                        A, B, C = consts[(j, k)]
                        new_a = abs(new_point[0] * A + new_point[1] * B + C)
                        if new_a > 0:
                            counts[new_a] += 1
                    
                    # Check score
                    # Finding max is O(K).
                    # We can maintain a heap? 
                    # Or just max(). K is small.
                    current_score = max(counts.values()) if counts else 0
                    
                    if current_score > best_move_score:
                        best_move_score = current_score
                        best_move = new_point
                    else:
                        # Rollback
                        for j, k in pairs_not_involving[i]:
                            old_a = get_area(i, j, k, points, consts)
                            if old_a > 0:
                                counts[old_a] += 1
                            
                            A, B, C = consts[(j, k)]
                            new_a = abs(new_point[0] * A + new_point[1] * B + C)
                            if new_a > 0:
                                counts[new_a] -= 1
                    
            if best_move is not None:
                # Apply the best move
                # We need to update points, point_set, and counts permanently
                # But the loop above rolled back all non-best moves.
                # The best move was NOT rolled back.
                # So points[i] is still current_point (OLD).
                # Wait, in the loop, I did NOT modify points[i].
                # I only modified counts.
                # So I need to apply the move to points and point_set.
                
                # But counts are already updated for the best move?
                # Yes, because the "else" branch (rollback) was not executed for the best move.
                # So counts reflect the state with best_move.
                
                points[i] = best_move
                point_set.remove(current_point)
                point_set.add(best_move)
                
                # Update consts?
                # Consts depend on points[j] and points[k].
                # Since only point i moved, consts for pairs involving i are NOT used in 
                # pairs_not_involving[i].
                # However, for the NEXT iteration of i (or if we move another point),
                # consts involving the NEW point i might be needed?
                # No, consts are keyed by (j, k).
                # If we move point i, the pairs (j, k) not involving i are unchanged.
                # But the pairs involving i (e.g., (i, k)) WOULD change if we were to move k later.
                # Since we recompute `get_area` using `points[i]` directly in the update loop,
                # we don't actually USE `consts` for the moving point's area calculation in the 
                # incremental step?
                # Wait, `get_area(i, j, k, points, consts)` uses `points[i]`.
                # And `consts[(j, k)]` uses `points[j]` and `points[k]`.
                # Since j, k != i, and we only move i, `points[j]` and `points[k]` are unchanged.
                # So `consts` remain valid for all pairs (j, k) where j, k != i.
                # What about pairs where one index is i?
                # `pairs_not_involving[i]` excludes them.
                # So `consts` are valid for the duration of the move of i.
                
                # However, if we move point i, the `consts` for pairs involving i are now STALE 
                # if we were to use them later.
                # But we only use `consts` for pairs (j, k) in `pairs_not_involving[current_i]`.
                # When we move to the next point `i+1`, we will use `pairs_not_involving[i+1]`.
                # This set includes pairs like `(i, k)`.
                # For these pairs, `consts[(i, k)]` would be stale because `points[i]` changed.
                # So we MUST recompute `consts` after any move?
                # Or, we can just recalculate the area for pairs involving the moved point 
                # without using `consts`?
                
                # Simpler: Recompute `consts` every time a point moves.
                # Cost: 435 pairs * small calcs. Negligible.
                
                consts = compute_constants(points)
                
                if best_move_score > best_score:
                    best_score = best_move_score
                    best_points = points[:]
                    improved = True
        if not improved:
            break

    return best_points
