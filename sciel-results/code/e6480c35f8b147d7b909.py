import numpy as np
import time
import heapq
import random

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 620
    p = 7
    q = p * p
    
    # Field arithmetic helpers
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    def f_add(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa + ya) % p, (xb + yb) % p)

    def f_neg(x):
        xa, xb = to_pair(x)
        return to_int((-xa) % p, (-xb) % p)

    def f_sub(x, y):
        return f_add(x, f_neg(y))

    def f_mul(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        real = (xa * ya + xb * yb * 3) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    # Directions
    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)
    
    # Map points to indices
    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}
    idx_to_point = {v: k for k, v in point_index.items()}
    
    in_N = np.ones(q * q, dtype=bool)
    
    # Precompute lines: for each point x, list of (y, v) for y on line through x with direction v, y != x
    # Actually, we need for each y, the list of (x, v) such that y is on line through x with dir v.
    # This is symmetric: y is on line x-v iff x is on line y-v (with same v).
    # So we can precompute for each point x, the neighbors on each line.
    
    # neighbors[x][v] = list of points y != x on line through x with direction v
    # There are 49 such points for each v.
    
    # To save memory/time, we can compute on the fly or precompute a list of affected (x, v) for each y.
    # affected_by[y] = list of (x, v) such that y is on line x-v.
    
    affected_by = [[] for _ in range(q * q)]
    for x_idx in range(q * q):
        px, py = idx_to_point[x_idx]
        for v_idx, (vx, vy) in enumerate(directions):
            # Generate points on line
            for t in range(1, q):
                tx = f_add(px, f_mul(t, vx))
                ty = f_add(py, f_mul(t, vy))
                y_idx = point_index[(tx, ty)]
                # y_idx is on line x_idx with direction v_idx
                # Also x_idx is on line y_idx with direction v_idx
                affected_by[y_idx].append((x_idx, v_idx))
                # Note: we only need one direction of the relation for the "essential" logic?
                # If we remove y, it affects x's coverage for direction v.
                # So for each y, we need to know which (x, v) pairs it supports.
    
    # Initialize valid directions for each point
    # valid_dirs[x] is a set of v indices
    valid_dirs = [set(range(n_dir)) for _ in range(q * q)]
    
    # essential_count[y] = number of x such that y is essential for x
    # y is essential for x if len(valid_dirs[x]) == 1 and the single direction v is such that y is on line x-v.
    # Initially, len(valid_dirs[x]) = 50, so no one is essential.
    essential_count = np.zeros(q * q, dtype=np.int32)
    
    # Heap of (essential_count[y], y)
    heap = [(0, i) for i in range(q * q)]
    heapq.heapify(heap)
    
    removed_count = 0
    
    while heap:
        if time.time() > deadline - 60:
            break
            
        e_count, y_idx = heapq.heappop(heap)
        
        # Check if still in N and if count is still 0 (might be stale)
        if not in_N[y_idx]:
            continue
        if essential_count[y_idx] != e_count:
            # Stale entry, skip
            continue
            
        if e_count > 0:
            # Cannot remove any more points with this greedy strategy
            break
            
        # Try to remove y_idx
        in_N[y_idx] = False
        removed_count += 1
        
        # Update valid_dirs and essential_count for affected x
        # affected_by[y_idx] contains (x_idx, v_idx)
        # For each such pair, remove v_idx from valid_dirs[x_idx]
        
        # We need to track changes to valid_dirs to update essential_count correctly
        # If valid_dirs[x] changes from size k to k-1:
        # If k-1 == 1, then the remaining direction v_rem makes all points on that line essential for x.
        # If k > 1, no new essentials created (unless k-1==1).
        # If k == 1, removal would have been invalid, but we checked e_count==0 for y, which implies y is not essential for any x.
        # Wait, if valid_dirs[x] has size 1, say {v}, and y is on line x-v, then y is essential for x.
        # So essential_count[y] would be at least 1.
        # So if essential_count[y] == 0, then for all x affected by y, len(valid_dirs[x]) > 1.
        # So removing y will reduce len(valid_dirs[x]) by 1.
        # It might drop to 1.
        
        new_essentials = []
        
        for x_idx, v_idx in affected_by[y_idx]:
            if not in_N[x_idx]:
                continue
                
            if v_idx in valid_dirs[x_idx]:
                valid_dirs[x_idx].remove(v_idx)
                new_size = len(valid_dirs[x_idx])
                
                if new_size == 1:
                    # The remaining direction is the only one left
                    # Find the remaining direction
                    v_rem = next(iter(valid_dirs[x_idx]))
                    # All points on line x_idx with direction v_rem (except x_idx) are now essential for x_idx
                    # We need to increment essential_count for those points
                    # But only if they are in N
                    px, py = idx_to_point[x_idx]
                    vx, vy = directions[v_rem]
                    for t in range(1, q):
                        tx = f_add(px, f_mul(t, vx))
                        ty = f_add(py, f_mul(t, vy))
                        z_idx = point_index[(tx, ty)]
                        if in_N[z_idx]:
                            # z_idx is now essential for x_idx
                            # But we only increment if it wasn't already essential?
                            # essential_count counts the number of x for which z is essential.
                            # So we just increment.
                            essential_count[z_idx] += 1
                            new_essentials.append(z_idx)
        
        # Push updated points to heap
        # We only need to push points whose essential_count changed to 0 or decreased?
        # Actually, we only care about points that might become removable (count 0).
        # But counts only increase in this step.
        # So no new points become removable in this step.
        # However, we might have skipped points earlier that are now removable? No, counts only go up.
        # So the heap only needs to be updated if we *decrease* counts, which we don't.
        # Wait, do we ever decrease essential_count?
        # If we remove a point x, then y is no longer essential for x.
        # But we are removing y, not x.
        # The points x are still in N.
        # So essential_count only increases.
        # Thus, once a point has essential_count > 0, it will never become 0 again.
        # So we don't need to push new entries to the heap!
        # We just pop from the initial heap.
        
    # Build result
    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])
            
    if not points:
        # Fallback
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                points.append([[xa, xb], [ya, yb]])
                
    return np.array(points, dtype=np.int64)
