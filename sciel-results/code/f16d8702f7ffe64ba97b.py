import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

def compute_2d_cost(positions_16d):
    """
    Computes the exact integral of the 2D union area over c in [0, 1].
    positions_16d is a flat array of shape (16,), representing 8 x-positions and 8 y-positions.
    The 2D tube k=(i,j) at height c is [x_i + c*i/8, x_i + c*i/8 + 1/8] x [y_j + c*j/8, y_j + c*j/8 + 1/8].
    The union area at height c is the area of the union of these 64 rectangles.
    We integrate this area over c from 0 to 1.
    
    Since the union area is piecewise quadratic in c (breakpoints where rectangle edges align),
    we can compute the integral exactly by splitting the c-interval at all breakpoints.
    """
    N = 8
    w = 1.0 / 8.0
    
    # Unpack positions
    x_pos = positions_16d[:N]
    y_pos = positions_16d[N:]
    
    # Generate all 64 tubes
    # i is 0..7, j is 0..7
    # Tube (i, j) has:
    # x_left = x_pos[i] + c*i/8
    # x_right = x_pos[i] + c*i/8 + w
    # y_left = y_pos[j] + c*j/8
    # y_right = y_pos[j] + c*j/8 + w
    
    # Breakpoints in c where any two edges of any two rectangles align.
    # Edges are of the form:
    # x: x_pos[i] + c*i/8 + offset (offset in {0, w})
    # y: y_pos[j] + c*j/8 + offset (offset in {0, w})
    
    # A breakpoint occurs when:
    # x_pos[i1] + c*i1/8 + o1 = x_pos[i2] + c*i2/8 + o2
    # => c * (i1-i2)/8 = x_pos[i2] - x_pos[i1] + o2 - o1
    # Similarly for y.
    
    breakpoints = set([0.0, 1.0])
    
    # X-dimension breakpoints
    for i1 in range(N):
        for i2 in range(N):
            if i1 == i2:
                continue
            denom = (i1 - i2) / 8.0
            for o1 in [0.0, w]:
                for o2 in [0.0, w]:
                    rhs = x_pos[i2] - x_pos[i1] + o2 - o1
                    c_val = rhs / denom
                    if 0.0 < c_val < 1.0:
                        breakpoints.add(c_val)
                        
    # Y-dimension breakpoints
    for j1 in range(N):
        for j2 in range(N):
            if j1 == j2:
                continue
            denom = (j1 - j2) / 8.0
            for o1 in [0.0, w]:
                for o2 in [0.0, w]:
                    rhs = y_pos[j2] - y_pos[j1] + o2 - o1
                    c_val = rhs / denom
                    if 0.0 < c_val < 1.0:
                        breakpoints.add(c_val)
                        
    breakpoints = sorted(breakpoints)
    
    total_cost = 0.0
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        c_mid = (c_start + c_end) / 2.0
        dc = c_end - c_start
        
        # At c_mid, compute the union area of the 64 rectangles.
        # Since the topology (which rectangles overlap) is constant in [c_start, c_end],
        # the union area is a quadratic function of c: A(c) = a*c^2 + b*c + d.
        # We can determine a, b, d by evaluating A at 3 points, or use the fact that
        # the area is piecewise quadratic.
        # Actually, simpler: sample A at c_start, c_mid, c_end? No, c_start and c_end are breakpoints.
        # The function is continuous.
        # Let's sample at 3 points to fit the quadratic, or just use Simpson's rule if we assume smoothness?
        # No, it's piecewise quadratic. Within an interval, it IS quadratic.
        # So A(c) = A0 + A1*(c-c0) + A2*(c-c0)^2.
        # We can evaluate A at c_start, c_mid, c_end (even if at breakpoints, the value is well-defined).
        
        def union_area_at(c_val):
            # Compute union of 64 rectangles
            rects = []
            for i in range(N):
                for j in range(N):
                    x_l = x_pos[i] + c_val * i / 8.0
                    y_l = y_pos[j] + c_val * j / 8.0
                    rects.append((x_l, y_l, x_l + w, y_l + w))
            
            # Sweep line or grid?
            # N=8, so 64 rects.
            # Coordinates involved:
            x_coords = set()
            y_coords = set()
            for r in rects:
                x_coords.add(r[0])
                x_coords.add(r[2])
                y_coords.add(r[1])
                y_coords.add(r[3])
            
            x_coords = sorted(x_coords)
            y_coords = sorted(y_coords)
            
            # Grid of cells
            area = 0.0
            for xi in range(len(x_coords) - 1):
                for yj in range(len(y_coords) - 1):
                    cx = (x_coords[xi] + x_coords[xi+1]) / 2.0
                    cy = (y_coords[yj] + y_coords[yj+1]) / 2.0
                    # Check if this cell center is covered by any rect
                    covered = False
                    for r in rects:
                        if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
                            covered = True
                            break
                    if covered:
                        area += (x_coords[xi+1] - x_coords[xi]) * (y_coords[yj+1] - y_coords[yj])
            return area
        
        A_start = union_area_at(c_start)
        A_mid = union_area_at(c_mid)
        A_end = union_area_at(c_end)
        
        # Integral of quadratic through 3 points
        # Using Simpson's rule which is exact for cubics, so exact for quadratics.
        integral_segment = (dc / 6.0) * (A_start + 4.0 * A_mid + A_end)
        total_cost += integral_segment
        
    return total_cost

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    N = 8
    
    # Initial guess: Keich configuration
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # The 16D vector is [x_0..x_7, y_0..y_7]
    initial_16d = np.concatenate([keich, keich])
    
    # Bounds for differential evolution
    # Positions should be within a reasonable range.
    # The tubes shear by at most 1 unit.
    # Let's say x_i in [-2, 2]
    bounds = [(-2.0, 2.0) for _ in range(16)]
    
    # Differential Evolution settings
    # We want to minimize the 2D union area integral.
    # This is a non-convex problem.
    # Use a relatively high population size for global exploration.
    # maxiter should be limited by time.
    
    # We will run DE, but we need to monitor time.
    # DE is stochastic. We can use a callback to stop early if time is up.
    # However, scipy DE doesn't have a clean "stop after X seconds" callback that aborts the current iteration quickly if it's long.
    # But our cost function is fast (exact integration over ~100-200 breakpoints, each involving 64 rects union area via grid).
    # Let's estimate cost function speed.
    # Breakpoints: ~ O(N^2) * 4 = 200.
    # Union area: 64 rects, grid ~ (2*8)^2 = 256 cells? No, coords are unique.
    # Max unique coords: 8*2 + 8*2 = 32. Grid 31x31 = ~1000 cells.
    # 200 breakpoints * 1000 cells * 64 rects = 12.8M ops per cost eval.
    # This might be a bit slow for DE with large pop.
    
    # Optimization: The union area calculation can be optimized.
    # Instead of checking every rect for every cell, we can use a sweep line or just check coverage more efficiently.
    # Or, since N=8 is small, we can precompute the structure.
    # Actually, the grid method is O(C^2 * R) where C is num coords, R is num rects.
    # C ~ 32, R=64. 32*32*64 = 65536 ops.
    # 200 breakpoints * 65536 = 13M ops.
    # In Python, this is slow. 13M simple ops might take 0.1-0.5s.
    # DE with pop=50, maxiter=100 -> 5000 evals -> 500-2500s. Too slow.
    
    # We need a faster cost function or fewer evals.
    # Let's optimize the union area calculation using vectorized numpy or a more efficient sweep.
    
    def fast_union_area_at(c_val, x_pos, y_pos):
        # Generate rects
        # x_l = x_pos[i] + c*i/8
        # y_l = y_pos[j] + c*j/8
        # We can compute the union area by sweeping x.
        # Sort all x-events.
        # Events: (x, type, y1, y2, rect_id)
        # type 0: start, type 1: end
        
        events = []
        for i in range(N):
            for j in range(N):
                x_l = x_pos[i] + c_val * i / 8.0
                x_r = x_l + w
                y_l = y_pos[j] + c_val * j / 8.0
                y_r = y_l + w
                events.append((x_l, 0, y_l, y_r))
                events.append((x_r, 1, y_l, y_r))
                
        events.sort(key=lambda e: e[0])
        
        # Active intervals in y
        # Use a simple list or sorted structure?
        # N=8, so at most 64 active intervals.
        # We can just keep a list of active (y_l, y_r) and recompute union length in y for each x-slab.
        
        active = []
        area = 0.0
        prev_x = events[0][0]
        
        for x, typ, y1, y2 in events:
            if x > prev_x:
                # Compute union length of active y-intervals
                if active:
                    # Sort by y1
                    active.sort(key=lambda t: t[0])
                    union_len = 0.0
                    cur_l = active[0][0]
                    cur_r = active[0][1]
                    for k in range(1, len(active)):
                        l, r = active[k]
                        if l <= cur_r:
                            if r > cur_r:
                                cur_r = r
                        else:
                            union_len += cur_r - cur_l
                            cur_l = l
                            cur_r = r
                    union_len += cur_r - cur_l
                    area += union_len * (x - prev_x)
            
            if typ == 0:
                active.append((y1, y2))
            else:
                # Remove the interval (y1, y2)
                # Since it's small, linear search is fine
                for k in range(len(active)):
                    if active[k][0] == y1 and active[k][1] == y2:
                        active.pop(k)
                        break
            
            prev_x = x
            
        return area
    
    def compute_2d_cost_fast(positions_16d):
        N = 8
        w = 1.0 / 8.0
        x_pos = positions_16d[:N]
        y_pos = positions_16d[N:]
        
        breakpoints = set([0.0, 1.0])
        
        for i1 in range(N):
            for i2 in range(N):
                if i1 == i2: continue
                denom = (i1 - i2) / 8.0
                for o1 in [0.0, w]:
                    for o2 in [0.0, w]:
                        rhs = x_pos[i2] - x_pos[i1] + o2 - o1
                        c_val = rhs / denom
                        if 0.0 < c_val < 1.0:
                            breakpoints.add(c_val)
                            
        for j1 in range(N):
            for j2 in range(N):
                if j1 == j2: continue
                denom = (j1 - j2) / 8.0
                for o1 in [0.0, w]:
                    for o2 in [0.0, w]:
                        rhs = y_pos[j2] - y_pos[j1] + o2 - o1
                        c_val = rhs / denom
                        if 0.0 < c_val < 1.0:
                            breakpoints.add(c_val)
                            
        breakpoints = sorted(breakpoints)
        
        total_cost = 0.0
        for k in range(len(breakpoints) - 1):
            c_start = breakpoints[k]
            c_end = breakpoints[k + 1]
            c_mid = (c_start + c_end) / 2.0
            dc = c_end - c_start
            
            A_start = fast_union_area_at(c_start, x_pos, y_pos)
            A_mid = fast_union_area_at(c_mid, x_pos, y_pos)
            A_end = fast_union_area_at(c_end, x_pos, y_pos)
            
            total_cost += (dc / 6.0) * (A_start + 4.0 * A_mid + A_end)
            
        return total_cost

    # Check speed of fast_union_area_at
    # If it's still too slow, we reduce pop/iter.
    
    best_pos = initial_16d
    best_cost = compute_2d_cost_fast(initial_16d)
    
    # Run Differential Evolution
    # We'll use a callback to check time.
    def callback(xk, convergence):
        if time.time() - start_time > deadline:
            raise TimeoutError()
            
    try:
        # DE parameters
        # popsize=15 (75 candidates), maxiter=50
        # Tolerances tight
        result = differential_evolution(
            compute_2d_cost_fast,
            bounds,
            init='sobol',
            maxiter=100,
            popsize=10,
            tol=1e-6,
            seed=12345,
            callback=callback,
            polish=True  # Use L-BFGS-B at the end for local refinement
        )
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_pos = result.x
            
    except TimeoutError:
        pass
    except Exception as e:
        # If DE fails, fall back to local optimization from initial guess
        pass
        
    # If we have time left, do a local refinement with L-BFGS-B
    if time.time() - start_time < deadline - 10:
        from scipy.optimize import minimize
        
        def objective(x):
            return compute_2d_cost_fast(x)
            
        try:
            res = minimize(objective, best_pos, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_cost:
                best_cost = res.fun
                best_pos = res.x
        except Exception:
            pass
            
    # Construct the 64x2 output array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_pos[:N][i_flat]
    base_y = best_pos[N:][j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
