import numpy as np
import itertools
from collections import Counter
import random
import time

def construct():
    def compute_best_area(pts):
        """
        Computes the maximum multiplicity of any non-zero doubled area.
        Returns (max_count, best_area, pts_copy)
        """
        n = len(pts)
        counts = Counter()
        # Optimization: Precompute x,y lists
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        for i in range(n):
            x1, y1 = xs[i], ys[i]
            for j in range(i+1, n):
                x2, y2 = xs[j], ys[j]
                dx12 = x2 - x1
                dy12 = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = xs[k], ys[k]
                    det = abs(dx12 * (y3 - y1) - dy12 * (x3 - x1))
                    if det > 0:
                        counts[det] += 1
                        
        if not counts:
            return 0, 0, pts
        best_area, best_count = max(counts.items(), key=lambda x: x[1])
        return best_count, best_area, pts

    def local_search(pts, max_iters=500):
        """
        Hill climbing / Simulated Annealing hybrid.
        """
        current_pts = pts[:]
        current_count, current_area, _ = compute_best_area(current_pts)
        global_best_count = current_count
        global_best_pts = current_pts[:]
        
        T = 10.0
        T_min = 0.1
        cooling = 0.99
        random.seed(42) # Fixed seed for reproducibility in tests? No, random is fine.
        
        for it in range(max_iters):
            if T < T_min:
                break
            
            # Random move: pick a point and move it to a random nearby location or swap
            # Let's try swapping two points or moving one point.
            # Moving one point to a random location in [-30, 30]
            idx = random.randint(0, 29)
            old_pt = current_pts[idx]
            
            # Generate a new point. To keep the set "dense", pick from a local neighborhood
            # or just a random point in the bounding box of current points expanded slightly.
            # For simplicity and speed, random point in [-30, 30]
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_pt = (new_x, new_y)
            
            # Ensure distinctness
            if new_pt in current_pts:
                # Try a few times to find a distinct point
                for _ in range(10):
                    new_x = random.randint(-30, 30)
                    new_y = random.randint(-30, 30)
                    new_pt = (new_x, new_y)
                if new_pt in current_pts:
                    continue
            
            new_pts = current_pts[:]
            new_pts[idx] = new_pt
            
            new_count, new_area, _ = compute_best_area(new_pts)
            
            delta = new_count - current_count
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts = new_pts
                current_count = new_count
                current_area = new_area
                
                if current_count > global_best_count:
                    global_best_count = current_count
                    global_best_pts = current_pts[:]
            
            T *= cooling
            
        return global_best_pts, global_best_count

    # Define multiple initial configurations
    inits = []
    
    # 1. 5x6 Grid
    inits.append([(x, y) for x in range(5) for y in range(6)])
    # 2. 6x5 Grid
    inits.append([(x, y) for x in range(6) for y in range(5)])
    # 3. 3x10 Grid
    inits.append([(x, y) for x in range(3) for y in range(10)])
    # 4. 10x3 Grid
    inits.append([(x, y) for x in range(10) for y in range(3)])
    # 5. Centered Grids (shifted to [-30, 30])
    # Shift 5x6 to center
    inits.append([(x-2, y-2) for x in range(5) for y in range(6)])
    inits.append([(x-2, y-2) for x in range(6) for y in range(5)])
    
    # 6. Random dense cluster
    random.seed(123)
    rand_pts = []
    while len(rand_pts) < 30:
        p = (random.randint(-10, 10), random.randint(-10, 10))
        if p not in rand_pts:
            rand_pts.append(p)
    inits.append(rand_pts)
    
    best_overall_pts = None
    best_overall_count = 0
    
    # Time management
    start_time = time.time()
    time_limit = 50.0 # Leave 10s buffer
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        # Run local search on this init
        # We can run multiple rounds of SA with different seeds/params
        for run in range(2):
            if time.time() - start_time > time_limit:
                break
            res_pts, res_count = local_search(init_pts, max_iters=1000)
            if res_count > best_overall_count:
                best_overall_count = res_count
                best_overall_pts = res_pts
                
    if best_overall_pts is None:
        # Fallback
        best_overall_pts = inits[0]
        
    return best_overall_pts
