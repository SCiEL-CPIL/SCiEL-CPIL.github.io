import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    max_time = 115.0  # 5s margin
    
    best_val = -np.inf
    best_circles = None
    
    # Try a range of aspect ratios
    # Perimeter is 2(W+H)=4 => W+H=2 => H=2-W
    # W can range from small to large.
    # If W is very small, H is large.
    # Let's try W in [0.5, 1.5] (symmetric around 1.0)
    
    aspect_ratios = np.linspace(0.6, 1.4, 9)
    
    for W in aspect_ratios:
        if time.time() - start_time > max_time:
            break
        H = 2.0 - W
        
        # Grid search for initialization
        # We need to place N circles.
        # Let's try a grid of potential centers.
        # Resolution: 20x20 grid inside the rectangle.
        grid_res = 20
        xs = np.linspace(0.05, W - 0.05, grid_res)
        ys = np.linspace(0.05, H - 0.05, grid_res)
        
        # Generate all potential points
        points = []
        for x in xs:
            for y in ys:
                points.append((x, y))
        
        # Sort points by "potential radius" (min dist to boundary)
        # This helps place larger circles first in tight spots or corners
        # Actually, for packing, we usually want to place circles where they fit.
        # Let's just iterate through a shuffled grid or sorted by distance to center?
        # Let's try sorting by distance to the closest corner or boundary to encourage boundary hugging?
        # Or just random shuffle for diversity.
        
        np.random.seed(42 + int(W*100)) # Deterministic but varies with W
        np.random.shuffle(points)
        
        circles = []
        # Greedy placement
        for px, py in points:
            if len(circles) >= N:
                break
            # Calculate max possible radius for a circle at (px, py)
            # limited by boundary
            r_max = min(px, py, W - px, H - py)
            if r_max < 1e-4:
                continue
            
            # Check overlap with existing circles
            valid = True
            for cx, cy, cr in circles:
                dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                if dist < r_max + cr - 1e-8:
                    valid = False
                    break
            
            if valid:
                circles.append((px, py, r_max))
        
        if len(circles) < N:
            # If we couldn't place N circles on this grid, try a denser grid or different W
            # But first, let's try to fill the remaining spots with smaller radii?
            # No, the greedy algorithm places max possible radius.
            # If we run out of points, we failed.
            # Let's try a slightly different strategy: 
            # Use a local search to improve the current set if we have N circles.
            
            # If we don't have N, this W/grid combo failed.
            # We could try a finer grid, but that's expensive.
            # Let's skip to next W.
            continue
            
        # We have N circles. Now refine.
        # Local search: try to move circles or expand radii.
        # Objective: maximize sum of radii.
        # Constraint: no overlap, inside boundary.
        
        current_circles = [c[:] for c in circles]
        current_val = sum(c[2] for c in current_circles)
        
        # Simple Hill Climbing / Simulated Annealing
        # Since N=21, we can do a few iterations.
        # Each iteration: pick a random circle, try to move it or change radius.
        
        T = 0.01
        for step in range(500): # Limit steps per aspect ratio
            if time.time() - start_time > max_time:
                break
            
            i = np.random.randint(N)
            cx, cy, cr = current_circles[i]
            
            # Try a random perturbation in position and radius
            # Move: small delta
            # Radius: small delta
            
            delta_x = np.random.normal(0, 0.05)
            delta_y = np.random.normal(0, 0.05)
            delta_r = np.random.normal(0, 0.01)
            
            new_x = cx + delta_x
            new_y = cy + delta_y
            new_r = cr + delta_r
            
            # Bounds check
            if new_r < 1e-4:
                continue
            if new_x < new_r or new_x > W - new_r:
                continue
            if new_y < new_r or new_y > H - new_r:
                continue
            
            # Overlap check with all other circles
            valid = True
            for j in range(N):
                if i == j:
                    continue
                ox, oy, or_ = current_circles[j]
                dist = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                if dist < new_r + or_ - 1e-8:
                    valid = False
                    break
            
            if valid:
                new_val = current_val - cr + new_r
                if new_val > current_val:
                    # Accept
                    current_circles[i] = [new_x, new_y, new_r]
                    current_val = new_val
                    T *= 0.95 # Cool down
                else:
                    # Reject, but maybe accept if T is high?
                    # For simplicity, let's just do hill climbing (only accept improvements)
                    # Or simple SA:
                    # prob = np.exp((new_val - current_val) / T)
                    # if np.random.rand() < prob: ...
                    pass # Just hill climbing for now to ensure stability
            
            if current_val > best_val:
                best_val = current_val
                best_circles = [c[:] for c in current_circles]
                
    if best_circles is None:
        # Fallback: Grid of equal circles
        # This is always valid.
        # 5x5 grid in 1x1 square?
        # W=1, H=1.
        W = 1.0
        H = 1.0
        cells = []
        for i in range(5):
            for j in range(5):
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.5 / 5.0 - 0.001
                cells.append([x, y, r])
                if len(cells) == N:
                    break
            if len(cells) == N:
                break
        return np.array(cells)
        
    return np.array(best_circles)
