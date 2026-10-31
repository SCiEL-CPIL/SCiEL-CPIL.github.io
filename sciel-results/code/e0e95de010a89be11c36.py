N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_result = None
    best_score = -np.inf
    best_w = 1.0
    best_h = 1.0

    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.05 or h < 0.05:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score, best_w, best_h
        if validate(w, h, x, y, r):
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0, None)
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
                best_w = w
                best_h = h
                return True
        return False

    def force_relax(w, h, x, y, r, max_iters=3000, dt=0.01, damping=0.9):
        """
        Force-directed relaxation: circles repel each other and are attracted to center,
        with wall constraints handled by forces.
        """
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        
        # Track best valid solution
        best_local_x = x.copy()
        best_local_y = y.copy()
        best_local_r = r.copy()
        best_local_score = -np.inf
        
        if validate(w, h, x, y, r):
            best_local_score = np.sum(r)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Repulsion between circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < 1e-10:
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx * dx + dy * dy)
                    
                    if dist < min_dist:
                        # Overlapping: strong repulsion
                        overlap = min_dist - dist
                        force = 10.0 * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                        # Push radii apart slightly
                        fr[i] -= 0.5 * overlap
                        fr[j] -= 0.5 * overlap
                    else:
                        # Mild attraction to fill space (only if not too far)
                        if dist < min_dist * 1.5:
                            force = 0.1 * (dist - min_dist)
                            fx[i] -= force * dx / dist
                            fy[i] -= force * dy / dist
                            fx[j] += force * dx / dist
                            fy[j] += force * dy / dist
            
            # Wall forces: push circles away from walls
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    fx[i] += 10.0 * (r[i] - x[i])
                    fr[i] -= 2.0 * (r[i] - x[i])
                # Right wall
                if w - x[i] < r[i]:
                    fx[i] -= 10.0 * (r[i] - (w - x[i]))
                    fr[i] -= 2.0 * (r[i] - (w - x[i]))
                # Bottom wall
                if y[i] < r[i]:
                    fy[i] += 10.0 * (r[i] - y[i])
                    fr[i] -= 2.0 * (r[i] - y[i])
                # Top wall
                if h - y[i] < r[i]:
                    fy[i] -= 10.0 * (r[i] - (h - y[i]))
                    fr[i] -= 2.0 * (r[i] - (h - y[i]))
            
            # Centripetal force: gently pull toward center to prevent drift
            cx, cy = w / 2.0, h / 2.0
            for i in range(N):
                fx[i] += 0.01 * (cx - x[i])
                fy[i] += 0.01 * (cy - y[i])
            
            # Update positions
            x += dt * fx * damping
            y += dt * fy * damping
            r += dt * fr * damping
            
            # Hard constraints: keep circles in bounds and radii positive
            for i in range(N):
                r[i] = max(0.01, r[i])
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
            
            # Check validity
            if validate(w, h, x, y, r):
                score = np.sum(r)
                if score > best_local_score:
                    best_local_score = score
                    best_local_x = x.copy()
                    best_local_y = y.copy()
                    best_local_r = r.copy()
        
        return best_local_x, best_local_y, best_local_r

    def greedy_place(w, h, order=None):
        """
        Sequential greedy placement: place circles one by one at the position
        that maximizes the radius while avoiding overlap with already placed circles.
        """
        if order is None:
            order = list(range(N))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Grid of candidate positions
        n_samples = 40
        xs = np.linspace(0, w, n_samples)
        ys = np.linspace(0, h, n_samples)
        
        for idx in order:
            best_r = 0.0
            best_x = w / 2.0
            best_y = h / 2.0
            
            # Try candidate positions
            for cx in xs:
                for cy in ys:
                    # Max radius at this position given walls
                    r_wall = min(cx, w - cx, cy, h - cy)
                    if r_wall <= 0:
                        continue
                    
                    # Max radius given already placed circles
                    r_max = r_wall
                    for j in range(len(order)):
                        if order[j] == idx:
                            break
                        j_idx = order[j]
                        dx = cx - x[j_idx]
                        dy = cy - y[j_idx]
                        dist = np.sqrt(dx * dx + dy * dy)
                        r_allowed = dist - r[j_idx]
                        if r_allowed < r_max:
                            r_max = r_allowed
                    
                    if r_max > best_r:
                        best_r = r_max
                        best_x = cx
                        best_y = cy
            
            r[idx] = max(0.01, best_r)
            x[idx] = best_x
            y[idx] = best_y
        
        return x, y, r

    def hexagonal_init(w, h):
        """
        Initialize circles in a hexagonal pattern.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Estimate number of columns
        # Hexagonal packing: rows offset by half spacing
        # Try to fit N circles
        spacing = 0.15
        rows = 0
        cols = 0
        count = 0
        while count < N:
            cols = int(w / spacing) + 1
            if rows % 2 == 0:
                cols = int(w / spacing) + 1
            else:
                cols = int(w / spacing)
            count += cols
            rows += 1
        
        idx = 0
        for row in range(rows):
            if idx >= N:
                break
            n_in_row = int(w / spacing) + 1 if row % 2 == 0 else int(w / spacing)
            offset = spacing / 2.0 if row % 2 == 1 else 0.0
            for col in range(n_in_row):
                if idx >= N:
                    break
                cx = offset + (col + 0.5) * spacing
                cy = (row + 0.5) * spacing * 0.866
                if cx > w or cy > h:
                    continue
                r_est = min(cx, w - cx, cy, h - cy, spacing * 0.45)
                x[idx] = cx
                y[idx] = cy
                r[idx] = max(0.01, r_est)
                idx += 1
        
        # If we didn't place all, fill remaining with small circles
        while idx < N:
            x[idx] = w / 2.0 + random.uniform(-0.2, 0.2)
            y[idx] = h / 2.0 + random.uniform(-0.2, 0.2)
            r[idx] = 0.05
            idx += 1
        
        return x, y, r

    def grid_init(w, h):
        """
        Initialize circles in a grid pattern.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        n_cols = int(np.round(np.sqrt(N * w / h)))
        n_rows = int(np.ceil(N / n_cols))
        while n_cols * n_rows < N:
            n_cols += 1
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        max_r = min(cell_w, cell_h) * 0.45
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = max_r
                idx += 1
        
        return x, y, r

    def random_init(w, h):
        """
        Random initialization.
        """
        x = np.random.uniform(0.1, w - 0.1, N)
        y = np.random.uniform(0.1, h - 0.1, N)
        r = np.random.uniform(0.05, 0.15, N)
        return x, y, r

    # Main search loop
    np.random.seed(42)
    random.seed(42)
    
    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Initialize with multiple strategies
    init_strategies = ['hex', 'grid', 'random', 'greedy']
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        # Calculate w, h from aspect ratio
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        for strategy in init_strategies:
            if time.time() > deadline - 10.0:
                break
            
            if strategy == 'hex':
                x0, y0, r0 = hexagonal_init(w, h)
            elif strategy == 'grid':
                x0, y0, r0 = grid_init(w, h)
            elif strategy == 'random':
                x0, y0, r0 = random_init(w, h)
            elif strategy == 'greedy':
                # Greedy placement with random order
                order = list(range(N))
                random.shuffle(order)
                x0, y0, r0 = greedy_place(w, h, order)
            else:
                continue
            
            # Run force relaxation
            x_best, y_best, r_best = force_relax(w, h, x0, y0, r0, max_iters=5000)
            
            update_best(w, h, x_best, y_best, r_best)
            
            # Also try greedy placement followed by force relaxation
            if strategy != 'greedy' and time.time() < deadline - 10.0:
                order = list(range(N))
                random.shuffle(order)
                x_g, y_g, r_g = greedy_place(w, h, order)
                x_g2, y_g2, r_g2 = force_relax(w, h, x_g, y_g, r_g, max_iters=3000)
                update_best(w, h, x_g2, y_g2, r_g2)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 3.0:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        w = best_w
        h = best_h
        
        x_polished, y_polished, r_polished = force_relax(w, h, x, y, r, max_iters=3000, dt=0.005, damping=0.95)
        update_best(w, h, x_polished, y_polished, r_polished)

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        n_cols = 5
        for i in range(N):
            row = i // 5
            col = i % 5
            x[i] = (col + 0.5) * 0.2
            y[i] = (row + 0.5) * 0.2
            r[i] = 0.09
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
