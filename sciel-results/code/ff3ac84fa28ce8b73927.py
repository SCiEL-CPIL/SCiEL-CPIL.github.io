"""Combined program: Topological graph model with contact constraints and force-directed relaxation."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    best_result = None
    best_score = -np.inf

    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            # Clip small numerical errors
            x_c = np.clip(x, 0, w)
            y_c = np.clip(y, 0, h)
            r_c = np.clip(r, 0, None)
            
            score = np.sum(r_c)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x_c, y_c, r_c])
                return True
        return False

    def build_contact_graph(x, y, r, w, h, threshold=1.05):
        """
        Build a contact graph where edges represent circles that are close to touching.
        Also includes wall contacts.
        Returns: list of edges (i, j, type) where type is 'circle' or 'wall'
                 Wall edges: (i, 'L'), (i, 'R'), (i, 'B'), (i, 'T')
        """
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist * threshold:
                    edges.append((i, j, 'circle'))
            # Wall contacts
            if x[i] < r[i] * threshold:
                edges.append((i, -1, 'L'))
            if w - x[i] < r[i] * threshold:
                edges.append((i, -2, 'R'))
            if y[i] < r[i] * threshold:
                edges.append((i, -3, 'B'))
            if h - y[i] < r[i] * threshold:
                edges.append((i, -4, 'T'))
        return edges

    def force_relax_topo(w, h, x, y, r, max_iter=500, step_size=0.02, growth_rate=0.005):
        """
        Force-directed relaxation using topological contact graph.
        The idea: maintain a contact graph, and apply forces to satisfy distance constraints
        for contact edges (dist = r_i + r_j for circle-circle, dist = r_i for wall).
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = len(x)
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Rebuild contact graph periodically
            if it % 10 == 0:
                edges = build_contact_graph(x, y, r, w, h, threshold=1.1)
            else:
                edges = build_contact_graph(x, y, r, w, h, threshold=1.1)
            
            # Reset forces
            fx.fill(0.0)
            fy.fill(0.0)
            
            # Apply forces based on contact constraints
            for edge in edges:
                i, j, etype = edge
                if etype == 'circle':
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    target_dist = r[i] + r[j]
                    if dist < 1e-8:
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    # Force to push apart if too close, or pull together if too far (within threshold)
                    error = dist - target_dist
                    force = error * 5.0  # Stiffness
                    fx[i] += force * (dx / dist)
                    fy[i] += force * (dy / dist)
                    fx[j] -= force * (dx / dist)
                    fy[j] -= force * (dy / dist)
                elif etype == 'L':
                    # Circle i should have x[i] = r[i]
                    error = x[i] - r[i]
                    fx[i] += error * 5.0
                elif etype == 'R':
                    # Circle i should have x[i] = w - r[i]
                    error = x[i] - (w - r[i])
                    fx[i] += error * 5.0
                elif etype == 'B':
                    # Circle i should have y[i] = r[i]
                    error = y[i] - r[i]
                    fy[i] += error * 5.0
                elif etype == 'T':
                    # Circle i should have y[i] = h - r[i]
                    error = y[i] - (h - r[i])
                    fy[i] += error * 5.0
            
            # Also add repulsion for non-contact pairs to prevent overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        if dist > 1e-6:
                            overlap = min_dist - dist
                            force = overlap * 10.0
                            fx[i] += force * (dx / dist)
                            fy[i] += force * (dy / dist)
                            fx[j] -= force * (dx / dist)
                            fy[j] -= force * (dy / dist)
                        else:
                            fx[i] += random.uniform(-1, 1)
                            fy[i] += random.uniform(-1, 1)
                            fx[j] -= random.uniform(-1, 1)
                            fy[j] -= random.uniform(-1, 1)
            
            # Wall repulsion
            for i in range(n):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10.0
                if w - x[i] < r[i]:
                    fx[i] -= (r[i] - (w - x[i])) * 10.0
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10.0
                if h - y[i] < r[i]:
                    fy[i] -= (r[i] - (h - y[i])) * 10.0
            
            # Update positions
            x += fx * step_size
            y += fy * step_size
            
            # Clamp positions
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            
            # Try to increase radii
            new_r = r * (1.0 + growth_rate)
            valid_growth = True
            for i in range(n):
                if new_r[i] > x[i] or new_r[i] > w - x[i] or new_r[i] > y[i] or new_r[i] > h - y[i]:
                    valid_growth = False
                    break
            if valid_growth:
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = new_r[i] + new_r[j]
                        if dist_sq < min_dist * min_dist:
                            valid_growth = False
                            break
                    if not valid_growth:
                        break
            
            if valid_growth:
                r = new_r
            
            # Adjust aspect ratio based on bounding box
            if it % 50 == 0:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                span_x = x_max - x_min + 2 * np.mean(r)
                span_y = y_max - y_min + 2 * np.mean(r)
                
                if span_x > 0 and span_y > 0:
                    target_aspect = span_x / span_y
                    target_w = 2.0 * target_aspect / (1.0 + target_aspect)
                    target_w = np.clip(target_w, 0.5, 1.5)
                    
                    w = 0.9 * w + 0.1 * target_w
                    h = 2.0 - w
                    
                    x = np.clip(x, 0, w)
                    y = np.clip(y, 0, h)
                    
                    if not validate(w, h, x, y, r, tol=1e-8):
                        r = r * 0.95
        
        return x, y, r, w, h

    def create_grid_init(w, seed=42):
        h = 2.0 - w
        aspect = w / h
        if aspect >= 1.0:
            n_cols = int(np.round(np.sqrt(N * aspect)))
            n_rows = int(np.ceil(N / n_cols))
        else:
            n_rows = int(np.round(np.sqrt(N / aspect)))
            n_cols = int(np.ceil(N / n_rows))
        
        while n_cols * n_rows < N:
            if aspect >= 1.0:
                n_cols += 1
            else:
                n_rows += 1
            
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
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
        return x, y, r, w, h

    # Multi-start with varied aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.85, 1.15]
    random.seed(42)
    np.random.seed(42)
    random.shuffle(aspect_ratios)
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        x, y, r, w, h = create_grid_init(w)
        r = r * 0.5
        
        # Run topological force relaxation
        x, y, r, w, h = force_relax_topo(w, h, x, y, r, max_iter=300, step_size=0.02, growth_rate=0.005)
        
        update_best(w, h, x, y, r)
        
        # Polish
        if time.time() < deadline - 5.0:
            x, y, r, w, h = force_relax_topo(w, h, x, y, r, max_iter=100, step_size=0.005, growth_rate=0.001)
            update_best(w, h, x, y, r)

    if best_result is None:
        x, y, r, w, h = create_grid_init(1.0)
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
