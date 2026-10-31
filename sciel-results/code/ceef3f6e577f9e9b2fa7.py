"""Circle packing in perimeter-4 rectangle using topological graph model."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_value = 0.0
    best_result = None
    
    # Try multiple random initializations with different contact graphs
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.default_rng(attempt * 12345 + 67890)
        
        # Create a random contact graph (Delaunay-like structure)
        contact_graph = _generate_contact_graph(N, rng)
        
        # Initialize with a reasonable arrangement
        centers, radii = _initialize_packing(N, contact_graph, rng)
        
        # Optimize using force-directed relaxation on the contact graph
        result = _topological_optimize(centers, radii, contact_graph, 
                                       deadline - (time.time() - start_time))
        if result is not None:
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_result = result
        
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _generate_contact_graph(n, rng):
    """Generate a random contact graph for dense packing."""
    # Start with a grid-based graph and add some random edges
    side = int(np.ceil(np.sqrt(n)))
    graph = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        row, col = divmod(i, side)
        # Connect to neighbors in grid
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < side and 0 <= nc < side:
                j = nr * side + nc
                if j < n:
                    graph[i, j] = True
                    graph[j, i] = True
    
    # Add some random edges to break symmetry
    for _ in range(n // 3):
        i, j = rng.choice(n, 2, replace=False)
        graph[i, j] = True
        graph[j, i] = True
    
    return graph


def _initialize_packing(n, graph, rng):
    """Initialize circle positions and radii based on contact graph."""
    # Start with equal radii in a grid arrangement
    side = int(np.ceil(np.sqrt(n)))
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.1)
    
    for i in range(n):
        row, col = divmod(i, side)
        centers[i, 0] = (col + 0.5) / side + rng.uniform(-0.02, 0.02)
        centers[i, 1] = (row + 0.5) / side + rng.uniform(-0.02, 0.02)
    
    return centers, radii


def _topological_optimize(centers, radii, graph, time_budget):
    """Force-directed relaxation using contact graph constraints."""
    start_time = time.time()
    n = len(centers)
    
    # Parameter vector: [w, h, x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    w_init = 1.0
    h_init = 1.0
    params_init = np.concatenate(([w_init, h_init], centers.flatten(), radii))
    
    def objective(params):
        w = params[0]
        h = params[1]
        if w <= 0 or h <= 0:
            return 1e6
        
        # Enforce perimeter constraint
        if w + h > 2.0:
            return 1e6 + (w + h - 2.0) * 100
        
        centers_opt = params[2:2+2*n].reshape(n, 2)
        radii_opt = params[2+2*n:2+3*n]
        
        # Penalty for circles outside rectangle
        penalty = 0.0
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            if x - r < 0:
                penalty += (x - r) * 100
            if x + r > w:
                penalty += (x + r - w) * 100
            if y - r < 0:
                penalty += (y - r) * 100
            if y + r > h:
                penalty += (y + r - h) * 100
        
        # Penalty for overlapping circles (all pairs, not just graph edges)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii_opt[i] + radii_opt[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) * 50
        
        # Additional penalty to encourage contact on graph edges
        # This helps guide the search toward dense packings
        for i in range(n):
            for j in range(i+1, n):
                if graph[i, j]:
                    dx = centers_opt[i, 0] - centers_opt[j, 0]
                    dy = centers_opt[i, 1] - centers_opt[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    target_dist = radii_opt[i] + radii_opt[j]
                    # Penalize deviation from target contact distance
                    penalty += (dist - target_dist) ** 2 * 0.1
        
        # Maximize sum of radii
        return -np.sum(radii_opt) + penalty
    
    # Bounds
    bounds = [(0.1, 1.9), (0.1, 1.9)]  # w, h
    for i in range(n):
        bounds.append((0, 2))  # x
        bounds.append((0, 2))  # y
        bounds.append((0.01, 0.5))  # r
    
    # Optimize with time budget
    maxiter = int(min(500, time_budget * 50))  # Scale iterations with time
    
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e4:
            w = res.x[0]
            h = res.x[1]
            if w + h <= 2.0 + 1e-6 and w > 0 and h > 0:
                centers_opt = res.x[2:2+2*n].reshape(n, 2)
                radii_opt = res.x[2+2*n:2+3*n]
                
                # Verify solution is valid
                valid = True
                for i in range(n):
                    x, y = centers_opt[i]
                    r = radii_opt[i]
                    if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                        valid = False
                        break
                    if r < 0:
                        valid = False
                        break
                
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = centers_opt[i, 0] - centers_opt[j, 0]
                            dy = centers_opt[i, 1] - centers_opt[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < radii_opt[i] + radii_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    # Scale to ensure perimeter <= 4
                    scale = min(1.0, 2.0 / (w + h))
                    centers_final = centers_opt * scale
                    radii_final = radii_opt * scale
                    
                    result = np.column_stack([centers_final, radii_final])
                    return result
    except Exception:
        pass
    
    return None
