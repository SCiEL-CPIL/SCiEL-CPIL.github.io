import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import linprog

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    def elapsed():
        return time.time() - start_time

    def validate(x, y, r, tol=1e-8):
        """Check if the packing is valid: within bounds and no overlaps."""
        if np.any(x - r < -tol) or np.any(x + r > 1.0 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1.0 + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def get_score(x, y, r):
        return float(np.sum(r))

    def solve_radii_for_graph(positions, edges, bounds_width=1.0, bounds_height=1.0):
        """
        Given fixed positions and a contact graph (edges), solve for maximum radii
        such that circles don't overlap and stay in bounds.
        
        This is a linear program:
        Maximize sum(r_i)
        Subject to:
            r_i + r_j <= dist(i,j) for all edges (i,j)
            r_i <= x_i (left boundary)
            r_i <= bounds_width - x_i (right boundary)
            r_i <= y_i (bottom boundary)
            r_i <= bounds_height - y_i (top boundary)
            r_i >= 0
        """
        n = len(positions)
        if n == 0:
            return np.array([])
        
        # Objective: maximize sum(r) -> minimize -sum(r)
        c = -np.ones(n)
        
        # Inequality constraints: A_ub @ r <= b_ub
        # For each edge (i,j): r_i + r_j <= dist(i,j)
        # For boundaries: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        
        A_rows = []
        b_rows = []
        
        for (i, j) in edges:
            d = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                       (positions[i][1] - positions[j][1])**2)
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_rows.append(d)
        
        for i in range(n):
            # Left: r_i <= x_i
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(positions[i][0])
            
            # Right: r_i <= 1 - x_i
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(bounds_width - positions[i][0])
            
            # Bottom: r_i <= y_i
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(positions[i][1])
            
            # Top: r_i <= 1 - y_i
            row = np.zeros(n)
            row[i] = 1
            A_rows.append(row)
            b_rows.append(bounds_height - positions[i][1])
        
        if not A_rows:
            return np.zeros(n)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                r = -res.x
                # Clip to non-negative
                r = np.maximum(r, 0)
                return r
            else:
                return np.zeros(n)
        except Exception:
            return np.zeros(n)

    def get_delaunay_edges(positions):
        """Get edges from Delaunay triangulation."""
        try:
            tri = Delaunay(positions)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            return list(edges)
        except Exception:
            return []

    def force_relaxation(x, y, r, iterations=200, dt=0.001):
        """
        Simple force-directed relaxation to improve packing.
        Pushes overlapping circles apart and pulls them toward center of available space.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for it in range(iterations):
            if elapsed() > deadline - 5:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Repulsion between overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-10:
                        # Force proportional to overlap
                        force = (min_dist - dist) * 0.5
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    elif dist < 1e-10:
                        # Random small perturbation
                        fx[i] += np.random.uniform(-0.01, 0.01)
                        fy[i] += np.random.uniform(-0.01, 0.01)
                        fx[j] -= fx[i]
                        fy[j] -= fy[i]
            
            # Boundary forces: push circles back into bounds
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 0.5
                if x[i] > 1 - r[i]:
                    fx[i] -= (x[i] - (1 - r[i])) * 0.5
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 0.5
                if y[i] > 1 - r[i]:
                    fy[i] -= (y[i] - (1 - r[i])) * 0.5
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Clamp to valid range (loose)
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
        
        return x, y, r

    def improve_with_lp(positions, edges, max_iters=50):
        """
        Iteratively: solve LP for radii given positions and graph,
        then update positions to maximize minimum clearance.
        """
        x = positions[:, 0].copy()
        y = positions[:, 1].copy()
        
        for it in range(max_iters):
            if elapsed() > deadline - 5:
                break
                
            # Solve for radii
            r = solve_radii_for_graph(positions, edges)
            
            if np.sum(r) < 1e-6:
                break
            
            # Check if we can improve by moving positions
            # Use a simple gradient ascent on sum(r) w.r.t. positions
            # This is complex, so we use force relaxation instead
            
            x_new, y_new, _ = force_relaxation(x, y, r, iterations=50, dt=0.0005)
            
            # Check if improvement
            positions_new = np.column_stack((x_new, y_new))
            
            # Re-solve with new positions
            r_new = solve_radii_for_graph(positions_new, edges)
            
            if np.sum(r_new) > np.sum(r) * (1 + 1e-6):
                x, y = x_new, y_new
                positions = positions_new
            else:
                # No improvement, try smaller steps
                break
        
        return x, y, r

    def newton_raphson_packing(x, y, r, edges, max_iters=100):
        """
        Newton-Raphson method for circle packing.
        Given a contact graph, solve for positions and radii such that
        all contacts are tight and no overlaps.
        
        Variables: x_i, y_i, r_i for i=0..N-1
        Constraints:
            (x_i - x_j)^2 + (y_i - y_j)^2 = (r_i + r_j)^2 for each edge (i,j)
            x_i = r_i (left wall contact) or x_i = 1 - r_i (right wall)
            y_i = r_i (bottom wall) or y_i = 1 - r_i (top wall)
        
        We use a simplified approach: fix which walls each circle touches,
        then solve the nonlinear system.
        """
        # This is complex; we use a penalty-based approach instead
        # For now, just return the LP solution
        return x, y, r

    # Main algorithm
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Strategy 1: Start with grid, then improve
    # Strategy 2: Random restarts with force relaxation + LP
    # Strategy 3: Delaunay-based topology optimization
    
    num_restarts = 15
    
    for restart in range(num_restarts):
        if elapsed() > deadline - 10:
            break
        
        if restart == 0:
            # Grid start
            side = int(np.ceil(np.sqrt(N)))
            x = []
            y = []
            for i in range(side):
                for j in range(side):
                    if len(x) < N:
                        x.append((i + 0.5) / side)
                        y.append((j + 0.5) / side)
            x = np.array(x)
            y = np.array(y)
            r = np.full(N, 0.1)
        elif restart == 1:
            # Better grid: 5x5 with some removed
            x = []
            y = []
            positions_5x5 = [((i + 0.5) / 5, (j + 0.5) / 5) for i in range(5) for j in range(5)]
            # Take first N positions
            for i in range(N):
                x.append(positions_5x5[i][0])
                y.append(positions_5x5[i][1])
            x = np.array(x)
            y = np.array(y)
            r = np.full(N, 0.09)
        else:
            # Random start
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.full(N, 0.05)
        
        positions = np.column_stack((x, y))
        
        # Get initial Delaunay edges
        edges = get_delaunay_edges(positions)
        
        # Solve for radii
        r = solve_radii_for_graph(positions, edges)
        
        # Force relaxation
        x_rel, y_rel, r_rel = force_relaxation(x, y, r, iterations=300, dt=0.001)
        positions_rel = np.column_stack((x_rel, y_rel))
        
        # Update Delaunay edges
        edges_new = get_delaunay_edges(positions_rel)
        
        # Solve again with new edges
        r_new = solve_radii_for_graph(positions_rel, edges_new)
        
        # Check validity
        if validate(x_rel, y_rel, r_new):
            score = get_score(x_rel, y_rel, r_new)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x_rel, y_rel, r_new))
        
        # Try to improve further with LP iteration
        if elapsed() < deadline - 15:
            x_improved, y_improved, r_improved = improve_with_lp(
                positions_rel, edges_new, max_iters=20
            )
            
            # Update edges again
            positions_improved = np.column_stack((x_improved, y_improved))
            edges_improved = get_delaunay_edges(positions_improved)
            r_improved = solve_radii_for_graph(positions_improved, edges_improved)
            
            if validate(x_improved, y_improved, r_improved):
                score = get_score(x_improved, y_improved, r_improved)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((x_improved, y_improved, r_improved))
    
    # Final fallback if no valid solution found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        best_result = np.array(cells[:N])
    
    return best_result
