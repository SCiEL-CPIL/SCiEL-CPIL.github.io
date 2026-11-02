"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import linprog


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def get_delaunay_pairs(pos):
        """Get pairs of circles that are close (Delaunay neighbors)."""
        n = len(pos)
        try:
            tri = Delaunay(pos)
            pairs = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i+1, len(simplex)):
                        a, b = simplex[i], simplex[j]
                        pairs.add((min(a, b), max(a, b)))
        except Exception:
            # Fallback: all pairs
            pairs = set()
            for i in range(n):
                for j in range(i+1, n):
                    pairs.add((i, j))
        return list(pairs)

    def solve_lp(pos, rad, width, height, pairs):
        """
        Solve LP to maximize sum of radii subject to:
        - Circle i stays in [r_i, width-r_i] x [r_i, height-r_i]
        - For each pair (i,j) in pairs: dist(pos_i, pos_j) >= r_i + r_j
        We linearize the distance constraint using the current positions as normals.
        Variables: r_0, ..., r_{n-1}
        Constraints:
        For each pair (i,j):
            (pos_i - pos_j) dot (1,0) * r_i + (pos_i - pos_j) dot (1,0) * r_j 
            Actually, let's use: |pos_i - pos_j|^2 >= (r_i + r_j)^2
            Linearize: 2*(pos_i - pos_j) dot (pos_i - pos_j) >= 2*(r_i + r_j)*(|pos_i - pos_j|)
            => (pos_i - pos_j) dot (pos_i - pos_j) / |pos_i - pos_j| >= r_i + r_j
            Let d_ij = |pos_i - pos_j|, n_ij = (pos_i - pos_j)/d_ij
            Then: n_ij dot (pos_i - pos_j) = d_ij, so constraint is d_ij >= r_i + r_j
            But that's not right for linearization.
            
            Better linearization: The constraint is |p_i - p_j| >= r_i + r_j
            At the current point, the gradient w.r.t. r_i is 1, w.r.t. r_j is 1.
            The gradient w.r.t. position is along the direction.
            For LP on radii only (positions fixed), the constraint is:
            r_i + r_j <= |p_i - p_j|
            This is a linear constraint in r_i, r_j!
        """
        n = len(pos)
        # Objective: maximize sum(r) => minimize -sum(r)
        c = -np.ones(n)
        
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= dist(i,j)
        for i, j in pairs:
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
        
        # Boundary constraints: r_i <= pos_i[0], r_i <= width - pos_i[0], r_i <= pos_i[1], r_i <= height - pos_i[1]
        for i in range(n):
            for bound in [pos[i][0], width - pos[i][0], pos[i][1], height - pos[i][1]]:
                row = np.zeros(n)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(bound)
        
        # r_i >= 0
        bounds = [(0, None) for _ in range(n)]
        
        if not A_ub:
            return rad.copy()
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                new_rad = res.x
                # Ensure non-negative
                new_rad = np.maximum(new_rad, 0)
                return new_rad
        except Exception:
            pass
        
        return rad.copy()

    def move_circles(pos, rad, width, height, pairs, alpha=0.5, max_iter=200):
        """
        Move circles to reduce overlaps and improve packing.
        Uses gradient information from constraint violations.
        """
        n = len(pos)
        pos = pos.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute forces
            force = np.zeros_like(pos)
            
            # Pair forces
            for i, j in pairs:
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    force[i][0] += fx
                    force[i][1] += fy
                    force[j][0] -= fx
                    force[j][1] -= fy
            
            # Boundary forces
            for i in range(n):
                if pos[i][0] < rad[i]:
                    force[i][0] += (rad[i] - pos[i][0])
                if pos[i][0] > width - rad[i]:
                    force[i][0] -= (pos[i][0] - (width - rad[i]))
                if pos[i][1] < rad[i]:
                    force[i][1] += (rad[i] - pos[i][1])
                if pos[i][1] > height - rad[i]:
                    force[i][1] -= (pos[i][1] - (height - rad[i]))
            
            # Check if we're done
            max_force = np.max(np.abs(force))
            if max_force < 1e-6:
                break
            
            # Update positions
            pos += alpha * force
            alpha *= 0.98  # decay step size
            
            # Clip to bounds
            for i in range(n):
                pos[i][0] = np.clip(pos[i][0], rad[i], width - rad[i])
                pos[i][1] = np.clip(pos[i][1], rad[i], height - rad[i])
            
            # Update Delaunay pairs periodically
            if it % 20 == 19:
                pairs = get_delaunay_pairs(pos)
        
        return pos, pairs

    def is_valid(pos, rad, width, height, tol=1e-7):
        """Check if the packing is valid."""
        n = len(pos)
        for i in range(n):
            if pos[i][0] < rad[i] - tol or pos[i][0] > width - rad[i] + tol:
                return False
            if pos[i][1] < rad[i] - tol or pos[i][1] > height - rad[i] + tol:
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rad[i] + rad[j] - tol:
                    return False
        return True

    def evaluate(pos, rad, width, height):
        """Evaluate the sum of radii if valid, else 0."""
        if is_valid(pos, rad, width, height):
            return np.sum(rad)
        return 0.0

    def optimize_for_aspect(aspect, init_pos=None, init_rad=None):
        """Run the full optimization for a given aspect ratio."""
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        if init_pos is None:
            # Initialize on a grid
            side = int(np.ceil(np.sqrt(N)))
            init_pos = []
            init_rad = []
            for i in range(side):
                for j in range(side):
                    if len(init_pos) >= N:
                        break
                    x = (i + 0.5) * w / side
                    y = (j + 0.5) * h / side
                    r = min(w, h) / (2 * side) * 0.9
                    init_pos.append([x, y])
                    init_rad.append(r)
            init_pos = np.array(init_pos[:N])
            init_rad = np.array(init_rad[:N])
        
        pos = init_pos.copy()
        rad = init_rad.copy() if init_rad is not None else np.full(N, min(w, h) / (2 * int(np.ceil(np.sqrt(N)))) * 0.9)
        
        best_score = 0
        best_pos = pos.copy()
        best_rad = rad.copy()
        
        # Iterative LP + move loop
        for outer in range(50):
            if time.time() > deadline:
                break
            
            # Get Delaunay pairs
            pairs = get_delaunay_pairs(pos)
            
            # Solve LP for radii
            new_rad = solve_lp(pos, rad, w, h, pairs)
            
            # Move circles
            pos, pairs = move_circles(pos, new_rad, w, h, pairs, alpha=0.3, max_iter=100)
            
            # Update radii
            rad = new_rad
            
            # Evaluate
            score = evaluate(pos, rad, w, h)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_rad = rad.copy()
        
        return best_pos, best_rad, best_score

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.75, 1.4, 0.7, 1.5, 0.65]
    
    best_overall_score = 0
    best_overall_result = None
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        pos, rad, score = optimize_for_aspect(aspect)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_result = np.column_stack([pos, rad])
    
    if best_overall_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_overall_result
# EVOLVE-BLOCK-END
