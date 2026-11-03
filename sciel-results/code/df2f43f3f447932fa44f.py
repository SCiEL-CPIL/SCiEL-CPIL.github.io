"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_packing_init(n_circles, aspect):
    """
    Generate a hexagonal packing of n_circles in a rectangle of given aspect ratio.
    Returns (centers, radii) where centers are (n,2) and radii are (n,).
    The rectangle has perimeter 4, so W + H = 2, H = aspect * W => W = 2/(1+aspect), H = 2*aspect/(1+aspect).
    """
    W = 2.0 / (1.0 + aspect)
    H = 2.0 * aspect / (1.0 + aspect)
    
    # Hexagonal lattice: rows of circles
    # For hex packing, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # We want to fit as many as possible in W x H
    
    # Try different numbers of rows
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    for n_rows in range(1, 8):
        # In hex packing, alternate rows are offset by r
        # Number of columns depends on row parity
        # Let's estimate: if we have n_rows rows, each row has approximately W/(2r) circles
        # But we need to figure out r first.
        
        # For a hexagonal packing in a rectangle W x H:
        # If we have m columns and n_rows rows:
        # Width needed: (m-1)*2r + 2r = m*2r (for aligned rows)
        # Actually for hex: horizontal spacing is 2r, so m circles need (m-1)*2r + 2r = m*2r? No.
        # m circles in a row: centers at r, 3r, 5r, ..., (2m-1)r. Width = (2m-1)r + r = 2mr? 
        # No: from first center at r to last at (2m-1)r, the extent is from 0 to 2mr, so width needed = 2mr.
        # Vertical: n_rows rows, spacing sqrt(3)*r, so height = (n_rows-1)*sqrt(3)*r + 2r
        
        # We want 2mr <= W and (n_rows-1)*sqrt(3)*r + 2r <= H
        # => r <= W/(2m) and r <= H / ((n_rows-1)*sqrt(3) + 2)
        
        # Total circles = n_rows * m (approx, with offset rows having m or m-1)
        # For simplicity, let's use m columns for all rows, with offset.
        
        for m in range(1, 12):
            r_w = W / (2.0 * m)
            r_h = H / ((n_rows - 1) * np.sqrt(3) + 2.0)
            r = min(r_w, r_h)
            if r <= 0:
                continue
            
            # Check total circles
            # Row 0 (even): m circles at x = r, 3r, 5r, ...
            # Row 1 (odd): m-1 or m circles? In standard hex, odd rows have one fewer if aligned.
            # Let's say even rows have m circles, odd rows have m-1 circles (offset by r).
            # Total = ceil(n_rows/2)*m + floor(n_rows/2)*(m-1)
            total = ((n_rows + 1) // 2) * m + (n_rows // 2) * (m - 1)
            if total < n_circles:
                continue
            
            # Generate centers
            centers = []
            for row in range(n_rows):
                y = r + row * np.sqrt(3) * r
                if row % 2 == 0:
                    # Even row: m circles
                    for col in range(m):
                        x = r + 2 * col * r
                        centers.append((x, y))
                else:
                    # Odd row: m-1 circles, offset by r
                    for col in range(m - 1):
                        x = 3 * r + 2 * col * r
                        centers.append((x, y))
            
            if len(centers) >= n_circles:
                centers = centers[:n_circles]
                radii = np.full(len(centers), r)
                s = sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = np.array(centers)
                    best_radii = radii
    
    if best_centers is None:
        # Fallback: square grid
        side = int(np.ceil(np.sqrt(n_circles)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < n_circles:
                    centers.append((W * (i + 0.5) / side, H * (j + 0.5) / side))
        r = min(W / side, H / side) / 2
        best_centers = np.array(centers)
        best_radii = np.full(n_circles, r)
    
    return best_centers, best_radii


def _build_constraints(n, aspect):
    """
    Build constraint matrices for:
    1. Non-overlap: ||c_i - c_j|| >= r_i + r_j for all i < j
       This is nonlinear, so we use a penalty approach or reformulate.
       For SLSQP, we can use the constraint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
       which is nonlinear. scipy SLSQP supports nonlinear constraints.
    2. Boundary: 0 <= x_i <= W, 0 <= y_i <= H, r_i >= 0
    3. Perimeter: W + H = 2 (fixed by aspect)
    
    We'll use nonlinear constraints for non-overlap.
    """
    W = 2.0 / (1.0 + aspect)
    H = 2.0 * aspect / (1.0 + aspect)
    
    # Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_{n-1}, y_{n-1}, r_{n-1}]
    # Total variables: 3n
    
    def non_overlap_constraints(x):
        """
        For each pair (i,j), constraint: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        We return the array of these values (should be >= 0).
        """
        x = x.reshape(n, 3)
        cx = x[:, 0]
        cy = x[:, 1]
        cr = x[:, 2]
        
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dr = cr[i] + cr[j]
                cons.append(dx * dx + dy * dy - dr * dr)
        return np.array(cons)
    
    def non_overlap_jacobian(x):
        """
        Jacobian of non_overlap_constraints.
        For pair (i,j): f = (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2
        df/dx_i = 2(x_i - x_j), df/dy_i = 2(y_i - y_j), df/dr_i = -2(r_i + r_j)
        df/dx_j = -2(x_i - x_j), df/dy_j = -2(y_i - y_j), df/dr_j = -2(r_i + r_j)
        """
        x = x.reshape(n, 3)
        cx = x[:, 0]
        cy = x[:, 1]
        cr = x[:, 2]
        
        n_pairs = n * (n - 1) // 2
        jac = np.zeros((n_pairs, 3 * n))
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dr = cr[i] + cr[j]
                
                # Row for pair (i,j)
                jac[pair_idx, 3 * i + 0] = 2 * dx
                jac[pair_idx, 3 * i + 1] = 2 * dy
                jac[pair_idx, 3 * i + 2] = -2 * dr
                jac[pair_idx, 3 * j + 0] = -2 * dx
                jac[pair_idx, 3 * j + 1] = -2 * dy
                jac[pair_idx, 3 * j + 2] = -2 * dr
                pair_idx += 1
        return jac
    
    return W, H, non_overlap_constraints, non_overlap_jacobian


def _optimize_aspect(aspect, time_budget):
    """
    Optimize the packing for a given aspect ratio.
    Returns (score, x_opt) or (None, None) if failed.
    """
    W = 2.0 / (1.0 + aspect)
    H = 2.0 * aspect / (1.0 + aspect)
    
    # Initialize with hex packing
    centers, radii = _hex_packing_init(N, aspect)
    
    # Build initial x vector: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3 * i + 0] = centers[i, 0]
        x0[3 * i + 1] = centers[i, 1]
        x0[3 * i + 2] = radii[i]
    
    # Objective: maximize sum of radii => minimize -sum(radii)
    def objective(x):
        r = x[2::3]
        return -np.sum(r)
    
    def objective_grad(x):
        g = np.zeros(3 * N)
        g[2::3] = -1.0
        return g
    
    # Constraints:
    # 1. Non-overlap (nonlinear, >= 0)
    # 2. Boundary constraints (linear)
    # 3. Radius positivity (linear)
    
    W, H, non_overlap_constraints, non_overlap_jacobian = _build_constraints(N, aspect)
    
    # Boundary: 0 <= x_i <= W, 0 <= y_i <= H
    # r_i >= 0
    
    # Linear constraints for boundaries
    # For each i: 0 <= x_i <= W => -x_i <= 0 and x_i <= W
    # 0 <= y_i <= H => -y_i <= 0 and y_i <= H
    # r_i >= 0 => -r_i <= 0
    
    A_ub = []
    b_ub = []
    
    for i in range(N):
        # x_i <= W
        row = np.zeros(3 * N)
        row[3 * i + 0] = 1
        A_ub.append(row)
        b_ub.append(W)
        
        # -x_i <= 0
        row = np.zeros(3 * N)
        row[3 * i + 0] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        # y_i <= H
        row = np.zeros(3 * N)
        row[3 * i + 1] = 1
        A_ub.append(row)
        b_ub.append(H)
        
        # -y_i <= 0
        row = np.zeros(3 * N)
        row[3 * i + 1] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        # -r_i <= 0
        row = np.zeros(3 * N)
        row[3 * i + 2] = -1
        A_ub.append(row)
        b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Nonlinear constraints
    cons = [
        {
            'type': 'ineq',
            'fun': non_overlap_constraints,
            'jac': non_overlap_jacobian
        }
    ]
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, W))  # x_i
        bounds.append((0, H))  # y_i
        bounds.append((0, min(W, H) / 2))  # r_i
    
    try:
        start_time = time.time()
        result = minimize(
            objective, x0, method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return None, None
        
        if result.success:
            x_opt = result.x
            score = -result.fun
            # Verify validity
            x = x_opt.reshape(N, 3)
            cx = x[:, 0]
            cy = x[:, 1]
            cr = x[:, 2]
            
            # Check non-overlap
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
                    if dist < cr[i] + cr[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            # Check boundaries
            if valid:
                if np.any(cx < -1e-9) or np.any(cx > W + 1e-9) or np.any(cy < -1e-9) or np.any(cy > H + 1e-9):
                    valid = False
            if valid and np.any(cr < -1e-9):
                valid = False
            
            if valid:
                return score, x_opt
            else:
                return None, None
        else:
            return None, None
    except Exception:
        return None, None


def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    best_score = 0.0
    best_x = None
    
    # Try multiple aspect ratios
    aspects = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for aspect in aspects:
        elapsed = time.time() - start_time
        time_budget = max(0, deadline - elapsed - 10.0)  # 10s safety margin
        if time_budget <= 0:
            break
        
        score, x_opt = _optimize_aspect(aspect, time_budget)
        
        if score is not None and score > best_score:
            best_score = score
            best_x = x_opt
    
    # If no optimization succeeded, fall back to hex packing
    if best_x is None:
        # Try a few aspect ratios with just hex packing (no optimization)
        for aspect in [1.0, 0.8, 1.25]:
            centers, radii = _hex_packing_init(N, aspect)
            if len(centers) == N:
                x = np.zeros(3 * N)
                for i in range(N):
                    x[3 * i + 0] = centers[i, 0]
                    x[3 * i + 1] = centers[i, 1]
                    x[3 * i + 2] = radii[i]
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_x = x
    
    if best_x is None:
        # Ultimate fallback
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append(((i + 0.5) / side, (j + 0.5) / side))
        r = 0.5 / side - 1e-3
        best_x = np.zeros(3 * N)
        for i in range(N):
            best_x[3 * i + 0] = centers[i][0]
            best_x[3 * i + 1] = centers[i][1]
            best_x[3 * i + 2] = r
    
    # Convert to output format: (n, 3) array of (x, y, r)
    result = best_x.reshape(N, 3)
    
    return result
# EVOLVE-BLOCK-END
