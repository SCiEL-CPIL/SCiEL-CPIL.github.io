"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def construct():
    """
    Rigid-body assembly using linearized constraint solving.
    """
    
    def hex_initial(n, seed=42):
        """Hexagonal lattice, centered and scaled to fit in [-0.5,0.5]^2."""
        np.random.seed(seed)
        pts = []
        # Generate hex lattice rows
        for row in range(12):
            for col in range(12):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2.0
                pts.append([x, y])
        pts = np.array(pts[:n])
        pts -= pts.mean(axis=0)
        max_abs = np.max(np.abs(pts))
        pts *= (0.48 / max_abs)
        # Add slight jitter to break symmetry
        pts += np.random.randn(n, 2) * 0.01
        return pts

    def compute_radius(pts):
        """Compute the packing radius for a configuration."""
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        r_bound = 0.5 - np.max(np.abs(pts), axis=1)
        r_bound = np.min(r_bound)
        return min(r_pair, r_bound)

    def solve_linear_step(pts, r_target, max_step=0.05):
        """
        Solve for a displacement delta_x that increases distances to meet r_target.
        Uses linearized constraints.
        """
        n = len(pts)
        # Variables: delta_x[0..2n-1]
        c = np.zeros(2 * n)
        
        A_ub = []
        b_ub = []
        
        # 1. Pairwise distance constraints
        # We only include pairs that are currently "close" to avoid too many constraints
        r_current = compute_radius(pts)
        threshold = 2 * r_current * 1.2 + 0.01
        
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = pts[i] - pts[j]
                dist = np.linalg.norm(d_vec)
                if dist < threshold:
                    # Linearization: 2 * d_vec . (dx_i - dx_j) >= (2*r_target)^2 - dist^2
                    # d_vec . dx_i - d_vec . dx_j >= 0.5 * ((2*r_target)^2 - dist^2)
                    row = np.zeros(2 * n)
                    row[2*i:2*i+2] = d_vec
                    row[2*j:2*j+2] = -d_vec
                    rhs = 0.5 * ((2 * r_target)**2 - dist**2)
                    A_ub.append(row)
                    b_ub.append(rhs)
        
        # 2. Boundary constraints
        # We want pts + dx to be inside [-0.5 + r_target, 0.5 - r_target]
        # So: pts_i + dx_i >= -0.5 + r_target  =>  dx_i >= -0.5 + r_target - pts_i
        #     pts_i + dx_i <= 0.5 - r_target  =>  -dx_i >= pts_i - (0.5 - r_target) => dx_i <= 0.5 - r_target - pts_i
        
        # Linprog handles A_ub @ x <= b_ub.
        # So for lower bound: -dx_i <= pts_i - (-0.5 + r_target)
        # For upper bound: dx_i <= 0.5 - r_target - pts_i
        
        for i in range(n):
            for k in range(2):
                # Lower bound
                row = np.zeros(2 * n)
                row[2*i+k] = -1
                A_ub.append(row)
                b_ub.append(pts[i, k] - (-0.5 + r_target))
                
                # Upper bound
                row = np.zeros(2 * n)
                row[2*i+k] = 1
                A_ub.append(row)
                b_ub.append(0.5 - r_target - pts[i, k])
        
        # 3. Step size limit: ||dx|| <= max_step
        # Approximate with box constraints: -max_step <= dx <= max_step
        bounds = [(-max_step, max_step) for _ in range(2 * n)]
        
        if not A_ub:
            return np.zeros(2 * n)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                return np.zeros(2 * n)
        except Exception:
            return np.zeros(2 * n)

    # Main Logic
    pts = hex_initial(N)
    current_r = compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    max_iter = 500
    r_target = current_r * 1.05
    step_size = 0.05
    stagnation = 0

    for it in range(max_iter):
        # Solve for step
        dx = solve_linear_step(pts, r_target, max_step=step_size)
        
        # If no step found or step is tiny, break or increase target
        if np.linalg.norm(dx) < 1e-6:
            stagnation += 1
            if stagnation > 5:
                break
            # Increase target
            r_target *= 1.01
            continue
        else:
            stagnation = 0
            
        # Update positions
        new_pts = pts + dx.reshape(N, 2)
        
        # Clip to square (safety)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        # Check new radius
        new_r = compute_radius(new_pts)
        
        if new_r > current_r - 1e-9:
            # Accept step
            pts = new_pts
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
                
            # Increase target radius slightly if we are making progress
            if it % 10 == 0:
                r_target = min(current_r * 1.05, 0.1) # Cap target
            else:
                r_target = current_r * 1.02
        else:
            # Reject step, reduce target
            r_target = current_r * 0.98
            step_size *= 0.9
            
            if r_target < current_r * 0.95:
                break
                
    # Final refinement: try to squeeze more
    # Use a smaller step and tighter target
    r_target = compute_radius(best_pts) * 1.01
    step_size = 0.01
    stagnation = 0
    for it in range(100):
        dx = solve_linear_step(best_pts, r_target, max_step=step_size)
        if np.linalg.norm(dx) < 1e-7:
            stagnation += 1
            if stagnation > 5:
                break
            r_target *= 1.005
            continue
        else:
            stagnation = 0
        new_pts = best_pts + dx.reshape(N, 2)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        new_r = compute_radius(new_pts)
        if new_r > compute_radius(best_pts):
            best_pts = new_pts
            r_target = new_r * 1.005
        else:
            r_target *= 0.99
            if r_target < compute_radius(best_pts) * 0.99:
                break

    return best_pts
# EVOLVE-BLOCK-END
