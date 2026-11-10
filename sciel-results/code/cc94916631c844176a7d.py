# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    """Return (best_margin, active_normal). best_margin>0 means separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _outer_side_for_points(pts, angle_deg=0.0):
    """Compute side length of min enclosing regular hexagon for points."""
    a = np.deg2rad(angle_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    normals = np.column_stack((np.cos(th), np.sin(th)))
    proj = pts @ normals.T
    maxproj = proj.max(axis=0)
    apothem = maxproj.max()
    s = 2.0 * apothem / np.sqrt(3.0)
    return s

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def _objective(x, angle_deg=0.0):
    """
    Objective function for L-BFGS-B.
    x: (22,) array of 11 centers (x, y).
    Returns: side length + penalty for overlaps.
    """
    centers = x.reshape(11, 2)
    
    # Compute side length of enclosing hexagon
    side = _outer_side_for_points(centers, angle_deg)
    
    # Compute overlap penalty
    penalty = 0.0
    Vs = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m < 0:
                # Penalize overlaps heavily
                penalty += (-m) ** 2 * 100.0
    
    return side + penalty

def _objective_grad(x, angle_deg=0.0):
    """
    Numerical gradient for the objective function.
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _objective(x, angle_deg)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        fp = _objective(xp, angle_deg)
        xm = x.copy()
        xm[i] -= eps
        fm = _objective(xm, angle_deg)
        grad[i] = (fp - fm) / (2 * eps)
    return grad

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0  # Safety margin
    
    # Generate multiple initial configurations
    def _make_initial_configs():
        configs = []
        
        # Config 0: Standard 7-flower + 4 satellites
        def axial_to_cartesian(q, r):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            return x, y
        
        centers0 = np.array([
            axial_to_cartesian(0, 0),
            axial_to_cartesian(1, 0),
            axial_to_cartesian(0, 1),
            axial_to_cartesian(-1, 1),
            axial_to_cartesian(-1, 0),
            axial_to_cartesian(0, -1),
            axial_to_cartesian(1, -1),
            axial_to_cartesian(2, 0),
            axial_to_cartesian(0, 2),
            axial_to_cartesian(-2, 0),
            axial_to_cartesian(0, -2),
        ])
        centers0 -= centers0.mean(axis=0)
        configs.append(centers0.flatten())
        
        # Config 1: Perturbed version
        rng = np.random.default_rng(42)
        centers1 = centers0 + rng.normal(0, 0.2, size=centers0.shape)
        centers1 -= centers1.mean(axis=0)
        configs.append(centers1.flatten())
        
        # Config 2: Different topology - more compact
        centers2 = np.array([
            (0, 0),
            (2, 0),
            (1, 1.732),
            (-1, 1.732),
            (-2, 0),
            (-1, -1.732),
            (1, -1.732),
            (3, 1.732),
            (-3, 1.732),
            (3, -1.732),
            (-3, -1.732),
        ])
        centers2 -= centers2.mean(axis=0)
        configs.append(centers2.flatten())
        
        # Config 3: Random
        centers3 = rng.normal(0, 2, size=(11, 2))
        centers3 -= centers3.mean(axis=0)
        configs.append(centers3.flatten())
        
        return configs
    
    best_val = -np.inf
    best_result = None
    
    configs = _make_initial_configs()
    
    for cfg_idx, x0 in enumerate(configs):
        if time.time() > deadline - 30.0:
            break
        
        # Optimize for several outer hexagon angles
        for angle_deg in [0.0, 30.0, 15.0, 45.0]:
            if time.time() > deadline - 20.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    x0,
                    args=(angle_deg,),
                    method='L-BFGS-B',
                    jac=_objective_grad,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 100:  # Check if result is reasonable
                    centers_opt = res.x.reshape(11, 2)
                    centers_opt -= centers_opt.mean(axis=0)  # Re-center
                    
                    inner = np.column_stack((centers_opt[:, 0], centers_opt[:, 1], np.zeros(11)))
                    
                    # Find best outer hexagon angle
                    best_s = np.inf
                    best_a = 0
                    for a_deg in np.linspace(0, 60, 361):
                        s_val = _outer_side_for_points(centers_opt, a_deg)
                        if s_val < best_s:
                            best_s = s_val
                            best_a = a_deg
                    
                    outer = np.array([0.0, 0.0, best_a])
                    
                    # Check validity
                    if _all_valid(inner, outer, best_s):
                        val = 1.0 / best_s
                        if val > best_val:
                            best_val = val
                            best_result = (inner, outer, best_s)
                            
            except Exception as e:
                pass
    
    if best_result is None:
        # Fallback: return a known valid loose packing
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0
    
    return best_result

# EVOLVE-BLOCK-END
