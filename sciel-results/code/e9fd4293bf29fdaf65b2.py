import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    # Hexagonal spiral: center + 6 neighbors + 4 outer ring (known optimal lattice packing)
    axial_spiral = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_spiral)

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # New candidates
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    # Additional candidates
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    # Additional promising configurations
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # More diverse configurations for tighter packing
    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_29)

    # More diverse compact configurations
    axial_30 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, -1), (3, 1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (0, 3), (-1, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2), (-1, 2),
        (0, 3), (1, 3), (-1, 3)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_40)

    # Additional high-quality configurations
    axial_41 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_43)

    axial_44 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (0, 2)
    ]
    candidates.append(axial_44)

    axial_45 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_45)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.05):
        """
        Smooth surrogate for the max-norm objective using log-sum-exp.
        Returns (S, D, cx, cy) where S = (D + APOTHEM) / APOTHEM.
        """
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        # Shape: (n_points, 6)
        projs = pts_centered @ normals.T
        
        # We need max over all |proj|
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), float('inf'), cx, cy
        
        # Soft max via log-sum-exp for numerical stability
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        
        S = (D + APOTHEM) / APOTHEM
        return S, D, cx, cy

    def check_overlap(pts):
        # Check if any two hexagons overlap.
        # All inner hexagons have the same rotation angle (relative to each other).
        # The Minkowski sum of two unit hexagons with the same orientation is a hexagon
        # of side 2, with apothem sqrt(3).
        # For non-overlap: for all 6 normals, |d . n| >= sqrt(3)
        
        normals = np.array([
            [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return False
        return True

    def optimize_shape(pts_init):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts_init
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        # First, find the best beta for the initial lattice configuration
        # Use finer grid for better initial estimate
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            S, center, D = calc_S_for_beta(pts_init, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with tighter tolerance
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts_init, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 1000, 'xatol': 1e-16})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts_init, best_beta)
        except Exception:
            pass
        
        # Additional refinement: try small perturbations around best_beta
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            for delta_beta in [0.001, -0.001, 0.0001, -0.0001, 0.0005, -0.0005, 0.00001, -0.00001]:
                test_beta = best_beta + delta_beta
                test_beta = ((test_beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                S_test, center_test, _ = calc_S_for_beta(pts_init, test_beta)
                if S_test < best_S:
                    best_S = S_test
                    best_beta = test_beta
                    best_center = center_test
        
        # Local relaxation of positions using smooth surrogate
        # We will optimize the positions of the inner hexagons to minimize S, 
        # subject to non-overlap.
        # We use a penalty method with smooth objective.
        
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def smooth_obj(params):
                # params: [x1, y1, ..., xN, yN, cx, cy, beta]
                n_pts = len(pts_init)
                pts = params[:2*n_pts].reshape(n_pts, 2)
                cx = params[2*n_pts]
                cy = params[2*n_pts+1]
                beta = params[2*n_pts+2]
                
                # Normalize beta
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                # Calculate smooth S
                S_smooth, D, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=0.05)
                if S_smooth == float('inf'):
                    return 1e10
                
                # Check overlap penalty
                penalty = 0.0
                normals = np.array([
                    [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
                    for k in range(6)
                ])
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        d = pts[i] - pts[j]
                        projs = np.abs(d @ normals.T)
                        min_proj = np.min(projs)
                        if min_proj < math.sqrt(3):
                            # Penalty proportional to violation
                            violation = math.sqrt(3) - min_proj
                            penalty += violation * 100.0
                
                return S_smooth + penalty

            # Initial guess: best_pts, best_center, best_beta
            init_params = np.concatenate([pts_init.ravel(), [best_center[0], best_center[1], best_beta]])
            
            # Run Nelder-Mead with smooth objective
            try:
                res = minimize(smooth_obj, init_params, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res.fun < best_S + 0.1:  # Allow some slack for smoothness
                    # Extract best pts
                    n_pts = len(pts_init)
                    candidate_pts = res.x[:2*n_pts].reshape(n_pts, 2)
                    candidate_center = (res.x[2*n_pts], res.x[2*n_pts+1])
                    candidate_beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    # Verify and recalculate exact S
                    S_exact, c_exact, _ = calc_S_for_beta(candidate_pts, candidate_beta)
                    if S_exact < best_S and check_overlap(candidate_pts):
                        best_S = S_exact
                        best_pts = candidate_pts
                        best_center = c_exact
                        best_beta = candidate_beta
            except Exception:
                pass
            
            # Final high-precision polish: use exact objective with L-BFGS-B
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def exact_obj(params):
                    n_pts = len(pts_init)
                    pts = params[:2*n_pts].reshape(n_pts, 2)
                    cx = params[2*n_pts]
                    cy = params[2*n_pts+1]
                    beta = params[2*n_pts+2]
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    S, c, D = calc_S_for_beta(pts, beta)
                    if S == float('inf'):
                        return 1e10
                    if not check_overlap(pts):
                        return 1e10
                    return S
                
                init_params3 = np.concatenate([best_pts.ravel(), [best_center[0], best_center[1], best_beta]])
                
                try:
                    res3 = minimize(exact_obj, init_params3, method='L-BFGS-B',
                                   options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-18})
                    
                    if res3.fun < best_S:
                        n_pts = len(pts_init)
                        candidate_pts3 = res3.x[:2*n_pts].reshape(n_pts, 2)
                        candidate_center3 = (res3.x[2*n_pts], res3.x[2*n_pts+1])
                        candidate_beta3 = ((res3.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        
                        S_exact3, c_exact3, _ = calc_S_for_beta(candidate_pts3, candidate_beta3)
                        if S_exact3 < best_S and check_overlap(candidate_pts3):
                            best_S = S_exact3
                            best_pts = candidate_pts3
                            best_center = c_exact3
                            best_beta = candidate_beta3
                except Exception:
                    pass
            
            # Ultra-high-precision final polish: use smooth surrogate with very tight temperature
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def ultra_smooth_obj(params):
                    n_pts = len(pts_init)
                    pts = params[:2*n_pts].reshape(n_pts, 2)
                    cx = params[2*n_pts]
                    cy = params[2*n_pts+1]
                    beta = params[2*n_pts+2]
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    S_smooth, D, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=0.0001)
                    if S_smooth == float('inf'):
                        return 1e10
                    
                    # Very strong overlap penalty
                    penalty = 0.0
                    normals = np.array([
                        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
                        for k in range(6)
                    ])
                    for i in range(n_pts):
                        for j in range(i+1, n_pts):
                            d = pts[i] - pts[j]
                            projs = np.abs(d @ normals.T)
                            min_proj = np.min(projs)
                            if min_proj < math.sqrt(3):
                                violation = math.sqrt(3) - min_proj
                                penalty += violation * 10000.0
                    
                    return S_smooth + penalty

                init_params4 = np.concatenate([best_pts.ravel(), [best_center[0], best_center[1], best_beta]])
                
                try:
                    res4 = minimize(ultra_smooth_obj, init_params4, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    
                    if res4.fun < best_S + 0.001:
                        n_pts = len(pts_init)
                        candidate_pts4 = res4.x[:2*n_pts].reshape(n_pts, 2)
                        candidate_center4 = (res4.x[2*n_pts], res4.x[2*n_pts+1])
                        candidate_beta4 = ((res4.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        
                        S_exact4, c_exact4, _ = calc_S_for_beta(candidate_pts4, candidate_beta4)
                        if S_exact4 < best_S and check_overlap(candidate_pts4):
                            best_S = S_exact4
                            best_pts = candidate_pts4
                            best_center = c_exact4
                            best_beta = candidate_beta4
                except Exception:
                    pass
            
            # Second pass: refine with tighter temperature and more iterations
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def smooth_obj_tight(params):
                    n_pts = len(pts_init)
                    pts = params[:2*n_pts].reshape(n_pts, 2)
                    cx = params[2*n_pts]
                    cy = params[2*n_pts+1]
                    beta = params[2*n_pts+2]
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    S_smooth, D, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=0.005)
                    if S_smooth == float('inf'):
                        return 1e10
                    
                    penalty = 0.0
                    normals = np.array([
                        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
                        for k in range(6)
                    ])
                    for i in range(n_pts):
                        for j in range(i+1, n_pts):
                            d = pts[i] - pts[j]
                            projs = np.abs(d @ normals.T)
                            min_proj = np.min(projs)
                            if min_proj < math.sqrt(3):
                                violation = math.sqrt(3) - min_proj
                                penalty += violation * 500.0
                    
                    return S_smooth + penalty

                # Start from current best
                init_params2 = np.concatenate([best_pts.ravel(), [best_center[0], best_center[1], best_beta]])
                
                try:
                    res2 = minimize(smooth_obj_tight, init_params2, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                    
                    if res2.fun < best_S + 0.01:
                        n_pts = len(pts_init)
                        candidate_pts2 = res2.x[:2*n_pts].reshape(n_pts, 2)
                        candidate_center2 = (res2.x[2*n_pts], res2.x[2*n_pts+1])
                        candidate_beta2 = ((res2.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        
                        S_exact2, c_exact2, _ = calc_S_for_beta(candidate_pts2, candidate_beta2)
                        if S_exact2 < best_S and check_overlap(candidate_pts2):
                            best_S = S_exact2
                            best_pts = candidate_pts2
                            best_center = c_exact2
                            best_beta = candidate_beta2
                except Exception:
                    pass
        
        return best_S, best_beta, best_center, best_pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by a quick heuristic: try compact ones first
    # Compute a quick bounding box size for each candidate
    def quick_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        # Quick estimate: max projection in 6 directions
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM

    # Sort by quick score to find good solutions early
    scored_candidates = [(quick_score(c), idx, c) for idx, c in enumerate(candidates)]
    scored_candidates.sort(key=lambda x: x[0])
    
    # Try top candidates first, then fill in with remaining
    sorted_candidates = [c for _, _, c in scored_candidates[:20]]
    remaining = [c for _, _, c in scored_candidates[20:]]
    all_sorted = sorted_candidates + remaining

    for axial in all_sorted:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center, optimized_pts = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = optimized_pts
            best_global_beta = beta
            best_global_center = center
        
        # If we've found a very good solution, stop early to save time
        if best_global_S < 3.5:
            break

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    # Final high-precision polish on the best solution found
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_global_center is not None:
        def final_smooth_obj(params):
            n_pts = len(best_global_pts)
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            S_smooth, D, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=0.00001)
            if S_smooth == float('inf'):
                return 1e10
            
            # Very strong overlap penalty
            penalty = 0.0
            normals = np.array([
                [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
                for k in range(6)
            ])
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    d = pts[i] - pts[j]
                    projs = np.abs(d @ normals.T)
                    min_proj = np.min(projs)
                    if min_proj < math.sqrt(3):
                        violation = math.sqrt(3) - min_proj
                        penalty += violation * 100000.0
            
            return S_smooth + penalty

        init_final = np.concatenate([best_global_pts.ravel(), [best_global_center[0], best_global_center[1], best_global_beta]])
        
        try:
            res_final = minimize(final_smooth_obj, init_final, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-16})
            
            if res_final.fun < best_global_S + 0.0001:
                n_pts = len(best_global_pts)
                candidate_pts_final = res_final.x[:2*n_pts].reshape(n_pts, 2)
                candidate_center_final = (res_final.x[2*n_pts], res_final.x[2*n_pts+1])
                candidate_beta_final = ((res_final.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                S_exact_final, c_exact_final, _ = calc_S_for_beta(candidate_pts_final, candidate_beta_final)
                if S_exact_final < best_global_S and check_overlap(candidate_pts_final):
                    best_global_S = S_exact_final
                    best_global_pts = candidate_pts_final
                    best_global_center = c_exact_final
                    best_global_beta = candidate_beta_final
        except Exception:
            pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Use a very small safety margin to ensure no overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
