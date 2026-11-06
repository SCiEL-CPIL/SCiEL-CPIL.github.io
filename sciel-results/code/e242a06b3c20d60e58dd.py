import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def _score(pts):
        """Compute Delta/n^n after scaling to diameter 2."""
        pts = np.asarray(pts, dtype=np.float64)
        n = len(pts)
        # Compute all pairwise distances
        max_d = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_d:
                    max_d = d
        if max_d < 1e-15:
            return 0.0
        # Scale to diameter 2
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        # Product of all ordered pair distances
        log_prod = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 0.0
                log_prod += np.log(d)
        # Delta = product over ordered pairs
        # Score = Delta / n^n = exp(log_prod) / n^n
        score = np.exp(log_prod - n * np.log(n))
        return score

    def _params_to_points(params):
        """
        Convert 8D parameter vector to 8 points with conjugate symmetry.
        params = [r1, a1, r2, a2, r3, a3, r4, a4]
        Points:
          P0 = (r0, 0) with r0 = 1 (fixed, one end of diameter)
          P4 = (-1, 0) (opposite end of diameter, fixed)
          P1 = (r1*cos(a1), r1*sin(a1))
          P7 = (r1*cos(-a1), r1*sin(-a1)) = conjugate of P1
          P2 = (r2*cos(a2), r2*sin(a2))
          P6 = (r2*cos(-a2), r2*sin(-a2)) = conjugate of P2
          P3 = (r3*cos(a3), r3*sin(a3))
          P5 = (r3*cos(-a3), r3*sin(-a3)) = conjugate of P3
        Then we need to ensure diameter is 2.
        Actually, let's fix P0=(1,0) and P4=(-1,0) so diameter is at least 2.
        The other points must be within distance 2 of all others.
        """
        r1, a1, r2, a2, r3, a3, r4, a4 = params
        # Clamp radii to be non-negative
        r1 = max(0.0, r1)
        r2 = max(0.0, r2)
        r3 = max(0.0, r3)
        r4 = max(0.0, r4)
        
        p0 = np.array([1.0, 0.0])
        p4 = np.array([-1.0, 0.0])
        p1 = np.array([r1 * np.cos(a1), r1 * np.sin(a1)])
        p7 = np.array([r1 * np.cos(a1), -r1 * np.sin(a1)])
        p2 = np.array([r2 * np.cos(a2), r2 * np.sin(a2)])
        p6 = np.array([r2 * np.cos(a2), -r2 * np.sin(a2)])
        p3 = np.array([r3 * np.cos(a3), r3 * np.sin(a3)])
        p5 = np.array([r3 * np.cos(a3), -r3 * np.sin(a3)])
        
        return np.array([p0, p1, p2, p3, p4, p5, p6, p7])

    def _neg_score_from_params(params):
        """Negative score for minimization."""
        pts = _params_to_points(params)
        # Check diameter constraint: all pairwise distances <= 2
        for i in range(8):
            for j in range(i + 1, 8):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > 2.0 + 1e-9:
                    # Penalty for violating diameter constraint
                    penalty = (d - 2.0) ** 2 * 1000.0
                    return penalty
        s = _score(pts)
        return -s

    # Regular octagon initial guess
    # Regular octagon with vertices at angles 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # On unit circle: P0=(1,0), P1=(cos(pi/4), sin(pi/4)), P2=(0,1), P3=(-cos(pi/4), sin(pi/4)),
    # P4=(-1,0), P5=(-cos(pi/4), -sin(pi/4)), P6=(0,-1), P7=(cos(pi/4), -sin(pi/4))
    c = np.cos(np.pi / 4)
    s_val = np.sin(np.pi / 4)
    
    # In our parameterization:
    # P0=(1,0), P4=(-1,0) fixed
    # P1=(c, s_val) -> r1=1, a1=pi/4
    # P7=(c, -s_val) -> conjugate of P1
    # P2=(0,1) -> r2=1, a2=pi/2
    # P6=(0,-1) -> conjugate of P2
    # P3=(-c, s_val) -> r3=1, a3=3pi/4
    # P5=(-c, -s_val) -> conjugate of P3
    
    x0 = np.array([1.0, np.pi/4, 1.0, np.pi/2, 1.0, 3*np.pi/4, 0.0, 0.0])
    # Wait, we have 8 params: r1,a1,r2,a2,r3,a3,r4,a4
    # But we only have 3 pairs (P1/P7, P2/P6, P3/P5). What is r4,a4 for?
    # Actually, we have 6 free points (P1,P2,P3,P5,P6,P7) = 3 conjugate pairs.
    # So we need 6 params: r1,a1,r2,a2,r3,a3.
    # Let me redefine.
    
    def _params_to_points_v2(params):
        """
        Convert 6D parameter vector to 8 points with conjugate symmetry.
        params = [r1, a1, r2, a2, r3, a3]
        Points:
          P0 = (1, 0) fixed
          P4 = (-1, 0) fixed
          P1 = (r1*cos(a1), r1*sin(a1))
          P7 = (r1*cos(a1), -r1*sin(a1))
          P2 = (r2*cos(a2), r2*sin(a2))
          P6 = (r2*cos(a2), -r2*sin(a2))
          P3 = (r3*cos(a3), r3*sin(a3))
          P5 = (r3*cos(a3), -r3*sin(a3))
        """
        r1, a1, r2, a2, r3, a3 = params
        r1 = max(0.0, r1)
        r2 = max(0.0, r2)
        r3 = max(0.0, r3)
        
        p0 = np.array([1.0, 0.0])
        p4 = np.array([-1.0, 0.0])
        p1 = np.array([r1 * np.cos(a1), r1 * np.sin(a1)])
        p7 = np.array([r1 * np.cos(a1), -r1 * np.sin(a1)])
        p2 = np.array([r2 * np.cos(a2), r2 * np.sin(a2)])
        p6 = np.array([r2 * np.cos(a2), -r2 * np.sin(a2)])
        p3 = np.array([r3 * np.cos(a3), r3 * np.sin(a3)])
        p5 = np.array([r3 * np.cos(a3), -r3 * np.sin(a3)])
        
        return np.array([p0, p1, p2, p3, p4, p5, p6, p7])

    def _neg_score_v2(params):
        """Negative score for minimization with diameter penalty."""
        pts = _params_to_points_v2(params)
        # Check diameter constraint
        max_penalty = 0.0
        for i in range(8):
            for j in range(i + 1, 8):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > 2.0:
                    max_penalty += (d - 2.0) ** 2
        if max_penalty > 1e-10:
            return max_penalty * 1000.0
        s = _score(pts)
        return -s

    # Initial guess for regular octagon
    c = np.cos(np.pi / 4)
    x0 = np.array([1.0, np.pi/4, 1.0, np.pi/2, 1.0, 3*np.pi/4])
    
    best_params = x0.copy()
    best_score = _score(_params_to_points_v2(x0))
    
    # Run L-BFGS-B optimization
    try:
        res = minimize(_neg_score_v2, x0, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < -best_score:
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass

    if time.time() > deadline - 30:
        pts = _params_to_points_v2(best_params)
        return pts.tolist()

    # Multi-start with perturbations
    np.random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Perturb the regular octagon
        noise = np.random.normal(0, 0.1, size=6)
        x0_pert = x0 + noise
        # Keep radii positive
        x0_pert[0] = max(0.1, x0_pert[0])
        x0_pert[2] = max(0.1, x0_pert[2])
        x0_pert[4] = max(0.1, x0_pert[4])
        
        try:
            res = minimize(_neg_score_v2, x0_pert, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            continue

    # Also try some asymmetric initial guesses
    # Two concentric squares or other configurations
    # Try: points at radius 0.5 and 1.0
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random initial guess within unit disk
        r1 = np.random.uniform(0.1, 1.5)
        a1 = np.random.uniform(0, 2*np.pi)
        r2 = np.random.uniform(0.1, 1.5)
        a2 = np.random.uniform(0, 2*np.pi)
        r3 = np.random.uniform(0.1, 1.5)
        a3 = np.random.uniform(0, 2*np.pi)
        x0_rand = np.array([r1, a1, r2, a2, r3, a3])
        
        try:
            res = minimize(_neg_score_v2, x0_rand, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            continue

    pts = _params_to_points_v2(best_params)
    
    # Ensure distinct points
    for i in range(8):
        for j in range(i+1, 8):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                # Nudge point j slightly
                pts[j] += np.array([1e-6, 1e-6])
    
    # Final scale to diameter 2
    max_d = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_d:
                max_d = d
    if max_d > 0:
        pts = pts * (2.0 / max_d)
    
    return pts.tolist()
