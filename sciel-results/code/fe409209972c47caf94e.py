import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    """
    Finds 8 points maximizing the product of pairwise distances normalized by diameter.
    Uses a parameterization based on 4 conjugate pairs of complex numbers.
    """
    
    def get_points(params):
        """
        params: array of 8 floats [r1, th1, r2, th2, r3, th3, r4, th4]
        Returns: array of 8 complex numbers
        """
        points = []
        for i in range(4):
            r = params[2*i]
            th = params[2*i+1]
            z = r * np.exp(1j * th)
            points.append(z)
            points.append(np.conjugate(z))
        return np.array(points)

    def log_objective(params):
        """
        Computes -log(J(P)) where J(P) = prod(d_ij) / D^56
        We minimize this.
        """
        pts = get_points(params)
        
        # Calculate all pairwise distances
        # Distance matrix
        # d_ij = |z_i - z_j|
        # We need sum over i != j of log(d_ij)
        
        # Efficient calculation:
        # diff = pts[:, None] - pts[None, :]
        # dists = np.abs(diff)
        # mask = ~np.eye(8, dtype=bool)
        # log_sum = np.sum(np.log(dists[mask] + 1e-12))
        
        # Diameter D = max(dists)
        
        # To avoid log(0) if points coincide, add epsilon.
        # Also, if points coincide, the score is 0 (log -inf), so -log is +inf.
        
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.abs(diff)
        
        # Exclude self-distances
        mask = ~np.eye(8, dtype=bool)
        valid_dists = dists[mask]
        
        if np.any(valid_dists < 1e-12):
            return 1e10 # Penalty for coincident points
        
        log_prod = np.sum(np.log(valid_dists))
        
        D = np.max(dists)
        
        if D < 1e-12:
            return 1e10
            
        log_J = log_prod - 56.0 * np.log(D)
        
        return -log_J

    # Initial guesses
    # 1. Regular Octagon
    # Radius 1, Angles 0, pi/4, pi/2, 3pi/4
    # Note: The order doesn't matter for the set, but for the param space it does.
    # Our param space is [r1, th1, r2, th2, ...]
    # Conjugates are handled automatically.
    # For regular octagon, points are at angles 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4.
    # Conjugate pairs:
    # 0 and pi (if r=1, z=1, -1 are conjugates? No. 1 and 1. -1 and -1.)
    # Actually, $e^{i 0} = 1$, $\bar{1}=1$. $e^{i \pi} = -1$, $\bar{-1}=-1$.
    # $e^{i \pi/4}$ and $e^{-i \pi/4} = e^{i 7\pi/4}$.
    # $e^{i \pi/2} = i$, $\bar{i} = -i = e^{i 3\pi/2}$.
    
    # So pairs are:
    # 1: r=1, th=0  -> {1, 1} (Wait, if th=0, z=1, conj=1. This gives duplicate points!)
    # We need distinct points.
    # If th=0, z=r, conj=r. Duplicate.
    # If th=pi, z=-r, conj=-r. Duplicate.
    # If th=pi/2, z=ir, conj=-ir. Distinct.
    
    # The regular octagon has points at:
    # 1, -1, i, -i, (1+i)/sqrt(2), (1-i)/sqrt(2), (-1+i)/sqrt(2), (-1-i)/sqrt(2).
    
    # Let's map these to our 4 pairs:
    # Pair 1: 1 and 1? No, we need distinct points.
    # The set {1, -1} is real.
    # The set {i, -i} is imaginary.
    # The set {(1+i)/sqrt(2), (1-i)/sqrt(2)} are conjugates.
    # The set {(-1+i)/sqrt(2), (-1-i)/sqrt(2)} are conjugates.
    
    # So:
    # Pair 1: r=1, th=0 -> {1, 1} -> BAD.
    # We must ensure that the generated points are distinct.
    # If th is 0 or pi, the point is real, so z = conj(z).
    # To avoid this, we can perturb or choose a different parameterization.
    # Or, we can just accept that the optimizer might hit this boundary and handle it.
    # But starting at the boundary is bad.
    
    # Let's start slightly off the real/imag axes or use a different initial configuration.
    # Or, we can define the pairs as (r, th) where th is in (0, pi).
    # If th is close to 0, z and conj(z) are close.
    
    # Let's use a "distorted octagon" start.
    # Start 1: Regular octagon, but slightly perturbed to avoid duplicates.
    # Actually, we can just use 4 complex numbers with non-zero imaginary parts initially?
    # No, the regular octagon *has* real points.
    
    # Alternative: Optimize over 8 real coordinates (x,y) directly?
    # 16 dimensions. More robust to symmetry breaking.
    # The prompt suggests the polynomial approach (8 dims), but if it fails due to symmetry constraints, 16 dims is safer.
    # However, 8 dims is much faster.
    
    # Let's try to make the 8-dim approach work by handling the duplicates.
    # If we start with:
    # r1=1, th1=pi/8 (45 deg? No, 22.5 deg)
    # The regular octagon vertices are at k * pi/4.
    # k=0: 0 deg.
    # k=1: 45 deg.
    # k=2: 90 deg.
    # k=3: 135 deg.
    # k=4: 180 deg.
    # k=5: 225 deg.
    # k=6: 270 deg.
    # k=7: 315 deg.
    
    # Conjugate pairs:
    # 0 and 0? No.
    # 45 and 315 (-45). Pair 1: r=1, th=pi/4. Points: e^{i pi/4}, e^{-i pi/4}.
    # 90 and 270 (-90). Pair 2: r=1, th=pi/2. Points: i, -i.
    # 135 and 225 (-135). Pair 3: r=1, th=3pi/4. Points: e^{i 3pi/4}, e^{-i 3pi/4}.
    # 0 and 180. These are real.
    # Pair 4: r=1, th=0. Points: 1, 1. DUPLICATE.
    # Pair 5: r=1, th=pi. Points: -1, -1. DUPLICATE.
    
    # So the regular octagon is NOT representable as 4 distinct conjugate pairs in the form $z, \bar{z}$ with $z \neq \bar{z}$.
    # It requires 2 real points and 3 conjugate pairs.
    # Total 2 + 3*2 = 8 points.
    # My parameterization forces 4 pairs of $z, \bar{z}$.
    # If $z$ is real, $z=\bar{z}$, so we get duplicates.
    
    # Fix: Allow the parameterization to handle real points by treating them as a single point with multiplicity? No, we need distinct points.
    # We can modify the parameterization:
    # 2 real points: $x_1, x_2$.
    # 3 conjugate pairs: $r_k e^{i \theta_k}$.
    # Total params: 2 (for x1, x2) + 3*2 (for r, th) = 8 params.
    # This covers the regular octagon perfectly.
    
    def get_points_v2(params):
        # params: [x1, x2, r1, th1, r2, th2, r3, th3]
        pts = []
        pts.append(complex(params[0], 0))
        pts.append(complex(params[1], 0))
        for i in range(3):
            r = params[2 + 2*i]
            th = params[3 + 2*i]
            z = r * np.exp(1j * th)
            pts.append(z)
            pts.append(np.conjugate(z))
        return np.array(pts)

    def log_objective_v2(params):
        pts = get_points_v2(params)
        
        # Check for duplicates
        # If any two points are too close, return high penalty
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.abs(diff)
        mask = ~np.eye(8, dtype=bool)
        valid_dists = dists[mask]
        
        if np.min(valid_dists) < 1e-10:
            return 1e10
            
        log_prod = np.sum(np.log(valid_dists))
        D = np.max(dists)
        
        if D < 1e-10:
            return 1e10
            
        log_J = log_prod - 56.0 * np.log(D)
        return -log_J

    # Initial guess for Regular Octagon:
    # x1 = 1, x2 = -1
    # Pair 1 (45 deg): r=1, th=pi/4
    # Pair 2 (90 deg): r=1, th=pi/2
    # Pair 3 (135 deg): r=1, th=3*pi/4
    init_reg = [1.0, -1.0, 1.0, np.pi/4, 1.0, np.pi/2, 1.0, 3.0*np.pi/4]

    # Run optimization
    best_score = -1e10
    best_params = init_reg
    
    # Multi-start
    starts = [init_reg]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(10):
        pert = rng.normal(0, 0.1, size=8)
        starts.append(init_reg + pert)
        
    # Add a "two squares" guess
    # Square 1: 1, -1, i, -i (r=1, th=0, pi, pi/2, 3pi/2) -> 2 real, 2 imag
    # Square 2: rotated 45 deg?
    # Let's just rely on the random starts and the regular one.

    for start in starts:
        try:
            res = minimize(log_objective_v2, start, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Get the points
    pts = get_points_v2(best_params)
    
    # Scale to diameter 2
    # Calculate current diameter
    diff = pts[:, np.newaxis] - pts[np.newaxis, :]
    dists = np.abs(diff)
    D = np.max(dists)
    
    if D > 1e-12:
        scale_factor = 2.0 / D
        pts = pts * scale_factor
        
    # Convert to list of [x, y]
    result = [[float(p.real), float(p.imag)] for p in pts]
    
    return result

if __name__ == "__main__":
    construct()
