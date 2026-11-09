"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Strategy: Powell's Method (Derivative-Free) for minimizing Phi(x) = 0.5 * ||F(x)||^2.
    
    Rationale:
    1. Avoids Jacobian estimation errors which can destabilize Broyden/Newton on non-smooth or stiff systems.
    2. Powell's method is robust in low dimensions (2D here) and requires only 1 F-call per line-search step.
    3. The directional set adapts to the local geometry of the potential well, often converging to the root
       with fewer total calls than finite-difference methods if the landscape is well-behaved.
    4. Includes multi-start restarts to handle cases where x0 is far from the root or in a different basin.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for the 150s limit

    def time_left():
        return deadline - time.time()

    def norm(vec):
        return np.linalg.norm(vec)

    def phi(x):
        """Scalar objective: 0.5 * ||F(x)||^2"""
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return np.inf
            return 0.5 * np.dot(Fx, Fx)
        except Exception:
            return np.inf

    def is_converged(x, tol=1e-8):
        """Check if ||F(x)|| <= tol"""
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return np.linalg.norm(Fx) <= tol
        except Exception:
            return False

    def line_search(F, x, d, f0, max_iter=50):
        """
        Golden section search or parabolic interpolation for minimizing f(x + t*d).
        Returns (best_t, best_f).
        """
        # Initial bracketing
        t = 1.0
        f1 = phi(x + t * d)
        
        if f1 >= f0:
            # Function increased, try smaller step
            t = 0.5
            for _ in range(10):
                f1 = phi(x + t * d)
                if f1 < f0:
                    break
                t *= 0.5
            else:
                return 0.0, f0
            
            # Now f(t) < f(0), bracket is [0, t*2] potentially
            t_low = 0.0
            t_high = t * 2.0
            f_low = f0
            f_high = phi(x + t_high * d)
        else:
            # Function decreased, expand bracket
            t_low = 0.0
            t_high = t
            f_low = f0
            f_high = f1
            while f_high < f_low and time_left() > 0.1:
                t_high *= 2.0
                f_high = phi(x + t_high * d)
                if t_high > 1e6:
                    break

        if not (np.isfinite(f_low) and np.isfinite(f_high)):
            return 0.0, f0

        # Golden section search within [t_low, t_high]
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        t1 = t_low + (1.0 - gr) * (t_high - t_low)
        t2 = t_low + gr * (t_high - t_low)
        f1 = phi(x + t1 * d)
        f2 = phi(x + t2 * d)

        for _ in range(max_iter):
            if time_left() < 0.1:
                break
            if f1 < f2:
                t_high = t2
                t2 = t1
                f2 = f1
                t1 = t_low + (1.0 - gr) * (t_high - t_low)
                f1 = phi(x + t1 * d)
            else:
                t_low = t1
                t1 = t2
                f1 = f2
                t2 = t_low + gr * (t_high - t_low)
                f2 = phi(x + t2 * d)
            
            if abs(t_high - t_low) < 1e-10 * max(1.0, abs(t1)):
                break

        # Return the best point
        best_t = (t_low + t_high) / 2.0
        best_f = phi(x + best_t * d)
        
        # Check endpoints just in case
        if f_low < best_f:
            best_t = t_low
            best_f = f_low
        if f_high < best_f:
            best_t = t_high
            best_f = f_high

        return best_t, best_f

    def powell_method(F, x0, max_iter=100, tol=1e-12):
        """
        Powell's method for minimizing phi(x).
        Returns (best_x, best_phi, converged).
        """
        x = np.array(x0, dtype=float)
        if not np.all(np.isfinite(x)):
            x = np.array([0.0, 0.0])
        
        f0 = phi(x)
        if not np.isfinite(f0):
            return x, np.inf, False
        
        best_x = x.copy()
        best_f = f0

        # Initial directions: standard basis
        dirs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
            
            x_old = x.copy()
            f_old = f0
            f_min = f0
            dir_min = 0
            
            # Minimize along each direction
            for j in range(len(dirs)):
                t, f_new = line_search(F, x, dirs[j], f_min)
                if f_new < f_min:
                    x = x + t * dirs[j]
                    f_min = f_new
                    dir_min = j
                # Update f0 for next direction? No, Powell uses the current x for each direction
                # But we track the best decrease
                f0 = f_min
            
            # Check convergence
            if abs(f_old - f_min) < tol:
                if f_min < best_f:
                    best_f = f_min
                    best_x = x.copy()
                return best_x, best_f, True

            if f_min < best_f:
                best_f = f_min
                best_x = x.copy()

            # Check if we made significant progress
            if f_min >= f_old:
                break

            # Update direction set
            # Replace the direction with the largest decrease with the new direction
            new_dir = x - x_old
            if norm(new_dir) > 1e-15:
                dirs[dir_min] = new_dir / norm(new_dir)
            
            # Optional: Add a new direction if the set is not full (not applicable for 2D fixed set usually,
            # but standard Powell replaces one direction. For 2D, we have 2 directions.
            # We replace the one that gave the largest improvement with the net displacement direction.)
            
        return best_x, best_f, False

    # Main Logic
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])

    # Phase 1: Powell from x0
    x_best, phi_best, converged = powell_method(F, x0)
    
    if phi_best <= 0.5 * (1e-8)**2: # ||F||^2 <= 1e-16 => ||F|| <= 1e-8
        return x_best

    # Phase 2: Restarts
    # Generate candidates around x0 and zero
    scale = max(1.0, norm(x0))
    
    candidates = []
    candidates.append(np.array([0.0, 0.0]))
    candidates.append(x0 * 0.5)
    candidates.append(x0 * 2.0)
    candidates.append(x0 * 0.1)
    candidates.append(x0 * 5.0)
    
    rng = np.random.default_rng(42)
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0]:
        for _ in range(3):
            u = rng.normal(size=2)
            u_norm = norm(u)
            if u_norm > 1e-10:
                u = u / u_norm
                p = x0 + r * u
                candidates.append(p)

    for c in candidates:
        if time_left() < 1.0:
            break
        
        if not np.all(np.isfinite(c)):
            continue
        
        x_try, phi_try, conv = powell_method(F, c)
        
        if phi_try <= 0.5 * (1e-8)**2:
            return x_try
        
        if phi_try < phi_best:
            x_best = x_try
            phi_best = phi_try
    
    return x_best
# EVOLVE-BLOCK-END
