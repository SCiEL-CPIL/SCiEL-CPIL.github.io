"""Root finder for 3 fixed 2D systems using Adaptive Broyden's Method with Line Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Deadline guard: 150s limit, use 140s for safety
    start_time = time.time()
    time_limit = 140.0
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
        
    # Initial evaluation
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            # If initial eval fails or is non-finite, try a small perturbation
            x = x + np.array([1e-3, 1e-3])
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return x # Best effort, likely to fail but avoids crash
    except Exception:
        return x

    norm_F = np.linalg.norm(Fx)
    if norm_F <= 1e-8:
        return x

    # Initialize Jacobian approximation using finite differences
    # Central differences are more accurate but cost 4 calls. 
    # Forward differences cost 2 calls. Let's use forward for lower cost.
    h = 1e-5
    J = np.zeros((2, 2))
    J[:, 0] = (F(x + np.array([h, 0])) - Fx) / h
    J[:, 1] = (F(x + np.array([0, h])) - Fx) / h
    
    # Check if J is singular or non-finite
    if not np.all(np.isfinite(J)):
        # Fallback to identity if Jacobian computation fails
        J = np.eye(2)

    # Broyden's Method with Line Search
    max_iter = 50
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, perturb J slightly or use pseudo-inverse
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(s)):
            break
            
        # Line Search
        alpha = 1.0
        x_new = x + alpha * s
        F_new = None
        norm_F_new = float('inf')
        
        # Backtracking line search
        for _ in range(20):
            try:
                F_new = F(x_new)
                F_new = np.asarray(F_new, dtype=float)
                if F_new.shape != (2,):
                    F_new = F_new.reshape(2)
                if np.all(np.isfinite(F_new)):
                    norm_F_new = np.linalg.norm(F_new)
                    if norm_F_new <= 1e-12:
                        x = x_new
                        return x
                    # Check for sufficient decrease (Armijo-like condition simplified)
                    # We just want reduction in norm for robustness
                    if norm_F_new < norm_F:
                        break
            except Exception:
                pass
            
            # Reduce step size
            alpha *= 0.5
            x_new = x + alpha * s
            
        if not np.all(np.isfinite(x_new)) or norm_F_new >= norm_F:
            # If line search failed to find a better point, try a smaller fixed step
            # or break if we can't improve
            # Try one more time with a very small step
            alpha = 1e-3
            x_new = x + alpha * s
            try:
                F_new = F(x_new)
                F_new = np.asarray(F_new, dtype=float)
                if F_new.shape != (2,):
                    F_new = F_new.reshape(2)
                if np.all(np.isfinite(F_new)):
                    norm_F_new = np.linalg.norm(F_new)
                    if norm_F_new < norm_F:
                        pass # Accept
                    else:
                        break # No improvement possible with this direction
                else:
                    break
            except Exception:
                break

        # Update x and F
        x = x_new
        Fx = F_new
        norm_F = norm_F_new
        
        if norm_F <= 1e-8:
            return x
            
        # Update Jacobian using Broyden's formula
        # J_new = J + (y - J*s) * s^T / (s^T * s)
        # where y = F(x_new) - F(x_old)
        y = F_new - (Fx - F_new) # Wait, Fx is old F. y = F_new - F_old
        # In the loop above, we updated Fx to F_new at the end? 
        # Let's trace: 
        # Start of loop: Fx is current.
        # End of loop: x = x_new, Fx = F_new.
        # So y = F_new - F_old. But we overwrote Fx.
        # Let's save F_old before update.
        
        # Correction: The code above updates x and Fx at the end.
        # We need F_old for the Broyden update.
        # Let's restructure slightly to ensure we have F_old.
        
        # Actually, the loop logic above is slightly messy regarding F_old.
        # Let's restart the loop logic in the code below to be cleaner.
        
        break # Break out of the messy loop above, the code below is the real implementation
        
    # The code above had a flaw in tracking F_old for Broyden update.
    # Let's implement a clean version.
    
    return x # Fallback

# The above draft was messy. Here is the clean implementation.

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 140.0
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
        
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            x = x + np.array([1e-3, 1e-3])
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return x
    except Exception:
        return x

    norm_F = np.linalg.norm(Fx)
    if norm_F <= 1e-8:
        return x

    # Initialize Jacobian
    h = 1e-5
    J = np.zeros((2, 2))
    try:
        J[:, 0] = (F(x + np.array([h, 0])) - Fx) / h
        J[:, 1] = (F(x + np.array([0, h])) - Fx) / h
        if not np.all(np.isfinite(J)):
            J = np.eye(2)
    except Exception:
        J = np.eye(2)

    for i in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(s)):
            break
            
        # Line Search
        alpha = 1.0
        x_new = None
        F_new = None
        norm_F_new = float('inf')
        
        for _ in range(30):
            x_try = x + alpha * s
            try:
                F_try = F(x_try)
                F_try = np.asarray(F_try, dtype=float)
                if F_try.shape != (2,):
                    F_try = F_try.reshape(2)
                if np.all(np.isfinite(F_try)):
                    norm_try = np.linalg.norm(F_try)
                    if norm_try < norm_F:
                        x_new = x_try
                        F_new = F_try
                        norm_F_new = norm_try
                        break
            except Exception:
                pass
            
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        if x_new is None:
            # No improvement found
            break

        # Update state
        F_old = Fx
        x = x_new
        Fx = F_new
        norm_F = norm_F_new
        
        if norm_F <= 1e-8:
            return x
            
        # Broyden Update
        y = F_new - F_old
        s_dot_s = np.dot(s, s)
        if s_dot_s > 1e-12:
            J = J + np.outer(y - J @ s, s) / s_dot_s
            
    return x
# EVOLVE-BLOCK-END
