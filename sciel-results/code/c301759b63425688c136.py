import time
import numpy as np

_DEADLINE = time.time() + 148.0

def _time_left():
    return _DEADLINE - time.time()

def _clean(v):
    try:
        a = np.asarray(v, dtype=float).reshape(2)
    except Exception:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a

class _Budget(object):
    __slots__ = ("F", "limit", "n", "best_x", "best_f")

    def __init__(self, F, limit):
        self.F = F
        self.limit = limit
        self.n = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, x):
        if self.n >= self.limit:
            return None
        self.n += 1
        try:
            v = self.F(x)
        except Exception:
            return None
        v = _clean(v)
        if v is None:
            return None
        f = float(np.linalg.norm(v))
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return v

    def best(self):
        return self.best_x

def _rk4_step(g, x, t, dt):
    """
    Perform one RK4 step for the homotopy ODE.
    H(x, t) = (1-t)F(x) + t*x = 0.
    Differentiating w.r.t t:
    (1-t) J_F dx/dt + (-1)F(x) + t*I dx/dt + x = 0
    [(1-t)J_F + t*I] dx/dt = F(x) - x
    
    We approximate J_F with a finite difference or a stored estimate.
    Here we use a simple finite difference for J_F at the current point, 
    but this is expensive. 
    Instead, let's use the secant-like update from the previous step if available, 
    or just use Identity as a rough guide if we don't have a good J.
    
    Actually, for a "different strategy" that is robust, let's use the 
    "pseudo-transient continuation" idea where we solve the ODE:
    dx/dt = -H(x, t) 
    This is a gradient flow of the homotopy residual, which is simpler but slower.
    
    Let's stick to the ODE: [(1-t)J_F + t*I] dx/dt = F(x) - x.
    To avoid expensive FD Jacobians, we will maintain a Broyden approximation J.
    """
    # This function is a placeholder for the structure, but the actual implementation 
    # will be inline in the main loop to manage state (J) efficiently.
    pass

def _homotopy_rk4(g, x0, max_steps=200):
    """
    Homotopy continuation using adaptive RK4-like steps.
    Starts at t=1, x=0 (root of x=0).
    Ends at t=0, x=x* (root of F(x)=0).
    
    We use a Broyden approximation for J_F to solve the ODE system.
    """
    x = np.zeros(2)
    t = 1.0
    
    # Initial evaluation
    F0 = g(x)
    if F0 is None:
        return x, False, np.inf
    
    f_norm = float(np.linalg.norm(F0))
    if f_norm <= 1e-8:
        return x, True, f_norm
        
    # Initialize Jacobian approximation for F
    # Use a finite difference for the first step to get a better J
    eps = 1e-6
    J = np.zeros((2, 2))
    for i in range(2):
        x_pert = x.copy()
        x_pert[i] += eps
        F_pert = g(x_pert)
        if F_pert is None:
            J = np.eye(2)
            break
        J[:, i] = (F_pert - F0) / eps
    else:
        # Check if J is singular
        if np.linalg.det(J) < 1e-10:
            J = np.eye(2)
    
    # If J is bad, fallback to Identity
    if not np.all(np.isfinite(J)):
        J = np.eye(2)
        
    x_prev = x.copy()
    F_prev = F0.copy()
    
    dt = 0.1
    min_dt = 1e-4
    
    for step in range(max_steps):
        if g.n >= g.limit or _time_left() < 2.0:
            break
            
        t_new = max(0.0, t - dt)
        
        # Define the ODE RHS: dx/dt = J_H^{-1} (F(x) - x)
        # J_H = (1-t)J + t*I
        
        def rhs(x_curr, t_curr):
            J_H = (1.0 - t_curr) * J + t_curr * np.eye(2)
            F_curr = g(x_curr)
            if F_curr is None:
                return None
            try:
                return np.linalg.solve(J_H, F_curr - x_curr)
            except np.linalg.LinAlgError:
                return F_curr - x_curr # Fallback
        
        # RK4 Step
        k1 = rhs(x, t)
        if k1 is None:
            dt = max(dt * 0.5, min_dt)
            continue
            
        k2 = rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        if k2 is None:
            dt = max(dt * 0.5, min_dt)
            continue
            
        k3 = rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        if k3 is None:
            dt = max(dt * 0.5, min_dt)
            continue
            
        k4 = rhs(x + dt * k3, t + dt)
        if k4 is None:
            dt = max(dt * 0.5, min_dt)
            continue
            
        x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Check residual at new point
        F_new = g(x_new)
        if F_new is None:
            dt = max(dt * 0.5, min_dt)
            continue
            
        f_new = float(np.linalg.norm(F_new))
        
        # Update Broyden J
        s = x_new - x
        y = F_new - F_prev
        denom = np.dot(s, s)
        if denom > 1e-15:
            J = J + np.outer(y - J @ s, s) / denom
            
        # Accept step if residual decreased or if we are close to t=0
        if f_new < f_norm * 1.1 or t_new < 1e-6:
            x = x_new
            F_prev = F_new
            f_norm = f_new
            t = t_new
            dt = min(dt * 1.5, 0.5) # Increase step
            
            if f_norm <= 1e-8:
                return x, True, f_norm
                
            if t <= 1e-6:
                return x, False, f_norm
        else:
            dt = max(dt * 0.5, min_dt)
            
    return x, False, f_norm

def _broyden_local(g, x0, max_it=50):
    """Standard Broyden polish."""
    x = np.array(x0, dtype=float)
    Fx = g(x)
    if Fx is None:
        return x, False, np.inf
        
    f = float(np.linalg.norm(Fx))
    if f <= 1e-8:
        return x, True, f
        
    # Initialize J with FD
    eps = 1e-6
    J = np.zeros((2, 2))
    for i in range(2):
        x_pert = x.copy()
        x_pert[i] += eps
        F_pert = g(x_pert)
        if F_pert is None:
            J = np.eye(2)
            break
        J[:, i] = (F_pert - Fx) / eps
    else:
        if np.linalg.det(J) < 1e-10:
            J = np.eye(2)
            
    if not np.all(np.isfinite(J)):
        J = np.eye(2)
        
    for _ in range(max_it):
        if g.n >= g.limit or _time_left() < 1.0:
            break
            
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx
            
        if not np.all(np.isfinite(dx)):
            break
            
        alpha = 1.0
        success = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = g(x_new)
            if Fx_new is None:
                break
            f_new = float(np.linalg.norm(Fx_new))
            
            if f_new <= 1e-8:
                return x_new, True, f_new
                
            if f_new < f:
                s = alpha * dx
                y = Fx_new - Fx
                denom = np.dot(s, s)
                if denom > 1e-15:
                    J = J + np.outer(y - J @ s, s) / denom
                x = x_new
                Fx = Fx_new
                f = f_new
                success = True
                break
            alpha *= 0.5
            
        if not success:
            break
            
    return x, False, f

def find_root_system(F, x0):
    x0 = _clean(x0)
    if x0 is None:
        x0 = np.zeros(2)
        
    limit = 480
    g = _Budget(F, limit)
    
    # 1. Homotopy RK4
    x_h, success, f_h = _homotopy_rk4(g, x0, max_steps=100)
    
    if success:
        return x_h
        
    if f_h < g.best_f:
        g.best_f = f_h
        g.best_x = x_h.copy()
        
    # 2. Polish if close
    if g.best_x is not None and g.best_f < 1e-3:
        x_res, success, f_res = _broyden_local(g, g.best_x, max_it=30)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    # 3. Fallback: Broyden from x0
    if g.best_f > 1e-3:
        x_res, success, f_res = _broyden_local(g, x0, max_it=50)
        if success:
            return x_res
        if f_res < g.best_f:
            g.best_f = f_res
            g.best_x = x_res.copy()
            
    # 4. Multi-start
    if g.best_f > 1e-3 and g.n < limit - 50:
        rng = np.random.default_rng(42)
        scale = max(1.0, np.linalg.norm(x0))
        for _ in range(5):
            if g.n >= limit - 20 or _time_left() < 5.0:
                break
            x_rand = x0 + rng.uniform(-5, 5, size=2) * scale
            x_res, success, f_res = _broyden_local(g, x_rand, max_it=20)
            if success:
                return x_res
            if f_res < g.best_f:
                g.best_f = f_res
                g.best_x = x_res.copy()
                
    if g.best_x is not None:
        return g.best_x
    return x0.copy()
