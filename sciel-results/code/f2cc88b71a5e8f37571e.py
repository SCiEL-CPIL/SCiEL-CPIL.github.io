<<<<<<< SEARCH
def _polish(pts, deadline):
    """Steepest descent polish with backtracking line search and adaptive learning rate."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    lr = 0.05
    min_lr = 1e-16
    max_lr = 1.0
    patience = 0
    
    while time.time() < deadline:
        e, g = _log_energy_grad_fast(pts)
        if e < -1e17: break
        
        # Project gradient to tangent space
        g_t = _tangent_project(pts, g)
        
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-14: break
        
        # Backtracking line search with Armijo condition
        alpha = lr
        found = False
        for _ in range(60):
            new_pts = pts - alpha * g_t
            new_pts = _project_to_sphere(new_pts)
            
            new_e = _log_energy_fast(new_pts)
            
            # Armijo condition: f(x+alpha*p) <= f(x) + c*alpha*grad^T*p
            c = 1e-4
            if new_e < current_energy + c * alpha * np.sum(g_t * g_t):
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            patience = 0
            lr = min(alpha * 1.5, max_lr)
        else:
            lr = max(alpha * 0.5, min_lr)
            patience += 1
            if patience > 20 or lr < min_lr:
                break
                
    return pts
=======
def _polish(pts, deadline):
    """Steepest descent polish with backtracking line search and adaptive learning rate."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    # Start with a larger lr to handle initial random starts better
    lr = 0.1 
    min_lr = 1e-16
    max_lr = 5.0
    patience = 0
    max_patience = 100 # Increased patience for primary optimization
    
    while time.time() < deadline:
        e, g = _log_energy_grad_fast(pts)
        if e < -1e17: break
        
        # Project gradient to tangent space
        g_t = _tangent_project(pts, g)
        
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-14: break
        
        # Backtracking line search with Armijo condition
        alpha = lr
        found = False
        for _ in range(80): # Increased line search steps
            new_pts = pts - alpha * g_t
            new_pts = _project_to_sphere(new_pts)
            
            new_e = _log_energy_fast(new_pts)
            
            # Armijo condition: f(x+alpha*p) <= f(x) + c*alpha*grad^T*p
            c = 1e-4
            if new_e < current_energy + c * alpha * np.sum(g_t * g_t):
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            patience = 0
            lr = min(alpha * 1.5, max_lr)
        else:
            lr = max(alpha * 0.5, min_lr)
            patience += 1
            if patience > max_patience or lr < min_lr:
                break
                
    return pts
>>>>>>> REPLACE
