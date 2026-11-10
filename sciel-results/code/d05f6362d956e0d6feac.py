import numpy as np
import scipy.optimize as opt
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coords for optimization
    # theta = arccos(z), phi = arctan2(y, x)
    theta_init = np.arccos(np.clip(x_init[:, 2], -1.0, 1.0))
    phi_init = np.arctan2(x_init[:, 1], x_init[:, 0])
    
    # Ensure phi is in [0, 2pi)
    phi_init = np.mod(phi_init, 2 * np.pi)
    
    # Flatten for optimizer
    x0 = np.concatenate([theta_init, phi_init])
    
    # If no time, return initial
    if time.time() > deadline - 5.0:
         return x_init
         
    # Define objective and gradient
    def energy_and_grad(x):
        theta = x[:N]
        phi = x[N:]
        
        # Convert to Cartesian
        st = np.sin(theta)
        ct = np.cos(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        
        X = st * cp
        Y = st * sp
        Z = ct
        pts = np.column_stack([X, Y, Z])
        
        # Compute pairwise distances
        # pts shape (N, 3)
        # diff shape (N, N, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero (though points should be distinct)
        dist_sq = np.maximum(dist_sq, 1e-15)
        dist = np.sqrt(dist_sq)
        
        # Energy: sum_{i<j} log(1/dist_ij)
        # Create mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = np.sum(np.log(1.0 / dist[mask]))
        
        # Gradient
        # grad_cart[i] = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # Note: diff[i,j] = x_i - x_j
        # Term is diff[i,j] / dist_sq[i,j]
        # We need to sum over j for each i.
        
        # Be careful with i=j, dist_sq is 0. We masked it or set to 1e-15.
        # The term for i=j is 0 in the sum because diff is 0.
        # So we can just sum over all j != i.
        
        # grad_term[i,j] = diff[i,j] / dist_sq[i,j]
        # We want sum_j grad_term[i,j]
        
        # To avoid 0/0 issues, we can set the diagonal of dist_sq to 1.
        # And the diagonal of diff is 0, so 0/1 = 0.
        
        # Let's compute grad_cart
        # diff shape (N, N, 3)
        # dist_sq shape (N, N)
        
        # We divide diff by dist_sq[..., np.newaxis]
        # But we must handle the diagonal.
        # Since diff[i,i] = 0, the result will be 0 regardless of the denominator, 
        # as long as it's not NaN/Inf. 0 / 1e-15 = 0.
        
        inv_dist_sq = 1.0 / dist_sq
        grad_terms = diff * inv_dist_sq[..., np.newaxis]
        
        grad_cart = np.sum(grad_terms, axis=1) # shape (N, 3)
        
        # Project to spherical coords
        # dE/dtheta_i = grad_cart[i] . d(x_i)/dtheta_i
        # d(x_i)/dtheta_i = (cos(theta_i)cos(phi_i), cos(theta_i)sin(phi_i), -sin(theta_i))
        
        dtheta = np.column_stack([ct * cp, ct * sp, -st]) # shape (N, 3)
        dphi = np.column_stack([-st * sp, st * cp, np.zeros(N)]) # shape (N, 3)
        
        grad_theta = np.sum(grad_cart * dtheta, axis=1)
        grad_phi = np.sum(grad_cart * dphi, axis=1)
        
        grad = np.concatenate([grad_theta, grad_phi])
        
        return energy, grad

    # Run optimization
    try:
        res = opt.minimize(
            energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        if res.success and time.time() < deadline:
             theta_opt = res.x[:N]
             phi_opt = res.x[N:]
             
             st = np.sin(theta_opt)
             ct = np.cos(theta_opt)
             cp = np.cos(phi_opt)
             sp = np.sin(phi_opt)
             
             X = st * cp
             Y = st * sp
             Z = ct
             x_opt = np.column_stack([X, Y, Z])
             
             # Normalize just in case of numerical drift
             norms = np.linalg.norm(x_opt, axis=1)
             x_opt = x_opt / norms[:, np.newaxis]
             
             return x_opt
        else:
             return x_init
    except Exception:
        return x_init
