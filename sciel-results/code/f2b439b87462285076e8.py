import numpy as np
import time

N = 100

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_grad_hess(P):
    """Compute energy, tangent gradient, and tangent Hessian on the sphere.
    P is (N,3) assumed unit norm. Returns (energy, G_tan (N,3), H_tan (N,3,N,3))."""
    # Pairwise differences
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]  # (N,N,3)
    dists = np.linalg.norm(diff, axis=-1)  # (N,N)
    np.fill_diagonal(dists, 1.0)
    
    # Energy
    triu_i, triu_j = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dists[triu_i, triu_j]))
    
    # inv_d2 (N,N)
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    
    # Gradient (Euclidean) for point i: sum_{j != i} (P_i - P_j) / d_ij^2 = sum_j diff[i,j] * inv_d2[i,j]
    # Wait, f_ij = -log(d_ij). d/dP_i (-log d_ij) = -(1/d_ij) * d(d_ij)/dP_i = -(1/d_ij)*(P_i - P_j)/d_ij = -(P_i - P_j)/d_ij^2
    # So gradient of total energy w.r.t P_i = sum_{j != i} -(P_i - P_j)/d_ij^2 = -sum_j diff[i,j]*inv_d2[i,j]
    G_euc = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (N,3)
    
    # Project to tangent space
    dot = np.sum(G_euc * P, axis=1, keepdims=True)
    G_tan = G_euc - dot * P  # (N,3)
    
    # Hessian (Euclidean) for pair (i,j):
    # d2/dP_i dP_i of (-log d_ij) = (I - r_ij r_ij^T / d^2) / d^2
    # where r_ij = P_i - P_j, d = |r_ij|
    # d2/dP_i dP_j of (-log d_ij) = -(I - r_ij r_ij^T / d^2) / d^2
    # Let's compute the "outer" term: (I - r r^T/d^2)/d^2 for each pair
    # r = diff (N,N,3), d^2 = dists^2 (N,N)
    # We need r r^T / d^2 for each pair: diff[:,:,i] * diff[:,:,j] / d^2
    # H_block = (I - r r^T/d^2)/d^2 = I/d^2 - r r^T/d^4
    
    d2 = dists ** 2  # (N,N)
    inv_d4 = np.zeros_like(d2)
    inv_d4[mask] = 1.0 / (d2[mask] ** 2)
    
    # r r^T / d^4: (N,N,3,3)
    rr = diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :]  # (N,N,3,3)
    rr_over_d4 = rr * inv_d4[:, :, np.newaxis, np.newaxis]
    
    # I/d^2 - rr/d^4
    I = np.eye(3)
    # H_pair (N,N,3,3) = (I/d^2 - rr/d^4) for the "diagonal" type, but we need to be careful
    # For pair (i,j): H_ii += (I/d^2 - rr/d^4), H_ij += -(I/d^2 - rr/d^4)
    # Actually the Euclidean Hessian of f_ij w.r.t (P_i, P_j):
    #   H_ii = (I - r r^T/d^2)/d^2
    #   H_ij = -(I - r r^T/d^2)/d^2
    #   H_ji = H_ij^T
    #   H_jj = (I - r r^T/d^2)/d^2  (since d/dP_j of (P_j - P_i)/d^2 ... let me verify)
    # d/dP_j (-log d_ij) = -(1/d_ij) * (P_j - P_i)/d_ij = (P_i - P_j)/d_ij^2 = -diff[i,j]*inv_d2[i,j]... 
    # Hmm, let me be careful. Actually by symmetry, H_jj for pair (i,j) is also (I - r r^T/d^2)/d^2 where r = P_i - P_j? 
    # Let r = P_i - P_j. d/dP_j f = (P_i - P_j)/d^2 = r/d^2. d/dP_j (r/d^2) = d/dP_j ((P_i-P_j)/d^2)
    # = (-I * d^2 - (P_i-P_j)*d(P_j)/dP_j)/d^4 ... this is getting complex. Let me just use the known result:
    # For f = -log|P_i - P_j|, the Hessian is:
    #   H_ii = (I - u u^T)/d^2, H_jj = (I - u u^T)/d^2, H_ij = H_ji = -(I - u u^T)/d^2
    # where u = (P_i - P_j)/d, d = |P_i - P_j|.
    # So (I - u u^T)/d^2 = (I - r r^T/d^2)/d^2 = I/d^2 - r r^T/d^4. Yes.
    
    H_pair_diag = I[np.newaxis, np.newaxis, :, :] / d2[:, :, np.newaxis, np.newaxis] - rr_over_d4
    # H_pair_diag (N,N,3,3) = (I - u u^T)/d^2 for pair (i,j)
    
    # Build full Euclidean Hessian (N,3,N,3)
    # H_euc[i,j] = sum over pairs
    # For pair (i,j) with i != j:
    #   H_euc[i,i] += H_pair_diag[i,j]
    #   H_euc[j,j] += H_pair_diag[i,j]
    #   H_euc[i,j] -= H_pair_diag[i,j]
    #   H_euc[j,i] -= H_pair_diag[i,j]
    # Note H_pair_diag[i,j] = H_pair_diag[j,i] since r^2 same.
    
    H_euc = np.zeros((N, 3, N, 3))
    # Use broadcasting: for each pair (i,j), add to 4 blocks
    # H_euc[i,i] += sum_j H_pair_diag[i,j]
    # H_euc[i,j] -= H_pair_diag[i,j]
    
    # H_ii: sum over j of H_pair_diag[i,j]
    H_ii = np.sum(H_pair_diag, axis=1)  # (N,3,3)
    # H_ij for i != j: -H_pair_diag[i,j]
    # So H_euc[i,j] = -H_pair_diag[i,j] for i != j, and H_euc[i,i] = H_ii[i]
    
    # Build: H_euc[i,j] for all i,j
    # H_euc[i,j] = -H_pair_diag[i,j] + (I if i==j else 0) * ... 
    # Actually H_euc[i,j] = -H_pair_diag[i,j] for i != j, and H_euc[i,i] = sum_k H_pair_diag[i,k]
    # Note: -H_pair_diag[i,i] is 0 (since diagonal of H_pair_diag is 0, d=1 set artificially but we don't use i==j)
    # Let's set H_pair_diag diagonal to 0 to be safe
    H_pair_diag[np.arange(N), np.arange(N), :, :] = 0.0
    
    # H_euc[i,j] = H_ii[i] * delta_ij - H_pair_diag[i,j]
    H_euc = -H_pair_diag  # (N,N,3,3) -> this is H_euc[i,j] for the off-diagonal part
    # Add diagonal: H_euc[i,i] should be H_ii[i]
    idx = np.arange(N)
    H_euc[idx, idx, :, :] = H_ii
    
    # Now project Hessian to tangent space
    # The Riemannian Hessian is the projection of the Euclidean Hessian applied to tangent vectors.
    # For tangent vectors v_i (perpendicular to P_i), the Hessian action:
    # (H_tan v)_i = sum_j (P_i P_i^T) H_euc[i,j] v_j ... 
    # Actually, the Riemannian Hessian at P of f in direction v (tangent) is:
    # Hess f(v) = (D^2 f)(v) - (D^2 f)(v, P) P - ... 
    # Simpler: the Riemannian Hessian matrix in tangent space is:
    # H_tan[i,j] = (I - P_i P_i^T) H_euc[i,j] (I - P_j P_j^T)
    # where (I - P_i P_i^T) is the projection to tangent space at P_i.
    
    # Build projection matrices
    # Proj_i = I - P_i P_i^T (3,3)
    PP = P[:, :, np.newaxis] * P[:, np.newaxis, :]  # (N,3,3) P_i P_i^T
    Proj = I[np.newaxis, :, :] - PP  # (N,3,3)
    
    # H_tan[i,j] = Proj_i @ H_euc[i,j] @ Proj_j
    # H_euc is (N,N,3,3). 
    # First: H_euc[i,j] @ Proj_j: (N,N,3,3) @ (N,3,3) -> need broadcasting
    # H_euc[i,j] @ Proj_j = sum_k H_euc[i,j,k,m] * Proj[j,m,l]
    H_euc_ProjJ = np.einsum('ijnm,jml->ijnl', H_euc, Proj)  # (N,N,3,3)
    H_tan = np.einsum('inm,ijnl->ijml', Proj, H_euc_ProjJ)  # (N,N,3,3)
    
    return energy, G_tan, H_tan

def _riemannian_newton(P, max_iter=200, time_limit=30.0, damping=1.0, anneal_temp=0.0, rng=None):
    """Riemannian Newton with damping and optional simulated annealing."""
    start = time.time()
    P = P.copy()
    best_P = P.copy()
    best_E = _energy_grad_hess(P)[0]
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        E, G, H = _energy_grad_hess(P)
        if E < best_E:
            best_E = E
            best_P = P.copy()
        
        # Check convergence
        gnorm = np.linalg.norm(G)
        if gnorm < 1e-10:
            break
        
        # Solve H v = -G in tangent space
        # Flatten H to (3N, 3N), G to (3N,)
        H_flat = H.transpose(0, 2, 1, 3).reshape(N * 3, N * 3)
        G_flat = G.reshape(N * 3)
        
        # Damped Newton: (H + lam I) v = -G
        # Choose lam based on eigenvalues or just a fixed damping
        try:
            # Add regularization for stability
            lam = 1e-6
            H_reg = H_flat + lam * np.eye(N * 3)
            v_flat = np.linalg.solve(H_reg, -G_flat)
        except np.linalg.LinAlgError:
            # Fallback to gradient step
            v_flat = -G_flat * 0.01
            v = v_flat.reshape(N, 3)
            # Project to tangent
            dot = np.sum(v * P, axis=1, keepdims=True)
            v = v - dot * P
            # Retraction: normalize
            P_new = P + v
            P_new /= np.linalg.norm(P_new, axis=1, keepdims=True)
            # Simulated annealing acceptance
            E_new = _energy_grad_hess(P_new)[0]
            if E_new < E or (anneal_temp > 0 and rng is not None and rng.random() < np.exp(-(E_new - E) / anneal_temp)):
                P = P_new
            continue
        
        v = v_flat.reshape(N, 3)
        # Project to tangent space (Newton step should already be tangent, but ensure)
        dot = np.sum(v * P, axis=1, keepdims=True)
        v = v - dot * P
        
        # Line search for damping
        alpha = 1.0
        P_new = P + alpha * v
        P_new /= np.linalg.norm(P_new, axis=1, keepdims=True)
        E_new = _energy_grad_hess(P_new)[0]
        
        # Backtracking line search
        for ls in range(20):
            if E_new < E:
                break
            alpha *= 0.5
            P_new = P + alpha * v
            P_new /= np.linalg.norm(P_new, axis=1, keepdims=True)
            E_new = _energy_grad_hess(P_new)[0]
        
        # Simulated annealing: sometimes accept worse
        if E_new < E:
            P = P_new
        elif anneal_temp > 0 and rng is not None:
            if rng.random() < np.exp(-(E_new - E) / anneal_temp):
                P = P_new
        else:
            # Try smaller step
            for ls in range(5):
                alpha *= 0.5
                P_new = P + alpha * v
                P_new /= np.linalg.norm(P_new, axis=1, keepdims=True)
                E_new = _energy_grad_hess(P_new)[0]
                if E_new < E:
                    P = P_new
                    break
    
    return best_P, best_E

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)
    
    rng = np.random.default_rng(42)
    
    # Initializations
    inits = []
    fib = _fibonacci_sphere()
    inits.append(fib)
    
    # Random inits
    for _ in range(8):
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)
    
    # Perturbed fib
    for _ in range(4):
        noise = rng.normal(size=(N, 3)) * 0.2
        p = fib + noise
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        inits.append(p)
    
    best_P = None
    best_E = float('inf')
    
    n_inits = len(inits)
    for i, P0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time() - 10
        alloc = remaining / max(1, n_inits - i)
        alloc = max(alloc, 5.0)  # Minimum 5s per init
        
        # Simulated annealing: start with high temp, cool down
        # Use annealing to escape local minima
        P_opt, E_opt = _riemannian_newton(P0, max_iter=100, time_limit=alloc, anneal_temp=0.5, rng=rng)
        
        if E_opt < best_E:
            best_E = E_opt
            best_P = P_opt.copy()
    
    # Polish the best solution with pure Newton (no annealing)
    if time.time() < deadline - 5 and best_P is not None:
        time_left = deadline - time.time() - 5
        P_opt, E_opt = _riemannian_newton(best_P, max_iter=100, time_limit=time_left, anneal_temp=0.0)
        if E_opt < best_E:
            best_E = E_opt
            best_P = P_opt.copy()
    
    if best_P is None:
        best_P = fib
    
    # Normalize final result
    best_P = best_P / np.linalg.norm(best_P, axis=1, keepdims=True)
    
    return best_P
