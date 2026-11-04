"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Strategy: Use structured initialization based on known algebraic decompositions
    # For 4x4 * 4x9, we can try various rank ranges with proper ALS
    
    best = None
    best_rank = nm * mp * pn  # schoolbook = 144
    
    # Try ranks from 80 down to 60 (known bounds suggest < 100 is possible)
    for r in range(80, 55, -1):
        for trial in range(5):
            random.seed(42 + trial * 1000 + r)
            np.random.seed(42 + trial * 1000 + r)
            
            # Initialize with structured patterns that respect the tensor structure
            # Use Kronecker product based initialization
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Try different initialization strategies
            strategy = trial % 3
            
            if strategy == 0:
                # Random half-integer initialization
                U = np.random.choice([-0.5, 0.0, 0.5], size=(nm, r))
                V = np.random.choice([-0.5, 0.0, 0.5], size=(mp, r))
                W = np.random.choice([-0.5, 0.0, 0.5], size=(pn, r))
            elif strategy == 1:
                # Structured: partition columns among (i,j,k) groups
                for c in range(r):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n*m)) % p
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    # Add some perturbation
                    if c < r//2:
                        U[(i+1)%m*m + j, c] = 0.5
                        V[(j+1)%m*p + k, c] = 0.5
                        W[(k+1)%p*n + i, c] = 0.5
            else:
                # Random with bias toward structured entries
                U = np.random.choice([-0.5, 0.0, 0.5, 1.0], size=(nm, r))
                V = np.random.choice([-0.5, 0.0, 0.5, 1.0], size=(mp, r))
                W = np.random.choice([-0.5, 0.0, 0.5, 1.0], size=(pn, r))
                # Zero out most entries
                for c in range(r):
                    mask_u = np.random.random(nm) < 0.3
                    mask_v = np.random.random(mp) < 0.3
                    mask_w = np.random.random(pn) < 0.3
                    U[~mask_u, c] = 0.0
                    V[~mask_v, c] = 0.0
                    W[~mask_w, c] = 0.0
            
            # ALS iterations
            converged = False
            for it in range(50):
                # Update U
                for c in range(r):
                    # Build RHS: for each u, sum_{v,w} T[u,v,w] * V[v,c] * W[w,c]
                    # Build LHS matrix: M[c',c] = <V[:,c'], V[:,c]> * <W[:,c'], W[:,c]>
                    Vc = V[:, c]
                    Wc = W[:, c]
                    
                    # RHS vector
                    rhs = np.zeros(nm)
                    for u in range(nm):
                        rhs[u] = np.sum(T[u, :, :] * Vc[None, :] * Wc[None, :])
                    
                    # LHS matrix
                    M = np.zeros((r, r))
                    for cp in range(r):
                        M[cp, c] = np.dot(V[:, cp], Vc) * np.dot(W[:, cp], Wc)
                    
                    # Solve M @ U[:,c] = rhs (but M should be r x r, we're solving for U[:,c])
                    # Actually this formulation is wrong. Let me redo.
                    # The correct ALS: fix V, W. For column c of U:
                    # We want to minimize ||T - sum_c' U_c' ⊗ V_c' ⊗ W_c'||_F^2
                    # Taking derivative w.r.t U[u,c]:
                    # sum_{v,w} T[u,v,w] V[v,c] W[w,c] = sum_{c'} U[u,c'] V[v,c']V[v,c] W[w,c']W[w,c]
                    # = sum_{c'} U[u,c'] * (V[:,c']^T V[:,c]) * (W[:,c']^T W[:,c])
                    
                    # So: for each u, rhs[u] = sum_{v,w} T[u,v,w] V[v,c] W[w,c]
                    # And the system is: sum_{c'} U[u,c'] * M[c',c] = rhs[u] for all u
                    # This means: U[:,c] is determined by... wait, this couples all c'.
                    
                    # Actually in ALS, when updating U[:,c], we treat other U[:,c'] as fixed.
                    # So: rhs_new[u] = sum_{v,w} T[u,v,w] V[v,c] W[w,c] - sum_{c'≠c} U[u,c'] * M[c',c]
                    # Then U[u,c] = rhs_new[u] / M[c,c] (if M[c,c] ≠ 0)
                    
                    M_diag = M[c, c]
                    if abs(M_diag) < 1e-10:
                        continue
                    
                    rhs_new = rhs.copy()
                    for cp in range(r):
                        if cp != c:
                            rhs_new -= U[:, cp] * M[cp, c]
                    
                    U[:, c] = rhs_new / M_diag
                
                # Update V similarly
                for c in range(r):
                    Uc = U[:, c]
                    Wc = W[:, c]
                    
                    rhs = np.zeros(mp)
                    for v in range(mp):
                        rhs[v] = np.sum(T[:, v, :] * Uc[:, None] * Wc[None, :])
                    
                    M = np.zeros((r, r))
                    for cp in range(r):
                        M[cp, c] = np.dot(U[:, cp], Uc) * np.dot(W[:, cp], Wc)
                    
                    M_diag = M[c, c]
                    if abs(M_diag) < 1e-10:
                        continue
                    
                    rhs_new = rhs.copy()
                    for cp in range(r):
                        if cp != c:
                            rhs_new -= V[:, cp] * M[cp, c]
                    
                    V[:, c] = rhs_new / M_diag
                
                # Update W similarly
                for c in range(r):
                    Uc = U[:, c]
                    Vc = V[:, c]
                    
                    rhs = np.zeros(pn)
                    for w in range(pn):
                        rhs[w] = np.sum(T[:, :, w] * Uc[:, None] * Vc[None, :])
                    
                    M = np.zeros((r, r))
                    for cp in range(r):
                        M[cp, c] = np.dot(U[:, cp], Uc) * np.dot(V[:, cp], Vc)
                    
                    M_diag = M[c, c]
                    if abs(M_diag) < 1e-10:
                        continue
                    
                    rhs_new = rhs.copy()
                    for cp in range(r):
                        if cp != c:
                            rhs_new -= W[:, cp] * M[cp, c]
                    
                    W[:, c] = rhs_new / M_diag
                
                # Check convergence
                recon = np.zeros_like(T)
                for c in range(r):
                    recon += np.einsum('u,v,w->uvw', U[:,c], V[:,c], W[:,c])
                residual = np.linalg.norm(recon - T)
                if residual < 1e-4:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded version
                recon_r = np.zeros_like(T)
                for c in range(r):
                    recon_r += np.einsum('u,v,w->uvw', U_rounded[:,c], V_rounded[:,c], W_rounded[:,c])
                
                if np.allclose(recon_r, T, atol=1e-6):
                    best = (r, U_rounded, V_rounded, W_rounded)
                    best_rank = r
                    break
        
        if best is not None:
            break
    
    if best is not None:
        r, U, V, W = best
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
