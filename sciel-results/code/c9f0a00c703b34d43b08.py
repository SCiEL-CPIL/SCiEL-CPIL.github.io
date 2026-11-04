"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid else 0
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    NM, MP, PN = n*m, m*p, p*n
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = np.array(U).reshape(n, m, rank)
        V_r = np.array(V).reshape(m, p, rank)
        W_r = np.array(W).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-8)
    
    def schoolbook_init():
        """Create the schoolbook (trivial) decomposition"""
        U = np.zeros((n, m, rank_schoolbook))
        V = np.zeros((m, p, rank_schoolbook))
        W = np.zeros((p, n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U, V, W
    
    def snap_to_halfint(X):
        """Snap values to nearest half-integer"""
        return np.round(X * 2.0) / 2.0
    
    def snap_to_int(X):
        """Snap values to nearest integer"""
        return np.round(X)
    
    def alternating_projection(rank, max_iters=300, seed=0, init=None, use_halfint=True):
        """
        Alternating projection: 
        1. Solve linear least-squares for each factor (affine projection)
        2. Snap to nearest half-integers
        Iterate until convergence or exact solution found.
        """
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = [x.copy() for x in init]
        else:
            # Initialize with small random values
            scale = 0.5
            U = np.random.randn(n, m, rank) * scale
            V = np.random.randn(m, p, rank) * scale
            W = np.random.randn(p, n, rank) * scale
            if use_halfint:
                U = snap_to_halfint(U)
                V = snap_to_halfint(V)
                W = snap_to_halfint(W)
        
        prev_err = np.inf
        stuck_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Check current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if current_err < 1e-10:
                # Check if already half-integer
                if use_halfint:
                    if np.allclose(U, snap_to_halfint(U), atol=1e-10) and \
                       np.allclose(V, snap_to_halfint(V), atol=1e-10) and \
                       np.allclose(W, snap_to_halfint(W), atol=1e-10):
                        return U, V, W
                    # Snap and recheck
                    U = snap_to_halfint(U)
                    V = snap_to_halfint(V)
                    W = snap_to_halfint(W)
                    T_rec2 = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    if np.allclose(T_rec2.flatten(), T_flat, atol=1e-8):
                        return U, V, W
                else:
                    # Try integer snap
                    Ui = snap_to_int(U)
                    Vi = snap_to_int(V)
                    Wi = snap_to_int(W)
                    T_rec2 = np.einsum('ijc,jkc,kic->ijk', Ui, Vi, Wi)
                    if np.allclose(T_rec2.flatten(), T_flat, atol=1e-8):
                        return Ui, Vi, Wi
                    # Try half-integer snap
                    Uh = snap_to_halfint(U)
                    Vh = snap_to_halfint(V)
                    Wh = snap_to_halfint(W)
                    T_rec3 = np.einsum('ijc,jkc,kic->ijk', Uh, Vh, Wh)
                    if np.allclose(T_rec3.flatten(), T_flat, atol=1e-8):
                        return Uh, Vh, Wh
                return None
            
            # Check for convergence/stuck
            if abs(prev_err - current_err) < 1e-14:
                stuck_count += 1
                if stuck_count > 20:
                    # Try snapping and checking
                    if use_halfint:
                        U = snap_to_halfint(U)
                        V = snap_to_halfint(V)
                        W = snap_to_halfint(W)
                    T_rec2 = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    if np.allclose(T_rec2.flatten(), T_flat, atol=1e-8):
                        return U, V, W
                    # Add small random perturbation to escape local minimum
                    np.random.seed(seed + it)
                    U += np.random.randn(n, m, rank) * 0.01
                    V += np.random.randn(m, p, rank) * 0.01
                    W += np.random.randn(p, n, rank) * 0.01
                    stuck_count = 0
            else:
                stuck_count = 0
            prev_err = current_err
            
            # Step 1: Solve for U (affine projection)
            for i in range(n):
                for j in range(m):
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])  # (rank, p)
                    b = T[i, j, :]  # (p,)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])  # (rank, n)
                    b = T[:, j, k]  # (n,)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])  # (rank, m)
                    b = T[i, :, k]  # (m,)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
            
            # Step 2: Snap to half-integers
            if use_halfint:
                U = snap_to_halfint(U)
                V = snap_to_halfint(V)
                W = snap_to_halfint(W)
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        if np.allclose(T_rec.flatten(), T_flat, atol=1e-8):
            if use_halfint:
                U = snap_to_halfint(U)
                V = snap_to_halfint(V)
                W = snap_to_halfint(W)
                T_rec2 = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                if np.allclose(T_rec2.flatten(), T_flat, atol=1e-8):
                    return U, V, W
            else:
                # Try both integer and half-integer
                for snap_func in [snap_to_int, snap_to_halfint]:
                    Us = snap_func(U)
                    Vs = snap_func(V)
                    Ws = snap_func(W)
                    T_rec2 = np.einsum('ijc,jkc,kic->ijk', Us, Vs, Ws)
                    if np.allclose(T_rec2.flatten(), T_flat, atol=1e-8):
                        return Us, Vs, Ws
        return None
    
    def sparse_init(rank, seed=0):
        """Create a sparse random initialization"""
        np.random.seed(seed)
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        
        # Each factor gets a few non-zero entries per row
        for i in range(n):
            for j in range(m):
                num_nonzero = max(1, min(rank, np.random.randint(1, 4)))
                indices = np.random.choice(rank, size=num_nonzero, replace=False)
                U[i, j, indices] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=num_nonzero)
        
        for j in range(m):
            for k in range(p):
                num_nonzero = max(1, min(rank, np.random.randint(1, 4)))
                indices = np.random.choice(rank, size=num_nonzero, replace=False)
                V[j, k, indices] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=num_nonzero)
        
        for k in range(p):
            for i in range(n):
                num_nonzero = max(1, min(rank, np.random.randint(1, 4)))
                indices = np.random.choice(rank, size=num_nonzero, replace=False)
                W[k, i, indices] = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=num_nonzero)
        
        return U, V, W
    
    def structured_init(rank, seed=0):
        """Create initialization based on Kronecker product structure"""
        np.random.seed(seed)
        # Try to use structure from smaller matrix multiplications
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        
        # Fill with random sparse patterns
        for c in range(rank):
            # Each rank-1 component has a pattern
            if c % 3 == 0:
                # Pattern 1: row-based
                i0 = c % n
                U[i0, :, c] = np.random.choice([-1.0, 1.0], size=m)
                V[:, :, c] = 0
                W[:, i0, c] = np.random.choice([-1.0, 1.0], size=p)
            elif c % 3 == 1:
                # Pattern 2: col-based
                j0 = c % m
                U[:, j0, c] = np.random.choice([-1.0, 1.0], size=n)
                V[j0, :, c] = np.random.choice([-1.0, 1.0], size=p)
                W[:, :, c] = 0
            else:
                # Pattern 3: mixed
                i0 = c % n
                j0 = c % m
                k0 = c % p
                U[i0, j0, c] = 1.0
                V[j0, k0, c] = 1.0
                W[k0, i0, c] = 1.0
        
        return U, V, W
    
    def try_rank(rank, num_restarts=30, use_halfint=True):
        """Try to find a rank-rank decomposition"""
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Try different initialization strategies
            if seed % 3 == 0:
                # Random dense
                init = None
            elif seed % 3 == 1:
                # Sparse
                U_init, V_init, W_init = sparse_init(rank, seed=seed)
                init = (U_init, V_init, W_init)
            else:
                # Structured
                U_init, V_init, W_init = structured_init(rank, seed=seed)
                init = (U_init, V_init, W_init)
            
            max_iters = 200 if rank > 60 else 300
            result = alternating_projection(rank, max_iters=max_iters, seed=seed, init=init, use_halfint=use_halfint)
            
            if result is not None:
                U, V, W = result
                return (U.reshape(NM, rank).tolist(), 
                       V.reshape(MP, rank).tolist(), 
                       W.reshape(PN, rank).tolist())
        
        return None
    
    # Start with schoolbook as guaranteed solution
    U_sb, V_sb, W_sb = schoolbook_init()
    best_result = {"U": U_sb.reshape(NM, rank_schoolbook).tolist(),
                   "V": V_sb.reshape(MP, rank_schoolbook).tolist(),
                   "W": W_sb.reshape(PN, rank_schoolbook).tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Focus on ranks that are more likely to work
    ranks_to_try = list(range(rank_schoolbook - 1, 59, -1))
    
    # Prioritize certain ranks
    # Ranks that are products or have special structure
    priority_ranks = [63, 60, 58, 56, 54, 52, 50]
    ranks_to_try = [r for r in priority_ranks if r < rank_schoolbook and r > 59] + \
                   [r for r in ranks_to_try if r not in priority_ranks]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try with half-integer constraint
        result = try_rank(rank, num_restarts=20, use_halfint=True)
        if result is not None:
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = rank
            break
        
        # Also try without half-integer constraint (might find integer solutions)
        if time.time() > deadline - 25:
            break
        result = try_rank(rank, num_restarts=10, use_halfint=False)
        if result is not None:
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
