"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # Known Rank 18 Solution (Fallback)
    # 2 Strassen blocks (rank 7 each) for first 4 cols, 1 Schoolbook (rank 4) for last col
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            # Strassen products M1..M7
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    # Check if a given set of factors is a valid decomposition
    def check_decomposition(U, V, W, rank, tol=1e-9):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Indices:
        # U[i*m + j, t] -> A_ij
        # V[j*p + k, t] -> B_jk
        # W[k*n + i, t] -> C_ki (row k, col i)
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for t in range(rank):
                            val += U[i*m + j][t] * V[jp*p + k][t] * W[k*n + i][t]
                        if abs(val - target) > tol:
                            return False
        return True

    def snap_to_half_integers(arr, tol=1e-5):
        """Snap values to nearest half-integer if close enough."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def snap_to_integers(arr, tol=1e-5):
        """Snap values to nearest integer if close enough."""
        snapped = np.round(arr)
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_find_223_decomposition(target_rank):
        if time.time() > DEADLINE - 15:
            return None
        
        n_sub = 2
        m_sub = 2
        p_sub = 3
        
        def error_223(params, rank):
            U = params[:n_sub*m_sub*rank].reshape((n_sub*m_sub, rank))
            V = params[n_sub*m_sub*rank : n_sub*m_sub*rank + m_sub*p_sub*rank].reshape((m_sub*p_sub, rank))
            W = params[n_sub*m_sub*rank + m_sub*p_sub*rank :].reshape((p_sub*n_sub, rank))
            
            err = 0.0
            for i in range(n_sub):
                for j in range(m_sub):
                    for jp in range(m_sub):
                        for k in range(p_sub):
                            target = 1.0 if j == jp else 0.0
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                            err += (val - target) ** 2
            return err

        def check_exact_223(U, V, W, rank, tol=1e-6):
            for i in range(n_sub):
                for j in range(m_sub):
                    for jp in range(m_sub):
                        for k in range(p_sub):
                            target = 1.0 if j == jp else 0.0
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                            if abs(val - target) > tol:
                                return False
            return True

        # Strategy 1: Numerical optimization (Powell/L-BFGS-B)
        # Strategy 2: Randomized algebraic elimination (Gaussian elimination over rationals)
        
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 15:
                break
            
            if restart % 2 == 0:
                # Numerical optimization
                np.random.seed(restart * 101 + target_rank)
                x0 = np.random.uniform(-1, 1, size=n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank)
                
                method = 'Powell'
                options = {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
                
                from scipy.optimize import minimize
                res = minimize(error_223, x0, args=(target_rank,), method=method, options=options)
                
                if res.fun < 1e-8:
                    U = res.x[:n_sub*m_sub*target_rank].reshape((n_sub*m_sub, target_rank))
                    V = res.x[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape((m_sub*p_sub, target_rank))
                    W = res.x[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape((p_sub*n_sub, target_rank))
                    
                    # Try snapping
                    U_r = snap_to_half_integers(U)
                    V_r = snap_to_half_integers(V)
                    W_r = snap_to_half_integers(W)
                    
                    if U_r is not None and V_r is not None and W_r is not None:
                        if check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-6):
                            return (U_r, V_r, W_r)
                    
                    U_i = snap_to_integers(U)
                    V_i = snap_to_integers(V)
                    W_i = snap_to_integers(W)
                    
                    if U_i is not None and V_i is not None and W_i is not None:
                        if check_exact_223(U_i, V_i, W_i, target_rank, tol=1e-6):
                            return (U_i, V_i, W_i)
            else:
                # Algebraic Elimination Strategy
                # We want to find sparse rational solutions.
                # Fix a pattern of non-zeros in U, V, W and solve for the values.
                # Since rank is small (9 or 10), we can try random sparse patterns.
                
                # Number of variables:
                # U: 4 * rank
                # V: 6 * rank
                # W: 6 * rank
                # Total: 16 * rank
                # Number of equations: 2*2*2*3 = 24
                # For rank 9: 144 vars, 24 eqs. Very underdetermined.
                # We can set most variables to 0 and solve for the rest.
                
                # Let's try to find a solution where each term has a specific structure.
                # Or just random sparsity.
                
                np.random.seed(restart * 200 + target_rank)
                
                # Choose a subset of variables to be non-zero.
                # Let's pick 24 variables to match the 24 equations (square system).
                # If we pick too few, it might be unsolvable. If too many, we fix some to 0.
                # Actually, we can pick 24 variables and solve the linear system.
                # But the system is nonlinear (trilinear).
                # However, if we fix 2 variables per term to be non-zero and 1 to be non-zero?
                # No, each equation involves 3 variables (one from U, one from V, one from W) for each term.
                # Wait, the equation is sum_t U[i,j,t] V[j',k,t] W[k,i,t] = target.
                # This is a multilinear system.
                
                # A common trick: Set one factor to be identity-like or basis vectors.
                # But we need low rank.
                
                # Let's try a different approach:
                # Iterate over possible "support" sets for the tensors.
                # For each term t, choose a set of indices for U, V, W that are non-zero.
                # Then solve for the values.
                
                # Given time constraints, let's stick to numerical optimization but with better initialization.
                # Or try to solve a linear system by fixing two tensors and solving for the third?
                # No, it's trilinear.
                
                # Let's try a simple heuristic:
                # Randomly initialize U, V, W with small integers (-1, 0, 1).
                # Check if it's a solution.
                # If not, perturb.
                
                U_int = np.zeros((n_sub*m_sub, target_rank), dtype=int)
                V_int = np.zeros((m_sub*p_sub, target_rank), dtype=int)
                W_int = np.zeros((p_sub*n_sub, target_rank), dtype=int)
                
                # Fill with random -1, 0, 1
                U_int[np.random.choice(U_int.size, size=U_int.size//2, replace=False)] = np.random.choice([-1, 1], size=U_int.size//2)
                V_int[np.random.choice(V_int.size, size=V_int.size//2, replace=False)] = np.random.choice([-1, 1], size=V_int.size//2)
                W_int[np.random.choice(W_int.size, size=W_int.size//2, replace=False)] = np.random.choice([-1, 1], size=W_int.size//2)
                
                if check_exact_223(U_int, V_int, W_int, target_rank, tol=1e-6):
                    return (U_int, V_int, W_int)
        
        return None

    # Try to find rank 9 for 2x2x3 (would give total rank 16)
    sub_decomp = None
    if time.time() < DEADLINE - 30:
        sub_decomp = try_find_223_decomposition(9)
    
    # Try to find rank 10 for 2x2x3 (would give total rank 17)
    if sub_decomp is None and time.time() < DEADLINE - 30:
        sub_decomp = try_find_223_decomposition(10)
        
    if sub_decomp:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = 7 + rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy Strassen block (cols 0,1 of B)
        def add_strassen_block(t_offset, k_start):
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        
        # Copy sub-decomposition (cols 2,3,4 of B)
        for i in range(2):
            for j in range(2):
                for c in range(rank_sub):
                    U[i*2 + j][7 + c] = U_sub[i*2 + j, c]
                    
        for j in range(2):
            for k_sub in range(3):
                k_global = 2 + k_sub
                for c in range(rank_sub):
                    V[j*p + k_global][7 + c] = V_sub[j*3 + k_sub, c]
                    
        for k_sub in range(3):
            k_global = 2 + k_sub
            for i in range(2):
                for c in range(rank_sub):
                    W[k_global*n + i][7 + c] = W_sub[k_sub*2 + i, c]
                    
        return {"U": U, "V": V, "W": W}
    else:
        return get_rank_18_solution()

# EVOLVE-BLOCK-END
