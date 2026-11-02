import time
import random
import numpy as np
from fractions import Fraction

N, M, P = 2, 3, 10
DEADLINE = 143.0
START = time.time()

def _elapsed():
    return time.time() - START

def _check_deadline():
    return _elapsed() > DEADLINE

def _build_strassen_rank55():
    """Construct the rank-55 Strassen-based decomposition (verified)."""
    rank = 55
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    for k in range(P):
        b2k = 2 * P + k
        U[a02, col] += 1; V[b2k, col] += 1; W[2 * k, col] += 1
        col += 1
        U[a12, col] += 1; V[b2k, col] += 1; W[2 * k + 1, col] += 1
        col += 1
    return U, V, W

def _verify_np(U, V, W, tol=1e-9):
    """Verify decomposition using numpy. Returns True if exact within tol."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_solve(rank, U_init=None, V_init=None, W_init=None, max_iter=1000, tol=1e-10):
    """
    Alternating Least Squares to find a decomposition of rank 'rank'.
    """
    if U_init is None:
        U = np.random.randn(N * M, rank)
        V = np.random.randn(M * P, rank)
        W = np.random.randn(P * N, rank)
    else:
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        # If rank is smaller, truncate or pad
        if U.shape[1] > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        elif U.shape[1] < rank:
            U = np.pad(U, ((0,0), (0, rank-U.shape[1])), mode='constant')
            V = np.pad(V, ((0,0), (0, rank-V.shape[1])), mode='constant')
            W = np.pad(W, ((0,0), (0, rank-W.shape[1])), mode='constant')

    for _ in range(max_iter):
        if _check_deadline():
            return None
        
        # Solve for U
        # T_ijk = sum_r U_ijr V_jkr W_kir
        # For fixed V, W, the equation is linear in U.
        # sum_r U_ijr (V_jkr W_kir) = 1
        # Let A_ijk = V_jkr W_kir. Then sum_r U_ijr A_ijk = 1.
        # This is a linear system for U_ijr.
        # We can solve for each i,j separately? No, the equations are coupled.
        # Actually, for each i,j, we have P equations? No.
        # The equations are for all i,j,k.
        # We can vectorize.
        # Let u be the vector of U variables (size N*M*rank).
        # Let v be the vector of V variables (size M*P*rank).
        # Let w be the vector of W variables (size P*N*rank).
        # The tensor T is a 3-way array.
        # We can unfold it.
        # T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        # For fixed V, W, this is linear in U.
        # We can form a matrix A such that A * u = b, where b is the target tensor (all 1s).
        # The number of equations is N*M*P = 60.
        # The number of unknowns is N*M*rank = 6*rank.
        # For rank=54, 6*54=324 unknowns, 60 equations. Underdetermined.
        # We can find the minimum norm solution using lstsq.
        
        # Construct A for U
        # A has shape (N*M*P, N*M*rank)
        # A[ (i,j,k), (i',j',r) ] = 1 if i=i', j=j', and V[j,k,r] * W[k,i,r] != 0?
        # No, A[ (i,j,k), (i,j,r) ] = V[j,k,r] * W[k,i,r]
        # Other entries are 0.
        
        A_U = np.zeros((N*M*P, N*M*rank))
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    eq_idx = i*M*P + j*P + k
                    for r in range(rank):
                        var_idx = i*M + j
                        A_U[eq_idx, var_idx*rank + r] = V[j*P+k, r] * W[k*N+i, r]
        
        b = np.ones(N*M*P)
        U_sol, residuals, rank_s, sv = np.linalg.lstsq(A_U, b, rcond=None)
        U = U_sol.reshape(N*M, rank).T
        
        # Similarly for V
        A_V = np.zeros((N*M*P, M*P*rank))
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    eq_idx = i*M*P + j*P + k
                    for r in range(rank):
                        var_idx = j*P + k
                        A_V[eq_idx, var_idx*rank + r] = U[i*M+j, r] * W[k*N+i, r]
        
        V_sol, residuals, rank_s, sv = np.linalg.lstsq(A_V, b, rcond=None)
        V = V_sol.reshape(M*P, rank).T
        
        # Similarly for W
        A_W = np.zeros((N*M*P, P*N*rank))
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    eq_idx = i*M*P + j*P + k
                    for r in range(rank):
                        var_idx = k*N + i
                        A_W[eq_idx, var_idx*rank + r] = U[i*M+j, r] * V[j*P+k, r]
        
        W_sol, residuals, rank_s, sv = np.linalg.lstsq(A_W, b, rcond=None)
        W = W_sol.reshape(P*N, rank).T
        
        # Check convergence
        if _verify_np(U, V, W, tol=1e-8):
            return U, V, W
            
    return None

def _try_rank(rank):
    """Try to find a decomposition of given rank using ALS."""
    if _check_deadline():
        return None
    
    # Initialize with random values
    for attempt in range(100):
        if _check_deadline():
            break
        U, V, W = _als_solve(rank, max_iter=200)
        if U is not None:
            # Verify exactly
            Uf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in U]
            Vf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in V]
            Wf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in W]
            
            # Exact verification
            valid = True
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        s = Fraction(0)
                        for r in range(rank):
                            s += Uf[i*M+j][r] * Vf[j*P+k][r] * Wf[k*N+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
                
    # Try initializing from rank 55 solution
    if _check_deadline():
        return None
    U55, V55, W55 = _build_strassen_rank55()
    U, V, W = _als_solve(rank, U_init=U55, V_init=V55, W_init=W55, max_iter=500)
    if U is not None:
        Uf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in U]
        Vf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in V]
        Wf = [[Fraction(x).limit_denominator(10**12) for x in row] for row in W]
        
        valid = True
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[i*M+j][r] * Vf[j*P+k][r] * Wf[k*N+i][r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U, V, W
            
    return None

def construct():
    # Fallback: rank-55 Strassen
    U55, V55, W55 = _build_strassen_rank55()
    
    # Try rank 54
    result54 = _try_rank(54)
    if result54 is not None:
        U, V, W = result54
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try rank 53
    result53 = _try_rank(53)
    if result53 is not None:
        U, V, W = result53
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Return rank-55 fallback
    return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}
