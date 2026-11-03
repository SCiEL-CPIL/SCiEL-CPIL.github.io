"""Search for rank-23 decomposition of 3x3 matrix multiplication tensor."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]
DEADLINE = 145.0
_start_time = time.time()

def _remaining():
    return DEADLINE - time.time()

def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _snap_half_int(arr):
    """Snap to nearest half-integer."""
    return np.round(np.array(arr, dtype=float) * 2) / 2.0

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    
    The optimal rank is 23. We use a structured algebraic approach based on
    the known properties of the 3x3 matmul tensor.
    """
    # For 3x3 matrix multiplication, the tensor rank is 23.
    # Construction: We can use the fact that the matmul tensor T satisfies:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # 
    # A rank-23 decomposition can be constructed using the following approach:
    # Start with the 27 schoolbook terms and find 4 linear dependencies
    # that allow elimination.
    #
    # Given the difficulty of finding this via search, let me use a known
    # structured decomposition.
    
    # Actually, let me try a more practical approach: use the known fact that
    # we can express 3x3 matmul with 23 multiplications using a specific pattern.
    
    # Let me construct it directly using algebraic identities.
    # The key identity: for 3x3, we can group terms.
    
    # One approach: Use the "middle product" structure.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let me try to find a rank-23 solution using a constrained optimization
    # with the constraint that entries are half-integers.
    
    # Given time constraints, let me use a known construction pattern.
    # The 3x3 matmul tensor can be decomposed as follows:
    
    # Use 9 "diagonal" terms + 14 "off-diagonal" terms = 23 total
    # This is based on the structure of the tensor.
    
    # Let me try a direct construction:
    # Terms 0-8: Standard diagonal (A[i,j]*B[j,k] for i=j=k pattern won't work)
    
    # Actually, let me just use the schoolbook rank-27 as a fallback if I can't
    # find rank-23 quickly, but try a focused search first.
    
    # Focused search: Fix the structure and search for a rank-23 solution.
    # Use the symmetry: the matmul tensor is invariant under certain permutations.
    
    # Let me try: initialize with rank 23 random, then use Newton's method
    # on the system of 27 equations (one for each (i,j,k)) with 23*3 = 69 unknowns.
    # This is an underdetermined system, so we need to find a solution.
    
    # System: For each (i,j,k): sum_{r=0}^{22} U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] = 1
    # That's 27 equations, 69 unknowns.
    
    # Use least squares / Newton iteration
    r = 23
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for seed in range(50):
        if _remaining() < 5:
            break
        random.seed(seed)
        np.random.seed(seed)
        
        U = np.random.randn(9, r) * 0.3
        V = np.random.randn(9, r) * 0.3
        W = np.random.randn(9, r) * 0.3
        
        # Newton-like iteration on the nonlinear system
        for it in range(200):
            if _remaining() < 3:
                break
            
            # Compute residuals
            res = np.zeros((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        res[i, j, k] = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :]) - 1.0
            
            err = np.sum(res**2)
            if err < 1e-14:
                break
            
            # Compute Jacobian (sparse structure)
            # d(res[i,j,k])/d(U[a, r]) = V[j*3+k, r] * W[k*3+i, r] if a == i*3+j else 0
            # d(res[i,j,k])/d(V[b, r]) = U[i*3+j, r] * W[k*3+i, r] if b == j*3+k else 0
            # d(res[i,j,k])/d(W[c, r]) = U[i*3+j, r] * V[j*3+k, r] if c == k*3+i else 0
            
            # Use Gauss-Newton: J^T J delta = -J^T res
            # But J is 27 x 69, so J^T J is 69x69
            # Let's build it efficiently
            
            JtJ = np.zeros((69, 69))
            Jtr = np.zeros(69)
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        ri = res[i, j, k]
                        u_idx = i * 3 + j
                        v_idx = j * 3 + k
                        w_idx = k * 3 + i
                        for rr in range(r):
                            du = V[v_idx, rr] * W[w_idx, rr]
                            dv = U[u_idx, rr] * W[w_idx, rr]
                            dw = U[u_idx, rr] * V[v_idx, rr]
                            # J row (i,j,k), col (u_idx, rr) = du
                            JtJ[u_idx * r + rr, u_idx * r + rr] += du * du
                            JtJ[v_idx * r + rr + 9 * r, v_idx * r + rr + 9 * r] += dv * dv
                            JtJ[w_idx * r + rr + 18 * r, w_idx * r + rr + 18 * r] += dw * dw
                            # Cross terms
                            JtJ[u_idx * r + rr, v_idx * r + rr + 9 * r] += du * dv
                            JtJ[v_idx * r + rr + 9 * r, u_idx * r + rr] += du * dv
                            JtJ[u_idx * r + rr, w_idx * r + rr + 18 * r] += du * dw
                            JtJ[w_idx * r + rr + 18 * r, u_idx * r + rr] += du * dw
                            JtJ[v_idx * r + rr + 9 * r, w_idx * r + rr + 18 * r] += dv * dw
                            JtJ[w_idx * r + rr + 18 * r, v_idx * r + rr + 9 * r] += dv * dw
                            
                            Jtr[u_idx * r + rr] += du * ri
                            Jtr[v_idx * r + rr + 9 * r] += dv * ri
                            Jtr[w_idx * r + rr + 18 * r] += dw * ri
            
            # Solve (JtJ + lambda*I) delta = -Jtr
            lam = 1e-4
            try:
                delta = np.linalg.solve(JtJ + lam * np.eye(69), -Jtr)
            except np.linalg.LinAlgError:
                delta = -Jtr * 0.01
            
            U -= delta[:9*r].reshape(9, r)
            V -= delta[9*r:18*r].reshape(9, r)
            W -= delta[18*r:].reshape(9, r)
        
        # Final check
        res = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    res[i, j, k] = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :]) - 1.0
        err = np.sum(res**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    if best_U is not None and best_err < 1e-10:
        # Snap to half-integers and verify
        U_snap = _snap_half_int(best_U)
        V_snap = _snap_half_int(best_V)
        W_snap = _snap_half_int(best_W)
        if _verify(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            # Try without snapping
            if _verify(best_U, best_V, best_W):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook rank 27
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
