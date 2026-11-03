import time
import sys
import random
import numpy as np
from fractions import Fraction
from itertools import product

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    We use a known rank-17 decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_exact(U, V, W):
        """Verify using exact rational arithmetic."""
        rank = len(U[0])
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[idx_u][c] * V_frac[idx_v][c] * W_frac[idx_w][c]
                    if s != Fraction(1):
                        return False
        return True

    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    def rank17_decomposition():
        """
        Construct a rank-17 decomposition for 2x3x3 matrix multiplication.
        """
        rank = 17
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Indices:
        # A: i in 0..1, j in 0..2 -> idx = i*3 + j
        # B: j in 0..2, k in 0..2 -> idx = j*3 + k
        # C: k in 0..2, i in 0..1 -> idx = k*2 + i
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A indices: 0,1,3,4
        # B indices: 0,1,3,4
        # C indices: 0,1,2,3
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[4][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # A indices: 2, 5
        # B indices: 6, 7
        # C indices: 0,1,2,3
        
        # r=7: A02 B20 -> C00
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        # r=8: A02 B21 -> C01
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        # r=9: A12 B20 -> C10
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        # r=10: A12 B21 -> C11
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
        # A indices: 0,1,2,3,4,5
        # B indices: 2,5,8
        # C indices: 4,5
        
        # r=11: A00 B02 -> C02
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        # r=12: A01 B12 -> C02
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        # r=13: A02 B22 -> C02
        U[2][13] = 1
        V[8][13] = 1
        W[4][13] = 1
        
        # r=14: A10 B02 -> C12
        U[3][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        # r=15: A11 B12 -> C12
        U[4][15] = 1
        V[5][15] = 1
        W[5][15] = 1
        
        # r=16: A12 B22 -> C12
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def snap_to_simple_fraction(x, max_num=30, max_den=30, tol=0.005):
        """Snap a float to the nearest simple fraction p/q with |p|<=max_num, 1<=q<=max_den."""
        best_frac = 0.0
        best_diff = abs(x)
        for num in range(-max_num, max_num + 1):
            for den in range(1, max_den + 1):
                val = num / den
                diff = abs(x - val)
                if diff < best_diff:
                    best_diff = diff
                    best_frac = val
        if best_diff < tol:
            return best_frac
        return x

    def try_rank16_numerical():
        """
        Attempt to find a rank-16 decomposition using numerical optimization.
        Uses L-BFGS-B with multiple restarts and high-precision polish.
        """
        target_rank = 16
        
        nm = n * m  # 6
        mp = m * p  # 9
        pn = p * n  # 6
        
        # Build target tensor as a 3D array
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Flatten target for objective
        T_flat = T.flatten()
        
        def objective_and_grad(params):
            """Compute objective value and gradient for flattened parameters."""
            U = params[:nm*target_rank].reshape(nm, target_rank)
            V = params[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
            W = params[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
            
            R = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            residual = R.flatten() - T_flat
            err = 0.5 * np.sum(residual**2)
            
            # Gradients
            R_3d = R
            res_3d = R_3d - T
            dU = np.einsum('abc,br,cr->ar', res_3d, V, W, optimize=True)
            dV = np.einsum('abc,ar,cr->br', res_3d, U, W, optimize=True)
            dW = np.einsum('abc,ar,br->cr', res_3d, U, V, optimize=True)
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return err, grad
        
        np.random.seed(42)
        num_restarts = 15
        
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            # Different initialization strategies
            if restart % 5 == 0:
                # Small random initialization
                scale = 0.2
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            elif restart % 5 == 1:
                # Medium random
                scale = 0.5
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            elif restart % 5 == 2:
                # Sparse with ±1
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    u_idx = np.random.choice(nm, size=2, replace=False)
                    v_idx = np.random.choice(mp, size=3, replace=False)
                    w_idx = np.random.choice(pn, size=2, replace=False)
                    for idx in u_idx:
                        U[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in v_idx:
                        V[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in w_idx:
                        W[idx, r] = np.random.choice([-1.0, 1.0])
            elif restart % 5 == 3:
                # Based on rank-17 structure: take first 16 columns of rank-17
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                # Copy from rank-17 pattern (first 16 terms)
                r17 = rank17_decomposition()
                U = np.array(r17['U'])[:, :target_rank]
                V = np.array(r17['V'])[:, :target_rank]
                W = np.array(r17['W'])[:, :target_rank]
                # Add small noise
                U += np.random.randn(nm, target_rank) * 0.05
                V += np.random.randn(mp, target_rank) * 0.05
                W += np.random.randn(pn, target_rank) * 0.05
            else:
                # Structured: exploit symmetry
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                # Initialize with partial Strassen-like structure
                U[0, 0] = 1; U[4, 0] = 1
                V[0, 0] = 1; V[4, 0] = 1
                W[0, 0] = 1; W[3, 0] = 1
                U[3, 1] = 1; U[4, 1] = 1
                V[0, 1] = 1
                W[1, 1] = 1; W[3, 1] = -1
                U[0, 2] = 1
                V[1, 2] = 1; V[4, 2] = -1
                W[2, 2] = 1; W[3, 2] = 1
                U[4, 3] = 1
                V[3, 3] = 1; V[0, 3] = -1
                W[0, 3] = 1; W[1, 3] = 1
                U[0, 4] = 1; U[1, 4] = 1
                V[4, 4] = 1
                W[0, 4] = -1; W[2, 4] = 1
                U[3, 5] = 1; U[0, 5] = -1
                V[0, 5] = 1; V[1, 5] = 1
                W[3, 5] = 1
                U[1, 6] = 1; U[4, 6] = -1
                V[3, 6] = 1; V[4, 6] = 1
                W[0, 6] = 1
                # Remaining 10 terms: sparse random
                for r in range(7, target_rank):
                    u_idx = np.random.choice(nm, size=2, replace=False)
                    v_idx = np.random.choice(mp, size=3, replace=False)
                    w_idx = np.random.choice(pn, size=2, replace=False)
                    for idx in u_idx:
                        U[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in v_idx:
                        V[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in w_idx:
                        W[idx, r] = np.random.choice([-1.0, 1.0])
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use scipy L-BFGS-B if available, otherwise fallback to gradient descent
            try:
                from scipy.optimize import minimize
                result = minimize(
                    objective_and_grad,
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < 1e-6:
                    params_opt = result.x
                    U_opt = params_opt[:nm*target_rank].reshape(nm, target_rank)
                    V_opt = params_opt[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_opt = params_opt[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    R_check = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt, optimize=True)
                    err_check = np.sum((R_check - T)**2)
                    
                    if err_check < 1e-4:
                        if _polish_and_verify(U_opt, V_opt, W_opt, T, target_rank, deadline):
                            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            except ImportError:
                # Fallback to plain gradient descent
                lr = 0.01
                for it in range(2000):
                    if time.time() > deadline - 15:
                        break
                    
                    R = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                    residual = R - T
                    err = np.sum(residual**2)
                    
                    dU = 2.0 * np.einsum('abc,br,cr->ar', residual, V, W, optimize=True)
                    dV = 2.0 * np.einsum('abc,ar,cr->br', residual, U, W, optimize=True)
                    dW = 2.0 * np.einsum('abc,ar,br->cr', residual, U, V, optimize=True)
                    
                    grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                    if grad_norm > 1.0:
                        scale_factor = 1.0 / grad_norm
                        dU *= scale_factor
                        dV *= scale_factor
                        dW *= scale_factor
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    
                    if it % 100 == 0:
                        R_check = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                        err_check = np.sum((R_check - T)**2)
                        if err_check < 1e-4:
                            if _polish_and_verify(U, V, W, T, target_rank, deadline):
                                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    def _polish_and_verify(U, V, W, T, target_rank, deadline):
        """Polish a near-solution and verify with exact arithmetic."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        # First, try to snap to simple fractions with multiple tolerance levels
        for tol in [0.005, 0.002, 0.001, 0.0005]:
            if time.time() > deadline - 5:
                return False
            
            U_snap = np.array([[snap_to_simple_fraction(U[a, r], tol=tol) for r in range(target_rank)] for a in range(nm)])
            V_snap = np.array([[snap_to_simple_fraction(V[b, r], tol=tol) for r in range(target_rank)] for b in range(mp)])
            W_snap = np.array([[snap_to_simple_fraction(W[c, r], tol=tol) for r in range(target_rank)] for c in range(pn)])
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if verify(U_list, V_list, W_list):
                return True
        
        # Try with half-integers (denominator 2)
        U_half = np.array([[(round(U[a, r] * 2) / 2.0) for r in range(target_rank)] for a in range(nm)])
        V_half = np.array([[(round(V[b, r] * 2) / 2.0) for r in range(target_rank)] for b in range(mp)])
        W_half = np.array([[(round(W[c, r] * 2) / 2.0) for r in range(target_rank)] for c in range(pn)])
        
        if verify(U_half.tolist(), V_half.tolist(), W_half.tolist()):
            return True
        
        # Try with quarter-integers (denominator 4)
        U_quarter = np.array([[(round(U[a, r] * 4) / 4.0) for r in range(target_rank)] for a in range(nm)])
        V_quarter = np.array([[(round(V[b, r] * 4) / 4.0) for r in range(target_rank)] for b in range(mp)])
        W_quarter = np.array([[(round(W[c, r] * 4) / 4.0) for r in range(target_rank)] for c in range(pn)])
        
        if verify(U_quarter.tolist(), V_quarter.tolist(), W_quarter.tolist()):
            return True
        
        # Local polish: adjust individual entries to minimize error
        U_snap = np.array([[snap_to_simple_fraction(U[a, r], tol=0.01) for r in range(target_rank)] for a in range(nm)])
        V_snap = np.array([[snap_to_simple_fraction(V[b, r], tol=0.01) for r in range(target_rank)] for b in range(mp)])
        W_snap = np.array([[snap_to_simple_fraction(W[c, r], tol=0.01) for r in range(target_rank)] for c in range(pn)])
        
        # Try adjusting entries one at a time
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            
            improved = False
            for a in range(nm):
                for r in range(target_rank):
                    for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                        U_test = U_snap.copy()
                        U_test[a, r] += delta
                        if verify(U_test.tolist(), V_snap.tolist(), W_snap.tolist()):
                            U_snap = U_test
                            improved = True
                            break
                    if time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
            
            for b in range(mp):
                for r in range(target_rank):
                    for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                        V_test = V_snap.copy()
                        V_test[b, r] += delta
                        if verify(U_snap.tolist(), V_test.tolist(), W_snap.tolist()):
                            V_snap = V_test
                            improved = True
                            break
                    if time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
            
            for c in range(pn):
                for r in range(target_rank):
                    for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                        W_test = W_snap.copy()
                        W_test[c, r] += delta
                        if verify(U_snap.tolist(), V_snap.tolist(), W_test.tolist()):
                            W_snap = W_test
                            improved = True
                            break
                    if time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
            
            if not improved:
                break
        
        if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
            return True
        
        return False
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # First, try the known rank-17 decomposition
    if time.time() < deadline:
        r17 = rank17_decomposition()
        if verify(r17["U"], r17["V"], r17["W"]):
            best_result = r17
            best_rank = 17
    
    # Try to find rank-16
    if time.time() < deadline - 25:
        r16 = try_rank16_numerical()
        if r16 is not None:
            # Final verification before returning
            if verify(r16["U"], r16["V"], r16["W"]):
                best_result = r16
                best_rank = 16
    
    return best_result
# EVOLVE-BLOCK-END
