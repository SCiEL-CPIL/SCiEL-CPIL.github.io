"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    deadline = time.time() + 115
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U[nm, r], V[mp, r], W[pn, r] such that
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Known results: For 2x3 x 3x5, the lower bound is max(2*3, 3*5, 5*2) = max(6,15,10) = 15
    # But actually we need to think about this more carefully.
    # The rank of matrix multiplication tensor for n x m times m x p is at least max(n*m, m*p, p*n)
    # For 2,3,5: max(6, 15, 10) = 15
    # Upper bound is n*m*p = 30
    # Strassen-like methods might help, but for small dimensions, let's try to find a low-rank decomposition.
    
    # Actually, the exact rank for 2x3 x 3x5 matrix multiplication...
    # Let me think about this differently. We can try rank values from the theoretical minimum upward.
    
    # Theoretical lower bound: The tensor T_{ijk} has the property that fixing k gives a 2x3 matrix,
    # fixing i gives a 3x5 matrix, fixing j gives a 2x5 matrix.
    # The rank must be at least max(6, 15, 10) = 15.
    
    # Let's try to find a decomposition with rank as low as possible.
    # We'll use a systematic search approach: try to construct U, V, W with decreasing rank.
    
    # For small problems, we can try to use the fact that we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # and sum_r U[i,j,r] * V[j,k,r] * W[k',i,r] = 0 for k != k' (but wait, that's not quite right)
    
    # Actually the condition is simply:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a parameterized approach. Since n=2 is small, let's exploit that.
    
    # Strategy: Try ranks from 15 upward, using random initialization + optimization.
    # But with 120s time limit, we need to be smart.
    
    # Alternative: Use the known result that for matrix multiplication A(2x3) * B(3x5),
    # we can potentially use rank less than 30.
    
    # Let me try a direct construction approach for specific ranks.
    
    # First, let's try the trivial upper bound of 30 and then attempt to reduce.
    # Actually, let's use numpy to try optimization-based approaches for ranks near the lower bound.
    
    best_result = None
    
    # Try ranks from 15 upward
    for rank in range(15, 31):
        if time.time() > deadline - 10:
            break
        if best_result is not None:
            break
            
        success = False
        attempts = 50 if rank <= 20 else 100
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            np.random.seed(42 + attempt * 137)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Optimize using gradient descent on the residual
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                    
                # Compute tensor: T_hat[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Residual: R[i,j,k] = T_hat[i,j,k] - 1
                # Gradient w.r.t. U[i*m+j,r] = sum_k V[j*p+k,r] * W[k*n+i,r] * R[i,j,k]
                
                # Build the full tensor
                T_hat = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_hat[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                
                R = T_hat - 1.0
                if np.max(np.abs(R)) < 1e-12:
                    success = True
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = R[i,j,k]
                            # dT/dU[i*m+j, r'] = V[j*p+k, r'] * W[k*n+i, r']
                            grad_U[i*m+j] += r * V[j*p+k] * W[k*n+i]
                            # dT/dV[j*p+k, r'] = U[i*m+j, r'] * W[k*n+i, r']
                            grad_V[j*p+k] += r * U[i*m+j] * W[k*n+i]
                            # dT/dW[k*n+i, r'] = U[i*m+j, r'] * V[j*p+k, r']
                            grad_W[k*n+i] += r * U[i*m+j] * V[j*p+k]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if it % 20 == 19:
                    err = np.max(np.abs(T_hat - 1.0))
                    if err > 1.0:
                        lr *= 0.5
                    elif err < 0.01:
                        lr = min(lr * 1.5, 0.1)
            
            if success:
                # Verify and convert
                U_int, V_int, W_int = _try_convert(U, V, W, nm, mp, pn, rank)
                if U_int is not None:
                    best_result = (rank, U_int, V_int, W_int)
                    break
                
                # Try with different scaling
                for scale in [0.5, 2.0, 0.1, 10.0]:
                    U2, V2, W2 = _try_convert(U * scale, V / scale, W, nm, mp, pn, rank)
                    if U2 is not None:
                        best_result = (rank, U2, V2, W2)
                        success = True
                        break
                    U2, V2, W2 = _try_convert(U, V * scale, W / scale, nm, mp, pn, rank)
                    if U2 is not None:
                        best_result = (rank, U2, V2, W2)
                        success = True
                        break
                    if success:
                        break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: return the trivial rank-30 decomposition
    rank = nm * mp // p  # This isn't right, just use full rank
    rank = n * m * p  # 30
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_convert(U, V, W, nm, mp, pn, rank):
    """Try to convert float arrays to integer/half-integer lists."""
    try:
        # Check if values are close to integers or half-integers
        U_list = []
        for row in U:
            U_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-8:
                    U_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-8:
                    if nearest_half == int(nearest_half):
                        U_row.append(int(nearest_half))
                    else:
                        U_row.append(float(nearest_half))
                else:
                    return None
            U_list.append(U_row)
        
        V_list = []
        for row in V:
            V_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-8:
                    V_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-8:
                    if nearest_half == int(nearest_half):
                        V_row.append(int(nearest_half))
                    else:
                        V_row.append(float(nearest_half))
                else:
                    return None
            V_list.append(V_row)
        
        W_list = []
        for row in W:
            W_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-8:
                    W_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-8:
                    if nearest_half == int(nearest_half):
                        W_row.append(int(nearest_half))
                    else:
                        W_row.append(float(nearest_half))
                else:
                    return None
            W_list.append(W_row)
        
        # Verify the decomposition
        n, m, p = 2, 3, 5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if abs(total - 1.0) > 1e-8:
                        return None
        
        return U_list, V_list, W_list
    except:
        return None

# EVOLVE-BLOCK-END
