"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s out of 120s budget
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    Nm = n * m
    Mp = m * p
    Pn = p * n
    T = np.zeros((Nm, Mp, Pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n * m * p  # start with schoolbook
    best_result = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions using random initialization + gradient descent
    # We'll try ranks from 79 down to some lower bound
    # Theoretical lower bound for 4x4x5: at least max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But realistically, the tensor rank is much higher.
    
    np.random.seed(42)
    
    for target_rank in range(79, 30, -1):
        if time.time() > deadline:
            break
        
        success = False
        for trial in range(30):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(Nm, target_rank) * 0.5
            V = np.random.randn(Mp, target_rank) * 0.5
            W = np.random.randn(Pn, target_rank) * 0.5
            
            # Normalize columns to help convergence
            for c in range(target_rank):
                nu = np.linalg.norm(U[:, c])
                nv = np.linalg.norm(V[:, c])
                nw = np.linalg.norm(W[:, c])
                if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                    U[:, c] /= (nu ** (1/3))
                    V[:, c] /= (nv ** (1/3))
                    W[:, c] /= (nw ** (1/3))
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute residual: R[i,j,k] = T[i,j,k] - sum_c U[i,c]*V[j,c]*W[k,c]
                # Efficiently: R = T - einsum('ic,jc,kc->ijk', U, V, W)
                approx = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
                residual = T - approx
                
                # Gradients
                # dL/dU[i,c] = -sum_{j,k} residual[i,j,k] * V[j,c] * W[k,c]
                # dL/dV[j,c] = -sum_{i,k} residual[i,j,k] * U[i,c] * W[k,c]
                # dL/dW[k,c] = -sum_{i,j} residual[i,j,k] * U[i,c] * V[j,c]
                
                # Use vectorized computation
                # gradU[i,c] = -sum_{j,k} residual[i,j,k] * V[j,c] * W[k,c]
                # = -einsum('ijk,jc,kc->ic', residual, V, W)
                gradU = np.einsum('ijk,jc,kc->ic', residual, V, W, optimize=True)
                gradV = np.einsum('ijk,ic,kc->jc', residual, U, W, optimize=True)
                gradW = np.einsum('ijk,ic,jc->kc', residual, U, V, optimize=True)
                
                # Scale learning rate by gradient norm
                gnorm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
                if gnorm < 1e-15:
                    break
                
                step_size = lr / max(gnorm, 1e-10)
                
                U -= step_size * gradU
                V -= step_size * gradV
                W -= step_size * gradW
                
                # Check convergence
                if step % 100 == 0:
                    rnorm = np.linalg.norm(residual)
                    if rnorm < 1e-6:
                        success = True
                        break
                    if rnorm < 1e-3:
                        # Try to polish
                        for polish_step in range(500):
                            approx = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
                            residual = T - approx
                            rnorm = np.linalg.norm(residual)
                            if rnorm < 1e-8:
                                success = True
                                break
                            if rnorm < 1e-4:
                                success = True
                                break
                            gradU = np.einsum('ijk,jc,kc->ic', residual, V, W, optimize=True)
                            gradV = np.einsum('ijk,ic,kc->jc', residual, U, W, optimize=True)
                            gradW = np.einsum('ijk,ic,jc->kc', residual, U, V, optimize=True)
                            gnorm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
                            if gnorm < 1e-15:
                                break
                            ps = 0.001 / max(gnorm, 1e-10)
                            U -= ps * gradU
                            V -= ps * gradV
                            W -= ps * gradW
                        if success:
                            break
            
            if success:
                # Verify and construct exact-ish result
                approx = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
                final_resid = np.abs(T - approx)
                if np.max(final_resid) < 1e-4:
                    best_rank = target_rank
                    best_result = _to_list_result(U, V, W)
                    break
        
        if success:
            # Try one rank lower
            continue
        else:
            # If we failed at this rank, lower ranks are unlikely to succeed easily
            # but let's try a couple more
            for tr2 in range(target_rank - 1, max(target_rank - 5, 30), -1):
                if time.time() > deadline:
                    break
                s2 = False
                for trial2 in range(10):
                    if time.time() > deadline:
                        break
                    U2 = np.random.randn(Nm, tr2) * 0.5
                    V2 = np.random.randn(Mp, tr2) * 0.5
                    W2 = np.random.randn(Pn, tr2) * 0.5
                    lr2 = 0.01
                    for step2 in range(1000):
                        if time.time() > deadline:
                            break
                        approx2 = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
                        resid2 = T - approx2
                        gradU2 = np.einsum('ijk,jc,kc->ic', resid2, V2, W2, optimize=True)
                        gradV2 = np.einsum('ijk,ic,kc->jc', resid2, U2, W2, optimize=True)
                        gradW2 = np.einsum('ijk,ic,jc->kc', resid2, U2, V2, optimize=True)
                        gn2 = np.sqrt(np.sum(gradU2**2) + np.sum(gradV2**2) + np.sum(gradW2**2))
                        if gn2 < 1e-15:
                            break
                        ss2 = lr2 / max(gn2, 1e-10)
                        U2 -= ss2 * gradU2
                        V2 -= ss2 * gradV2
                        W2 -= ss2 * gradW2
                        if step2 % 200 == 0:
                            rn2 = np.linalg.norm(resid2)
                            if rn2 < 1e-6:
                                s2 = True
                                break
                    if s2:
                        break
                if s2:
                    approx2 = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
                    if np.max(np.abs(T - approx2)) < 1e-4:
                        best_rank = tr2
                        best_result = _to_list_result(U2, V2, W2)
                        break
            break
    
    return best_result


def _schoolbook(n, m, p):
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


def _to_list_result(U, V, W):
    """Convert numpy arrays to lists, rounding small values to 0 and keeping precision."""
    def clean(arr):
        # Round to 6 decimal places, set tiny values to 0
        arr = np.round(arr, 6)
        arr[np.abs(arr) < 1e-6] = 0.0
        return arr.tolist()
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
# EVOLVE-BLOCK-END
