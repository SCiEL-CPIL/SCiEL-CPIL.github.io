import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill

    # Target tensor T[i, j, k] = 1 for all i,j,k
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    best_rank = 17
    best_U, best_V, best_W = None, None, None

    # Build rank-17 Strassen-based solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U(i, j, r, val): U17[i * m + j, r] = val
    def set_V(j, k, r, val): V17[j * p + k, r] = val
    def set_W(k, i, r, val): W17[k * n + i, r] = val

    # Strassen for 2x2 blocks (columns 0-6 handle i=0,1; j=0,1; k=0,1)
    # Standard Strassen: M1=(A11+A22)(B11+B22), M2=(A21+A22)B11,
    # M3=A11(B11-B22), M4=A22(B22-B11), M5=(A11+A12)B22,
    # M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6

    # M1: (U[0,0]+U[1,1])*(V[0,0]+V[1,1]) -> contributes to W[0,0] and W[1,1]
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)

    # M2: (U[1,0]+U[1,1])*V[0,0] -> contributes to W[1,0] and W[1,1]
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(1, 0, 1, 1); set_W(1, 1, 1, -1)

    # M3: U[0,0]*(V[0,0]-V[1,1]) -> contributes to W[0,0] and W[0,1]
    set_U(0, 0, 2, 1)
    set_V(0, 0, 2, 1); set_V(1, 1, 2, -1)
    set_W(0, 0, 2, 1); set_W(0, 1, 2, -1)

    # M4: U[1,1]*(V[1,1]-V[0,0]) -> contributes to W[1,1] and W[1,0]
    set_U(1, 1, 3, 1)
    set_V(1, 1, 3, 1); set_V(0, 0, 3, -1)
    set_W(1, 1, 3, 1); set_W(1, 0, 3, -1)

    # M5: (U[0,0]+U[0,1])*V[1,1] -> contributes to W[0,0] and W[1,0]
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, 1); set_W(1, 0, 4, -1)

    # M6: (U[1,0]-U[0,0])*(V[0,0]+V[0,1]) -> contributes to W[1,0] and W[1,1]
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 0, 5, 1); set_W(1, 1, 5, -1)

    # M7: (U[0,1]-U[1,1])*(V[1,0]+V[1,1]) -> contributes to W[0,1] and W[1,1]
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 1, 6, 1); set_W(1, 1, 6, -1)

    # Direct terms for k=2 (column 2 of B): U[i,j] * B[j,2]
    set_U(0, 0, 7, 1); set_V(0, 2, 7, 1); set_W(2, 0, 7, 1)
    set_U(0, 1, 8, 1); set_V(1, 2, 8, 1); set_W(2, 0, 8, 1)
    set_U(1, 0, 9, 1); set_V(0, 2, 9, 1); set_W(2, 1, 9, 1)
    set_U(1, 1, 10, 1); set_V(1, 2, 10, 1); set_W(2, 1, 10, 1)

    # Direct terms for j=2 (column 2 of A): U[i,2] * B[2,k]
    set_U(0, 2, 11, 1); set_V(2, 0, 11, 1); set_W(0, 0, 11, 1)
    set_U(0, 2, 12, 1); set_V(2, 1, 12, 1); set_W(1, 0, 12, 1)
    set_U(0, 2, 13, 1); set_V(2, 2, 13, 1); set_W(2, 0, 13, 1)
    set_U(1, 2, 14, 1); set_V(2, 0, 14, 1); set_W(0, 1, 14, 1)
    set_U(1, 2, 15, 1); set_V(2, 1, 15, 1); set_W(1, 1, 15, 1)
    set_U(1, 2, 16, 1); set_V(2, 2, 16, 1); set_W(2, 1, 16, 1)

    # Verify rank-17 solution
    def compute_tensor(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return T

    T17 = compute_tensor(U17, V17, W17, 17)
    if np.allclose(T17, np.ones((n, m, p)), atol=1e-10):
        best_U, best_V, best_W = U17, V17, W17
    else:
        # Fallback: schoolbook (rank 18)
        best_rank = 18
        U18 = np.zeros((n * m, 18))
        V18 = np.zeros((m * p, 18))
        W18 = np.zeros((p * n, 18))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U18[i * m + j, r] = 1
                    V18[j * p + k, r] = 1
                    W18[k * n + i, r] = 1
        best_U, best_V, best_W = U18, V18, W18

    def error_and_grad(U, V, W, rank):
        """Vectorized error and gradient computation."""
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        R = T - 1.0  # residual
        
        err = np.sum(R ** 2)
        
        # Gradients:
        # dE/dU[i,j,r] = 2 * sum_k R[i,j,k] * V[j,k,r] * W[k,i,r]
        # dE/dV[j,k,r] = 2 * sum_i R[i,j,k] * U[i,j,r] * W[k,i,r]
        # dE/dW[k,i,r] = 2 * sum_j R[i,j,k] * U[i,j,r] * V[j,k,r]
        
        dU_t = 2.0 * np.einsum('ijk,jkr,kir->ijr', R, V_t, W_t)
        dV_t = 2.0 * np.einsum('ijk,ijr,kir->jkr', R, U_t, W_t)
        dW_t = 2.0 * np.einsum('ijk,ijr,jkr->kir', R, U_t, V_t)
        
        return err, dU_t.reshape(n*m, rank), dV_t.reshape(m*p, rank), dW_t.reshape(p*n, rank)

    def is_exact(U, V, W, rank, tol=1e-9):
        T = compute_tensor(U, V, W, rank)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def snap_half(U, V, W):
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    # Try rank 16: start from rank-17, zero out one column, optimize
    for target_rank in [16, 15]:
        if time.time() > deadline:
            break
            
        found = False
        num_seeds = 30 if target_rank == 16 else 15
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + target_rank)
            
            # Strategy: start from rank-17, zero a random column, then optimize
            U = U17[:, :target_rank].copy()
            V = V17[:, :target_rank].copy()
            W = W17[:, :target_rank].copy()
            
            # Add small random perturbation
            U += np.random.randn(n * m, target_rank) * 0.05
            V += np.random.randn(m * p, target_rank) * 0.05
            W += np.random.randn(p * n, target_rank) * 0.05
            
            # Adam-like optimizer
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            b1, b2, eps = 0.9, 0.999, 1e-8
            t_adam = 0
            lr = 0.05
            
            for step in range(3000):
                if time.time() > deadline:
                    break
                
                err, dU, dV, dW = error_and_grad(U, V, W, target_rank)
                
                if err < 1e-14:
                    break
                
                t_adam += 1
                mU = b1*mU + (1-b1)*dU; vU = b2*vU + (1-b2)*dU**2
                mV = b1*mV + (1-b1)*dV; vV = b2*vV + (1-b2)*dV**2
                mW = b1*mW + (1-b1)*dW; vW = b2*vW + (1-b2)*dW**2
                mUh = mU/(1-b1**t_adam); vUh = vU/(1-b2**t_adam)
                mVh = mV/(1-b1**t_adam); vVh = vV/(1-b2**t_adam)
                mWh = mW/(1-b1**t_adam); vWh = vW/(1-b2**t_adam)
                
                U -= lr * mUh / (np.sqrt(vUh) + eps)
                V -= lr * mVh / (np.sqrt(vVh) + eps)
                W -= lr * mWh / (np.sqrt(vWh) + eps)
                
                # Periodic snap to half-integers
                if step % 200 == 199:
                    U, V, W = snap_half(U, V, W)
                    err, _, _, _ = error_and_grad(U, V, W, target_rank)
                    if err < 1e-10:
                        break
                
                # Check exactness
                if step % 50 == 0 and is_exact(U, V, W, target_rank):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break

    # Final polish: if we have a near-exact solution, do high-precision Newton steps
    if best_rank < 17:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        rank = best_rank
        for _ in range(50):
            if time.time() > deadline:
                break
            err, dU, dV, dW = error_and_grad(U, V, W, rank)
            if err < 1e-20:
                break
            # Simple gradient step with tiny lr
            U -= 0.001 * dU
            V -= 0.001 * dV
            W -= 0.001 * dW
            U, V, W = snap_half(U, V, W)
            if is_exact(U, V, W, rank):
                best_U, best_V, best_W = U, V, W
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
