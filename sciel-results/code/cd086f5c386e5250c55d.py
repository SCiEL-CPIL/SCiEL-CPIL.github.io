import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    # Schoolbook decomposition (rank = n*m*p = 60)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    # Strassen-based block decomposition
    # Split A (2x5) into [A1(2x2), A2(2x2), A3(2x1)]
    # Split B (5x6) into [B1(2x6); B2(2x6); B3(1x6)]
    # C = A1*B1 + A2*B2 + A3*B3
    # Each 2x2 * 2x6 can be done with 3 Strassen 2x2x2 blocks (rank 21)
    # A3*B3 is rank 1 (outer product)
    # Total: 21 + 21 + 1 = 43
    def strassen_block_decomposition():
        # Strassen factors for 2x2x2
        # U: (4,7), V: (4,7), W: (4,7)
        # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        sU = np.zeros((4, 7))
        sV = np.zeros((4, 7))
        sW = np.zeros((4, 7))
        
        # P1 = (A00+A11)(B00+B11)
        sU[0,0] = 1; sU[3,0] = 1
        sV[0,0] = 1; sV[3,0] = 1
        sW[0,0] = 1; sW[3,0] = 1
        
        # P2 = (A10+A11)B00
        sU[2,1] = 1; sU[3,1] = 1
        sV[0,1] = 1
        sW[1,1] = 1; sW[3,1] = 1
        
        # P3 = A00(B01-B11)
        sU[0,2] = 1
        sV[1,2] = 1; sV[3,2] = -1
        sW[1,2] = 1; sW[3,2] = -1
        
        # P4 = A11(B00-B01)
        sU[3,3] = 1
        sV[0,3] = 1; sV[1,3] = -1
        sW[0,3] = 1; sW[3,3] = 1
        
        # P5 = (A00+A01)B11
        sU[0,4] = 1; sU[1,4] = 1
        sV[3,4] = 1
        sW[0,4] = -1; sW[3,4] = 1
        
        # P6 = (A10-A00)(B00+B01)
        sU[2,5] = 1; sU[0,5] = -1
        sV[0,5] = 1; sV[1,5] = 1
        sW[2,5] = 1
        
        # P7 = (A01-A11)(B10+B11)
        sU[1,6] = 1; sU[3,6] = -1
        sV[2,6] = 1; sV[3,6] = 1
        sW[0,6] = 1; sW[2,6] = 1
        
        total_rank = 21 + 21 + 1
        R = total_rank
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        col = 0
        # Add Strassen blocks for A1*B1 (j in {0,1}, k in {0..5})
        for k_off in [0, 2, 4]:
            for r in range(7):
                for ui in range(4):
                    il = ui // 2
                    jl = ui % 2
                    if sU[ui, r] != 0:
                        U[il*m + jl, col] = sU[ui, r]
                for vi in range(4):
                    jl = vi // 2
                    kl = vi % 2
                    if sV[vi, r] != 0:
                        V[jl*p + (k_off + kl), col] = sV[vi, r]
                for wi in range(4):
                    kl = wi // 2
                    il = wi % 2
                    if sW[wi, r] != 0:
                        W[(k_off + kl)*n + il, col] = sW[wi, r]
                col += 1
        
        # Add Strassen blocks for A2*B2 (j in {2,3}, k in {0..5})
        for k_off in [0, 2, 4]:
            for r in range(7):
                for ui in range(4):
                    il = ui // 2
                    jl = ui % 2
                    if sU[ui, r] != 0:
                        U[il*m + (2 + jl), col] = sU[ui, r]
                for vi in range(4):
                    jl = vi // 2
                    kl = vi % 2
                    if sV[vi, r] != 0:
                        V[(2 + jl)*p + (k_off + kl), col] = sV[vi, r]
                for wi in range(4):
                    kl = wi // 2
                    il = wi % 2
                    if sW[wi, r] != 0:
                        W[(k_off + kl)*n + il, col] = sW[wi, r]
                col += 1
        
        # A3*B3: rank 1
        for gi in range(n):
            U[gi*m + 4, col] = 1
        for gk in range(p):
            V[4*p + gk, col] = 1
        for gi in range(n):
            for gk in range(p):
                W[gk*n + gi, col] = 1
        col += 1
        
        return {'U': U, 'V': V, 'W': W}

    # Verify decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    def verify(result):
        U = result['U']
        V = result['V']
        W = result['W']
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Try Strassen block decomposition first
    best_result = None
    best_rank = float('inf')
    
    try:
        result = strassen_block_decomposition()
        if verify(result):
            best_result = result
            best_rank = result['U'].shape[1]
    except Exception:
        pass

    # If Strassen didn't work or we have time, try to improve with ALS
    if time.time() < deadline - 20:
        try:
            # Try to find a lower rank decomposition using ALS
            # Start from the Strassen decomposition and try to reduce rank
            if best_result is not None:
                U0 = best_result['U'].copy()
                V0 = best_result['V'].copy()
                W0 = best_result['W'].copy()
                R0 = U0.shape[1]
                
                # Try ranks below current best
                for target_rank in range(max(1, best_rank - 15), best_rank):
                    if time.time() > deadline - 10:
                        break
                    
                    # Initialize with random projection
                    U = U0[:, :target_rank].copy()
                    V = V0[:, :target_rank].copy()
                    W = W0[:, :target_rank].copy()
                    
                    # ALS iterations
                    converged = False
                    for it in range(50):
                        if time.time() > deadline - 10:
                            break
                        
                        # Update U: for each (i,j), U[i*m+j,:] should satisfy
                        # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
                        # Fix V, W, solve for U
                        for i in range(n):
                            for j in range(m):
                                # Build system: for each k, sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                                # This is a linear system in U[i*m+j,:]
                                A_sys = np.zeros((p, target_rank))
                                b_sys = np.ones(p)
                                for k in range(p):
                                    A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                                # Solve least squares
                                try:
                                    U[i*m+j, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                                except:
                                    pass
                        
                        # Update V
                        for j in range(m):
                            for k in range(p):
                                A_sys = np.zeros((n, target_rank))
                                b_sys = np.ones(n)
                                for i in range(n):
                                    A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                                try:
                                    V[j*p+k, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                                except:
                                    pass
                        
                        # Update W
                        for k in range(p):
                            for i in range(n):
                                A_sys = np.zeros((m, target_rank))
                                b_sys = np.ones(m)
                                for j in range(m):
                                    A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                                try:
                                    W[k*n+i, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                                except:
                                    pass
                        
                        if it % 10 == 9:
                            # Check convergence
                            max_err = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        s = 0.0
                                        for r in range(target_rank):
                                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                        max_err = max(max_err, abs(s - 1.0))
                            if max_err < 1e-8:
                                converged = True
                                break
                    
                    if converged:
                        result = {'U': U, 'V': V, 'W': W}
                        if verify(result):
                            best_result = result
                            best_rank = target_rank
                            break
        except Exception:
            pass

    # Fallback to schoolbook if nothing else worked
    if best_result is None:
        best_result = schoolbook()
        best_rank = n * m * p

    return best_result
