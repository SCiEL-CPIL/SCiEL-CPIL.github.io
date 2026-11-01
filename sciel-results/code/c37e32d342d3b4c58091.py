import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build rank-32 solution as fallback
    def build_rank32():
        terms = []
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            U1 = {0: 1, 3: 1}
            V1 = {k0: 1, 9+k1: 1}
            W1 = {2*k0: 1, 2*k1+1: 1}
            terms.append((U1, V1, W1))
            U2 = {2: 1, 3: 1}
            V2 = {k0: 1}
            W2 = {2*k0+1: 1, 2*k1+1: -1}
            terms.append((U2, V2, W2))
            U3 = {0: 1}
            V3 = {k1: 1, 9+k1: -1}
            W3 = {2*k1: 1, 2*k1+1: 1}
            terms.append((U3, V3, W3))
            U4 = {3: 1}
            V4 = {9+k0: 1, k0: -1}
            W4 = {2*k0: 1, 2*k0+1: 1}
            terms.append((U4, V4, W4))
            U5 = {0: 1, 1: 1}
            V5 = {9+k1: 1}
            W5 = {2*k0: -1, 2*k1: 1}
            terms.append((U5, V5, W5))
            U6 = {2: 1, 0: -1}
            V6 = {k0: 1, k1: 1}
            W6 = {2*k1+1: 1}
            terms.append((U6, V6, W6))
            U7 = {1: 1, 3: -1}
            V7 = {9+k0: 1, 9+k1: 1}
            W7 = {2*k0: 1}
            terms.append((U7, V7, W7))
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        
        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    u_idx = i*2 + j
                    v_idx = j*9 + k
                    w_idx = k*2 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    U32, V32, W32 = build_rank32()
    assert verify(U32, V32, W32)
    
    # Try rank 31: perturb the rank-32 solution and try to collapse one dimension
    # Use a simple approach: for each term t, try setting U[:,t]=0, V[:,t]=0, W[:,t]=0
    # and see if we can adjust other terms to compensate.
    
    # More practically: try a 2x2x3 block with rank 11 for the last 3 columns.
    # If rank(2x2x3) = 11, then total = 3*7 + 11 = 32. No improvement.
    # If rank(2x2x3) = 10, then total = 3*7 + 10 = 31. Improvement!
    
    # Let me try to find a rank-10 decomposition for 2x2x3.
    # This is a small problem that we can search over.
    
    # For 2x2x3, the tensor T has shape (4, 6, 6) with T[i*2+j, j*3+k, k*2+i] = 1
    # We need U (4x10), V (6x10), W (6x10) such that:
    # sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    # This is 12 equations with 10*4+10*6+10*6 = 40+60+60 = 160 unknowns.
    # Should be feasible to search.
    
    # Use random initialization and gradient-free optimization
    np.random.seed(42)
    
    def try_rank10_2x2x3(max_iters=5000):
        """Try to find rank-10 decomposition for 2x2x3"""
        U = np.random.randn(4, 10) * 0.5
        V = np.random.randn(6, 10) * 0.5
        W = np.random.randn(6, 10) * 0.5
        
        def loss():
            total = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        u_idx = i*2 + j
                        v_idx = j*3 + k
                        w_idx = k*2 + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        total += (val - 1.0)**2
            return total
        
        best_loss = loss()
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.1
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Compute gradients numerically (expensive but simple)
            # Actually, let's use a simpler approach: coordinate descent
            
            # Try updating U
            for i in range(4):
                for r in range(10):
                    old = U[i, r]
                    U[i, r] = old + 0.1
                    l1 = loss()
                    U[i, r] = old - 0.1
                    l2 = loss()
                    U[i, r] = old
                    if l1 < l2:
                        U[i, r] = old + 0.1
                    elif l2 < l1:
                        U[i, r] = old - 0.1
                    else:
                        break
                if time.time() > deadline - 5:
                    break
            
            l = loss()
            if l < best_loss:
                best_loss = l
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-8:
                break
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W, best_loss
        return None, None, None, best_loss
    
    # This coordinate descent is too slow. Let me try a different approach.
    # Use the fact that we can fix two matrices and solve for the third.
    
    # Actually, let me just try a few random initializations with a simple
    # multiplicative update rule.
    
    def try_rank10_fast():
        """Try rank-10 for 2x2x3 with simple updates"""
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            np.random.seed(attempt)
            U = np.random.randn(4, 10) * 0.3
            V = np.random.randn(6, 10) * 0.3
            W = np.random.randn(6, 10) * 0.3
            
            for it in range(200):
                if time.time() > deadline - 5:
                    return None, None, None, float('inf')
                # Update U: for each (i,j) and r, U[i*2+j, r] should be
                # adjusted to make sum_r U[i*2+j,r]*V[j*3+k,r]*W[k*2+i,r] = 1
                # This is still nonlinear. Let's use a different strategy.
                
                # Fix V and W, solve for U:
                # For each (i,j): sum_r U[i*2+j, r] * (V[j*3+k, r] * W[k*2+i, r]) = 1
                # This gives 3 equations (for k=0,1,2) per (i,j), so 12 equations
                # for 40 unknowns in U. Underdetermined!
                
                # Let's fix U and V, solve for W:
                # For each (k,i): sum_r W[k*2+i, r] * (U[i*2+j, r] * V[j*3+k, r]) = 1
                # For fixed (k,i), we have 2 equations (j=0,1) for 10 unknowns in W[k*2+i,:]
                # Still underdetermined.
                
                # Let's just do a simple iterative approach.
                for _ in range(3):
                    # Update W given U,V
                    for k in range(3):
                        for i in range(2):
                            w_idx = k*2 + i
                            # Need: sum_r U[i*2+j, r] * V[j*3+k, r] * W[w_idx, r] = 1 for j=0,1
                            # This is a 2x10 system: A @ W[w_idx,:] = [1, 1]
                            # where A[j, r] = U[i*2+j, r] * V[j*3+k, r]
                            A = np.array([[U[i*2+j, r] * V[j*3+k, r] for r in range(10)] for j in range(2)])
                            b = np.array([1.0, 1.0])
                            # Solve least squares
                            W[w_idx, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Update U given V,W
                    for i in range(2):
                        for j in range(2):
                            u_idx = i*2 + j
                            A = np.array([[V[j*3+k, r] * W[k*2+i, r] for r in range(10)] for k in range(3)])
                            b = np.array([1.0, 1.0, 1.0])
                            U[u_idx, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Update V given U,W
                    for j in range(2):
                        for k in range(3):
                            v_idx = j*3 + k
                            A = np.array([[U[i*2+j, r] * W[k*2+i, r] for r in range(10)] for i in range(2)])
                            b = np.array([1.0, 1.0])
                            V[v_idx, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check loss
                loss = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            u_idx = i*2 + j
                            v_idx = j*3 + k
                            w_idx = k*2 + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            loss += (val - 1.0)**2
                
                if loss < 1e-8:
                    return U, V, W, loss
        
        return None, None, None, float('inf')
    
    U10, V10, W10, loss10 = try_rank10_fast()
    
    if U10 is not None and loss10 < 1e-8:
        # Found rank-10 for 2x2x3! Build full rank-31 solution.
        # First 6 columns: 3 Strassen blocks (rank 7 each) = 21
        # Last 3 columns (k=6,7,8): rank-10 block
        terms = []
        for block_idx in range(3):
            k0 = 2 * block_idx
            k1 = k0 + 1
            U1 = {0: 1, 3: 1}
            V1 = {k0: 1, 9+k1: 1}
            W1 = {2*k0: 1, 2*k1+1: 1}
            terms.append((U1, V1, W1))
            U2 = {2: 1, 3: 1}
            V2 = {k0: 1}
            W2 = {2*k0+1: 1, 2*k1+1: -1}
            terms.append((U2, V2, W2))
            U3 = {0: 1}
            V3 = {k1: 1, 9+k1: -1}
            W3 = {2*k1: 1, 2*k1+1: 1}
            terms.append((U3, V3, W3))
            U4 = {3: 1}
            V4 = {9+k0: 1, k0: -1}
            W4 = {2*k0: 1, 2*k0+1: 1}
            terms.append((U4, V4, W4))
            U5 = {0: 1, 1: 1}
            V5 = {9+k1: 1}
            W5 = {2*k0: -1, 2*k1: 1}
            terms.append((U5, V5, W5))
            U6 = {2: 1, 0: -1}
            V6 = {k0: 1, k1: 1}
            W6 = {2*k1+1: 1}
            terms.append((U6, V6, W6))
            U7 = {1: 1, 3: -1}
            V7 = {9+k0: 1, 9+k1: 1}
            W7 = {2*k0: 1}
            terms.append((U7, V7, W7))
        
        # Add rank-10 terms for k=6,7,8
        # Local indices: k_local = 0,1,2 correspond to global k=6,7,8
        # U local indices: 0,1,2,3 (same as global)
        # V local indices: j*3+k_local, global: j*9 + (6+k_local)
        # W local indices: k_local*2+i, global: (6+k_local)*2+i
        
        for r in range(10):
            Ud = {}
            Vd = {}
            Wd = {}
            for idx in range(4):
                if U10[idx, r] != 0:
                    Ud[idx] = U10[idx, r]
            for j in range(2):
                for k_local in range(3):
                    v_local = j*3 + k_local
                    v_global = j*9 + (6 + k_local)
                    if V10[v_local, r] != 0:
                        Vd[v_global] = V10[v_local, r]
            for k_local in range(3):
                for i in range(2):
                    w_local = k_local*2 + i
                    w_global = (6+k_local)*2 + i
                    if W10[w_local, r] != 0:
                        Wd[w_global] = W10[w_local, r]
            terms.append((Ud, Vd, Wd))
        
        rank = len(terms)
        assert rank == 31, f"Expected rank 31, got {rank}"
        
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 32
    U_list = U32.tolist()
    V_list = V32.tolist()
    W_list = W32.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
