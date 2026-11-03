import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = 4, 4, 6

    # Try to find a rank-10 decomposition for 2x2x3
    # If successful, use 7 outer steps x 10 inner = 70
    # Otherwise fall back to rank-11 inner = 77 total
    
    def try_rank10_223():
        """Try to find a rank-10 decomposition of 2x2x3 multiplication."""
        # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # U: 4x10, V: 6x10, W: 6x10 (W indexed as [k*2+i, r])
        
        def residuals(params):
            # params: flattened U (40), V (60), W (60) = 160 total
            U = params[0:40].reshape(2, 2, 10)
            V = params[40:100].reshape(2, 3, 10)
            W = params[100:160].reshape(3, 2, 10)  # W[k,i,r]
            
            res = []
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                        res.append(val - 1.0)
            return res
        
        # Initialize with a mix of Strassen-like structure
        np.random.seed(42)
        
        # Start from a rank-11 solution and try to merge
        # Actually, let's just try random initialization with multiple restarts
        best_residual = float('inf')
        best_params = None
        
        for restart in range(20):
            if time.time() > deadline - 30:
                break
            
            # Initialize: use Strassen for first 2 cols, schoolbook for 3rd
            params = np.zeros(160)
            U = np.zeros((2, 2, 10))
            V = np.zeros((2, 3, 10))
            W = np.zeros((3, 2, 10))
            
            # Strassen for cols 0,1 (7 terms)
            # Term 0: (A00+A11)(B00+B11) -> C00+C11
            U[0,0,0] += 1; U[1,1,0] += 1
            V[0,0,0] += 1; V[1,1,0] += 1
            W[0,0,0] += 1; W[1,1,0] += 1
            
            # Term 1: (A10+A11)B00 -> C10-C11
            U[1,0,1] += 1; U[1,1,1] += 1
            V[0,0,1] += 1
            W[1,0,1] += 1; W[1,1,1] -= 1
            
            # Term 2: A00(B01-B11) -> C01+C11
            U[0,0,2] += 1
            V[0,1,2] += 1; V[1,1,2] -= 1
            W[0,1,2] += 1; W[1,1,2] += 1
            
            # Term 3: A11(B10-B00) -> C00+C10
            U[1,1,3] += 1
            V[1,0,3] += 1; V[0,0,3] -= 1
            W[0,0,3] += 1; W[1,0,3] += 1
            
            # Term 4: (A00+A01)B11 -> C01-C00
            U[0,0,4] += 1; U[0,1,4] += 1
            V[1,1,4] += 1
            W[0,1,4] += 1; W[0,0,4] -= 1
            
            # Term 5: (A10-A00)(B00+B01) -> C11
            U[1,0,5] += 1; U[0,0,5] -= 1
            V[0,0,5] += 1; V[0,1,5] += 1
            W[1,1,5] += 1
            
            # Term 6: (A01-A11)(B10+B11) -> C00
            U[0,1,6] += 1; U[1,1,6] -= 1
            V[1,0,6] += 1; V[1,1,6] += 1
            W[0,0,6] += 1
            
            # Terms 7,8,9: for column 2
            # Use schoolbook-like but try to optimize
            U[0,0,7] += 1; V[0,2,7] += 1; W[2,0,7] += 1
            U[0,1,8] += 1; V[1,2,8] += 1; W[2,0,8] += 1
            U[1,0,9] += 1; V[0,2,9] += 1; W[2,1,9] += 1
            # Missing: U[1,1] * V[1,2] * W[2,1]
            # We only have 10 terms, so we need to combine
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Add random perturbation
            params += 0.1 * np.random.randn(160)
            
            try:
                result = least_squares(residuals, params, method='lm', 
                                       max_nfev=5000, xtol=1e-12, ftol=1e-12)
                if result.cost < best_residual:
                    best_residual = result.cost
                    best_params = result.x
            except:
                pass
        
        if best_params is not None and best_residual < 1e-10:
            U = best_params[0:40].reshape(2, 2, 10)
            V = best_params[40:100].reshape(2, 3, 10)
            W = best_params[100:160].reshape(3, 2, 10)
            # Round to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            # Verify after rounding
            max_err = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-8:
                return U, V, W
        return None
    
    # Try to find rank-10 decomposition
    inner_223 = try_rank10_223() if time.time() < deadline - 40 else None
    
    if inner_223 is not None:
        U_in, V_in, W_in = inner_223
        inner_rank = 10
    else:
        # Fallback to rank-11: Strassen (7) + schoolbook (4)
        inner_rank = 11
        U_in = np.zeros((2, 2, 11))
        V_in = np.zeros((2, 3, 11))
        W_in = np.zeros((3, 2, 11))
        
        # Strassen terms 0-6
        U_in[0,0,0] += 1; U_in[1,1,0] += 1
        V_in[0,0,0] += 1; V_in[1,1,0] += 1
        W_in[0,0,0] += 1; W_in[1,1,0] += 1
        
        U_in[1,0,1] += 1; U_in[1,1,1] += 1
        V_in[0,0,1] += 1
        W_in[1,0,1] += 1; W_in[1,1,1] -= 1
        
        U_in[0,0,2] += 1
        V_in[0,1,2] += 1; V_in[1,1,2] -= 1
        W_in[0,1,2] += 1; W_in[1,1,2] += 1
        
        U_in[1,1,3] += 1
        V_in[1,0,3] += 1; V_in[0,0,3] -= 1
        W_in[0,0,3] += 1; W_in[1,0,3] += 1
        
        U_in[0,0,4] += 1; U_in[0,1,4] += 1
        V_in[1,1,4] += 1
        W_in[0,1,4] += 1; W_in[0,0,4] -= 1
        
        U_in[1,0,5] += 1; U_in[0,0,5] -= 1
        V_in[0,0,5] += 1; V_in[0,1,5] += 1
        W_in[1,1,5] += 1
        
        U_in[0,1,6] += 1; U_in[1,1,6] -= 1
        V_in[1,0,6] += 1; V_in[1,1,6] += 1
        W_in[0,0,6] += 1
        
        # Schoolbook terms 7-10 for column 2
        U_in[0,0,7] += 1; V_in[0,2,7] += 1; W_in[2,0,7] += 1
        U_in[0,1,8] += 1; V_in[1,2,8] += 1; W_in[2,0,8] += 1
        U_in[1,0,9] += 1; V_in[0,2,9] += 1; W_in[2,1,9] += 1
        U_in[1,1,10] += 1; V_in[1,2,10] += 1; W_in[2,1,10] += 1
    
    rank = 7 * inner_rank
    
    # Build global decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen steps for 4x4 * 4x6 with 2x2 blocks (A) and 2x3 blocks (B, C)
    outer_steps = [
        (([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)])),
        (([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])),
        (([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)])),
        (([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)])),
        (([(0,1),(1,1)], [(3,1)], [(1,1),(0,-1)])),
        (([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)])),
        (([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]))
    ]
    
    def global_A(block_id, r, s):
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J
    
    def global_B(block_id, s, k):
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K
    
    def global_C(block_id, r, k):
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K
    
    for oi, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base = oi * inner_rank
        for ii in range(inner_rank):
            c = base + ii
            for r in range(2):
                for s in range(2):
                    for (bid, ca) in A_outer:
                        I, J = global_A(bid, r, s)
                        U[I * m + J][c] += U_in[r, s, ii] * ca
            for s in range(2):
                for k in range(3):
                    for (bid, cb) in B_outer:
                        J, K = global_B(bid, s, k)
                        V[J * p + K][c] += V_in[s, k, ii] * cb
            for r in range(2):
                for k in range(3):
                    for (bid, cc) in C_outer:
                        I, K = global_C(bid, r, k)
                        W[K * n + I][c] += W_in[k, r, ii] * cc
    
    # Verification
    if time.time() < deadline - 3.0:
        Um = np.array(U, dtype=np.float64)
        Vm = np.array(V, dtype=np.float64)
        Wm = np.array(W, dtype=np.float64)
        U_r = Um.reshape(n, m, rank)
        V_r = Vm.reshape(m, p, rank)
        W_r = Wm.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        T_target = np.ones((n, m, p))
        if not np.allclose(T_recon, T_target, atol=1e-8):
            # Fallback: schoolbook
            r2 = n * m * p
            U2 = [[0] * r2 for _ in range(n * m)]
            V2 = [[0] * r2 for _ in range(m * p)]
            W2 = [[0] * r2 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U2[i * m + j][r] = 1
                        V2[j * p + k][r] = 1
                        W2[k * n + i][r] = 1
            return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
