import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Dimensions for U, V, W
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # ------------------------------------------------------------------
    # Structural idea: Use Strassen 2x2x2 blocks as seeds for optimization.
    # Build a 4x5x6 decomposition by combining:
    #   - Four 2x2x2 Strassen blocks covering the 4x4x2 sub-tensor (rank 28)
    #   - Schoolbook for the remaining 4x1x6 strip (rank 24)
    #   - Schoolbook for the 4x4x4 remainder (rank 64)
    # Total seed rank: 28 + 24 + 64 = 116
    # Then optimize down to lower ranks.
    # ------------------------------------------------------------------
    
    def _build_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # P1=(a11+a22)(b11+b22)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # P2=(a21+a22)b11
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
        # P3=a11(b12-b22)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
        # P4=a22(b21-b11)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        # P5=(a11+a12)b22
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
        # P6=(a21-a11)(b11+b12)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # P7=(a12-a22)(b21+b22)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    def _build_seed_4x5x6():
        """Build a rank-116 seed using Strassen blocks + schoolbook."""
        U2, V2, W2 = _build_strassen_2x2()
        rank = 116
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        # Place 4 Strassen 2x2x2 blocks for the 4x4x2 part
        # A rows 0-3, B cols 0-3 (j), C cols 0-1 (k)
        block_starts = [(0, 0, 0), (0, 2, 0), (2, 0, 0), (2, 2, 0)]
        for bi, (i_off, j_off, k_off) in enumerate(block_starts):
            base = bi * 7
            for t in range(7):
                col = base + t
                # U part: rows (i_off+li)*5 + (j_off+lj)
                for li in range(2):
                    for lj in range(2):
                        coeff = U2[li*2+lj, t]
                        if coeff != 0:
                            U[(i_off+li)*5 + (j_off+lj), col] += coeff
                # V part: rows (j_off+lj)*6 + (k_off+lk)
                for lj in range(2):
                    for lk in range(2):
                        coeff = V2[lj*2+lk, t]
                        if coeff != 0:
                            V[(j_off+lj)*6 + (k_off+lk), col] += coeff
                # W part: rows (k_off+lk)*4 + (i_off+li)
                for lk in range(2):
                    for li in range(2):
                        coeff = W2[lk*2+li, t]
                        if coeff != 0:
                            W[(k_off+lk)*4 + (i_off+li), col] += coeff
        
        # Schoolbook for j=4, all k (4x1x6 = 24 terms)
        base = 28
        col = base
        for i in range(4):
            for k in range(6):
                U[i*5 + 4, col] = 1.0
                V[4*6 + k, col] = 1.0
                W[k*4 + i, col] = 1.0
                col += 1
        
        # Schoolbook for j=0..3, k=2..5 (4x4x4 = 64 terms)
        base = 52
        col = base
        for i in range(4):
            for j in range(4):
                for k in range(2, 6):
                    U[i*5 + j, col] = 1.0
                    V[j*6 + k, col] = 1.0
                    W[k*4 + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    seed_U, seed_V, seed_W = _build_seed_4x5x6()
    seed_rank = 116
    
    # Verify seed
    U3 = seed_U.reshape(n, m, seed_rank)
    V3 = seed_V.reshape(m, p, seed_rank)
    W3 = seed_W.reshape(p, n, seed_rank)
    T_check = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if np.max(np.abs(T_check - 1.0)) < 1e-9:
        # Seed is valid, use it as initial best
        best_U = seed_U.tolist()
        best_V = seed_V.tolist()
        best_W = seed_W.tolist()
        best_rank = seed_rank
    
    # ------------------------------------------------------------------
    # Refinement: Try to reduce rank below seed_rank via optimization
    # ------------------------------------------------------------------
    
    def residual(x, R):
        U_flat = x[0 : R * dim_U]
        V_flat = x[R * dim_U : R * dim_U + R * dim_V]
        W_flat = x[R * dim_U + R * dim_V : ]
        
        U = U_flat.reshape(R, dim_U)
        V = V_flat.reshape(R, dim_V)
        W = W_flat.reshape(R, dim_W)
        
        U_3d = U.T.reshape(n, m, R)
        V_3d = V.T.reshape(m, p, R)
        W_3d = W.T.reshape(p, n, R)
        
        tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        res = tensor_sum - T
        return res.flatten()
    
    def try_optimize(target_rank, seed, max_nfev=200):
        """Try to optimize from seed to target rank."""
        if target_rank >= seed_rank:
            return None
        
        num_params = target_rank * (dim_U + dim_V + dim_W)
        
        # Initialize by taking first target_rank columns of seed
        # and adding small random perturbation
        U_seed = seed[0]
        V_seed = seed[1]
        W_seed = seed[2]
        
        x0_parts = []
        for r in range(target_rank):
            x0_parts.append(U_seed[:, r].flatten())
        for r in range(target_rank):
            x0_parts.append(V_seed[:, r].flatten())
        for r in range(target_rank):
            x0_parts.append(W_seed[:, r].flatten())
        x0 = np.concatenate(x0_parts)
        
        # Add small noise to break symmetry
        x0 += np.random.normal(0, 0.01, num_params)
        
        try:
            res = least_squares(residual, x0, args=(target_rank,), method='lm', 
                               max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            
            if res.cost < 1e-8:
                U_flat = res.x[0 : target_rank * dim_U]
                V_flat = res.x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
                W_flat = res.x[target_rank * dim_U + target_rank * dim_V : ]
                
                U = U_flat.reshape(dim_U, target_rank)
                V = V_flat.reshape(dim_V, target_rank)
                W = W_flat.reshape(dim_W, target_rank)
                
                # Verify
                U3 = U.T.reshape(n, m, target_rank)
                V3 = V.T.reshape(m, p, target_rank)
                W3 = W.T.reshape(p, n, target_rank)
                T_out = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                error = np.max(np.abs(T_out - 1.0))
                
                if error < 1e-8:
                    # Try to round to half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    U3h = U_half.T.reshape(n, m, target_rank)
                    V3h = V_half.T.reshape(m, p, target_rank)
                    W3h = W_half.T.reshape(p, n, target_rank)
                    T_out_h = np.einsum('ijr,jkr,kir->ijk', U3h, V3h, W3h)
                    error_h = np.max(np.abs(T_out_h - 1.0))
                    
                    if error_h < 1e-8:
                        return U_half, V_half, W_half
                    else:
                        # Return unrounded if close enough
                        return U, V, W
        except Exception:
            pass
        
        return None
    
    # Try to reduce rank from seed_rank downward
    for target_rank in range(seed_rank - 1, max(seed_rank - 10, 110), -1):
        if time.time() > deadline:
            break
        
        np.random.seed(1000 + target_rank)
        result = try_optimize(target_rank, (seed_U, seed_V, seed_W), max_nfev=150)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_U = U_r.tolist()
            best_V = V_r.tolist()
            best_W = W_r.tolist()
            best_rank = target_rank
            break  # Found improvement, stop
    
    # If no improvement from seed, try random restarts at rank 115
    if best_rank == seed_rank and time.time() < deadline:
        target_rank = 115
        num_params = target_rank * (dim_U + dim_V + dim_W)
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(2000 + attempt)
            x0 = np.random.uniform(-1, 1, num_params)
            
            try:
                res = least_squares(residual, x0, args=(target_rank,), method='lm', 
                                   max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                
                if res.cost < 1e-8:
                    U_flat = res.x[0 : target_rank * dim_U]
                    V_flat = res.x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
                    W_flat = res.x[target_rank * dim_U + target_rank * dim_V : ]
                    
                    U = U_flat.reshape(dim_U, target_rank)
                    V = V_flat.reshape(dim_V, target_rank)
                    W = W_flat.reshape(dim_W, target_rank)
                    
                    U3 = U.T.reshape(n, m, target_rank)
                    V3 = V.T.reshape(m, p, target_rank)
                    W3 = W.T.reshape(p, n, target_rank)
                    T_out = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                    error = np.max(np.abs(T_out - 1.0))
                    
                    if error < 1e-8:
                        U_half = np.round(U * 2) / 2
                        V_half = np.round(V * 2) / 2
                        W_half = np.round(W * 2) / 2
                        
                        U3h = U_half.T.reshape(n, m, target_rank)
                        V3h = V_half.T.reshape(m, p, target_rank)
                        W3h = W_half.T.reshape(p, n, target_rank)
                        T_out_h = np.einsum('ijr,jkr,kir->ijk', U3h, V3h, W3h)
                        error_h = np.max(np.abs(T_out_h - 1.0))
                        
                        if error_h < 1e-8:
                            best_U = U_half.tolist()
                            best_V = V_half.tolist()
                            best_W = W_half.tolist()
                            best_rank = target_rank
                            break
            except Exception:
                pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
