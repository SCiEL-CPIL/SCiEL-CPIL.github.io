"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_strassen_decomposition():
        """Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere."""
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def find_laderman_3x3x3():
        """Find a rank-23 decomposition of the 3x3x3 matrix multiplication tensor using ALS."""
        if time.time() > deadline:
            return None
        
        sn, sm, sp = 3, 3, 3
        rank = 23
        
        # Target tensor
        T = np.zeros((sn*sm, sm*sp, sp*sn))
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    T[i*sm+j, j*sp+k, k*sn+i] = 1.0
        
        best_result = None
        best_err = float('inf')
        
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(seed)
            
            U = np.random.randn(sn*sm, rank) * 0.1
            V = np.random.randn(sm*sp, rank) * 0.1
            W = np.random.randn(sp*sn, rank) * 0.1
            
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U
                for c in range(rank):
                    R = T.copy()
                    for c2 in range(rank):
                        if c2 != c:
                            for i in range(sn):
                                for j in range(sm):
                                    for k in range(sp):
                                        R[i*sm+j, j*sp+k, k*sn+i] -= U[i*sm+j, c2] * V[j*sp+k, c2] * W[k*sn+i, c2]
                    
                    VW = np.zeros((sn*sm, sm*sp, sp*sn))
                    for i in range(sn):
                        for j in range(sm):
                            for k in range(sp):
                                VW[i*sm+j, j*sp+k, k*sn+i] = V[j*sp+k, c] * W[k*sn+i, c]
                    
                    denom = (VW**2).sum()
                    if denom > 1e-10:
                        U[:,c] = (R * VW).sum(axis=(1,2)) / denom
                
                # Update V
                for c in range(rank):
                    R = T.copy()
                    for c2 in range(rank):
                        if c2 != c:
                            for i in range(sn):
                                for j in range(sm):
                                    for k in range(sp):
                                        R[i*sm+j, j*sp+k, k*sn+i] -= U[i*sm+j, c2] * V[j*sp+k, c2] * W[k*sn+i, c2]
                    
                    UW = np.zeros((sn*sm, sm*sp, sp*sn))
                    for i in range(sn):
                        for j in range(sm):
                            for k in range(sp):
                                UW[i*sm+j, j*sp+k, k*sn+i] = U[i*sm+j, c] * W[k*sn+i, c]
                    
                    denom = (UW**2).sum()
                    if denom > 1e-10:
                        V[:,c] = (R * UW).sum(axis=(0,2)) / denom
                
                # Update W
                for c in range(rank):
                    R = T.copy()
                    for c2 in range(rank):
                        if c2 != c:
                            for i in range(sn):
                                for j in range(sm):
                                    for k in range(sp):
                                        R[i*sm+j, j*sp+k, k*sn+i] -= U[i*sm+j, c2] * V[j*sp+k, c2] * W[k*sn+i, c2]
                    
                    UV = np.zeros((sn*sm, sm*sp, sp*sn))
                    for i in range(sn):
                                for j in range(sm):
                                    for k in range(sp):
                                        UV[i*sm+j, j*sp+k, k*sn+i] = U[i*sm+j, c] * V[j*sp+k, c]
                    
                    denom = (UV**2).sum()
                    if denom > 1e-10:
                        W[:,c] = (R * UV).sum(axis=(0,1)) / denom
            
            # Check error
            err = 0.0
            for i in range(sn):
                for j in range(sm):
                    for k in range(sp):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*sm+j, c] * V[j*sp+k, c] * W[k*sn+i, c]
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-6:
                break
        
        if best_result is None:
            return None
        
        U, V, W = best_result
        
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify rounded version
        err_r = 0.0
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    val = 0.0
                    for c in range(rank):
                        val += U_r[i*sm+j, c] * V_r[j*sp+k, c] * W_r[k*sn+i, c]
                    err_r += (val - 1.0)**2
        
        if err_r < 1e-9:
            return (U_r, V_r, W_r)
        
        # Try integer rounding
        U_r2 = np.round(U)
        V_r2 = np.round(V)
        W_r2 = np.round(W)
        
        err_r2 = 0.0
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    val = 0.0
                    for c in range(rank):
                        val += U_r2[i*sm+j, c] * V_r2[j*sp+k, c] * W_r2[k*sn+i, c]
                    err_r2 += (val - 1.0)**2
        
        if err_r2 < 1e-9:
            return (U_r2, V_r2, W_r2)
        
        return None

    def build_laderman_decomposition():
        """Build a decomposition using Laderman on 3x3x3 blocks and schoolbook elsewhere."""
        if time.time() > deadline:
            return None
        
        laderman = find_laderman_3x3x3()
        if laderman is None:
            return None
        
        U_l, V_l, W_l = laderman
        rank_l = U_l.shape[1]
        
        # Partition: i:[3], j:[3,2], k:[3,3]
        i_groups = [[0, 1, 2]]
        j_groups = [[0, 1, 2], [3, 4]]
        k_groups = [[0, 1, 2], [3, 4, 5]]
        
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 3 and sj == 3 and sk == 3:
                        r = rank_l
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 3 and sj == 3 and sk == 3:
                for c in range(rank_l):
                    for li in range(si):
                        for lj in range(sj):
                            gi, gj = ig[li], jg[lj]
                            U[gi * m + gj][col] = U_l[li*3+lj, c]
                    for lj in range(sj):
                        for lk in range(sk):
                            gj, gk = jg[lj], kg[lk]
                            V[gj * p + gk][col] = V_l[lj*3+lk, c]
                    for lk in range(sk):
                        for li in range(si):
                            gk, gi = kg[lk], ig[li]
                            W[gk * n + gi][col] = W_l[lk*3+li, c]
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_improved_decomposition():
        """Try to find a better decomposition."""
        if time.time() > deadline:
            return None
        
        # Try Laderman-based decomposition
        laderman_result = build_laderman_decomposition()
        if laderman_result is not None:
            rank = laderman_result.pop("rank")
            if verify(laderman_result["U"], laderman_result["V"], laderman_result["W"], rank):
                return laderman_result
        
        # Fall back to Strassen-based
        strassen_result = build_strassen_decomposition()
        rank = strassen_result.pop("rank")
        if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], rank):
            return strassen_result
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition
    improved = try_improved_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
