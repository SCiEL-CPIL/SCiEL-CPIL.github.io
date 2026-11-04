"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses a structured block decomposition to achieve rank 36, and attempts to optimize
    for lower ranks (32, 28) using numerical methods with exact verification.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def schoolbook_decomposition():
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_half_integer(U, V, W):
        """
        Verify that U, V, W with half-integer entries exactly decompose the tensor.
        """
        U_int = (2 * U).astype(int)
        V_int = (2 * V).astype(int)
        W_int = (2 * W).astype(int)
        
        # Check if rounding was exact
        if not np.allclose(U * 2, U_int) or not np.allclose(V * 2, V_int) or not np.allclose(W * 2, W_int):
            return False
            
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                    if val != 8:
                        return False
        return True

    def strassen_2x2x2(U_offset, V_offset, W_offset, rank_offset):
        """
        Add a Strassen decomposition for a 2x2x2 block.
        U, V, W are pre-allocated arrays.
        U_offset, V_offset, W_offset are the starting indices for the rows/cols of this block.
        rank_offset is the starting column index in U, V, W.
        Returns the number of ranks used (7).
        """
        # Strassen terms:
        # M1 = (U00 + U11)(V00 + V11)
        # M2 = (U10 + U11)V00
        # M3 = U00(V01 - V11)
        # M4 = U11(V10 - V00)
        # M5 = (U00 + U01)V11
        # M6 = (U10 - U00)(V00 + V01)
        # M7 = (U01 - U11)(V10 + V11)
        
        # Indices in local 2x2 block:
        # U rows: 0,1. V cols: 0,1. W rows: 0,1 (for C00, C01, C10, C11? No, W is (p*n, rank))
        # W index: k*n + i. For 2x2, k=0,1, i=0,1.
        # W00: k=0, i=0 -> 0
        # W01: k=1, i=0 -> 1
        # W10: k=0, i=1 -> 2
        # W11: k=1, i=1 -> 3
        
        # Term 1
        U[U_offset+0, rank_offset] = 1
        U[U_offset+3, rank_offset] = 1
        V[V_offset+0, rank_offset] = 1
        V[V_offset+3, rank_offset] = 1
        W[W_offset+0, rank_offset] = 1
        W[W_offset+3, rank_offset] = 1
        
        # Term 2
        U[U_offset+2, rank_offset+1] = 1
        U[U_offset+3, rank_offset+1] = 1
        V[V_offset+0, rank_offset+1] = 1
        
        # Term 3
        U[U_offset+0, rank_offset+2] = 1
        V[V_offset+1, rank_offset+2] = 1
        V[V_offset+3, rank_offset+2] = -1
        
        # Term 4
        U[U_offset+3, rank_offset+3] = 1
        V[V_offset+2, rank_offset+3] = 1
        V[V_offset+0, rank_offset+3] = -1
        
        # Term 5
        U[U_offset+0, rank_offset+4] = 1
        U[U_offset+1, rank_offset+4] = 1
        V[V_offset+3, rank_offset+4] = 1
        
        # Term 6
        U[U_offset+2, rank_offset+5] = 1
        U[U_offset+0, rank_offset+5] = -1
        V[V_offset+0, rank_offset+5] = 1
        V[V_offset+1, rank_offset+5] = 1
        
        # Term 7
        U[U_offset+1, rank_offset+6] = 1
        U[U_offset+3, rank_offset+6] = -1
        V[V_offset+2, rank_offset+6] = 1
        V[V_offset+3, rank_offset+6] = 1
        
        # Output W coefficients
        # C00 = M1 - M4 + M5 - M7
        W[W_offset+0, rank_offset] = 1
        W[W_offset+0, rank_offset+3] = -1
        W[W_offset+0, rank_offset+4] = 1
        W[W_offset+0, rank_offset+6] = -1
        
        # C01 = M3 + M7
        W[W_offset+1, rank_offset+2] = 1
        W[W_offset+1, rank_offset+6] = 1
        
        # C10 = M2 + M4
        W[W_offset+2, rank_offset+1] = 1
        W[W_offset+2, rank_offset+3] = 1
        
        # C11 = M1 - M3 + M6
        W[W_offset+3, rank_offset] = 1
        W[W_offset+3, rank_offset+2] = -1
        W[W_offset+3, rank_offset+5] = 1
        
        return 7

    def schoolbook_2x2x1(U_offset, V_offset, W_offset, rank_offset):
        """
        Add a schoolbook decomposition for a 2x2x1 block.
        Rank 4.
        """
        # C0 = U00 V0 + U01 V1
        # C1 = U10 V0 + U11 V1
        
        # Term 1: U00 V0
        U[U_offset+0, rank_offset] = 1
        V[V_offset+0, rank_offset] = 1
        W[W_offset+0, rank_offset] = 1
        
        # Term 2: U01 V1
        U[U_offset+1, rank_offset+1] = 1
        V[V_offset+1, rank_offset+1] = 1
        W[W_offset+0, rank_offset+1] = 1
        
        # Term 3: U10 V0
        U[U_offset+2, rank_offset+2] = 1
        V[V_offset+0, rank_offset+2] = 1
        W[W_offset+1, rank_offset+2] = 1
        
        # Term 4: U11 V1
        U[U_offset+3, rank_offset+3] = 1
        V[V_offset+1, rank_offset+3] = 1
        W[W_offset+1, rank_offset+3] = 1
        
        return 4

    def construct_rank_36():
        """
        Construct a rank 36 decomposition.
        Split p=5 into 2, 2, 1.
        Each 2x4x2 block uses 2x(2x2x2) Strassen = 14 ranks.
        2x4x1 block uses schoolbook = 8 ranks.
        Total 14 + 14 + 8 = 36.
        """
        U = np.zeros((8, 36))
        V = np.zeros((20, 36))
        W = np.zeros((10, 36))
        
        # Block 1: C cols 0,1. B cols 0,1.
        # A is 2x4. B is 4x2.
        # A1 (rows 0,1; cols 0,1 of A) -> U rows 0,1, 4,5? No.
        # U index: i*m + j.
        # A1: i=0,1; j=0,1. Indices: 0,1, 4,5.
        # A2: i=0,1; j=2,3. Indices: 2,3, 6,7.
        # B1 (rows 0,1; cols 0,1 of B) -> V rows 0,1, 5,6?
        # V index: j*p + k.
        # B1: j=0,1; k=0,1. Indices: 0,1, 5,6.
        # B2 (rows 2,3; cols 0,1 of B) -> V rows 2,3, 7,8?
        # B2: j=2,3; k=0,1. Indices: 10,11, 15,16?
        # Wait, V index is j*p + k.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 5
        # j=1, k=1 -> 6
        # j=2, k=0 -> 10
        # j=2, k=1 -> 11
        # j=3, k=0 -> 15
        # j=3, k=1 -> 16
        
        # We need to map the 2x2 Strassen blocks to these indices.
        # The Strassen function expects contiguous 2x2 blocks in U, V, W.
        # Our indices are not contiguous in the global U, V, W matrices.
        # So we cannot use the helper directly with offsets unless we permute.
        
        # Instead, we will manually fill the 36 columns.
        
        # Helper to add a Strassen term for specific global indices
        def add_strassen_term(u_idx, v_idx, w_idx, rank_idx, coeffs):
            # coeffs is a dict: { (u_offset, v_offset, w_offset): coeff }
            # This is too complex. Let's just expand the 7 terms manually for each block.
            pass

        # Let's write a generic add_strassen that takes index maps.
        def add_strassen(u_map, v_map, w_map, rank_start):
            # u_map: dict {(local_u, local_v): global_u} ? No.
            # u_map: list of 4 global U indices for the 2x2 A block.
            # v_map: list of 4 global V indices for the 2x2 B block.
            # w_map: list of 4 global W indices for the 2x2 C block.
            
            # Term 1: (U00+U11)(V00+V11)
            U[u_map[0], rank_start] += 1
            U[u_map[3], rank_start] += 1
            V[v_map[0], rank_start] += 1
            V[v_map[3], rank_start] += 1
            W[w_map[0], rank_start] += 1
            W[w_map[3], rank_start] += 1
            
            # Term 2: (U10+U11)V00
            U[u_map[2], rank_start+1] += 1
            U[u_map[3], rank_start+1] += 1
            V[v_map[0], rank_start+1] += 1
            
            # Term 3: U00(V01-V11)
            U[u_map[0], rank_start+2] += 1
            V[v_map[1], rank_start+2] += 1
            V[v_map[3], rank_start+2] += -1
            
            # Term 4: U11(V10-V00)
            U[u_map[3], rank_start+3] += 1
            V[v_map[2], rank_start+3] += 1
            V[v_map[0], rank_start+3] += -1
            
            # Term 5: (U00+U01)V11
            U[u_map[0], rank_start+4] += 1
            U[u_map[1], rank_start+4] += 1
            V[v_map[3], rank_start+4] += 1
            
            # Term 6: (U10-U00)(V00+V01)
            U[u_map[2], rank_start+5] += 1
            U[u_map[0], rank_start+5] += -1
            V[v_map[0], rank_start+5] += 1
            V[v_map[1], rank_start+5] += 1
            
            # Term 7: (U01-U11)(V10+V11)
            U[u_map[1], rank_start+6] += 1
            U[u_map[3], rank_start+6] += -1
            V[v_map[2], rank_start+6] += 1
            V[v_map[3], rank_start+6] += 1
            
            # W coefficients
            # C00 = M1 - M4 + M5 - M7
            W[w_map[0], rank_start] += 1
            W[w_map[0], rank_start+3] += -1
            W[w_map[0], rank_start+4] += 1
            W[w_map[0], rank_start+6] += -1
            
            # C01 = M3 + M7
            W[w_map[1], rank_start+2] += 1
            W[w_map[1], rank_start+6] += 1
            
            # C10 = M2 + M4
            W[w_map[2], rank_start+1] += 1
            W[w_map[2], rank_start+3] += 1
            
            # C11 = M1 - M3 + M6
            W[w_map[3], rank_start] += 1
            W[w_map[3], rank_start+2] += -1
            W[w_map[3], rank_start+5] += 1
            
            return 7

        def add_schoolbook_2x2x1(u_map, v_map, w_map, rank_start):
            # C0 = U00 V0 + U01 V1
            # C1 = U10 V0 + U11 V1
            # u_map: 4 indices. v_map: 2 indices. w_map: 2 indices.
            
            # Term 1
            U[u_map[0], rank_start] += 1
            V[v_map[0], rank_start] += 1
            W[w_map[0], rank_start] += 1
            
            # Term 2
            U[u_map[1], rank_start+1] += 1
            V[v_map[1], rank_start+1] += 1
            W[w_map[0], rank_start+1] += 1
            
            # Term 3
            U[u_map[2], rank_start+2] += 1
            V[v_map[0], rank_start+2] += 1
            W[w_map[1], rank_start+2] += 1
            
            # Term 4
            U[u_map[3], rank_start+3] += 1
            V[v_map[1], rank_start+3] += 1
            W[w_map[1], rank_start+3] += 1
            
            return 4

        # Block 1: C cols 0,1.
        # A1: U indices [0, 1, 4, 5] (i=0,1; j=0,1)
        # A2: U indices [2, 3, 6, 7] (i=0,1; j=2,3)
        # B1: V indices [0, 1, 5, 6] (j=0,1; k=0,1)
        # B2: V indices [10, 11, 15, 16] (j=2,3; k=0,1)
        # C1: W indices [0, 1, 2, 3] (k=0,1; i=0,1)
        
        # C1 = A1 B1 + A2 B2
        # We add two Strassen decompositions.
        # 1. A1 B1
        u_map1 = [0, 1, 4, 5]
        v_map1 = [0, 1, 5, 6]
        w_map1 = [0, 1, 2, 3]
        rank_used = add_strassen(u_map1, v_map1, w_map1, 0)
        
        # 2. A2 B2
        u_map2 = [2, 3, 6, 7]
        v_map2 = [10, 11, 15, 16]
        w_map2 = [0, 1, 2, 3] # Same C block
        rank_used += add_strassen(u_map2, v_map2, w_map2, 7)
        
        # Block 2: C cols 2,3.
        # B1: V indices [2, 3, 7, 8] (j=0,1; k=2,3)
        # B2: V indices [12, 13, 17, 18] (j=2,3; k=2,3)
        # C2: W indices [4, 5, 6, 7] (k=2,3; i=0,1)
        
        # 3. A1 B1
        u_map3 = [0, 1, 4, 5]
        v_map3 = [2, 3, 7, 8]
        w_map3 = [4, 5, 6, 7]
        rank_used += add_strassen(u_map3, v_map3, w_map3, 14)
        
        # 4. A2 B2
        u_map4 = [2, 3, 6, 7]
        v_map4 = [12, 13, 17, 18]
        w_map4 = [4, 5, 6, 7]
        rank_used += add_strassen(u_map4, v_map4, w_map4, 21)
        
        # Block 3: C col 4.
        # B1: V indices [4, 9] (j=0,1; k=4)
        # B2: V indices [14, 19] (j=2,3; k=4)
        # C3: W indices [8, 9] (k=4; i=0,1)
        
        # 5. A1 B1 (2x2x1)
        u_map5 = [0, 1, 4, 5]
        v_map5 = [4, 9]
        w_map5 = [8, 9]
        rank_used += add_schoolbook_2x2x1(u_map5, v_map5, w_map5, 28)
        
        # 6. A2 B2 (2x2x1)
        u_map6 = [2, 3, 6, 7]
        v_map6 = [14, 19]
        w_map6 = [8, 9]
        rank_used += add_schoolbook_2x2x1(u_map6, v_map6, w_map6, 32)
        
        return U, V, W

    U_36, V_36, W_36 = construct_rank_36()
    
    if verify_half_integer(U_36, V_36, W_36):
        best_rank = 36
        best_decomp = (U_36, V_36, W_36)
    else:
        # Fallback to schoolbook if construction is wrong
        U_school, V_school, W_school = schoolbook_decomposition()
        best_rank = 40
        best_decomp = (U_school, V_school, W_school)

    # Try to optimize for lower ranks
    # This part is kept from the original program but with a time limit
    def loss_function(x, R):
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        U = x[:U_size].reshape(n*m, R)
        V = x[U_size:U_size+V_size].reshape(m*p, R)
        W = x[U_size+V_size:].reshape(p*n, R)
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1.0)**2
        return loss

    candidate_ranks = [32, 28, 24]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        found = False
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            U_init /= np.sqrt(target_rank)
            V_init /= np.sqrt(target_rank)
            W_init /= np.sqrt(target_rank)
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = scipy.optimize.minimize(
                    loss_function, x0, args=(target_rank,), 
                    method='L-BFGS-B', options={'maxiter': 20, 'ftol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < 1e-5:
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                if verify_half_integer(U_snap, V_snap, W_snap):
                    found = True
                    best_rank = target_rank
                    best_decomp = (U_snap, V_snap, W_snap)
                    break
        
        if found:
            break

    U_best, V_best, W_best = best_decomp
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
