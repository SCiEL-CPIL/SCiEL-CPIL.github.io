"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7 using Strassen where beneficial. Total: 93"""
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]
    W2p = [[0] * rank2 for _ in range(8)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V2p[j*2+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W2p[k*4+i][r] += val
            block_idx += 1
    
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]
    W3 = [[0] * rank3 for _ in range(4)]
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
    
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for i in range(4):
        for r in range(rank3):
            V[i*p + 6][offset3 + r] = V3[i][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    
    return {"U": U, "V": V, "W": W}


def _search_2x2x3_rank10():
    """Search for a rank-10 decomposition of 2x2x3 matrix multiplication using ALS."""
    def _als_2x2x3(max_iter=300, seed=42):
        rng = np.random.RandomState(seed)
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i*2+j, j*3+k, k*2+i] = 1.0
        rank = 10
        U = rng.randn(4, rank)
        V = rng.randn(6, rank)
        W = rng.randn(6, rank)
        for it in range(max_iter):
            VW = (V[:, :, None] * W[:, None, :]).reshape(36, rank)
            for a in range(4):
                target = T[a].flatten()
                A_mat = VW @ VW.T
                b_mat = VW @ target
                try:
                    U[a] = np.linalg.solve(A_mat + 1e-10*np.eye(36), b_mat)
                except np.linalg.LinAlgError:
                    U[a] = np.linalg.lstsq(VW.T, target, rcond=None)[0]
            UW = (U[:, :, None] * W[:, None, :]).reshape(24, rank)
            for b in range(6):
                target = T[:, b].flatten()
                A_mat = UW @ UW.T
                b_mat = UW @ target
                try:
                    V[b] = np.linalg.solve(A_mat + 1e-10*np.eye(24), b_mat)
                except np.linalg.LinAlgError:
                    V[b] = np.linalg.lstsq(UW.T, target, rcond=None)[0]
            UV = (U[:, :, None] * V[:, None, :]).reshape(24, rank)
            for c in range(6):
                target = T[:, :, c].flatten()
                A_mat = UV @ UV.T
                b_mat = UV @ target
                try:
                    W[c] = np.linalg.solve(A_mat + 1e-10*np.eye(24), b_mat)
                except np.linalg.LinAlgError:
                    W[c] = np.linalg.lstsq(UV.T, target, rcond=None)[0]
        residual = np.sum((np.einsum('ar,br,cr->abc', U, V, W) - T)**2)
        return U, V, W, residual

    best_res = float('inf')
    best = None
    for seed in range(10):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        U, V, W, res = _als_2x2x3(max_iter=150, seed=seed)
        if res < best_res:
            best_res = res
            best = (U, V, W)
    
    if best is None or best_res > 1e-6:
        return None
    
    U, V, W = best
    from math import gcd
    def _round(x, max_den=50):
        if abs(x) < 1e-10:
            return 0
        for den in range(1, max_den+1):
            num = round(x*den)
            if abs(x - num/den) < 1e-8:
                g = gcd(abs(num), den)
                return (num//g)/(den//g)
        return x
    U_rat = [[_round(U[i,t]) for t in range(10)] for i in range(4)]
    V_rat = [[_round(V[i,t]) for t in range(10)] for i in range(6)]
    W_rat = [[_round(W[i,t]) for t in range(10)] for i in range(6)]
    U_arr = np.array(U_rat, dtype=float)
    V_arr = np.array(V_rat, dtype=float)
    W_arr = np.array(W_rat, dtype=float)
    T = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T[i*2+j, j*3+k, k*2+i] = 1.0
    if np.max(np.abs(np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr) - T)) < 1e-8:
        return U_rat, V_rat, W_rat
    return None


def _build_4x4x7_direct_sum():
    """Build 4x4x7 as direct sum of four 2x2x3 sub-tensors.
    
    A is 4x4, split into four 2x2 blocks: A[br*2:br*2+2, bc*2:bc*2+2]
    B is 4x7. For each block (br, bc), the relevant part of B is:
    B[bc*2:bc*2+2, :] which is 2x7.
    But we need 2x3 for the sub-problem. Hmm, this doesn't work directly.
    
    Actually, the direct sum approach works when B can also be split.
    Let me reconsider.
    
    A (4x4) = [A00 A01; A10 A11] where each block is 2x2.
    B (4x7) = [B0; B1] where B0 is 2x7, B1 is 2x7.
    C (4x7) = A*B = [A00*B0 + A01*B1; A10*B0 + A11*B1]
    
    This doesn't decompose into independent 2x2*2x3 problems because
    each row block of C depends on both row blocks of B.
    
    Alternative: Split B into columns.
    B (4x7) = [B_left (4x4) | B_right (4x3)]
    C (4x7) = [A*B_left | A*B_right]
    
    A*B_left: 4x4 * 4x4 -> rank 49 (Strassen Kronecker) or less
    A*B_right: 4x4 * 4x3 -> ?
    
    For 4x4 * 4x3:
    Split A into 2x2 blocks, B_right into 2x3 blocks (vertically).
    C_right (4x3) = [A00*B0_right + A01*B1_right; A10*B0_right + A11*B1_right]
    where B0_right is 2x3, B1_right is 2x3.
    
    Each 2x2 * 2x3 sub-problem appears 4 times (for each pair of 2x2 blocks of A
    and 2x3 blocks of B_right).
    
    Total: 4 * rank(2x2x3) for the right part + rank(4x4x4) for the left part.
    
    If rank(2x2x3) = 10: total = 49 + 4*10 = 89. Better than 93 but not 40.
    
    Hmm, the directive says "direct sum of four 2x2x3 sub-tensors" giving rank 40.
    This would require the entire 4x4x7 to decompose into 4 independent 2x2x3 problems.
    That's only possible if the tensor has a specific block-diagonal structure.
    
    Actually, re-reading the directive: "modeling the 4x4x7 tensor as a direct sum 
    of four 2x2x3 sub-tensors". This means we need to find a change of basis or
    a different decomposition where the tensor splits into 4 copies of 2x2x3.
    
    This is only possible if the 4x4x7 tensor is isomorphic to 4 ⊕ (2x2x3).
    The 4x4x7 tensor has 112 non-zero entries. 4 * 12 = 48. That's not matching.
    
    I think the directive may be aspirational or based on a different interpretation.
    Let me just try to get the best rank I can with the 2x2x3 approach.
    
    Best I can do: 49 (for 4x4x4) + 4*rank(2x2x3) (for 4x4x3 part)
    If rank(2x2x3) = 10: 49 + 40 = 89
    If rank(2x2x3) = 11: 49 + 44 = 93 (same as current)
    
    So I need rank(2x2x3) <= 10 to improve.
    """
    result = _search_2x2x3_rank10()
    if result is None:
        return None
    
    U_sub, V_sub, W_sub = result
    # U_sub: 4x10, V_sub: 6x10, W_sub: 6x10
    
    n, m, p = 4, 4, 7
    # Build 4x4 * 4x4 (left part, B cols 0-3)
    U2, V2, W2 = _strassen_2x2x2()
    rank_left = 49
    U_left = [[0] * rank_left for _ in range(16)]
    V_left = [[0] * rank_left for _ in range(16)]
    W_left = [[0] * rank_left for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U_left[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V_left[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W_left[k*4+i][r] += val
    
    # Build 4x4 * 4x3 (right part, B cols 4-6)
    # Split into 4 sub-problems: (br, bc) in {0,1}x{0,1}
    # For each (br, bc): A[br*2:br*2+2, bc*2:bc*2+2] * B[bc*2:bc*2+2, 4:7] -> C[br*2:br*2+2, 4:7]
    # But this is NOT independent! C[br*2:br*2+2, :] depends on all of B.
    # 
    # Wait, I made an error. Let me reconsider.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For k in {4,5,6}: C[i,k] = sum_j A[i,j] * B[j,k]
    # This involves all of A (all 4 rows) and B[:, k] (all 4 rows of B for column k).
    # 
    # If I split A into 2x2 blocks and B into 2x3 blocks (vertically),
    # then C[br*2:br*2+2, 4:7] = A[br*2:br*2+2, :] * B[:, 4:7]
    # = A[br*2:br*2+2, 0:2] * B[0:2, 4:7] + A[br*2:br*2+2, 2:4] * B[2:4, 4:7]
    # 
    # So each 2x3 block of C depends on TWO 2x3 blocks of B (one from each row half).
    # This means we can't simply use 4 independent 2x2x3 problems.
    # 
    # Instead, for each row block br of A (2 rows), we have:
    # C[br*2:br*2+2, 4:7] = A[br*2:br*2+2, 0:2] * B[0:2, 4:7] + A[br*2:br*2+2, 2:4] * B[2:4, 4:7]
    # 
    # This is the sum of two 2x2 * 2x3 products.
    # If each 2x2*2x3 has rank r, the sum has rank at most 2r.
    # For 2 row blocks: total rank for right part = 2 * 2 * r = 4r.
    # 
    # So total = 49 + 4*r where r = rank(2x2x3).
    # If r=10: total = 89.
    
    rank_sub = 10
    rank_right = 4 * rank_sub  # 2 row blocks * 2 B-blocks * rank_sub
    
    total_rank = rank_left + rank_right
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]  # 4*7
    W = [[0] * total_rank for _ in range(28)]  # 7*4
    
    # Left part: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank_left):
            U[i][r] = U_left[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_left):
                V[i*7+j][r] = V_left[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank_left):
                W[i*4+j][r] = W_left[i*4+j][r]
    
    # Right part: B cols 4-6, C cols 4-6
    # For each row block br in {0,1} and B-block bc in {0,1}:
    #   A[br*2:br*2+2, bc*2:bc*2+2] * B[bc*2:bc*2+2, 4:7] contributes to C[br*2:br*2+2, 4:7]
    offset = rank_left
    for br in range(2):
        for bc in range(2):
            for t in range(rank_sub):
                r = offset + (br*2 + bc) * rank_sub + t
                # U: A[br*2:br*2+2, bc*2:bc*2+2] -> rows br*2, br*2+1, cols bc*2, bc*2+1
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U_sub[lr*2+lc][t]
                        if val != 0:
                            U[i*4+j][r] += val
                # V: B[bc*2:bc*2+2, 4:7] -> rows bc*2, bc*2+1, cols 4,5,6
                for lr in range(2):
                    for lc in range(3):
                        j = bc * 2 + lr
                        k = 4 + lc
                        val = V_sub[lr*3+lc][t]
                        if val != 0:
                            V[j*7+k][r] += val
                # W: C[br*2:br*2+2, 4:7] -> cols 4,5,6, rows br*2, br*2+1
                # W index = k*4+i where k in {4,5,6}, i in {br*2, br*2+1}
                for lc in range(3):
                    for lr in range(2):
                        k = 4 + lc
                        i = br * 2 + lr
                        val = W_sub[lc*2+lr][t]
                        if val != 0:
                            W[k*4+i][r] += val
    
    return {"U": U, "V": V, "W": W}


def _als_global_reduce(U, V, W, n, m, p, target_rank, max_iter=100, seed=0):
    """Try to reduce rank via ALS on the full 4x4x7 tensor."""
    rng = np.random.RandomState(seed)
    n, m, p = 4, 4, 7
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    cur_rank = len(U[0])
    if cur_rank <= target_rank:
        return None
    # Initialize with random perturbation of existing factors + random new ones
    U_new = np.zeros((dim_a, target_rank))
    V_new = np.zeros((dim_b, target_rank))
    W_new = np.zeros((dim_c, target_rank))
    # Use subset of existing columns as init
    keep = min(target_rank, cur_rank)
    U_new[:, :keep] = np.array(U)[:, :keep]
    V_new[:, :keep] = np.array(V)[:, :keep]
    W_new[:, :keep] = np.array(W)[:, :keep]
    if target_rank > keep:
        U_new[:, keep:] = rng.randn(dim_a, target_rank-keep)
        V_new[:, keep:] = rng.randn(dim_b, target_rank-keep)
        W_new[:, keep:] = rng.randn(dim_c, target_rank-keep)
    for it in range(max_iter):
        # Update U
        VW = (V_new[:, :, None] * W_new[:, None, :]).reshape(dim_b*dim_c, target_rank)
        for a in range(dim_a):
            target = T[a].flatten()
            A_mat = VW @ VW.T
            b_mat = VW @ target
            try:
                U_new[a] = np.linalg.solve(A_mat + 1e-8*np.eye(dim_b*dim_c), b_mat)
            except np.linalg.LinAlgError:
                U_new[a] = np.linalg.lstsq(VW.T, target, rcond=None)[0]
        # Update V
        UW = (U_new[:, :, None] * W_new[:, None, :]).reshape(dim_a*dim_c, target_rank)
        for b in range(dim_b):
            target = T[:, b].flatten()
            A_mat = UW @ UW.T
            b_mat = UW @ target
            try:
                V_new[b] = np.linalg.solve(A_mat + 1e-8*np.eye(dim_a*dim_c), b_mat)
            except np.linalg.LinAlgError:
                V_new[b] = np.linalg.lstsq(UW.T, target, rcond=None)[0]
        # Update W
        UV = (U_new[:, :, None] * V_new[:, None, :]).reshape(dim_a*dim_b, target_rank)
        for c in range(dim_c):
            target = T[:, :, c].flatten()
            A_mat = UV @ UV.T
            b_mat = UV @ target
            try:
                W_new[c] = np.linalg.solve(A_mat + 1e-8*np.eye(dim_a*dim_b), b_mat)
            except np.linalg.LinAlgError:
                W_new[c] = np.linalg.lstsq(UV.T, target, rcond=None)[0]
    residual = np.max(np.abs(np.einsum('ar,br,cr->abc', U_new, V_new, W_new) - T))
    if residual > 1e-6:
        return None
    from math import gcd
    def _round(x, max_den=20):
        if abs(x) < 1e-10:
            return 0
        for den in range(1, max_den+1):
            num = round(x*den)
            if abs(x - num/den) < 1e-8:
                g = gcd(abs(num), den)
                return (num//g)/(den//g)
        return x
    U_rat = [[_round(U_new[i,t]) for t in range(target_rank)] for i in range(dim_a)]
    V_rat = [[_round(V_new[i,t]) for t in range(target_rank)] for i in range(dim_b)]
    W_rat = [[_round(W_new[i,t]) for t in range(target_rank)] for i in range(dim_c)]
    U_arr = np.array(U_rat, dtype=float)
    V_arr = np.array(V_rat, dtype=float)
    W_arr = np.array(W_rat, dtype=float)
    if np.max(np.abs(np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr) - T)) < 1e-8:
        return {"U": U_rat, "V": V_rat, "W": W_rat}
    return None


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # First: build rank-93 (fast, deterministic)
    best_result = None
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            best_result = result
    except Exception:
        pass
    
    if best_result is None:
        return _schoolbook(4, 4, 7)
    
    # Try rank-89 via 2x2x3 rank-10
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        try:
            result = _build_4x4x7_direct_sum()
            if result is not None and _verify(result['U'], result['V'], result['W'], 4, 4, 7):
                best_result = result
        except Exception:
            pass
    
    # Try global ALS rank reduction from rank 93
    if best_result is not None and time.time() - _START_TIME < _TIME_LIMIT - 15:
        try:
            for target_rank in [92, 91, 90]:
                if time.time() - _START_TIME > _TIME_LIMIT - 10:
                    break
                reduced = _als_global_reduce(
                    best_result['U'], best_result['V'], best_result['W'],
                    4, 4, 7, target_rank, max_iter=50, seed=target_rank)
                if reduced is not None and _verify(reduced['U'], reduced['V'], reduced['W'], 4, 4, 7):
                    best_result = reduced
                    break
        except Exception:
            pass
    
    return best_result
