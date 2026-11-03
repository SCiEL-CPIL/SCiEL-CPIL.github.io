"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly for integer/half-integer entries."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    if np.max(np.abs(T_computed - target)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b, col] = 1
                V[b*p + c, col] = 1
                W[c*n + a, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A22(B21-B11) -> C21 += M4, C11 += M4
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A11+A12)B22 -> C12 += M5, C11 -= M5
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A21-A11)(B11+B12) -> C21 += M6
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A12-A22)(B21+B22) -> C12 += M7
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _block_decompose_5x5():
    """
    5x5x5 decomposition via block partition 5=2+2+1.
    For 2x2x2 blocks use Strassen (rank 7), else schoolbook.
    """
    sizes = [2, 2, 1]
    U_terms = []
    V_terms = []
    W_terms = []
    rank_offset = 0

    for i in range(3):
        for j in range(3):
            for k in range(3):
                si, sj, sk = sizes[i], sizes[j], sizes[k]
                if si == 0 or sj == 0 or sk == 0:
                    continue

                if si == 2 and sj == 2 and sk == 2:
                    U_blk, V_blk, W_blk = _strassen_2x2()
                    r = 7
                else:
                    r = si * sj * sk
                    U_blk = np.zeros((si*sj, r), dtype=np.int64)
                    V_blk = np.zeros((sj*sk, r), dtype=np.int64)
                    W_blk = np.zeros((sk*si, r), dtype=np.int64)
                    col = 0
                    for a in range(si):
                        for b in range(sj):
                            for c in range(sk):
                                U_blk[a*sj+b, col] = 1
                                V_blk[b*sk+c, col] = 1
                                W_blk[c*si+a, col] = 1
                                col += 1

                row_start_A = sum(sizes[:i])
                row_start_B = sum(sizes[:j])
                row_start_C = sum(sizes[:k])

                for rr in range(r):
                    for a_idx in range(si*sj):
                        val = U_blk[a_idx, rr]
                        if val != 0:
                            U_terms.append((row_start_A + a_idx, rank_offset + rr, val))
                    for b_idx in range(sj*sk):
                        val = V_blk[b_idx, rr]
                        if val != 0:
                            V_terms.append((row_start_B + b_idx, rank_offset + rr, val))
                    for c_idx in range(sk*si):
                        val = W_blk[c_idx, rr]
                        if val != 0:
                            W_terms.append((row_start_C + c_idx, rank_offset + rr, val))
                rank_offset += r

    total_rank = rank_offset
    U = np.zeros((25, total_rank), dtype=np.int64)
    V = np.zeros((25, total_rank), dtype=np.int64)
    W = np.zeros((25, total_rank), dtype=np.int64)
    for row, col, val in U_terms:
        U[row, col] += val
    for row, col, val in V_terms:
        V[row, col] += val
    for row, col, val in W_terms:
        W[row, col] += val
    return U.tolist(), V.tolist(), W.tolist(), total_rank

def _try_random_rank(n, m, p, target_rank, max_iters=50, seed_base=0):
    """Try to find a valid decomposition of given rank via random U,V + least-squares W."""
    T = _build_target(n, m, p)
    # Flatten: T[i,j,k] where i in [0,nm), j in [0,mp), k in [0,pn)
    # We want: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Fix U,V, solve for W: for each k, T[:,:,k] = sum_r U*V[:,r]*W[k,r]
    # This is linear in W[k,r] for each k.
    nm, mp, pn = n*m, m*p, p*n
    # Build design matrix: for each k, and each (i,j), the coefficient is sum_r U[i,r]*V[j,r]*W[k,r]
    # So T[i,j,k] = sum_r (U[i,r]*V[j,r]) * W[k,r]
    # Let M[i,j,r] = U[i,r]*V[j,r], then T[i,j,k] = sum_r M[i,j,r]*W[k,r]
    # Reshape: T_flat[k, i*j] = sum_r M_flat[r, i*j] * W[k,r]
    # So T_flat = W @ M_flat^T  =>  T_flat^T = M_flat @ W^T
    # Solve: M_flat @ W^T = T_flat^T  =>  W^T = lstsq(M_flat, T_flat^T)
    
    for it in range(max_iters):
        seed = seed_base + it
        np.random.seed(seed)
        # Use small integer entries for U and V
        U_r = np.random.randint(-2, 3, size=(nm, target_rank))
        V_r = np.random.randint(-2, 3, size=(mp, target_rank))
        
        # M[i,j,r] = U[i,r]*V[j,r], shape (nm, mp, target_rank)
        # M_flat[r, i*mp+j] = U[i,r]*V[j,r]
        M_flat = np.zeros((target_rank, nm*mp), dtype=np.float64)
        for r in range(target_rank):
            M_flat[r] = np.outer(U_r[:, r], V_r[:, r]).flatten()
        
        T_flat = T.transpose(2, 0, 1).reshape(pn, nm*mp)  # T_flat[k, i*mp+j]
        
        try:
            # Solve M_flat @ W^T = T_flat^T
            # W^T has shape (target_rank, pn)
            W_T, _, _, _ = np.linalg.lstsq(M_flat, T_flat.T, rcond=1e-10)
            W_sol = W_T.T  # shape (pn, target_rank)
            
            # Check if W_sol is close to half-integers
            W_q = np.round(W_sol * 2) / 2.0
            if np.max(np.abs(W_sol - W_q)) > 1e-6:
                continue
            
            # Verify
            if _verify(U_r.tolist(), V_r.tolist(), W_q.tolist(), n, m, p, target_rank):
                return U_r.tolist(), V_r.tolist(), W_q.tolist()
        except:
            continue
    return None

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try block decomposition
    if time.time() < deadline - 60:
        U_blk, V_blk, W_blk, blk_rank = _block_decompose_5x5()
        if _verify(U_blk, V_blk, W_blk, n, m, p, blk_rank):
            best_rank = blk_rank
            best_result = {"U": U_blk, "V": V_blk, "W": W_blk}
            print(f"Block decomposition rank: {best_rank}")
        else:
            print(f"Block decomposition rank {blk_rank} FAILED verification")
    
    # Try random search for lower rank
    if best_rank > 50 and time.time() < deadline - 10:
        for target_rank in range(min(best_rank - 1, 80), 48, -1):
            if time.time() > deadline - 5:
                break
            result = _try_random_rank(n, m, p, target_rank, max_iters=30, seed_base=target_rank)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                print(f"Found valid rank {best_rank}")
                break
    
    return best_result
# EVOLVE-BLOCK-END
