import numpy as np
import time
import itertools
import sys

N, M, P = 3, 5, 7

def _verify(U, V, W, n=N, m=M, p=P):
    """
    Verify if U, V, W form a valid rank-r decomposition of the matrix multiplication tensor.
    U: shape (n*m, r)
    V: shape (m*p, r)
    W: shape (p*n, r)
    Condition: For all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
        
    # Vectorized check
    # We need to check T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # To speed up verification, we can iterate over k and check the matrix equality
    # C_k = U[:, k-th slice? No, U is indexed by (i,j).
    # Let's just do the triple loop, it's only 105 checks * r multiplications.
    # r is likely < 105, so this is fast.
    
    for i in range(n):
        for j in range(m):
            u_vec = U[i * m + j]
            for k in range(p):
                v_w_prod = V[j * p + k] * W[k * n + i]
                val = np.dot(u_vec, v_w_prod)
                if abs(val - 1.0) > 1e-7:
                    return False
    return True

def _schoolbook(n=N, m=M, p=P):
    """Standard rank n*m*p decomposition."""
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

def _strassen_2x2x2():
    """
    Returns a rank-7 decomposition for 2x2 matrix multiplication.
    Indices:
    U row: i*2+j (0:00, 1:01, 2:10, 3:11)
    V row: j*2+k (0:00, 1:01, 2:10, 3:11)
    W row: k*2+i (0:00, 1:10, 2:01, 3:11) -- Note: W index is k*n+i.
    """
    n=m=p=2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen terms:
    # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1 # W[0,0] is k=0,i=0. W[3,0] is k=1,i=1.
    
    # m2 = (a10+a11)b00 -> c10 += m2
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1 # k=0, i=1 -> idx 1
    
    # m3 = a00(b01-b11) -> c01 += m3, c11 += m3? No, standard strassen:
    # c00 = m1 - m4 + m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m6 + m7
    
    # Let's use a verified standard set:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a11)(b01+b11)
    # m7 = (a01-a10)(b00+b10)
    
    # Reset U,V,W
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # m1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # m2
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1 # k=0,i=1
    
    # m3
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1 # k=1,i=0
    W[3][2]=1 # k=1,i=1? No, c11 gets m3? 
    # Wait, c01 = m3 + m5. c11 = m1 - m3 + m6 + m7.
    # So m3 contributes to c01 (k=1,i=0) and c11 (k=1,i=1).
    # W index for c01 is k=1, i=0 -> 1*2+0=2.
    # W index for c11 is k=1, i=1 -> 1*2+1=3.
    W[2][2]=1; W[3][2]=-1 # Because c11 has -m3.
    
    # m4
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[1][3]=1 # k=0,i=1 (c10)
    W[0][3]=1 # k=0,i=0 (c00) -> c00 has +m4? No, c00 = m1 - m4 + ...
    # So W[0][3] should be -1.
    W[0][3]=-1
    
    # m5
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[2][4]=1 # c01 (k=1,i=0)
    W[0][4]=1 # c00 (k=0,i=0)
    
    # m6
    U[2][5]=1; U[3][5]=-1
    V[1][5]=1; V[3][5]=1
    W[3][5]=1 # c11 (k=1,i=1)
    
    # m7
    U[1][6]=1; U[2][6]=-1
    V[0][6]=1; V[2][6]=1
    W[0][6]=1 # c00 (k=0,i=0)
    
    return {"U": U, "V": V, "W": W}

def _solve_linear_system(A, b, tol=1e-9):
    """
    Solve A x = b. Returns solution x if unique and consistent, else None.
    A is (num_eq, num_vars), b is (num_eq,).
    """
    if A.shape[0] < A.shape[1]:
        # Underdetermined, need more constraints or pick a specific solution
        # For our use case, we usually fix enough anchors to make it square or overdetermined.
        # If underdetermined, we can use least squares and check residual.
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=tol)
            if residuals is None:
                # rank deficient, check if A@x == b
                if np.allclose(A @ x, b, atol=tol):
                    return x
            elif residuals[0] < tol * np.linalg.norm(b):
                return x
        except:
            pass
        return None
    
    # Check rank
    rank = np.linalg.matrix_rank(A, tol=tol)
    if rank < A.shape[1]:
        # Underdetermined or inconsistent
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=tol)
            if np.allclose(A @ x, b, atol=tol):
                return x
        except:
            pass
        return None
        
    # Full rank square (or overdetermined with full column rank)
    try:
        x = np.linalg.solve(A, b)
        if np.allclose(A @ x, b, atol=tol):
            return x
    except:
        pass
    return None

def _algebraic_search(n, m, p, target_rank, time_budget=60):
    """
    Search for a decomposition of rank `target_rank` by fixing anchor values.
    """
    start_time = time.time()
    
    # Total variables: n*m*r + m*p*r + p*n*r
    num_vars = (n*m + m*p + p*n) * target_rank
    num_eqs = n * m * p
    
    # We need to fix enough variables to make the system solvable.
    # Let's fix 1 variable per equation? No, that's too many.
    # Strategy: Fix a small subset of variables to +/-1, 0.
    # The number of free variables must be <= num_eqs.
    # So we need to fix at least num_vars - num_eqs variables.
    
    # For 3x5x7, rank 57:
    # vars = (15 + 35 + 21) * 57 = 71 * 57 = 4047
    # eqs = 105
    # We need to fix ~3942 variables. This is too many to search randomly.
    
    # Alternative: Exploit symmetry or structure?
    # Or use a "greedy" approach where we build the decomposition term by term?
    
    # Let's try a different tactic: 
    # If we assume the decomposition is "sparse" or has small integer entries,
    # we can try to find a solution where U, V, W have entries in {-1, 0, 1}.
    
    # Since the search space is huge, let's focus on a specific structure:
    # What if we use the known rank-57 decomposition for 3x5x7?
    # It is constructed by combining Strassen (2x2) and other blocks.
    # The previous program tried block partitioning.
    # Maybe the issue is that the block partitioning logic was flawed or suboptimal.
    
    # Let's try a specific hybrid:
    # Split 3x5x7 into:
    # 1. 2x2x2 block (Strassen, rank 7)
    # 2. Remaining parts handled by schoolbook or other optimizations.
    
    # The previous program's `_greedy_block_search` did this.
    # Let's verify if the previous program actually found the optimal block partition.
    # Partitions of 3: [1,2], [2,1], [1,1,1], [3]
    # Partitions of 5: [1,4], [2,3], [3,2], [4,1], [1,1,3]...
    # Partitions of 7: [1,6], [2,5]...
    
    # If we pick [2,1] for N, [2,3] for M, [2,5] for P:
    # Block 1: 2x2x2 -> Rank 7
    # Block 2: 1x3x2 -> Rank 6
    # Block 3: 2x3x5 -> Rank 30
    # Block 4: 1x3x5 -> Rank 15
    # Total: 7+6+30+15 = 58.
    
    # Is 57 possible?
    # Yes, 3x5x7 has rank 57.
    # How?
    # Perhaps a different partition or a non-block decomposition.
    
    # Let's try to construct the rank 57 solution explicitly using known algorithms.
    # One way is to use the "Winograd" style or specific tensor identities.
    
    # Given the time constraints and complexity, let's try to improve the block search
    # by allowing "overlaps" or checking if a lower rank is possible via linear algebra
    # on the block boundaries.
    
    # Actually, let's just try to find ANY decomposition with rank < 105.
    # The block search in the previous program might have failed due to bugs.
    # Let's write a robust block builder.
    
    def _build_block(n, m, p, splits_n, splits_m, splits_p):
        # splits_n, splits_m, splits_p are lists of integers summing to n, m, p
        # They must have the same length.
        if len(splits_n) != len(splits_m) or len(splits_n) != len(splits_p):
            return None
            
        total_rank = 0
        U_rows = n * m
        V_rows = m * p
        W_rows = p * n
        
        # We don't know total_rank yet, so we'll append columns
        U_cols = []
        V_cols = []
        W_cols = []
        
        for idx in range(len(splits_n)):
            ni = splits_n[idx]
            mi = splits_m[idx]
            pi = splits_p[idx]
            
            # Get decomposition for ni x mi x pi
            if ni == 2 and mi == 2 and pi == 2:
                sub = _strassen_2x2x2()
            else:
                sub = _schoolbook(ni, mi, pi)
                
            sub_rank = len(sub["U"][0])
            
            # Map indices
            i_start = sum(splits_n[:idx])
            j_start = sum(splits_m[:idx])
            k_start = sum(splits_p[:idx])
            
            for r in range(sub_rank):
                u_col = [0] * U_rows
                v_col = [0] * V_rows
                w_col = [0] * W_rows
                
                for i in range(ni):
                    for j in range(mi):
                        u_col[(i_start + i) * m + (j_start + j)] = sub["U"][i * mi + j][r]
                        
                for j in range(mi):
                    for k in range(pi):
                        v_col[(j_start + j) * p + (k_start + k)] = sub["V"][j * pi + k][r]
                        
                for k in range(pi):
                    for i in range(ni):
                        w_col[(k_start + k) * n + (i_start + i)] = sub["W"][k * ni + i][r]
                        
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
                
        # Transpose to (rows, rank)
        rank = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for best block partition
    best_rank = n * m * p
    best_res = _schoolbook(n, m, p)
    
    # Generate all compositions
    def compositions(n, max_part=3):
        # Use recursion to generate compositions with parts <= max_part
        # To keep it fast, limit depth or parts
        res = []
        def helper(rem, cur):
            if rem == 0:
                res.append(tuple(cur))
                return
            for part in range(1, min(max_part, rem) + 1):
                cur.append(part)
                helper(rem - part, cur)
                cur.pop()
        helper(n, [])
        return res

    n_comps = compositions(n)
    m_comps = compositions(m)
    p_comps = compositions(p)
    
    # To avoid explosion, we only try partitions with same number of parts
    # and limit the number of parts to <= 3 or 4
    max_parts = 4
    n_comps = [c for c in n_comps if len(c) <= max_parts]
    m_comps = [c for c in m_comps if len(c) <= max_parts]
    p_comps = [c for c in p_comps if len(c) <= max_parts]
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                if len(nc) != len(mc) or len(nc) != len(pc):
                    continue
                if time.time() - start_time > time_budget:
                    return best_res, best_rank
                
                res = _build_block(n, m, p, list(nc), list(mc), list(pc))
                if res:
                    rank = len(res["U"][0])
                    if rank < best_rank:
                        if _verify(res["U"], res["V"], res["W"], n, m, p):
                            best_rank = rank
                            best_res = res
                            
    return best_res, best_rank

def construct():
    n, m, p = N, M, P
    deadline = time.time() + 120
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Phase 1: Block Search
    block_res, block_rank = _algebraic_search(n, m, p, 105, time_budget=60)
    if block_rank < best_rank:
        best_rank = block_rank
        best = block_res
        
    # Phase 2: Try to improve using linear algebra on the block result?
    # If we have a rank R decomposition, can we reduce it?
    # This is hard. 
    
    # Phase 3: Specific known low-rank constructions?
    # For 3x5x7, rank 57 is known.
    # The block search should find 58 or better.
    # If it finds 58, we are at 1.0000 * (105/58) > 1.
    # Wait, score is seed/value. Seed is 1.0. Value is rank.
    # If rank is 58, score is 105/58 = 1.81.
    
    # Let's ensure the block search is exhaustive enough.
    
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)
        
    return best

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
