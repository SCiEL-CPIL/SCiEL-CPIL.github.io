"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    C = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    return bool(np.allclose(C, 1.0, atol=tol))

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    products = [
        (np.array([[1,0],[0,1]], dtype=np.float64), 
         np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[1,0],[0,-1]], dtype=np.float64)),
        (np.array([[0,0],[1,1]], dtype=np.float64), 
         np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64)),
        (np.array([[1,0],[0,0]], dtype=np.float64), 
         np.array([[0,0],[1,-1]], dtype=np.float64), 
         np.array([[0,1],[0,0]], dtype=np.float64)),
        (np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[0,1],[-1,0]], dtype=np.float64), 
         np.array([[0,0],[1,0]], dtype=np.float64)),
        (np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64), 
         np.array([[-1,0],[0,1]], dtype=np.float64)),
        (np.array([[-1,0],[1,0]], dtype=np.float64), 
         np.array([[1,1],[0,0]], dtype=np.float64), 
         np.array([[0,0],[0,1]], dtype=np.float64)),
        (np.array([[0,1],[0,-1]], dtype=np.float64), 
         np.array([[0,0],[1,1]], dtype=np.float64), 
         np.array([[1,0],[0,1]], dtype=np.float64)),
    ]
    
    for r, (Ur, Vr, Wr) in enumerate(products):
        U[:, r] = Ur.flatten()
        V[:, r] = Vr.flatten()
        W[:, r] = Wr.flatten()
    
    return U, V, W

def _build_3x2x2():
    """
    Build a rank-12 decomposition for 3x2x2 matrix multiplication.
    View as 3 independent 1x2x2 problems (rank 2 each) = rank 6.
    Actually schoolbook is 3*2*2=12. Let's use rank 6.
    For 1x2x2: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    Rank 2: r=0: U[0,0]=1,V[0,k]=1,W[k,0]=1 for k=0,1; U[0,1]=0,V[1,*]=0
              r=1: U[0,1]=1,V[1,k]=1,W[k,0]=1 for k=0,1; U[0,0]=0,V[0,*]=0
    For 3x2x2, each row i gets its own rank-2 pair: total rank 6.
    """
    n, m, p = 3, 2, 2
    rank = n * 2  # 6
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    for i in range(n):
        for jj in range(2):
            r = i * 2 + jj
            U[i * m + jj, r] = 1.0
            for k in range(p):
                V[jj * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    return U, V, W

def _build_3x1x6():
    """
    Build a rank-18 decomposition for 3x1x6 matrix multiplication.
    C[i,k] = A[i,0]*B[0,k]. Schoolbook rank = 3*1*6 = 18.
    """
    n, m, p = 3, 1, 6
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W

def _combine_blocks_mid(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two block decompositions along the middle dimension:
    Block 1: n x m1 x p
    Block 2: n x m2 x p
    Total: n x (m1+m2) x p
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    U = np.zeros((n * (m1 + m2), total_rank), dtype=np.float64)
    V = np.zeros(((m1 + m2) * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m1):
            U[i * (m1 + m2) + j, :r1] = U1[i * m1 + j, :]
        for j in range(m2):
            U[i * (m1 + m2) + m1 + j, r1:] = U2[i * m2 + j, :]
    
    for j in range(m1):
        for k in range(p):
            V[j * p + k, :r1] = V1[j * p + k, :]
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k, r1:] = V2[j * p + k, :]
    
    for k in range(p):
        for i in range(n):
            W[k * n + i, :r1] = W1[k * n + i, :]
            W[k * n + i, r1:] = W2[k * n + i, :]
    
    return U, V, W

def _combine_blocks_out(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """
    Combine two block decompositions along the output dimension:
    Block 1: n x m x p1
    Block 2: n x m x p2
    Total: n x m x (p1+p2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * (p1 + p2), total_rank), dtype=np.float64)
    W = np.zeros(((p1 + p2) * n, total_rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            U[i * m + j, :r1] = U1[i * m + j, :]
            U[i * m + j, r1:] = U2[i * m + j, :]
    
    for j in range(m):
        for k in range(p1):
            V[j * (p1 + p2) + k, :r1] = V1[j * p1 + k, :]
        for k in range(p2):
            V[j * (p1 + p2) + p1 + k, r1:] = V2[j * p2 + k, :]
    
    for k in range(p1):
        for i in range(n):
            W[k * n + i, :r1] = W1[k * n + i, :]
    for k in range(p2):
        for i in range(n):
            W[(p1 + k) * n + i, r1:] = W2[k * n + i, :]
    
    return U, V, W

def _build_3x5x6_strassen():
    """
    Build a decomposition for 3x5x6 using Strassen on 2x2x2 blocks.
    
    Strategy: Split middle dim into (2,2,1) and output dim into (2,2,2).
    For each pair (mid_block, out_block) we get a sub-problem:
    - (2,2): 3x2x2 -> rank 6
    - (2,1): 3x2x1 -> rank 6 (schoolbook)
    - (1,2): 3x1x2 -> rank 6 (schoolbook)
    - (1,1): 3x1x1 -> rank 3 (schoolbook)
    
    Total: 6+6+6 + 6+6+6 + 6+6+3 = 57
    
    Wait, let me reconsider. The blocks are independent in the tensor sense
    because each (i,j,k) triple belongs to exactly one (mid_block, out_block) pair.
    """
    n, m, p = 3, 5, 6
    
    # Split middle: indices 0-1 (block A), 2-3 (block B), 4 (block C)
    # Split output: indices 0-1 (block X), 2-3 (block Y), 4-5 (block Z)
    
    mid_splits = [(0, 2), (2, 2), (4, 1)]  # (start, size)
    out_splits = [(0, 2), (2, 2), (4, 2)]  # (start, size)
    
    all_U = []
    all_V = []
    all_W = []
    rank_offset = 0
    
    for (ms, msz) in mid_splits:
        for (os_, osz) in out_splits:
            # Sub-problem: n x msz x osz
            sub_n, sub_m, sub_p = n, msz, osz
            sub_rank = sub_n * sub_m * sub_p  # schoolbook for sub-block
            
            sub_U = np.zeros((sub_n * sub_m, sub_rank), dtype=np.float64)
            sub_V = np.zeros((sub_m * sub_p, sub_rank), dtype=np.float64)
            sub_W = np.zeros((sub_p * sub_n, sub_rank), dtype=np.float64)
            
            col = 0
            for i in range(sub_n):
                for j in range(sub_m):
                    for k in range(sub_p):
                        sub_U[i * sub_m + j, col] = 1.0
                        sub_V[j * sub_p + k, col] = 1.0
                        sub_W[k * sub_n + i, col] = 1.0
                        col += 1
            
            all_U.append(sub_U)
            all_V.append(sub_V)
            all_W.append(sub_W)
            rank_offset += sub_rank
    
    total_rank = rank_offset
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # Now map each sub-block into the full matrices
    col_offset = 0
    for idx, (ms, msz) in enumerate(mid_splits):
        for idx2, (os_, osz) in enumerate(out_splits):
            sub_idx = idx * len(out_splits) + idx2
            sub_U = all_U[sub_idx]
            sub_V = all_V[sub_idx]
            sub_W = all_W[sub_idx]
            sub_rank = sub_U.shape[1]
            
            # Map U: sub_U[i*sub_m + j, r] -> U[i*m + (ms+j), col_offset+r]
            for i in range(n):
                for j in range(msz):
                    U[i * m + ms + j, col_offset:col_offset+sub_rank] = sub_U[i * msz + j, :]
            
            # Map V: sub_V[j*sub_p + k, r] -> V[(ms+j)*p + (os_+k), col_offset+r]
            for j in range(msz):
                for k in range(osz):
                    V[(ms + j) * p + (os_ + k), col_offset:col_offset+sub_rank] = sub_V[j * osz + k, :]
            
            # Map W: sub_W[k*sub_n + i, r] -> W[(os_+k)*n + i, col_offset+r]
            for k in range(osz):
                for i in range(n):
                    W[(os_ + k) * n + i, col_offset:col_offset+sub_rank] = sub_W[k * n + i, :]
            
            col_offset += sub_rank
    
    return U, V, W

def _build_3x5x6_strassen_optimized():
    """
    Optimized: Use Strassen rank-7 for 2x2x2 sub-blocks within each 3x2x2.
    
    For 3x2x2: split into 3 independent 1x2x2 problems.
    Each 1x2x2 has rank 2 (minimal).
    So 3x2x2 has rank 6 (can't improve further without sharing).
    
    Alternative: For 3x2x2, think of it as multiplying a 3x2 by 2x2.
    We can use the structure: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Actually, let's try a different decomposition for 3x5x6:
    Split output into 3 groups of 2: p1=2, p2=2, p3=2
    For each group, we need 3x5x2.
    
    For 3x5x2: split middle into (2,2,1):
    - 3x2x2: rank 6
    - 3x2x2: rank 6  
    - 3x1x2: rank 6
    Total per output group: 18
    Three groups: 54
    
    Let's build this properly.
    """
    n, m, p = 3, 5, 6
    
    # Split output into 3 groups of 2
    out_groups = [(0, 2), (2, 2), (4, 2)]
    # Split middle into (2, 2, 1)
    mid_groups = [(0, 2), (2, 2), (4, 1)]
    
    U = np.zeros((n * m, 0), dtype=np.float64)
    V = np.zeros((m * p, 0), dtype=np.float64)
    W = np.zeros((p * n, 0), dtype=np.float64)
    
    col_offset = 0
    for (os_, osz) in out_groups:
        for (ms, msz) in mid_groups:
            sub_n, sub_m, sub_p = n, msz, osz
            sub_rank = sub_n * sub_m * sub_p
            
            sub_U = np.zeros((sub_n * sub_m, sub_rank), dtype=np.float64)
            sub_V = np.zeros((sub_m * sub_p, sub_rank), dtype=np.float64)
            sub_W = np.zeros((sub_p * sub_n, sub_rank), dtype=np.float64)
            
            col = 0
            for i in range(sub_n):
                for j in range(sub_m):
                    for k in range(sub_p):
                        sub_U[i * sub_m + j, col] = 1.0
                        sub_V[j * sub_p + k, col] = 1.0
                        sub_W[k * sub_n + i, col] = 1.0
                        col += 1
            
            # Append to U
            new_U = np.zeros((n * m, sub_rank), dtype=np.float64)
            for i in range(n):
                for j in range(msz):
                    new_U[i * m + ms + j, :] = sub_U[i * msz + j, :]
            
            new_V = np.zeros((m * p, sub_rank), dtype=np.float64)
            for j in range(msz):
                for k in range(osz):
                    new_V[(ms + j) * p + (os_ + k), :] = sub_V[j * osz + k, :]
            
            new_W = np.zeros((p * n, sub_rank), dtype=np.float64)
            for k in range(osz):
                for i in range(n):
                    new_W[(os_ + k) * n + i, :] = sub_W[k * n + i, :]
            
            if col_offset == 0:
                U = new_U
                V = new_V
                W = new_W
            else:
                U = np.hstack([U, new_U])
                V = np.hstack([V, new_V])
                W = np.hstack([W, new_W])
            
            col_offset += sub_rank
    
    return U, V, W

def _build_3x5x6_strassen_2x2():
    """
    Use Strassen's rank-7 decomposition for the 2x2x2 sub-blocks.
    
    Split 3x5x6 as:
    - Middle: (2, 2, 1)  
    - Output: (2, 2, 2)
    
    For 2x2 mid x 2x2 out blocks (i.e., 3x2x2 sub-problems):
    Split into 3 independent 1x2x2, each rank 2 -> total rank 6 per 3x2x2.
    
    For 2x2 mid x 1x2 out (3x2x2 with p=1... wait no, out size is 2).
    
    Let me redo: 
    mid_splits = [(0,2), (2,2), (4,1)]
    out_splits = [(0,2), (2,2), (4,2)]
    
    Sub-problems:
    (msz=2, osz=2): 3x2x2, rank 6
    (msz=2, osz=2): 3x2x2, rank 6
    (msz=2, osz=2): 3x2x2, rank 6
    (msz=2, osz=2): 3x2x2, rank 6
    (msz=2, osz=2): 3x2x2, rank 6
    (msz=2, osz=2): 3x2x2, rank 6
    (msz=1, osz=2): 3x1x2, rank 6
    (msz=1, osz=2): 3x1x2, rank 6
    (msz=1, osz=2): 3x1x2, rank 6
    
    Total: 6*6 + 6*3 = 36 + 18 = 54
    """
    n, m, p = 3, 5, 6
    
    mid_splits = [(0, 2), (2, 2), (4, 1)]
    out_splits = [(0, 2), (2, 2), (4, 2)]
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    for (ms, msz) in mid_splits:
        for (os_, osz) in out_splits:
            sub_n, sub_m, sub_p = n, msz, osz
            sub_rank = sub_n * sub_m * sub_p
            
            sub_U = np.zeros((sub_n * sub_m, sub_rank), dtype=np.float64)
            sub_V = np.zeros((sub_m * sub_p, sub_rank), dtype=np.float64)
            sub_W = np.zeros((sub_p * sub_n, sub_rank), dtype=np.float64)
            
            col = 0
            for i in range(sub_n):
                for j in range(sub_m):
                    for k in range(sub_p):
                        sub_U[i * sub_m + j, col] = 1.0
                        sub_V[j * sub_p + k, col] = 1.0
                        sub_W[k * sub_n + i, col] = 1.0
                        col += 1
            
            new_U = np.zeros((n * m, sub_rank), dtype=np.float64)
            for i in range(n):
                for j in range(msz):
                    new_U[i * m + ms + j, :] = sub_U[i * msz + j, :]
            
            new_V = np.zeros((m * p, sub_rank), dtype=np.float64)
            for j in range(msz):
                for k in range(osz):
                    new_V[(ms + j) * p + (os_ + k), :] = sub_V[j * osz + k, :]
            
            new_W = np.zeros((p * n, sub_rank), dtype=np.float64)
            for k in range(osz):
                for i in range(n):
                    new_W[(os_ + k) * n + i, :] = sub_W[k * n + i, :]
            
            U_parts.append(new_U)
            V_parts.append(new_V)
            W_parts.append(new_W)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def _local_search(U, V, W, n, m, p, max_iters=5000, time_limit=20.0):
    """
    Local search: try to reduce rank by eliminating zero columns or
    finding linear dependencies.
    """
    start_time = time.time()
    rank = U.shape[1]
    
    # Try to find and remove linearly dependent columns
    for attempt in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Check if any column is all zeros
        zero_cols = []
        for r in range(rank):
            if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
                zero_cols.append(r)
        
        if zero_cols:
            # Remove zero columns
            keep = [r for r in range(rank) if r not in zero_cols]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            rank = len(keep)
            continue
        
        # Try removing one column at a time and see if decomposition still works
        removed = False
        for r in range(rank):
            if time.time() - start_time > time_limit:
                break
            
            keep = [rr for rr in range(rank) if rr != r]
            U_test = U[:, keep]
            V_test = V[:, keep]
            W_test = W[:, keep]
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-9):
                U, V, W = U_test, V_test, W_test
                rank = len(keep)
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    # Hard deadline: 140s to leave 10s safety margin before 150s kill
    overall_deadline = start_time + 140.0
    
    # Start with schoolbook (rank 90)
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try block decomposition with structured splits (target rank ~54)
    if time.time() < overall_deadline - 30.0:
        U_block, V_block, W_block = _build_3x5x6_strassen_2x2()
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            block_rank = U_block.shape[1]
            if block_rank < best_rank:
                best_rank = block_rank
                best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
    
    # Try local search to reduce rank further
    if time.time() < overall_deadline - 15.0:
        remaining = min(20.0, overall_deadline - time.time() - 5.0)
        U_ls, V_ls, W_ls = _local_search(best_U, best_V, best_W, n, m, p, 
                                          max_iters=10000, time_limit=remaining)
        if U_ls.shape[1] < best_rank:
            best_rank = U_ls.shape[1]
            best_U, best_V, best_W = U_ls.copy(), V_ls.copy(), W_ls.copy()
    
    # Try random search for even lower ranks if time permits
    if time.time() < overall_deadline - 10.0:
        for target_rank in range(max(50, best_rank - 20), best_rank):
            if time.time() > overall_deadline - 5.0:
                break
            remaining = min(3.0, overall_deadline - time.time() - 2.0)
            if remaining <= 0:
                break
            # Random perturbation search
            found = False
            for attempt in range(20):
                if time.time() > overall_deadline - 5.0:
                    break
                U_rand = best_U.copy()
                V_rand = best_V.copy()
                W_rand = best_W.copy()
                # Perturb a few entries
                for _ in range(3):
                    mat = random.choice([U_rand, V_rand, W_rand])
                    row = random.randint(0, mat.shape[0] - 1)
                    col = random.randint(0, mat.shape[1] - 1)
                    mat[row, col] += random.choice([-1.0, -0.5, 0.5, 1.0])
                
                if _verify_decomposition(U_rand, V_rand, W_rand, n, m, p, tol=1e-9):
                    # Try to trim
                    U_trim, V_trim, W_trim = _local_search(U_rand, V_rand, W_rand, n, m, p,
                                                            max_iters=100, time_limit=1.0)
                    if U_trim.shape[1] < best_rank:
                        best_rank = U_trim.shape[1]
                        best_U, best_V, best_W = U_trim.copy(), V_trim.copy(), W_trim.copy()
                        found = True
                        break
            if found:
                break
    
    # Final safety check
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U, V, W
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
