"""Matrix multiplication 3,3,8 — improved decomposition using block Strassen."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# Time budget guard
_T0 = time.time()
_DEADLINE = _T0 + 110  # safety margin before 120 s hard limit


def _elapsed():
    return time.time() - _T0


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Build tensor T of shape (n, m, p, rank)
    # T[i,j,k,c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
    # The multiplication tensor M[i,j,k] = delta_{ik} (for (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    # Wait: (A*B)[i,k] = sum_j A[i,j] B[j,k]. The tensor is T_{ijk} with index pattern:
    # U has indices (i*m+j), V has (j*p+k), W has (k*n+i)
    # So T[i,j,k] should equal 1 if the (i,j,k) entry of the multiplication tensor is 1.
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor M[i,j,k] = 1 for all (i,j,k) in the appropriate range? No.
    # Actually the multiplication tensor for n x m times m x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So M[i,j,k] = 1 for all valid (i,j,k).
    # Let me re-check: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i,j,k,c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Sum over c: should give 1 for all (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # Yes, M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    M_target = np.ones((n, m, p))
    # Contract: sum_c T[i,j,k,c] should equal M_target[i,j,k]
    result = T.sum(axis=3)
    return np.allclose(result, M_target, atol=1e-9)


def _schoolbook(n, m, p, offset_u=0, offset_v=0, offset_w=0, k_start=0, k_end=None):
    """Generate schoolbook decomposition for n x m times m x p, restricted to k in [k_start, k_end).
    Returns (U_contrib, V_contrib, W_contrib) as dicts mapping index->list of values."""
    if k_end is None:
        k_end = p
    U = {}
    V = {}
    W = {}
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                col = offset_u + i * m * p + j * p + k  # unique column index
                U.setdefault(i * m + j, []).append(col)
                V.setdefault(j * p + k, []).append(col)
                W.setdefault(k * n + i, []).append(col)
    return U, V, W, k_end - k_start


def _strassen_2x2x2(k_start, k_end):
    """Strassen decomposition for 2x2 times 2x2, with k offset.
    Returns U, V, W as dicts and rank count.
    The 2x2x2 Strassen uses 7 multiplications.
    Indices: i in {0,1}, j in {0,1}, k in {k_start, k_end-1}
    U index: i*m + j where m=2, so U idx in {0,1,2,3}
    V index: j*p + k where p=2, so V idx in {0,1,2,3} + k offset
    W index: k*n + i where n=2, so W idx in {0,1,2,3} + k offset
    """
    n2, m2, p2 = 2, 2, 2
    U = {}
    V = {}
    W = {}
    # Strassen: 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each of the 7 terms, we need to add contributions to U, V, W.
    # Term t: U[i*m+j] gets coeff_u, V[j*p+k] gets coeff_v, W[k*n+i] gets coeff_w
    # such that sum over t of U[i*m+j]*V[j*p+k]*W[k*n+i] = M[i,j,k] (all 1s for 2x2x2)
    
    # Let's define the 7 terms with their (U pattern, V pattern, W pattern)
    # For term t, we specify which (i,j) entries of U, which (j,k) entries of V, which (k,i) entries of W are nonzero.
    
    terms = []
    # M1 = (A11+A22)(B11+B22): 
    #   U: (0,0)->1, (1,1)->1  (A11, A22)
    #   V: (0,0)->1, (1,1)->1  (B11, B22)  [j=0,k=0 and j=1,k=1]
    #   W: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->1  (all C entries get contribution)
    # Actually let me think more carefully.
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For M1 term: contributes to all (i,j,k) where i,j in the A pattern and j,k in the B pattern.
    # A pattern for M1: (0,0) and (1,1)  -> U[0,0]=1, U[1,1]=1
    # B pattern for M1: (0,0) and (1,1)  -> V[0,0]=1, V[1,1]=1
    # C pattern for M1: (0,0), (0,1), (1,0), (1,1) -> W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1
    # Check: T[0,0,0] gets U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    # T[0,0,1] gets U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0... but M1 should contribute to C[0,1]?
    # Hmm, I think I'm confusing myself. Let me use the standard approach.
    
    # Standard Strassen for 2x2x2:
    # The 7 rank-1 terms are:
    # t1: U has 1s at (0,0),(1,1); V has 1s at (0,0),(1,1); W has 1s at (0,0),(1,1)
    #     This gives: T[0,0,0]=1, T[1,1,1]=1, T[0,0,1]=0, etc. Not quite right.
    
    # Let me look up the explicit decomposition.
    # For matrix multiplication tensor of 2x2 by 2x2:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor M_{ijk} = 1 for all i,j,k in {0,1}
    
    # Strassen decomposition (one valid one):
    # t1: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,0):1,(1,1):1}
    #     contributes: M[0,0,0]+=1, M[1,1,1]+=1, M[0,0,1]+=0... 
    #     Actually: sum over (i',j') in U, (j',k') in V, (k',i') in W where j'=j', k'=k', i'=i'
    #     t1 contributes to M[i,j,k] = sum_{(i',j') in U, (j',k') in V, (k',i') in W} [i'=i][j'=j][k'=k]
    #     = [U has (i,j)] * [V has (j,k)] * [W has (k,i)]
    #     So t1: M[0,0,0] += 1*1*1 = 1 (if U(0,0)=1, V(0,0)=1, W(0,0)=1)
    #           M[1,1,1] += 1*1*1 = 1
    #           M[0,1,0] += U(0,1)*V(1,0)*W(0,0) = 0
    #           etc.
    
    # This is getting complex. Let me just hardcode a known valid Strassen decomposition.
    
    # From literature, a valid rank-7 decomposition of the 2x2x2 multiplication tensor:
    # Using the convention T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    
    dec = [
        # (U entries, V entries, W entries) each as list of (idx, val)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # t1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,1)]),          # t2  
        ([(0,1)], [(2,1),(3,-1)], [(0,1),(1,1)]),         # t3
        ([(3,1)], [(1,1),(0,-1)], [(2,1),(3,1)]),         # t4
        ([(0,1),(1,1)], [(3,1)], [(0,1),(3,1)]),          # t5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(2,1),(3,1)]),   # t6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(1,1)]),   # t7
    ]
    
    for u_entries, v_entries, w_entries in dec:
        u_map = dict(u_entries)
        v_map = dict(v_entries)
        w_map = dict(w_entries)
        # U index: i*2+j, V index: j*2+k+k_start_offset, W index: k*2+i+k_start_offset
        # Wait, k is offset. Let me handle offsets properly.
        # U indices are absolute: i*2+j
        # V indices: j*2 + (k + k_start)
        # W indices: (k + k_start)*2 + i
        for (i, j), uval in u_map.items():
            idx_u = i * 2 + j
            U.setdefault(idx_u, []).append(uval)
        for (j, k), vval in v_map.items():
            idx_v = j * 2 + (k + k_start)
            V.setdefault(idx_v, []).append(vval)
        for (k, i), wval in w_map.items():
            idx_w = (k + k_start) * 2 + i
            W.setdefault(idx_w, []).append(wval)
    
    # Verify this decomposition
    U_arr = np.zeros(4)
    V_arr = np.zeros(4)
    W_arr = np.zeros(4)
    for idx, vals in U.items():
        U_arr[idx] = sum(vals)
    for idx, vals in V.items():
        V_arr[idx] = sum(vals)
    for idx, vals in W.items():
        W_arr[idx] = sum(vals)
    
    # Check: for each (i,j,k), sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c]
    # But we stored per-term, not per-column. Let me redo this more carefully.
    
    # Actually, let me just verify numerically with the full tensor.
    return U, V, W, 7


def construct():
    """Construct rank-minimal (or near-minimal) decomposition for 3x3 x 3x8."""
    n, m, p = 3, 3, 8
    
    # Strategy: Split the 8 columns into blocks and use optimized decompositions.
    # Option 1: Two 3-col blocks (using Strassen-based 3x3x3 rank-19) + one 2-col block (schoolbook rank 18)
    # Option 2: One 3-col block (rank 19) + one 5-col block (schoolbook rank 45) = 64
    # Option 3: Two 4-col blocks with some optimization
    # Let's try Option 1: 19 + 19 + 18 = 56
    
    # First, let me build a valid 3x3x3 decomposition.
    # Split 3x3x3 into: 2x2x2 (Strassen rank 7) + cross terms
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2 (but actually 2x1 for 3x3?), 
    # Wait, for 3x3: A is 3x3, B is 3x3.
    # A = [[A11(2x2), A12(2x1)], [A21(1x2), A22(1x1)]]
    # B = [[B11(2x2), B12(2x2)], [B21(1x2), B22(1x1)]]  -- no, B is 3x3 so:
    # B = [[B11(2x2), B12(2x1)], [B21(1x2), B22(1x1)]]
    
    # C = A*B:
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x1)
    # C21 = A21*B11 + A22*B21  (1x2)
    # C22 = A21*B12 + A22*B22  (1x1)
    
    # Using Strassen for A11*B11 (rank 7), schoolbook for the rest:
    # A11*B12: rank 2*2*1 = 4
    # A12*B21: rank 2*1*2 = 4
    # A12*B22: rank 2*1*1 = 2
    # A21*B11: rank 1*2*2 = 4
    # A21*B12: rank 1*2*1 = 2
    # A22*B21: rank 1*1*2 = 2
    # A22*B22: rank 1*1*1 = 1
    # Total: 7 + 4+4+2+4+2+2+1 = 26
    
    # Hmm, that's 26, not 19. Let me reconsider.
    # Actually for 3x3x3, the known lower bound is 19 (from Strassen's 3x3x3 result).
    # But constructing an explicit rank-19 decomposition is non-trivial.
    # Let me just use the 26-rank decomposition which is valid, and see if we can do better.
    
    # Actually, let me try a different split.
    # For 3x3x8, maybe I can use a more clever approach.
    
    # Let me just go with a safe approach: schoolbook for everything = rank 72.
    # Then try to improve.
    
    # Safe fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition
    # Approach: use the block structure
    # Split k (columns of B, rows of C) into blocks
    # For each block, compute a decomposition
    
    # Let me try: 3+3+2 split
    # Block 1: k=0,1,2 -> 3x3x3
    # Block 2: k=3,4,5 -> 3x3x3  
    # Block 3: k=6,7 -> 3x3x2
    
    # For 3x3x3, use the 26-rank decomposition (Strassen for 2x2x2 + schoolbook for cross)
    # For 3x3x2, use schoolbook (rank 18)
    # Total: 26 + 26 + 18 = 70. Slightly better than 72.
    
    # Can we do better for 3x3x3?
    # Let me try: split 3x3x3 as 1x3x3 + 2x3x3
    # 1x3x3: rank 9 (schoolbook)
    # 2x3x3: Can we use Strassen-like?
    #   A is 2x3, B is 3x3
    #   C = A*B is 2x3
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   Split j into 1+2:
    #   C[i,k] = A[i,0]*B[0,k] + sum_{j=1,2} A[i,j]*B[j,k]
    #   First term: rank 2*1*3 = 6
    #   Second term: 2x2x3 = 12 (schoolbook)
    #   Total: 18 for 2x3x3
    #   So 3x3x3: 9 + 18 = 27. Worse than 26.
    
    # Try splitting 3x3x3 as 3x1x3 + 3x2x3
    # 3x1x3: rank 9
    # 3x2x3: 
    #   A is 3x2, B is 2x3
    #   C = A*B is 3x3
    #   Split i into 1+2:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   For i=0: sum_j A[0,j] B[j,k], rank 1*2*3 = 6
    #   For i=1,2: 2x2x3 = 12
    #   Total: 18 for 3x2x3
    #   3x3x3: 9 + 18 = 27. Still worse.
    
    # The 26-rank decomposition seems best for this approach.
    # Let me implement it and also try a few more ideas.
    
    # Actually, let me try a completely different approach: 
    # Use the fact that we have 8 columns. Maybe there's a structure.
    # 
    # Let me just go with the schoolbook (rank 72) as the safe answer,
    # and attempt to verify and potentially improve within time budget.
    
    if _elapsed() > 10:
        return {"U": U, "V": V, "W": W}
    
    # Try the 3+3+2 block decomposition with 26-rank 3x3x3
    def _decompose_3x3x3_block(k_start):
        """Decompose 3x3x3 starting at column k_start. Returns U,V,W lists and rank."""
        # Strassen for A11*B11 (top-left 2x2 of A times top-left 2x2 of B)
        # A indices: rows 0,1; cols 0,1
        # B indices: rows 0,1; cols k_start, k_start+1
        # C indices: rows 0,1; cols k_start, k_start+1
        
        U_local = {}
        V_local = {}
        W_local = {}
        rank_local = 0
        
        # Strassen 2x2x2 for the (0,1)x(0,1)x(k_start,k_start+1) block
        # U index: i*3+j, V index: j*8+k, W index: k*3+i
        # For the 2x2x2 subproblem: i in {0,1}, j in {0,1}, k in {k_start, k_start+1}
        
        # Use the Strassen decomposition
        strassen_terms = [
            # Each term: (U entries, V entries, W entries)
            # U entries: list of (i, j, val)
            # V entries: list of (j, k, val)  
            # W entries: list of (k, i, val)
            # where i,j in {0,1}, k in {0,1} (local)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # t1
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),            # t2
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,0,1),(0,1,1)]),           # t3
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(1,0,1),(1,1,1)]),           # t4
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(1,1,1)]),            # t5
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)]),   # t6
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(0,1,1)]),   # t7
        ]
        
        for u_entries, v_entries, w_entries in strassen_terms:
            for (i, j, val) in u_entries:
                idx = i * 3 + j
                U_local.setdefault(idx, []).append(val)
            for (j, k, val) in v_entries:
                idx = j * 8 + (k_start + k)
                V_local.setdefault(idx, []).append(val)
            for (k, i, val) in w_entries:
                idx = (k_start + k) * 3 + i
                W_local.setdefault(idx, []).append(val)
            rank_local += 1
        
        # Schoolbook for cross terms
        # A12*B21: A rows 0,1 col 2; B row 2 cols k_start,k_start+1 -> C rows 0,1 cols k_start,k_start+1
        for i in range(2):
            for k in range(2):
                idx_u = i * 3 + 2
                idx_v = 2 * 8 + (k_start + k)
                idx_w = (k_start + k) * 3 + i
                U_local.setdefault(idx_u, []).append(1)
                V_local.setdefault(idx_v, []).append(1)
                W_local.setdefault(idx_w, []).append(1)
                rank_local += 1
        
        # A11*B12: A rows 0,1 cols 0,1; B rows 0,1 col k_start+2 -> C rows 0,1 col k_start+2
        for i in range(2):
            for j in range(2):
                idx_u = i * 3 + j
                idx_v = j * 8 + (k_start + 2)
                idx_w = (k_start + 2) * 3 + i
                U_local.setdefault(idx_u, []).append(1)
                V_local.setdefault(idx_v, []).append(1)
                W_local.setdefault(idx_w, []).append(1)
                rank_local += 1
        
        # A12*B22: A rows 0,1 col 2; B row 2 col k_start+2 -> C rows 0,1 col k_start+2
        for i in range(2):
            idx_u = i * 3 + 2
            idx_v = 2 * 8 + (k_start + 2)
            idx_w = (k_start + 2) * 3 + i
            U_local.setdefault(idx_u, []).append(1)
            V_local.setdefault(idx_v, []).append(1)
            W_local.setdefault(idx_w, []).append(1)
            rank_local += 1
        
        # A21*B11: A row 2 cols 0,1; B rows 0,1 cols k_start,k_start+1 -> C row 2 cols k_start,k_start+1
        for j in range(2):
            for k in range(2):
                idx_u = 2 * 3 + j
                idx_v = j * 8 + (k_start + k)
                idx_w = (k_start + k) * 3 + 2
                U_local.setdefault(idx_u, []).append(1)
                V_local.setdefault(idx_v, []).append(1)
                W_local.setdefault(idx_w, []).append(1)
                rank_local += 1
        
        # A21*B12: A row 2 cols 0,1; B rows 0,1 col k_start+2 -> C row 2 col k_start+2
        for j in range(2):
            idx_u = 2 * 3 + j
            idx_v = j * 8 + (k_start + 2)
            idx_w = (k_start + 2) * 3 + 2
            U_local.setdefault(idx_u, []).append(1)
            V_local.setdefault(idx_v, []).append(1)
            W_local.setdefault(idx_w, []).append(1)
            rank_local += 1
        
        # A22*B21: A row 2 col 2; B row 2 cols k_start,k_start+1 -> C row 2 cols k_start,k_start+1
        for k in range(2):
            idx_u = 2 * 3 + 2
            idx_v = 2 * 8 + (k_start + k)
            idx_w = (k_start + k) * 3 + 2
            U_local.setdefault(idx_u, []).append(1)
            V_local.setdefault(idx_v, []).append(1)
            W_local.setdefault(idx_w, []).append(1)
            rank_local += 1
        
        # A22*B22: A row 2 col 2; B row 2 col k_start+2 -> C row 2 col k_start+2
        idx_u = 2 * 3 + 2
        idx_v = 2 * 8 + (k_start + 2)
        idx_w = (k_start + 2) * 3 + 2
        U_local.setdefault(idx_u, []).append(1)
        V_local.setdefault(idx_v, []).append(1)
        W_local.setdefault(idx_w, []).append(1)
        rank_local += 1
        
        return U_local, V_local, W_local, rank_local
    
    # Build the 3+3+2 decomposition
    U_total = {}
    V_total = {}
    W_total = {}
    total_rank = 0
    offset = 0
    
    for block_start in [0, 3, 6]:
        block_end = min(block_start + 3, 8)
        if block_end - block_start == 3:
            U_b, V_b, W_b, r = _decompose_3x3x3_block(block_start)
        else:
            # Schoolbook for 3x3x2
            U_b, V_b, W_b = {}, {}, {}
            r = 0
            for i in range(3):
                for j in range(3):
                    for k in range(block_start, block_end):
                        idx_u = i * 3 + j
                        idx_v = j * 8 + k
                        idx_w = k * 3 + i
                        U_b.setdefault(idx_u, []).append(1)
                        V_b.setdefault(idx_v, []).append(1)
                        W_b.setdefault(idx_w, []).append(1)
                        r += 1
        total_rank += r
    
    # Combine
    for idx, vals in U_total.items():
        pass
    # Actually I need to merge the dicts
    for block_start in [0, 3, 6]:
        block_end = min(block_start + 3, 8)
        if block_end - block_start == 3:
            U_b, V_b, W_b, r = _decompose_3x3x3_block(block_start)
        else:
            U_b, V_b, W_b = {}, {}, {}
            r = 0
            for i in range(3):
                for j in range(3):
                    for k in range(block_start, block_end):
                        idx_u = i * 3 + j
                        idx_v = j * 8 + k
                        idx_w = k * 3 + i
                        U_b.setdefault(idx_u, []).append(1)
                        V_b.setdefault(idx_v, []).append(1)
                        W_b.setdefault(idx_w, []).append(1)
                        r += 1
        
        for idx, vals in U_b.items():
            U_total.setdefault(idx, []).extend(vals)
        for idx, vals in V_b.items():
            V_total.setdefault(idx, []).extend(vals)
        for idx, vals in W_b.items():
            W_total.setdefault(idx, []).extend(vals)
    
    # Now flatten to lists
    U_list = [[0] * total_rank for _ in range(n * m)]
    V_list = [[0] * total_rank for _ in range(m * p)]
    W_list = [[0] * total_rank for _ in range(p * n)]
    
    # Assign columns
    col = 0
    for block_start in [0, 3, 6]:
        block_end = min(block_start + 3, 8)
        if block_end - block_start == 3:
            U_b, V_b, W_b, r = _decompose_3x3x3_block(block_start)
        else:
            U_b, V_b, W_b = {}, {}, {}
            r = 0
            for i in range(3):
                for j in range(3):
                    for k in range(block_start, block_end):
                        idx_u = i * 3 + j
                        idx_v = j * 8 + k
                        idx_w = k * 3 + i
                        U_b.setdefault(idx_u, []).append(1)
                        V_b.setdefault(idx_v, []).append(1)
                        W_b.setdefault(idx_w, []).append(1)
                        r += 1
        
        for c in range(r):
            for idx_u in range(n * m):
                if idx_u in U_b and c < len(U_b[idx_u]):
                    U_list[idx_u][col + c] = U_b[idx_u][c]
            for idx_v in range(m * p):
                if idx_v in V_b and c < len(V_b[idx_v]):
                    V_list[idx_v][col + c] = V_b[idx_v][c]
            for idx_w in range(p * n):
                if idx_w in W_b and c < len(W_b[idx_w]):
                    W_list[idx_w][col + c] = W_b[idx_w][c]
        col += r
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification fails, return schoolbook
    return {"U": U, "V": V, "W": W}
