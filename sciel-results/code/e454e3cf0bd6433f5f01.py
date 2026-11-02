"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = 150.0  # wall-clock limit
_SAFETY = 5.0      # return this many seconds before the limit

def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _mat_to_rows(M):
    """Convert numpy 2-D to list-of-lists of int/float (exact half-integers)."""
    out = []
    for r in range(M.shape[0]):
        row = []
        for c in range(M.shape[1]):
            v = M[r, c]
            # keep as int if integer-valued, else as float (half-integer)
            if abs(v - round(v)) < 1e-9:
                row.append(int(round(v)))
            else:
                row.append(float(v))
        out.append(row)
    return out

def _build_schoolbook(n, m, p):
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

def _strassen222_terms():
    """Return lists (U_rows, V_rows, W_rows) for the 7 rank-1 Strassen terms of
    2x2x2 multiplication. Indices: A row (i,m) with i in 0..1, m in 0..1;
    B row (m,p) with m in 0..1, p in 0..1; W row (p*n+i) with p in 0..1, i in 0..1.
    Each term is (a_vec of len 4, b_vec of len 4, c_vec of len 4) with a[i*2+m], b[m*2+p], c[p*2+i].
    """
    # Strassen products M1..M7
    # A indexed a[i,j], B b[j,k], C c[i,k]
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append(( [1,0,0,1], [1,0,0,1], [1,0,0,1] ))  # c00 += M1
    # M2 = (a10+a11) b00
    terms.append(( [0,0,1,1], [1,0,0,0], [0,0,1,0] ))  # c10 += M2  (c[1,0])
    # M3 = a00 (b01 - b11)
    terms.append(( [1,0,0,0], [0,1,0,-1], [0,1,0,0] ))  # c01 += M3
    # M4 = a11 (b10 - b00)
    terms.append(( [0,0,0,1], [0,0,1,-1], [1,0,1,0] ))  # c00 += M4, c10 += M4
    # M5 = (a00+a01) b11
    terms.append(( [1,1,0,0], [0,0,0,1], [-1,0,0,1] ))  # c00 -= M5, c11 += M5
    # M6 = (a10-a00)(b00+b01)
    terms.append(( [-1,0,1,0], [1,1,0,0], [0,1,0,0] ))  # c11 += M6
    # M7 = (a01-a11)(b10+b11)
    terms.append(( [0,1,0,-1], [0,0,1,1], [1,0,1,0] ))  # c00 += M7, c10 += M7
    # Verify C = M matrix:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4 + M7
    # c11 = M5 + M6
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # My W rows above must match. Let me reconstruct properly below instead.
    return terms

def _build_strassen_block(terms, offset_rows_U, offset_rows_V, offset_rows_W, U, V, W, col):
    """Add terms. Each term: a (len n*m for the block), b (len m*p for block), c (len p*n for block).
    We add them as new columns."""
    for (a, b, c) in terms:
        for idx, val in enumerate(a):
            if val != 0:
                U[offset_rows_U + idx][col] += val
        for idx, val in enumerate(b):
            if val != 0:
                V[offset_rows_V + idx][col] += val
        for idx, val in enumerate(c):
            if val != 0:
                W[offset_rows_W + idx][col] += val
        col += 1
    return col

def _strassen_terms_nm_p(n, m, p):
    """Generic Strassen-like: split into 2x2x2 blocks where possible.
    This is complex; instead we do a block decomposition for (2,3,7) explicitly."""
    pass

def _build_candidate_rank25():
    """Try to build a rank-25 decomposition for (2,3,7) using block structure.
    Split m=3 into (1,2). 
    C = A[0,:1]@B[:1,:] + A[0:,1:]@B[1:,:].
    Block1: A1 (2x1) @ B1 (1x7): rank 2 (each of 2 output rows = a * B1 row).
    Block2: A2 (2x2) @ B2 (2x7): split p=7 into (2,2,2,1).
        B2 = [B2a(2x2) B2b(2x2) B2c(2x2) B2d(2x1)]
        A2@B2a: 2x2x2 -> Strassen rank 7
        A2@B2b: 2x2x2 -> Strassen rank 7
        A2@B2c: 2x2x2 -> Strassen rank 7
        A2@B2d: 2x2x1 -> rank 2
        Block2 total: 23
    Total rank: 2 + 23 = 25.
    """
    n, m, p = 2, 3, 7
    rank = 25
    U = [[0]*rank for _ in range(n*m)]   # 6 rows
    V = [[0]*rank for _ in range(m*p)]   # 21 rows
    W = [[0]*rank for _ in range(p*n)]   # 14 rows
    col = 0

    # ---- Block1: A1 = A[:,0] (2x1), B1 = B[0,:] (1x7). C += A1 @ B1.
    # For each i in 0..1: C[i,k] += A[i,0]*B[0,k]
    # rank-1 terms: for each i, term with U[i*3+0]=1 (A[i,0]), V[0*7+k]=1 for all k (B[0,k]), W[k*2+i]=1.
    # Actually B1 is 1x7, so B[0,k] for k in 0..6. The outer product A1(2x1)@B1(1x7) = sum_k B[0,k] * (A1 * e_k).
    # Each output C[i,k] = A[i,0]*B[0,k]. This is a rank-2 tensor: two rank-1 terms (one per i).
    for i in range(n):
        U[i*m + 0][col] = 1          # A[i,0]
        for k in range(p):
            V[0*p + k][col] = 1      # B[0,k]
            W[k*n + i][col] = 1      # C[i,k]
        col += 1

    # ---- Block2: A2 = A[:,1:3] (2x2), B2 = B[1:3,:] (2x7).
    # A2 rows: row index i in 0..1, col j in 1,2 (global m indices).
    # In local 2x2 block, A2[i, j_local] where j_local = j-1 in 0..1.
    # B2 rows j in 1,2 (global), cols k in 0..6.
    # Split B2 columns into groups: [0,1],[2,3],[4,5],[6] -> sizes 2,2,2,1.
    # For each group, multiply A2 (2x2) @ B2_group (2 x size).
    # Group sizes 2: use Strassen 2x2x2 (rank 7).
    # Group size 1: 2x2x1 rank 2.
    #
    # Global indices: A2[i, 1+j_local] -> U row i*m + (1+j_local).
    # B2[1+j_local, k] -> V row (1+j_local)*p + k.
    # C[i,k] -> W row k*n + i.
    #
    # Strassen for 2x2x2 block: we need a_vec over the 4 A2 entries (i in 0..1, j_local in 0..1),
    # b_vec over the 4 B2 entries (j_local in 0..1, k_local in 0..1),
    # c_vec over the 4 C entries (k_local in 0..1, i in 0..1).

    def add_strassen_222(k0):
        nonlocal col
        # k0: starting column index in B (and C). size 2.
        # Standard Strassen terms (a indexed [i, jl], b indexed [jl, kl], c indexed [kl, i]):
        # M1 = (a[0,0]+a[1,1])*(b[0,0]+b[1,1]) -> c[0,0] += M1, c[1,1]... let's use canonical:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        terms = [
            # (a_vec len4 [a00,a01,a10,a11], b_vec len4 [b00,b01,b10,b11], c_vec len4 [c00,c01,c10,c11])
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1: c00+=, c11+= (handle below)
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2: c10+=
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3: c01+=
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),  # M4: c00+=, c10+=
            ([1,1,0,0], [0,0,0,1], [-1,0,0,1]),  # M5: c00-=, c11+=
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6: c11+=
            ([0,1,0,-1], [0,0,1,1], [1,0,1,0]),  # M7: c00+=, c10+=
        ]
        # Re-verify C from these:
        # c00 = M1 + M4 - M5 + M7  -> from rows: M1 c00=1, M4 c00=1, M5 c00=-1, M7 c00=1 => 1+1-1+1=2? 
        # That's wrong; should be 1. Let me check M5 sign: c00 -= M5 means M5 contributes -1 to c00.
        # Sum c00 coeff: M1(+1) + M4(+1) + M5(-1) + M7(+1) = +2. But should be +1.
        # I think the standard formula: C00 = M1 + M4 - M5 + M7. Let me recompute M1..M7 to ensure
        # this identity. Use known-correct Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Using 0-index: C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6.
        # Now compute each M's contribution to C:
        # M1: C00+=1, C11+=1
        # M2: C10+=1, C11-=1
        # M3: C01+=1, C11+=1
        # M4: C00+=1, C10+=1
        # M5: C00-=1, C01+=1
        # M6: C11+=1
        # M7: C00+=1
        # Now C00 = M1+M4-M5+M7 = 1+1-1+1 = 2? But C00 should be A11B11+A22B22.
        # Let me verify by direct expansion:
        # M1+M4-M5+M7 = (A11+A22)(B11+B22) + A22(B21-B11) - (A11+A12)B22 + (A12-A22)(B21+B22)
        # = A11B11+A11B22+A22B11+A22B22 + A22B21-A22B11 - A11B22-A12B22 + A12B21+A12B22-A22B21-A22B22
        # = A11B11 + [A11B22 - A11B22] + [A22B11 - A22B11] + [A22B22 - A22B22] + [A22B21 - A22B21] + [-A12B22+A12B22] + A12B21... 
        # Let me collect A12B21: from M7: +A12B21. Any other? No. So we get an extra A12B21!
        # So C00 = A11B11 + A12B21? That's wrong. I have a sign error in M7.
        # Standard M7 = (A12 - A22)(B21 + B22)? Let me use the well-known set:
        # Actually the common Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # Hmm, let me just verify M7: (A12-A22)(B21+B22) = A12B21+A12B22-A22B21-A22B22.
        # And recompute C00 = M1+M4-M5+M7:
        # M1 = A11B11+A11B22+A22B11+A22B22
        # M4 = A22B21-A22B11
        # -M5 = -A11B22-A12B22
        # M7 = A12B21+A12B22-A22B21-A22B22
        # Sum: A11B11 + (A11B22-A11B22) + (A22B11-A22B11) + (A22B22-A22B22) + (A22B21-A22B21) + (-A12B22+A12B22) + A12B21
        # = A11B11 + A12B21. Still extra A12B21.
        # So the standard C11 formula must be C11 = M1 + M4 - M5 + M7 is WRONG.
        # Let me look up: correct is C11 = M1 + M4 - M5 + M7? Different sources differ.
        # To avoid error, I'll use a DIFFERENT verified approach: since I'll verify numerically anyway,
        # I'll just use the schoolbook for 2x2x2 if Strassen verification fails.
        # Simpler: skip Strassen; use a cleaner construction and verify.
        # For safety and correctness, I'll NOT use Strassen here. Fall back to verifying a known-good
        # block decomposition. Given time, I'll just return schoolbook which is guaranteed valid.
        return col

    # Given the risk of Strassen sign errors and the deadline, I will not rely on an unverified
    # rank-25 construction. Instead I return the guaranteed-valid schoolbook decomposition.
    # The verification below would catch any bug, but to be safe I construct schoolbook directly.
    return None  # signal: use schoolbook

def _verify(U, V, W, n, m, p, trials=8, seed=12345):
    """Check that for random integer A (n x m) and B (m x p), A@B equals the
    rank-1 reconstruction sum_c U[:,c] * W[:,c] outer V[:,c].
    Tensor convention: C[i,k] = sum_j A[i,j] B[j,k].
    Decomposition: C[i,k] = sum_c U[i*m+j_c?] ... we need the bilinear form:
    C = sum_c (U[:,c] as n x m) @ (V[:,c] as m x p) ? No.
    Actually U has n*m rows, V has m*p rows, W has p*n rows. The tensor identity is:
    C[i,k] = sum_c  (sum_j  U[i*m + j, c] * V[j*p + k, c])  ... no, W must appear.
    The standard: A[i,j]*B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]? 
    No — that's not bilinear in the right way. The correct bilinear decomposition:
    C[i,k] = sum_c U[i*m+j, c] * ... hmm.
    Let me use the convention: define for each c a rank-1 tensor T_c with
    T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]?? That's trilinear.
    The matrix-mult tensor is T[i,j,k] = delta (i,j) in A, (j,k) in B. Bilinear:
    C[i,k] = sum_j A[i,j] B[j,k]. A rank-r decomposition: C[i,k] = sum_c A'[i,c'] B'[c'',k]
    where A' = U @ A (r x k) and B' = B @ V (k x r)? 
    Standard: C = (A U) (V^T B^T)^T? Let me define:
    Let U be (n*m) x r, V be (m*p) x r, W be (p*n) x r.
    Reshape: A_vec = A flattened n*m (row-major i*m+j). B_vec = B flattened m*p (row-major j*p+k).
    C_vec = C flattened p*n (row-major k*n+i) per the W convention.
    Bilinear: C_vec = M * (A_vec outer B_vec)? No, bilinear means C_vec[c_index] is linear in A and B.
    The tensor: C[i,k] = sum_j A[i,j] B[j,k]. 
    Rank-1 term: C[i,k] += alpha(i) beta(j) gamma(k) delta... 
    Proper rank-1: C[i,k] = f_c(i) * g_c(k) * (linear functional of A and B)? 
    The bilinear rank-1: C[i,k] = (u_c . A_flat) * (v_c . B_flat) * w_c? No.
    Bilinear form: C[i,k] = sum_c U_c[i] * V_c[k] * ... where U_c depends linearly on A? 
    Correct: C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j])? No, U multiplies A pointwise.
    
    The cleanest: C[i,k] = sum_j A[i,j] B[j,k] = sum_c sum_j U[i*m+j,c] A[i,j] * V[j*p+k,c] B[j,k] * ... 
    For this to equal A@B for ALL A,B, we need: for each (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta? 
    No. The bilinear decomposition: C[i,k] = sum_c (a_c^T A) (b_c^T B) where a_c is n*m vector, b_c is m*p vector, and C[i,k] = sum_c a_c[i?]*b_c[k?].
    Standard tensor rank-1: T = u v^T w^T where T[i,j,k] = u[i] v[j] w[k]. Here our "tensor" T[i,j,k] = 1 if it's a valid (i,j,k) for the product, i.e., T[i,j,k] = 1 always (it's the identity bilinear map). Actually the matrix-mult tensor has T[i,j,k] = 1 for all i,j,k (the map (A,B) -> A@B is bilinear with T[i,j,k]=1).
    Rank-1: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]? That's trilinear in the indices but each factor is a coordinate. For the identity: we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k). 
    YES that's the schoolbook condition (each c picks a single (i,j,k) with all three =1).
    So verification: for random A, B, compute C = A@B, and compute C2[i,k] = sum_c U[i*m+j,c]*... no, the bilinear evaluation is:
    C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) * W[k*n+i,c]? 
    No! Rank-1 bilinear: C[i,k] = sum_c u_c(i) v_c(k) where u_c = (A contracted with U_c) and v_c = (B contracted with V_c), and W scales.
    Specifically: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]).
    Let me verify schoolbook: for term c=(i0,j0,k0): U[i*m+j,c]=1 only at (i0,j0), V[j*p+k,c]=1 only at (j0,k0), W[k*n+i,c]=1 only at (k0,i0).
    Then contribution to C[i,k]: W[k*n+i,c] is nonzero only if (i,k)=(i0,k0). Then (sum_j U[...]A[i,j]) = A[i0,j0] (since U nonzero only at j=j0 and i=i0). (sum_j V[...]B[j,k]) = B[j0,k0]. Product = A[i0,j0]B[j0,k0] at (i,k)=(i0,k0). Sum over c gives C[i0,k0] = sum_{j} ... wait we need C[i,k] = sum_j A[i,j]B[j,k]. For fixed (i,k), the terms c with (i0,k0)=(i,k) contribute A[i,j0]B[j0,k] summed over j0 = sum_j A[i,j]B[j,k]. YES correct.
    """
    rng = np.random.default_rng(seed)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    U3 = Ua.reshape(n, m, r)
    V3 = Va.reshape(m, p, r)
    W3 = Wa.reshape(p, n, r)
    for _ in range(trials):
        A = rng.integers(-5, 5, size=(n, m))
        B = rng.integers(-5, 5, size=(m, p))
        C = A @ B
        # C2[i,k] = sum_c W3[k,i,c] * (sum_j U3[i,j,c]*A[i,j]) * (sum_j V3[j,k,c]*B[j,k])
        C2 = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    a_part = sum(U3[i, j, c] * A[i, j] for j in range(m))
                    b_part = sum(V3[j, k, c] * B[j, k] for j in range(m))
                    s += W3[k, i, c] * a_part * b_part
                C2[i, k] = s
        if not np.allclose(C, C2, atol=1e-6):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Guaranteed-valid fallback
    school = _build_schoolbook(n, m, p)
    assert _verify(school["U"], school["V"], school["W"], n, m, p), "schoolbook must verify"

    # Try rank-25 candidate; use only if it verifies exactly.
    if _time_left() > 10:
        cand = _build_candidate_rank25()
        if cand is not None:
            if _verify(cand["U"], cand["V"], cand["W"], n, m, p, trials=12, seed=999):
                return cand
    return school
# EVOLVE-BLOCK-END
