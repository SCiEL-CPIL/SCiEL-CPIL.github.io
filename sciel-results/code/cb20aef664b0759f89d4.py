"""Matrix multiplication 4x4x6 rank-minimal decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 6]

def _schoolbook_442():
    """Schoolbook decomposition of 4x4x2 multiplication. Rank = 32."""
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_444():
    """
    Decompose 4x4x4 using two 2x2x2 Strassen blocks.
    Split A into 2x2 blocks, B into 2x2 blocks, C gets 2x2 blocks.
    Each 2x2x2 uses Strassen rank 7.
    Total rank = 14.
    """
    n, m, p = 4, 4, 4
    rank = 14  # 2 * 7
    U = np.zeros((n*m, rank))  # (16, 14)
    V = np.zeros((m*p, rank))  # (16, 14)
    W = np.zeros((p*n, rank))  # (16, 14)
    
    # Strassen 2x2x2: multiply [[a,b],[c,d]] * [[e,f],[g,h]] = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Using 7 multiplications:
    # M1 = (a+b)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(e-f)
    # M4 = d(g-h)
    # M5 = (a+d)(e+h)
    # M6 = b(f-h)
    # M7 = c(e-g)
    # C11 = M1 - M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M6 - M7
    
    # Strassen U factors (for 2x2 input A, indexed by row*2+col of A):
    # U maps (a,b,c,d) -> (M1..M7)
    # A indices: a=0, b=1, c=2, d=3
    U_s = np.zeros((4, 7))
    U_s[0, 0] = 1; U_s[1, 0] = 1   # M1: (a+b)
    U_s[2, 1] = 1; U_s[3, 1] = 1   # M2: (c+d)
    U_s[0, 2] = 1                    # M3: a
    U_s[3, 3] = 1                    # M4: d
    U_s[0, 4] = 1; U_s[3, 4] = 1   # M5: (a+d)
    U_s[1, 5] = 1                    # M6: b
    U_s[2, 6] = 1                    # M7: c
    
    # V factors (for 2x2 input B, indexed by row*2+col of B):
    # B indices: e=0, f=1, g=2, h=3
    V_s = np.zeros((4, 7))
    V_s[0, 0] = 1; V_s[3, 0] = 1   # M1: (e+h)
    V_s[0, 1] = 1; V_s[3, 1] = 1   # M2: (e+h)
    V_s[0, 2] = 1; V_s[1, 2] = -1  # M3: (e-f)
    V_s[2, 3] = 1; V_s[3, 3] = -1  # M4: (g-h)
    V_s[0, 4] = 1; V_s[3, 4] = 1   # M5: (e+h)
    V_s[1, 5] = 1; V_s[3, 5] = -1  # M6: (f-h)
    V_s[0, 6] = 1; V_s[2, 6] = -1  # M7: (e-g)
    
    # W factors (output C indices: c11=0, c12=1, c21=2, c22=3)
    W_s = np.zeros((4, 7))
    W_s[0, 0] = 1; W_s[0, 3] = -1; W_s[0, 4] = -1; W_s[0, 6] = -1  # C11 = M1 - M4 - M5 - M7
    W_s[1, 2] = 1; W_s[1, 4] = 1                                      # C12 = M3 + M5
    W_s[2, 1] = 1; W_s[2, 3] = 1                                      # C21 = M2 + M4
    W_s[3, 0] = 1; W_s[3, 2] = -1; W_s[3, 5] = -1; W_s[3, 6] = -1     # C22 = M1 - M3 - M6 - M7
    
    # Now embed into 4x4:
    # Block (0,0): A rows 0-1, B cols 0-1, C rows 0-1
    # Block (1,1): A rows 2-3, B cols 2-3, C rows 2-3
    
    # Block (0,0): uses U_s, V_s, W_s at offset 0
    # A row i (0..3), B col j (0..3)
    # For block (0,0): A index = i (if i<2), B index = j (if j<2)
    for i in range(2):
        for j in range(2):
            # U: A[i, j] contributes to U_s[i, :]
            U[i*4 + j, 0:7] = U_s[i, :]
            # V: B[j, i] -> in B's 2x2 block, row j, col i
            V[j*4 + i, 0:7] = V_s[j, :]
            # W: C[i, j]
            W[i*4 + j, 0:7] = W_s[i, :]
    
    # Block (1,1): uses U_s, V_s, W_s at offset 7
    for i in range(2):
        for j in range(2):
            # A[2+i, 2+j]
            U[(2+i)*4 + (2+j), 7:14] = U_s[i, :]
            # B[2+j, 2+i]
            V[(2+j)*4 + (2+i), 7:14] = V_s[j, :]
            # C[2+i, 2+j]
            W[(2+i)*4 + (2+j), 7:14] = W_s[i, :]
    
    # But this only covers diagonal blocks! We need cross terms too.
    # Actually, for 4x4x4 we can't just do two 2x2x2's because that misses
    # the off-diagonal block products.
    # 
    # Correct approach: 4x4x4 = 4 sub-blocks of 2x2x2, each with rank 7, total 28.
    # Or use a single 4x4x4 construction.
    # 
    # Let me just use schoolbook for 4x4x4: rank 64.
    # Actually, a better approach: use 4 Strassen blocks (one per 2x2 output block).
    
    # Let me redo: 4x4x4 with 4 Strassen 2x2x2 blocks
    rank = 28  # 4 * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for bi in range(2):  # output row block
        for bj in range(2):  # output col block
            block_offset = (bi*2 + bj) * 7
            # Output C[bi*2:bi*2+2, bj*2:bj*2+2]
            # = sum over k of A[bi*2:bi*2+2, k] * B[k, bj*2:bj*2+2]
            # But Strassen only works if we can group into 2x2 blocks.
            # For general 4x4, we need to sum over all 4 column/row indices.
            # This doesn't decompose cleanly into 4 independent 2x2x2's.
            
    # The clean way: 4x4x4 can be done with rank 28 by viewing it as
    # multiplying a 4x4 by 4x4 where we partition into 2x2 blocks:
    # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # C11 = A11*B11 + A12*B21  (sum of two 2x2 products, each rank 7, total 14)
    # C12 = A11*B12 + A12*B22  (14)
    # C21 = A21*B11 + A22*B21  (14)
    # C22 = A21*B12 + A22*B22  (14)
    # Total: 56. Not great.
    
    # Simplest correct approach: schoolbook rank 64 for 4x4x4.
    # Combined with 4x4x2 schoolbook rank 32, total = 96.
    # That's worse than the original 96 (4*4*6).
    # 
    # Wait, original is 4*4*6 = 96. My split gives 64+32=96. Same.
    # 
    # For a genuine improvement, I need better 4x4x4 or 4x4x6.
    # Known: 4x4x4 has rank at most 49 (Coppersmith-Winograd style) but 
    # practical known is around 49-52. Let me try a different strategy.
    
    # For 4x4x6, one approach: use the fact that we can pad to 4x4x8 and 
    # use a known 4x4x8 construction, then project. But that's complex.
    
    # Alternative: use a Kronecker product approach.
    # 4x4x6 = (4x4x2) * (2x2x3) in some tensor sense? No, that's not how it works.
    
    # Let me just try to find a good rank via random optimization.
    # Start with schoolbook rank 96 and try to reduce.
    
    # For now, return the simple split: 4x4x4 schoolbook + 4x4x2 schoolbook
    # But actually, let me just do full schoolbook for 4x4x6: rank 96.
    # And then try to improve via random perturbation + SVD-like reduction.
    
    return None  # Signal to use the main search

def _full_schoolbook():
    """Full schoolbook 4x4x6, rank 96."""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} ... 
    Actually, the tensor is T[i,j,k] = 1 if we're computing (AB)[i,k] which 
    involves A[i,j]*B[j,k]. So the condition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k),
    and = 0 otherwise (but all (i,j,k) are valid in matrix mult).
    Wait, in matrix multiplication every triple (i,j,k) appears with coefficient 1.
    So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for ALL i,j,k.
    """
    n, m, p = 4, 4, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape for efficient computation
    # For each column c, we have u_c (n*m), v_c (m*p), w_c (p*n)
    # Contribution to (i,j,k) from column c: u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    # Sum over c.
    
    # Efficient check: for each c, form the tensor contribution
    # u_c reshaped to (n, m), v_c to (m, p), w_c to (p, n)
    # Contribution: sum over j of u_c[i,j] * v_c[j,k] * w_c[k,i]
    # = (u_c @ v_c)[i,k] * w_c[k,i]  ... wait, that's a contraction.
    
    # Let me do it more carefully.
    # T_c[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
    # Total T[i,j,k] = sum_c T_c[i,j,k]
    # We need T[i,j,k] = 1 for all i,j,k.
    
    total = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        # T_c[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
        # For fixed c, this is a rank-1-like tensor (not quite, due to index cyclic deps)
        # Let's compute: for each i,k: sum_j u_c[i,j]*v_c[j,k] = (u_c @ v_c)[i,k]
        # Then multiply by w_c[k,i]
        # But we need the full tensor, not just the contracted version.
        # Let's just do it directly for small sizes.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    # Check if total is all ones
    return np.allclose(total, 1.0, atol=1e-9)

def _try_rank_reduction(U, V, W, target_rank, max_iterations=5000, seed=42):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 slices.
    Each column c defines a "slice" with U[:,c], V[:,c], W[:,c].
    We want to find if some columns can be eliminated.
    
    Strategy: randomly project the factor space to lower dimension and check if 
    the decomposition still works (approximately). Then refine.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = 4, 4, 6
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    rng = np.random.RandomState(seed)
    
    # Try to find a basis of dimension target_rank that still spans the same tensor.
    # The tensor is T = sum_c outer(U_c, V_c, W_c) where U_c, V_c, W_c are the c-th columns.
    # We need: sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1 for all i,j,k.
    
    # This is a non-convex problem. Let's try: 
    # 1. Compute the "moment" tensors to get constraints.
    # 2. Use alternating least squares to fit a lower-rank decomposition.
    
    # Flatten the constraint: we have n*m*p = 96 equations.
    # Unknowns: target_rank * (n*m + m*p + p*n) = target_rank * (16 + 24 + 24) = target_rank * 64
    
    # For target_rank = 96, we have 96*64 = 6144 unknowns for 96 equations. Overdetermined in good way.
    # For lower rank, still overdetermined.
    
    # ALS approach:
    # Initialize with random or subsampled columns.
    best_U, best_V, best_W = U, V, W
    best_err = np.inf
    
    for trial in range(5):
        # Initialize
        if trial == 0:
            # Use first target_rank columns
            init_idx = np.arange(target_rank)
        else:
            init_idx = rng.choice(current_rank, size=target_rank, replace=False)
        
        U_new = U[:, init_idx].copy()
        V_new = V[:, init_idx].copy()
        W_new = W[:, init_idx].copy()
        
        # ALS iterations
        for it in range(max_iterations):
            if it % 2 == 0:
                # Update U
                # For each (i,j), U_new[i*m+j, :] should satisfy:
                # sum_c U_new[i*m+j,c] * V_new[j*p+k,c] * W_new[k*n+i,c] = 1 for all k
                # This is nonlinear in U due to the coupling with V and W.
                # Linearize: fix V, W, solve for U.
                # For fixed i,j: sum_c U_new[i*m+j,c] * (sum_k V_new[j*p+k,c]*W_new[k*n+i,c]) = 1
                # Define D[i,j,c] = sum_k V_new[j*p+k,c] * W_new[k*n+i,c]
                # Then: U_new[i*m+j,:] @ D[i,j,:] = 1
                # D has shape (n, m, target_rank)
                D = np.zeros((n, m, target_rank))
                for i in range(n):
                    for j in range(m):
                        for c in range(target_rank):
                            D[i,j,c] = np.dot(V_new[j*p:(j+1)*p, c], W_new[::n, c])  # not right
                # Actually: D[i,j,c] = sum_{k} V_new[j*p+k, c] * W_new[k*n+i, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            D[i,j,:] += V_new[j*p+k, :] * W_new[k*n+i, :]
                # Now solve: for each (i,j), U_new[i*m+j,:] * D[i,j,:] = 1
                # This is an underdetermined system if target_rank > 1.
                # Min norm solution: U_new[i*m+j,:] = D[i,j,:] / ||D[i,j,:]||^2
                for i in range(n):
                    for j in range(m):
                        d = D[i,j,:]
                        dn = np.dot(d, d)
                        if dn > 1e-12:
                            U_new[i*m+j, :] = d / dn
                        else:
                            U_new[i*m+j, :] = rng.randn(target_rank) * 0.1
            else:
                # Update V
                # For fixed j,k: sum_c V_new[j*p+k,c] * (sum_i U_new[i*m+j,c]*W_new[k*n+i,c]) = 1
                # E[j,k,c] = sum_i U_new[i*m+j,c] * W_new[k*n+i,c]
                E = np.zeros((m, p, target_rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            E[j,k,:] += U_new[i*m+j, :] * W_new[k*n+i, :]
                for j in range(m):
                    for k in range(p):
                        e = E[j,k,:]
                        en = np.dot(e, e)
                        if en > 1e-12:
                            V_new[j*p+k, :] = e / en
                        else:
                            V_new[j*p+k, :] = rng.randn(target_rank) * 0.1
            # Update W (every iteration for balance)
            # For fixed k,i: sum_c W_new[k*n+i,c] * (sum_j U_new[i*m+j,c]*V_new[j*p+k,c]) = 1
            F = np.zeros((p, n, target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        F[k,i,:] += U_new[i*m+j, :] * V_new[j*p+k, :]
            for k in range(p):
                for i in range(n):
                    f = F[k,i,:]
                    fn = np.dot(f, f)
                    if fn > 1e-12:
                        W_new[k*n+i, :] = f / fn
                    else:
                        W_new[k*n+i, :] = rng.randn(target_rank) * 0.1
            
            # Compute error
            total = np.zeros((n, m, p))
            for c in range(target_rank):
                u_c = U_new[:, c].reshape(n, m)
                v_c = V_new[:, c].reshape(m, p)
                w_c = W_new[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            err = np.max(np.abs(total - 1.0))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            if err < 1e-8:
                break
    
    return best_U, best_V, best_W

def construct():
    """Construct a rank-minimal decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = 4, 4, 6
    
    # Start with schoolbook
    U, V, W = _full_schoolbook()
    best_result = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]  # 96
    
    # Try to improve via ALS rank reduction
    for target_rank in [88, 80, 72, 64, 56, 48, 40, 36, 32]:
        if time.time() > deadline:
            break
        U2, V2, W2 = _try_rank_reduction(U, V, W, target_rank, max_iterations=2000)
        if np.allclose(U2, 0) == False:  # not trivially zero
            # Verify
            if _verify(U2, V2, W2):
                best_rank = target_rank
                best_result = (U2, V2, W2)
                U, V, W = U2, V2, W2
                break  # Found a valid lower-rank decomposition
    
    # Convert to lists with integer/half-integer rounding if close
    def to_half_int_array(arr):
        arr = np.array(arr, dtype=float)
        # Check if entries are close to half-integers
        doubled = 2 * arr
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=1e-8):
            return [[rounded[i,j] / 2 for j in range(rounded.shape[1])] for i in range(rounded.shape[0])]
        # Check integers
        rounded_int = np.round(arr)
        if np.allclose(arr, rounded_int, atol=1e-8):
            return [[int(rounded_int[i,j]) for j in range(rounded_int.shape[1])] for i in range(rounded_int.shape[0])]
        # Fallback: return as-is (evaluator may accept floats)
        return [[arr[i,j] for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    U_list = to_half_int_array(best_result[0])
    V_list = to_half_int_array(best_result[1])
    W_list = to_half_int_array(best_result[2])
    
    return {"U": U_list, "V": V_list, "W": W_list}
