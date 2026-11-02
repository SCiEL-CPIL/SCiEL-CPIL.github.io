"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    
    Convention: W[k*n+i, r]
    The decomposition satisfies:
      C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    
    Equivalently, for the tensor T[i,j,k] = 1 (the multiplication tensor):
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    Wait, that's not right. Let me re-derive.
    
    The bilinear form for matrix multiplication C = A @ B where C[i,k] = sum_j A[i,j]*B[j,k]:
    
    We want: C[i,k] = sum_r (something involving A) * (something involving B) * W[k*n+i, r]
    
    Actually the standard tensor decomposition is:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    
    But that's not quite right either. The multiplication tensor has:
    T[i, j, k] = 1 meaning A[i,j] * B[j,k] contributes to C[i,k].
    
    So the tensor is actually a 4-tensor if we separate inputs and outputs, but for bilinear rank we use:
    
    C[i,k] = sum_r w_r[i,k] * (sum_j u_r[i,j] A[i,j]) * (sum_j' v_r[j',k] B[j',k])
    
    No wait, that doesn't work dimensionally. Let me think more carefully.
    
    Standard bilinear decomposition of matrix multiplication:
    We write each output entry as:
    C[i,k] = sum_{r=1}^{R} alpha_{r,i,k} * (sum_j beta_{r,i,j} A[i,j]) * (sum_j' gamma_{r,j',k} B[j',k])
    
    But actually the cleaner formulation is:
    C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    
    Let me verify with the schoolbook: U[i*m+j, r] = 1 if r corresponds to (i,j,k), V[j*p+k, r] = 1 if r corresponds to (i,j,k), W[k*n+i, r] = 1 if r corresponds to (i,j,k).
    
    Then: sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    = sum_r [1 if r=(i,j,k)] * A[i,j] * [1 if r=(i,j,k)] * B[j,k]
    = A[i,j] * B[j,k] when r=(i,j,k), 0 otherwise
    = A[i,j] * B[j,k]
    
    But C[i,k] = sum_j A[i,j]*B[j,k], so we need to sum over j too. The formula above only gives one term.
    
    Hmm, I think the correct interpretation is different. Let me look at this from the perspective of what makes the schoolbook work.
    
    In the schoolbook, for each triple (i,j,k), we have one rank-1 term. The total rank is n*m*p.
    
    The reconstruction should be:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j'*p+k', r] * B[j',k'])
    
    No, that's trilinear in A and B separately but the products are scalars.
    
    Actually, I think the correct formula based on the tensor decomposition is:
    
    The tensor T for n×m × m×p multiplication is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    with T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise.
    
    Wait no. Let me think about this differently.
    
    The matrix multiplication map is: (A, B) -> C where C = A @ B
    A is n×m, B is m×p, C is n×p.
    
    Flatten: A -> vector of length n*m (row-major: A[i*m+j]), B -> vector of length m*p (row-major: B[j*p+k]), C -> vector of length n*p.
    
    The bilinear map is: C_vec = M(A_vec, B_vec) where M is a third-order tensor.
    
    M[a_idx, b_idx, c_idx] = 1 if a_idx = i*m+j, b_idx = j*p+k, c_idx = i*p+k for some i,j,k.
    
    Wait, but c_idx = i*p+k (row-major for C which is n×p).
    
    But the convention says W[k*n+i, r], which suggests C is stored column-major or with a different indexing.
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W has shape (p*n, rank) and W[k*n+i, r] corresponds to output entry C[i,k] but indexed as k*n+i. This means the output is stored in column-major order (k is the slow index, i is the fast index), or equivalently, the output vector has index k*n+i for entry C[i,k].
    
    So the tensor M has:
    M[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    And the rank decomposition is: M = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r has length n*m, v_r has length m*p, w_r has length p*n.
    
    So: M[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the schoolbook: U[i*m+j, r] = 1 if r = (i,j,k), V[j*p+k, r] = 1 if r = (i,j,k), W[k*n+i, r] = 1 if r = (i,j,k).
    
    Then: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if there exists r = (i,j,k) matching all three, which is true when the indices match.
    
    For a fixed (i,j,k): M[i*m+j, j*p+k, k*n+i] = sum_r [U[i*m+j,r]=1 iff r=(i,j,k)] * [V[j*p+k,r]=1 iff r=(i,j,k)] * [W[k*n+i,r]=1 iff r=(i,j,k)] = 1. ✓
    
    For non-matching indices, it's 0. ✓
    
    So the verification is:
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    And for all other combinations of (a_idx, b_idx, c_idx) where a_idx = i*m+j, b_idx = j'*p+k, c_idx = k'*n+i' with j' != j or k' != k or i' != i, the sum should be 0.
    
    Actually, it's simpler: the tensor M is sparse, with M[a,b,c] = 1 only when a=i*m+j, b=j*p+k, c=k*n+i for some consistent (i,j,k).
    
    Let me just verify numerically by checking the tensor entries.
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Check all tensor entries
    # M[a, b, c] should be 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # and 0 otherwise
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    # Also check that off-diagonal entries are 0
    # For efficiency, check a sample or all
    # The full tensor has n*m * m*p * p*n entries, but most are 0
    # Let's check all to be safe
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    for i2 in range(n):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k2 * n + i2
                        if (i2 != i or k2 != k):
                            val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                            err = abs(val)
                            if err > max_err:
                                max_err = err
                for j2 in range(m):
                    if j2 != j:
                        for k2 in range(p):
                            a_idx = i * m + j
                            b_idx = j2 * p + k2
                            c_idx = k2 * n + i
                            val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                            err = abs(val)
                            if err > max_err:
                                max_err = err
    
    return max_err < tol


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Faster verification using matrix operations.
    
    The condition is: M = U ⊗ V ⊗ W where M[i*m+j, j*p+k, k*n+i] = 1
    
    Equivalently: for each r, the outer product structure must match.
    
    Let's use a different approach: 
    For each r, define matrices:
    U_r = U[:, r].reshape(n, m)  # shape (n, m)
    V_r = V[:, r].reshape(m, p)  # shape (m, p)  
    W_r = W[:, r].reshape(p, n)  # shape (p, n), so W_r[k, i] = W[k*n+i, r]
    
    The condition M[i*m+j, j*p+k, k*n+i] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    
    This can be written as: sum_r (U_r * V_r^T * W_r^T) [i,j,k] ... no.
    
    Actually: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k
    
    Let me define for each r: S_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    We need sum_r S_r[i,j,k] = 1.
    
    This is equivalent to: the tensor sum_r U_r ⊗ V_r ⊗ W_r (with appropriate index permutation) equals the all-ones tensor in the right subspace.
    
    For verification, let's just compute it directly but vectorized.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape to 3D: U_r[i,j], V_r[j,k], W_r[k,i]
    U3 = U.reshape(n, m, rank)  # U3[i, j, r]
    V3 = V.reshape(m, p, rank)  # V3[j, k, r]
    W3 = W.reshape(p, n, rank)  # W3[k, i, r]
    
    # Compute sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] for all i,j,k
    # This is a contraction over r
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # We can compute this as:
    # For each r: term_r = U3[:,:,r] * V3[:,:,r] * W3[:,:,r] (with broadcasting)
    # But the indices don't align directly.
    
    # U3[i,j,r], V3[j,k,r], W3[k,i,r]
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Let's compute step by step:
    # temp[i,j,k,r] = U3[i,j,r] * V3[j,k,r] -- this needs broadcasting
    # U3 has shape (n, m, rank), V3 has shape (m, p, rank)
    # We want temp[i,j,k,r] = U3[i,j,r] * V3[j,k,r]
    
    # U3 expanded: (n, m, 1, rank)
    # V3 expanded: (1, m, p, rank)
    # Product: (n, m, p, rank) -- but this sums over j implicitly? No.
    
    # Actually: temp[i,j,k,r] = U3[i,j,r] * V3[j,k,r]
    # U3[:, :, r] is (n, m), V3[:, :, r] is (m, p)
    # We need element-wise product with matching j index.
    
    # temp[i,j,k,r] = U3[i,j,r] * V3[j,k,r]
    # This is: for each i,k,r: temp[i,:,k,r] = U3[i,:,r] * V3[:,k,r] (element-wise over j)
    
    # Then result[i,j,k] = sum_r temp[i,j,k,r] * W3[k,i,r]
    
    # Let's do it with einsum:
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3)
    
    result = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3)
    
    max_err = np.max(np.abs(result - 1.0))
    return max_err < tol


def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Block 1: A_1 B_1
    # A_1 is 2x1 (col 0 of A), B_1 is 1x8 (row 0 of B)
    # C[i,k] += A[i,0] B[0,k]
    col = 0
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1.0
            V[0*p + k, col] = 1.0
            W[k*n + i, col] = 1.0
            col += 1
            
    # Block 2: A_2 B_2
    # A_2 is 2x2 (cols 1,2 of A), B_2 is 2x8 (rows 1,2 of B)
    # Partition B_2 into four 2x2 blocks B_{2,j} for j in 0..3
    # Use Strassen for each 2x2x2 multiplication
    
    # Strassen coefficients for U (A entries)
    # a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
    # U indices: a->1, b->2, c->4, d->5
    strassen_u = [
        [1, 1],    # M1: a+d
        [1, 1],    # M2: c+d
        [1],       # M3: a
        [1],       # M4: d
        [1, 1],    # M5: a+b
        [-1, 1],   # M6: c-a (c=1, a=-1) -> idx 4, 1
        [1, -1]    # M7: b-d (b=1, d=-1) -> idx 2, 5
    ]
    strassen_u_idx = [
        [1, 5],    # M1
        [4, 5],    # M2
        [1],       # M3
        [5],       # M4
        [1, 2],    # M5
        [4, 1],    # M6
        [2, 5]     # M7
    ]
    
    # Strassen coefficients for V (B entries)
    # e=B[1,2j], f=B[1,2j+1], g=B[2,2j], h=B[2,2j+1]
    # V indices for block j:
    # e -> 8+2j, f -> 9+2j, g -> 16+2j, h -> 17+2j
    strassen_v = [
        [1, 1],    # M1: e+h
        [1],       # M2: e
        [1, -1],   # M3: f-h
        [1, -1],   # M4: g-e
        [1],       # M5: h
        [1, 1],    # M6: e+f
        [1, 1]     # M7: g+h
    ]
    strassen_v_idx_base = [
        [0, 3],    # e, h
        [0],       # e
        [1, 3],    # f, h
        [2, 0],    # g, e
        [3],       # h
        [0, 1],    # e, f
        [2, 3]     # g, h
    ]
    
    # Strassen coefficients for W (C entries)
    # C[0,2j], C[0,2j+1], C[1,2j], C[1,2j+1]
    # W indices for block j:
    # C[0,2j] -> 4j
    # C[0,2j+1] -> 4j+2
    # C[1,2j] -> 4j+1
    # C[1,2j+1] -> 4j+3
    strassen_w = [
        [1, 1],    # M1: C00, C11
        [1, -1],   # M2: C10, -C11
        [1, 1],    # M3: C01, C11
        [1, 1],    # M4: C00, C10
        [-1, 1],   # M5: -C00, C01
        [1],       # M6: C11
        [1]        # M7: C00
    ]
    strassen_w_idx_base = [
        [0, 3],    # C00, C11
        [1, 3],    # C10, C11
        [2, 3],    # C01, C11
        [0, 1],    # C00, C10
        [0, 2],    # C00, C01
        [3],       # C11
        [0]        # C00
    ]
    
    for j in range(4):
        for s in range(7):
            r = col + s
            
            # Fill U
            for k, idx in enumerate(strassen_u_idx[s]):
                U[idx, r] = strassen_u[s][k]
                
            # Fill V
            for k, idx_base in enumerate(strassen_v_idx_base[s]):
                v_idx = 8 + 2*j + idx_base # Wait, base indices are relative to B_2 start?
                # e=0, f=1, g=2, h=3 relative to B_2 (which is 2x2 flattened)
                # B_2 is rows 1,2 of B.
                # B[1, 2j] is V[8+2j]
                # B[1, 2j+1] is V[9+2j]
                # B[2, 2j] is V[16+2j]
                # B[2, 2j+1] is V[17+2j]
                # My base indices:
                # 0 -> e -> 8+2j
                # 1 -> f -> 9+2j
                # 2 -> g -> 16+2j
                # 3 -> h -> 17+2j
                # This mapping is not linear if I just add.
                # Let's use explicit mapping.
                pass
            
            # Explicit V mapping
            v_map = {
                0: 8 + 2*j,
                1: 9 + 2*j,
                2: 16 + 2*j,
                3: 17 + 2*j
            }
            for k, idx_base in enumerate(strassen_v_idx_base[s]):
                V[v_map[idx_base], r] = strassen_v[s][k]
                
            # Explicit W mapping
            w_map = {
                0: 4*j,
                1: 4*j + 1,
                2: 4*j + 2,
                3: 4*j + 3
            }
            for k, idx_base in enumerate(strassen_w_idx_base[s]):
                W[w_map[idx_base], r] = strassen_w[s][k]
                
        col += 7
        
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if _verify_decomposition_fast(U, V, W, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fb = n*m*p
        U_fb = np.zeros((n*m, rank_fb), dtype=float)
        V_fb = np.zeros((m*p, rank_fb), dtype=float)
        W_fb = np.zeros((p*n, rank_fb), dtype=float)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1
                    V_fb[j*p+k, col_fb] = 1
                    W_fb[k*n+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
