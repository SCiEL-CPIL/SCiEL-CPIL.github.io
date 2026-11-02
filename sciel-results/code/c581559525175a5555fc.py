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
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with schoolbook rank 48
    best_rank = n * m * p
    best_result = None
    
    # Build schoolbook decomposition
    rank = best_rank
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Model 2×3×8 as Kronecker product of 2×3×2 and 1×1×4
    # The 2×3×2 tensor can potentially be decomposed at rank < 6
    # If we find a rank-r decomposition of 2×3×2, then 2×3×8 has rank r*4
    # Schoolbook for 2×3×2 is rank 6, so 2×3×8 schoolbook is 24... wait no.
    
    # Actually 2×3×8: n=2, m=3, p=8
    # Schoolbook rank = n*m*p = 2*3*8 = 48
    
    # If we view it as (2×3×2) ⊗ (1×1×4):
    # The 2×3×2 part has dimensions n=2, m=3, p=2, schoolbook rank 12
    # The 1×1×4 part has dimensions n=1, m=1, p=4, schoolbook rank 4
    # Total schoolbook: 12 * 4 = 48 ✓
    
    # If we can find a rank-r decomposition for 2×3×2, then 2×3×8 has rank r*4
    # We need r < 12 to improve.
    
    # Let's try to find a low-rank decomposition for 2×3×2.
    # The tensor T[i,j,k] for 2×3×2: T[i*m+j, j*p+k, k*n+i] = 1
    # n=2, m=3, p=2
    # U has shape (6, r), V has shape (6, r), W has shape (4, r)
    
    # Let's try to find a rank-11 or lower decomposition for 2×3×2.
    
    def build_2x3x2_decomposition(target_rank):
        """Try to find a rank decomposition for 2×3×2."""
        nn, mm, pp = 2, 3, 2
        
        # Verify function for 2×3×2
        def verify_232(U_t, V_t, W_t, tol=1e-9):
            U_t = np.asarray(U_t, dtype=float)
            V_t = np.asarray(V_t, dtype=float)
            W_t = np.asarray(W_t, dtype=float)
            U3 = U_t.reshape(nn, mm, target_rank)
            V3 = V_t.reshape(mm, pp, target_rank)
            W3 = W_t.reshape(pp, nn, target_rank)
            result = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3)
            return np.max(np.abs(result - 1.0)) < tol
        
        # Try random search
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            
            # Random entries from {-1, 0, 1}
            U_t = np.random.choice([-1, 0, 1], size=(nn*mm, target_rank))
            V_t = np.random.choice([-1, 0, 1], size=(mm*pp, target_rank))
            W_t = np.random.choice([-1, 0, 1], size=(pp*nn, target_rank))
            
            if verify_232(U_t, V_t, W_t):
                return U_t, V_t, W_t
        
        return None
    
    # Try ranks for 2×3×2: 11, 10, 9, ...
    # If we find rank r for 2×3×2, we get rank 4r for 2×3×8
    
    for r232 in range(11, 5, -1):
        if time.time() > deadline - 20:
            break
        
        target_rank_238 = r232 * 4
        if target_rank_238 >= best_rank:
            continue
        
        result_232 = build_2x3x2_decomposition(r232)
        if result_232 is not None:
            U_232, V_232, W_232 = result_232
            
            # Build 2×3×8 from 2×3×2 ⊗ 1×1×4
            # The Kronecker product structure:
            # For 2×3×8, we have n=2, m=3, p=8
            # Decompose p=8 into p2=2 * p4=4
            # For each (i,j,k) in 2×3×2, and each k4 in 0..3:
            #   i' = i, j' = j, k' = k*4 + k4
            #   The tensor entry is 1
            #   
            # U[i*m+j, r] for 2×3×8:
            #   r = (r232_index, k4) where r232_index is 0..r232-1 and k4 is 0..3
            #   U[i*3+j, r232_idx*4+k4] = U_232[i*3+j, r232_idx]
            #
            # V[j*p+k, r] for 2×3×8:
            #   k = k2*4 + k4 where k2 is 0..1 (from 2×3×2) and k4 is 0..3
            #   V[j*8+k, r232_idx*4+k4] = V_232[j*2+k2, r232_idx]
            #
            # W[k*n+i, r] for 2×3×8:
            #   k = k2*4 + k4
            #   W[k*2+i, r232_idx*4+k4] = W_232[k2*2+i, r232_idx]
            
            U_238 = np.zeros((n*m, target_rank_238), dtype=float)
            V_238 = np.zeros((m*p, target_rank_238), dtype=float)
            W_238 = np.zeros((p*n, target_rank_238), dtype=float)
            
            for r_idx in range(r232):
                for k4 in range(4):
                    r_238 = r_idx * 4 + k4
                    
                    # U: same for all k4
                    U_238[:, r_238] = U_232[:, r_idx]
                    
                    # V: for each j and k2, V[j*8 + k2*4 + k4, r_238] = V_232[j*2 + k2, r_idx]
                    for j in range(m):
                        for k2 in range(2):
                            V_238[j*8 + k2*4 + k4, r_238] = V_232[j*2 + k2, r_idx]
                    
                    # W: for each k2 and i, W[(k2*4+k4)*2 + i, r_238] = W_232[k2*2 + i, r_idx]
                    for k2 in range(2):
                        for i in range(n):
                            W_238[(k2*4 + k4)*2 + i, r_238] = W_232[k2*2 + i, r_idx]
            
            # Verify the 2×3×8 decomposition
            if _verify_decomposition_fast(U_238, V_238, W_238, n, m, p):
                best_rank = target_rank_238
                best_result = {"U": U_238.tolist(), "V": V_238.tolist(), "W": W_238.tolist()}
                break
    
    # Also try direct random search for 2×3×8 at various ranks
    for target_rank in range(47, best_rank, -1):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V_rand = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W_rand = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            if _verify_decomposition_fast(U_rand, V_rand, W_rand, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                break
    
    return best_result

# EVOLVE-BLOCK-END
