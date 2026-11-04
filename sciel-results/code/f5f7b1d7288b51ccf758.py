"""
Matrix multiplication 2x3 by 3x5: rank-minimal exact tensor decomposition.

Strategy:
  1. Start from the schoolbook (rank 30) decomposition.
  2. Attempt to reduce rank via structured linear algebra:
     - Try to find a rank-29 decomposition by solving for a linear dependence
       among the 30 basis tensors, using integer/rational search.
     - More generally, use a randomized algebraic approach: fix a candidate
       rank r, set up the tensor equation as a system of linear equations
       in the factor entries, and attempt to solve.
  3. Because the space of rank-r decompositions is non-linear, we use a
     heuristic: iteratively merge pairs of schoolbook terms and check if the
     resulting rank-(30-1) factorization is still valid by re-deriving one
     factor. If the merged tensor is rank 1, we can combine.
  4. As a practical matter, for small dimensions we can also brute-force
     search over low-rank structures with rational entries.

  The key insight: the matrix multiplication tensor T_{i,j,k} = e_{i,j} ⊗
  e_{j,k} ⊗ e_{k,i} (with appropriate index conventions) may have a tensor
  rank lower than n*m*p. For 2x3x5, we search for rank < 30.

  We implement a structured search:
  - Represent the tensor as a 3D array T[i,j,k] (size 2x3x5).
  - A rank-r decomposition gives:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Wait, that's not right. The tensor for matrix multiplication is:
      T_{i,j,k} = 1 if the multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    The tensor is T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}... 
    Actually, the matrix multiplication tensor is:
      T[i, j, k] = 1  for all valid (i,j,k)
    No. The tensor is T[i,j,k] such that (A B)[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    So T[i,j,k] = 1 for all i,j,k. That's just the all-ones tensor, which has rank 1.
    
    That's not right either. Let me re-derive.
    
    (A B)[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has components T[i,j,k] = 1 for all i,j,k? No.
    T is defined so that (A B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k].
    For this to equal sum_j A[i,j] B[j,k], we need T[i,j,k] = 1 for all i,j,k.
    
    But that would make the tensor rank 1, which is clearly wrong.
    
    The issue is the index structure. The tensor is:
      T[i, j, k] = delta_{i,i} * delta_{j,j} * delta_{k,k}
    No. Let's think in terms of the three matrix spaces.
    
    A is n x m, B is m x p, C = A B is n x p.
    (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor T is a map from (R^n)^* x (R^m)^* x ... 
    Actually, T is an element of (R^{n x m})^* ⊗ (R^{m x p})^* ⊗ (R^{n x p})^*? No.
    
    The matrix multiplication tensor is:
      T ∈ (R^{n × m})^* ⊗ (R^{m × p})^* ⊗ (R^{n × p})^*
    No, it's T ∈ R^{n × m} ⊗ R^{m × p} ⊗ R^{n × p} ... no.
    
    Let me use the standard definition. The tensor T satisfies:
      <T, A ⊗ B ⊗ C> = tr(C^T A B) for all A ∈ R^{n×m}, B ∈ R^{m×p}, C ∈ R^{n×p}
    
    So T_{i,j,k} where i indexes rows of A (0..n-1), j indexes columns of A / rows of B (0..m-1), k indexes columns of B / rows of C (0..p-1)... 
    
    Actually, the standard tensor for (n×m)·(m×p) → (n×p) multiplication is:
      T[i, j, k] = 1  where i ∈ [0,n), j ∈ [0,m), k ∈ [0,p)
    and the multiplication is:
      (A B)[i, k] = sum_j T[i, j, k] A[i, j] B[j, k]
    
    For this to hold, T[i, j, k] must be 1 for all i, j, k. But that's rank 1.
    
    I think the issue is that the tensor has 3 indices but they map to different matrix entries. The correct tensor is:
      T[i, j, k, l] ... no.
    
    Let me look at this differently. The task says:
      U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
      W[k*n+i, r]  -- trace convention
    
    So the decomposition is:
      For each r in range(rank):
        u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    
    And the multiplication satisfies:
      (A B)[i, k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * ... 
    
    No, that's not how it works. The tensor decomposition means:
      T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, but T should be the matrix multiplication tensor. The matrix multiplication tensor has:
      T[(i,j), (j,k), (k,i)] ... 
    
    I think the correct formulation is:
      The tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} where:
      T[i*m+j, j*p+k, k*n+i] = 1  for all i,j,k
    
    And a rank-r decomposition is:
      T[a, b, c] = sum_{r=0}^{r-1} U[a,r] V[b,r] W[c,r]
    
    Where T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, and 0 otherwise.
    
    So T is a 3-tensor of size (n*m) x (m*p) x (p*n) with mostly zeros and some ones.
    
    Specifically, T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    The schoolbook decomposition sets U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1 for r = i*m*p + j*p + k (or similar).
    
    So the task is to find the tensor rank of this specific sparse 3-tensor.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
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

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition exactly reproduces the tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Reconstruct tensor: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T: (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    # Check against expected tensor
    T_expected = _build_tensor(n, m, p)
    return np.allclose(T, T_expected, atol=1e-9)

def _try_rank_reduction(n, m, p, U, V, W, target_rank, deadline):
    """
    Try to reduce the rank from current to target_rank by finding a linear
    combination of two rank-1 terms that can be merged.
    
    Approach: For each pair of terms (r1, r2), try to see if their sum
    can be expressed as a single rank-1 term. This is checking if the
    sum tensor has rank 1, which for a 3-tensor means all 2x2 minors
    of each matricization are zero.
    
    More practically: we try to merge by solving for new u, v, w such that
    u ⊗ v ⊗ w = u1⊗v1⊗w1 + u2⊗v2⊗w2.
    
    This is generally not possible, but we can try a structured approach:
    fix one factor and solve for the others.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Try to merge pairs
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline:
                return None
            
            # Sum tensor: T_sum[a,b,c] = U[a,r1]*V[b,r1]*W[c,r1] + U[a,r2]*V[b,r2]*W[c,r2]
            # Try to express as u⊗v⊗w
            # This is possible iff the tensor has rank 1.
            # A sufficient check: pick a nonzero entry, say T_sum[a0,b0,c0] != 0
            # Then u[a] = T_sum[a,b0,c0] / T_sum[a0,b0,c0], etc.
            # But this only works if the tensor is separable.
            
            # Matricize along first mode: M1[a, (b,c)] = T_sum[a,b,c]
            # Rank 1 means M1 has rank 1.
            T1_mat = np.einsum('a,b,c->abc', U[:, r1], V[:, r1], W[:, r1])
            T2_mat = np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
            Tsum_mat = T1_mat + T2_mat
            
            M1 = Tsum_mat.reshape(n*m, -1)
            # Check if M1 has rank <= 1
            # Use SVD
            try:
                sv = np.linalg.svd(M1, compute_uv=False)
            except:
                continue
            if len(sv) > 1 and sv[1] > 1e-9:
                continue
            # Also check other matricizations
            M2 = Tsum_mat.transpose(1, 0, 2).reshape(m*p, -1)
            sv2 = np.linalg.svd(M2, compute_uv=False)
            if len(sv2) > 1 and sv2[1] > 1e-9:
                continue
            M3 = Tsum_mat.transpose(2, 0, 1).reshape(p*n, -1)
            sv3 = np.linalg.svd(M3, compute_uv=False)
            if len(sv3) > 1 and sv3[1] > 1e-9:
                continue
            
            # All matricizations have rank 1, so the tensor is rank 1.
            # Extract u, v, w from M1
            a0 = np.argmax(M1)
            a0 = a0 // M1.shape[1]
            b0c0 = a0 % M1.shape[1]
            # Find where M1[a0, b0c0] is max
            # Actually, let's find a nonzero entry
            flat = M1.flatten()
            idx = np.argmax(np.abs(flat))
            if abs(flat[idx]) < 1e-15:
                continue
            a_idx = idx // M1.shape[1]
            bc_idx = idx % M1.shape[1]
            val = M1[a_idx, bc_idx]
            
            u_new = M1[:, bc_idx] / val
            # v_new[b] = M1[a_idx, b*p*n + c0] ... this is getting complicated
            # Let's just use the first nonzero entry approach
            # For now, skip this and use a different approach
            
    return None

def _brute_force_small_rank(n, m, p, max_rank, deadline):
    """
    Brute-force search for a rank-r decomposition with small integer/half-integer entries.
    Only feasible for very small ranks.
    """
    # This is not practical for rank 28-30. Skip.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    
    # Verify schoolbook
    if _verify(n, m, p, schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best = schoolbook
    else:
        # Shouldn't happen, but fallback
        best = schoolbook
    
    # Try to improve
    U_cur, V_cur, W_cur = schoolbook['U'], schoolbook['V'], schoolbook['W']
    rank_cur = len(U_cur[0])
    
    # Attempt rank reduction via pair merging
    while rank_cur > 1 and time.time() < deadline:
        U = np.array(U_cur, dtype=np.float64)
        V = np.array(V_cur, dtype=np.float64)
        W = np.array(W_cur, dtype=np.float64)
        rank = rank_cur
        merged = False
        
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline:
                    break
                
                # Check if the sum of the two rank-1 tensors has rank 1
                T1 = np.einsum('a,b,c->abc', U[:, r1], V[:, r1], W[:, r1])
                T2 = np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                Tsum = T1 + T2
                
                # Check rank via matricizations
                # Matricize along mode 0: shape (n*m, m*p * p*n)
                M0 = Tsum.reshape(n * m, m * p * p * n)
                # Use SVD to check rank
                try:
                    sv = np.linalg.svd(M0, compute_uv=False)
                except np.linalg.LinAlgError:
                    continue
                
                if len(sv) > 1 and sv[1] > 1e-10:
                    continue
                
                # Also verify with mode 1 and 2
                M1 = Tsum.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                sv1 = np.linalg.svd(M1, compute_uv=False)
                if len(sv1) > 1 and sv1[1] > 1e-10:
                    continue
                
                M2 = Tsum.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                sv2 = np.linalg.svd(M2, compute_uv=False)
                if len(sv2) > 1 and sv2[1] > 1e-10:
                    continue
                
                # Rank 1 confirmed. Extract new factor.
                # Find a nonzero entry
                Tsum_flat = Tsum.flatten()
                nonzero = np.nonzero(np.abs(Tsum_flat) > 1e-15)[0]
                if len(nonzero) == 0:
                    # Tsum is zero, just remove both
                    U_new = np.delete(U, [r1, r2], axis=1)
                    V_new = np.delete(V, [r1, r2], axis=1)
                    W_new = np.delete(W, [r1, r2], axis=1)
                    U_cur = U_new.tolist()
                    V_cur = V_new.tolist()
                    W_cur = W_new.tolist()
                    rank_cur -= 2
                    merged = True
                    break
                
                idx = nonzero[0]
                i, j, k = np.unravel_index(idx, Tsum.shape)
                val = Tsum[i, j, k]
                
                u_new = Tsum[i, :, :] / val  # This gives w_new * v_new... 
                # Actually: Tsum[a, b, c] = val * (Tsum[a, j, k] / val) * (Tsum[i, b, k] / val) * (Tsum[i, j, c] / val)
                # No, that's not right for rank 1.
                # For rank 1: Tsum[a,b,c] = u[a] * v[b] * w[c]
                # So u[a] = Tsum[a, b0, c0] / Tsum[i, b0, c0] where b0=j, c0=k
                u_new = Tsum[:, j, k] / val
                v_new = Tsum[i, :, k] / val
                w_new = Tsum[i, j, :] / val
                
                # Verify: u_new[a] * v_new[b] * w_new[c] should equal Tsum[a,b,c]
                Tcheck = np.outer(np.outer(u_new, v_new), w_new)
                if not np.allclose(Tcheck, Tsum, atol=1e-10):
                    continue
                
                # Replace the two columns with one
                U_new = np.delete(U, [r1, r2], axis=1)
                V_new = np.delete(V, [r1, r2], axis=1)
                W_new = np.delete(W, [r1, r2], axis=1)
                
                if r2 < r1:
                    r2 = r1
                r_insert = min(r1, r2)
                U_new = np.column_stack([U_new[:, :r_insert], u_new.reshape(-1, 1), U_new[:, r_insert:]])
                V_new = np.column_stack([V_new[:, :r_insert], v_new.reshape(-1, 1), V_new[:, r_insert:]])
                W_new = np.column_stack([W_new[:, :r_insert], w_new.reshape(-1, 1), W_new[:, r_insert:]])
                
                # Round to nearest half-integer
                U_new = np.round(U_new * 2) / 2
                V_new = np.round(V_new * 2) / 2
                W_new = np.round(W_new * 2) / 2
                
                # Verify the full decomposition
                U_cur_new = U_new.tolist()
                V_cur_new = V_new.tolist()
                W_cur_new = W_new.tolist()
                if _verify(n, m, p, U_cur_new, V_cur_new, W_cur_new):
                    U_cur, V_cur, W_cur = U_cur_new, V_cur_new, W_cur_new
                    rank_cur -= 1
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            break
    
    # Convert to list of lists with proper types
    U_final = np.array(U_cur)
    V_final = np.array(V_cur)
    W_final = np.array(W_cur)
    
    # Convert to integer or half-integer representation
    def _to_list(mat):
        mat = np.array(mat)
        # Check if all entries are integers or half-integers
        doubled = mat * 2
        if np.allclose(doubled, np.round(doubled), atol=1e-10):
            mat = np.round(doubled) / 2
            # Represent as float if half-integers, int if integers
            result = []
            for row in mat:
                result.append([float(x) for x in row])
            return result
        else:
            return mat.tolist()
    
    U_list = _to_list(U_final)
    V_list = _to_list(V_final)
    W_list = _to_list(W_final)
    
    # Final verification
    if not _verify(n, m, p, U_list, V_list, W_list):
        # Fall back to schoolbook
        return schoolbook
    
    return {"U": U_list, "V": V_list, "W": W_list}
