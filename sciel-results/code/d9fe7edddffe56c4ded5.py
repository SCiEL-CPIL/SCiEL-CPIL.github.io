import time
import fractions
import itertools

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]

    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]

    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,i'} delta_{j,j'} delta_{k,k'}
    for the tensor T[i,j,k; i',j',k'] = (A*B)[i,k] where A is n x m, B is m x p.
    The multiplication tensor has T[i,j,k, i',j',k'] = 1 if i=i', j=j', k=k' else 0.
    Wait - the standard matrix multiplication tensor for (n x m) * (m x p) -> (n x p):
    T[i, j, k] for input indices i (row of A), j (col of A / row of B), k (col of B)
    and output index (i, k). The tensor is:
    M[i,j,k; i',k'] = delta_{j,j'} ... 
    Actually the multiplication tensor M has entries M[i,j,k, i',k'] where:
    (A*B)[i,k] = sum_j A[i,j] B[j,k]
    So the tensor T[i,j,k] = e_i e_j e_k (as a rank-1 in the input space) maps to e_i e_k in output.
    The full tensor: T[i, j, k, i', j', k'] = 1 if (i=i' and j=j' and k=k')... no.
    
    The correct tensor: for multiplying A (n x m) by B (m x p):
    (A*B)[i, k] = sum_j A[i, j] * B[j, k]
    The tensor M in the space (R^n x R^m) (x R^m x R^p) -> R^(n x p):
    M[i, j, k, i', j', k'] = delta_{i, i'} * delta_{j, j'} * delta_{k, k'}
    Wait, that's the identity tensor, not multiplication.
    
    Let me think again. The matrix multiplication tensor for (n x m) * (m x p) is:
    T in (R^n (x R^m) (x R^m (x R^p))^* (x R^(n x p))^*
    T(i, j, k; i', k') = 1 if j is the shared index... 
    
    Standard: T[i, j, k, i', k'] = delta_{i, i'} * delta_{j, j} * delta_{k, k}... 
    
    No. The multiplication map is:
    mult: R^(n x m) (x R^(m x p) -> R^(n x p)
    mult(A, B)[i, k] = sum_j A[i, j] B[j, k]
    
    The tensor T represents this bilinear map:
    T[i, j, k, i', k'] = 1 if i = i' and k = k' ... no, that's not right either.
    
    T is a 5-index tensor: T[i, j, k, i', k'] where i,i' in [0,n), j in [0,m), k,k' in [0,p)
    T[i, j, k, i', k'] = 1 if (i = i' and j = j and k = k')... 
    
    I think the convention is:
    T[i, j, k, i', j', k'] = 1 iff i=i', j=j', k=k'
    But that's the identity, not matrix multiplication.
    
    For matrix multiplication (n x m) * (m x p):
    The tensor has 5 indices: (i, j, k, i', k') where
    T[i, j, k, i', k'] = 1 if i = i' and k = k' ... 
    
    Actually, I recall now. The multiplication tensor M_{n,m,p} is:
    M[i, j, k, i', k'] = delta_{i,i'} delta_{j,j} delta_{k,k'}... 
    
    Let me just use the standard: 
    M[i, j, k, i', k'] = 1 if (i, j, k) and (i', k') satisfy the multiplication relation.
    
    The bilinear form: <M, A (x B> = sum_{i,j,k} M[i,j,k,i',k'] A[i,j] B[j,k] = (A*B)[i',k']
    
    So M[i,j,k,i',k'] * A[i,j] B[j,k] summed = (AB)[i',k'] = sum_j A[i',j] B[j,k']
    
    Therefore M[i,j,k,i',k'] = delta_{i,i'} delta_{j,j} delta_{k,k'}... 
    
    sum_{i,j,k} M[i,j,k,i',k'] A[i,j] B[j,k] = sum_j A[i',j] B[j,k']
    
    This means M[i,j,k,i',k'] = delta_{i,i'} delta_{j,j} delta_{k,k'}... but j appears in both A and B.
    
    Actually: M[i,j,k,i',k'] should satisfy:
    sum_{i,j,k} M[i,j,k,i',k'] A[i,j] B[j,k] = (AB)[i',k'] = sum_j A[i',j] B[j,k']
    
    So M[i,j,k,i',k'] = delta_{i,i'} delta_{k,k'} and the j index is summed over... 
    
    No, M is a tensor with 5 indices. The contraction is:
    (AB)[i',k'] = sum_{j} A[i',j] B[j,k']
    
    If we write A = sum_{i,j} A[i,j] e_i (x e_j and B = sum_{j,k} B[j,k] e_j (x e_k, then:
    (AB)[i',k'] = sum_{i,j,k} A[i,j] B[j,k] M[i,j,k,i',k']
    
    And this should equal sum_j A[i',j] B[j,k']
    
    So M[i,j,k,i',k'] = 1 if i=i', k=k', and j is free (summed over). But M can't depend on j in a way that's summed... 
    
    I think the correct statement is: M[i,j,k,i',k'] = delta_{i,i'} delta_{k,k'} and the j index is the same j in A[i,j] and B[j,k], so:
    
    sum_{i,j,k} delta_{i,i'} delta_{k,k'} A[i,j] B[j,k] = sum_j A[i',j] B[j,k'] = (AB)[i',k']  ✓
    
    So M[i,j,k,i',k'] = delta_{i,i'} delta_{k,k'} (independent of j).
    
    Now the decomposition: M = sum_r U_r (x V_r (x W_r where:
    U_r[i,j] * V_r[j,k] * W_r[i',k'] summed over r = M[i,j,k,i',k']
    
    Wait, that's not how it works. The tensor M is in (R^(n x m))^* (x (R^(m x p))^* (x (R^(n x p))^*
    
    A rank-1 decomposition: M = sum_r u_r (x v_r (x w_r where u_r in R^(n x m), v_r in R^(m x p), w_r in R^(n x p)
    
    And <u_r, A> <v_r, B> <w_r, AB> ... no.
    
    The standard: M = sum_r u_r (x v_r (x w_r where
    M[i,j,k,i',k'] = sum_r U_r[i,j] V_r[j,k] W_r[i',k']
    
    And we need: sum_r U_r[i,j] V_r[j,k] W_r[i',k'] = delta_{i,i'} delta_{k,k'}
    
    With the trace convention W[k*n + i, r], W is indexed by (k, i) for output position.
    
    So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} delta_{k,k'}... 
    
    But wait, there's no i' or k' on the left. The left side is a function of (i,j,k) only, but the right side depends on (i',k') too. That can't be right.
    
    I think I'm confusing myself. Let me use the standard convention from the problem:
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W has shape (p*n, rank), indexed by (k, i) -> W[k*n + i, r].
    
    The decomposition should satisfy:
    For all i in [0,n), j in [0,m), k in [0,p), i' in [0,n), k' in [0,p):
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k'*n + i', r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    No wait, that's the identity tensor. For matrix multiplication:
    
    (A*B)[i', k'] = sum_j A[i', j] B[j, k']
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give the multiplication tensor.
    
    I believe the correct verification is:
    For the multiplication tensor T[i,j,k,i',k'] = delta_{i,i'} delta_{k,k'} (with j being the shared index that gets summed):
    
    Actually, I think the way to verify is:
    Construct the bilinear form: for random A (n x m) and B (m x p), check that
    sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k]) * (sum_{i,k} W[k*n+i,r] C[i,k])
    equals sum_{i,k} (AB)[i,k] C[i,k] for all C.
    
    Or more simply: the decomposition is valid if for all i,j,k,i',k':
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i=i' and k=k') else 0... 
    
    But the left side only has (i,j,k) and the right side has (i',k'). This doesn't make sense unless the tensor has 5 indices and the decomposition is:
    
    T[i,j,k,i',k'] = sum_r U_r[i,j] V_r[j,k] W_r[i',k']
    
    And T[i,j,k,i',k'] = delta_{i,i'} delta_{k,k'} (the j is implicit in the product structure).
    
    Hmm, but then j appears in both U and V but not in W, and the right side doesn't depend on j. So:
    sum_r U_r[i,j] V_r[j,k] W_r[i',k'] = delta_{i,i'} delta_{k,k'}
    
    This must hold for all i,j,k,i',k'. Let me just verify numerically with the known rank-56 solution.
    
    I'll verify by checking: for all (i,j,k,i',k'), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k'*n+i',r] == (1 if i==i' and k==k' else 0)
    
    Wait, but W is indexed by (k, i) per the trace convention W[k*n+i, r]. So W[k'*n+i', r] is the entry for output position (i', k').
    
    Let me just do a numerical check with the rank-56 solution.
    """
    n, m, p = n, m, p
    rank = len(U[0])
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] == delta_{i,i'} delta_{k,k'}
    # for all i,j,k,i',k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        expected = 1 if (i == ip and k == kp) else 0
                        actual = 0
                        for r in range(rank):
                            actual += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        if actual != expected:
                            return False
    return True


def _try_reduce_rank(U, V, W, n=2, m=4, p=8, deadline=None):
    """Try to find a rank reduction by 1 from the given decomposition.
    We look for a non-trivial linear combination of the rank-1 terms that sums to zero.
    That is, find coefficients c_0, ..., c_{R-1} not all zero such that
    sum_r c_r U_r (x V_r (x W_r = 0 as a tensor.
    
    This is equivalent to: for all (i,j,k,i',k'):
    sum_r c_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] = 0
    
    We can set up a linear system. The number of equations is n*m*p*n*p = 2*4*8*2*8 = 1024.
    The number of unknowns is R = 56.
    
    We look for a vector c in the null space of the 1024 x 56 matrix A where
    A[equation, r] encodes the tensor entry for that equation and rank r.
    
    But the tensor entry for equation (i,j,k,i',k') and rank r is:
    U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]
    
    This is not linear in c... wait, it IS linear in c. The equation is:
    sum_r c_r * (U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]) = 0
    
    So the coefficient of c_r in equation (i,j,k,i',k') is:
    U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]
    
    We want a non-trivial c in the null space. The null space dimension is at least 56 - rank(A).
    If rank(A) < 56, we have a non-trivial null space vector, and we can eliminate one term.
    
    But for a valid rank-56 decomposition of a tensor with tensor rank 56, the 56 rank-1 terms should be linearly independent, so rank(A) = 56 and the null space is trivial.
    
    However, if the true tensor rank is 55, then there exists a rank-55 decomposition, and the rank-56 decomposition might have a dependency.
    
    Actually, even if the tensor rank is 56, a particular rank-56 decomposition might have linearly dependent terms (i.e., it's not a minimal decomposition in the sense of linear independence). But if it's a valid decomposition, the terms must span the tensor, so they should be independent.
    
    Let me try a different approach: use the known structure. The rank-56 solution is built from 8 blocks of 7 (Strassen). Maybe we can find a dependency across blocks.
    
    For now, let me just check if the 56 columns of the "coefficient matrix" are linearly independent. If not, we can reduce the rank.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    n, m, p = n, m, p
    R = len(U[0])
    
    # Build the 1024 x R matrix
    # Number of equations: n * m * p * n * p
    num_eq = n * m * p * n * p
    
    # We'll use fractions for exact arithmetic, but this might be slow.
    # Let's use numpy first to check rank, then verify with fractions if needed.
    import numpy as np
    
    A = np.zeros((num_eq, R), dtype=float)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        for r in range(R):
                            A[eq_idx, r] = U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                        eq_idx += 1
    
    # Check rank
    rank_A = np.linalg.matrix_rank(A, tol=1e-9)
    
    if rank_A < R:
        # There's a null space vector
        # Find it
        _, _, Vt = np.linalg.svd(A)
        # The last row of Vt is the null space vector
        c = Vt[-1, :]
        
        # Find the index of the largest magnitude coefficient
        max_idx = np.argmax(np.abs(c))
        
        # Eliminate term max_idx: for each other term r, add (c[r]/c[max_idx]) * term_r to term_max_idx
        # But we need to be careful: we're modifying the decomposition.
        # Actually, if sum_r c_r * T_r = 0, then T_max = -sum_{r != max} (c_r / c_max) * T_r
        # So we can replace U_max with -sum_{r != max} (c_r / c_max) * U_r, etc.
        
        # But this might not give integer/half-integer entries. Let's check.
        
        # For now, just report that a reduction is possible
        # We need to construct the new decomposition
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        
        # Remove column max_idx and adjust others
        # Actually, it's easier to just construct a new (R-1) rank decomposition
        # by expressing term max_idx as a combination of the others.
        
        # New rank is R - 1
        new_rank = R - 1
        new_U2 = [[0] * new_rank for _ in range(n * m)]
        new_V2 = [[0] * new_rank for _ in range(m * p)]
        new_W2 = [[0] * new_rank for _ in range(p * n)]
        
        # Map old indices to new indices (skipping max_idx)
        new_idx_map = {}
        new_i = 0
        for old_i in range(R):
            if old_i != max_idx:
                new_idx_map[old_i] = new_i
                new_i += 1
        
        # Copy all terms except max_idx
        for old_r in range(R):
            if old_r == max_idx:
                continue
            new_r = new_idx_map[old_r]
            for row in range(n * m):
                new_U2[row][new_r] = U[row][old_r]
            for row in range(m * p):
                new_V2[row][new_r] = V[row][old_r]
            for row in range(p * n):
                new_W2[row][new_r] = W[row][old_r]
        
        # Now add the contribution of term max_idx distributed to other terms
        # T_max = -sum_{r != max} (c_r / c_max) * T_r
        # So for each r != max, we add (c_r / c_max) * T_r to... 
        # Wait, we need: sum_{new_r} new_T_new_r = sum_{old_r != max} T_old_r + T_max
        # = sum_{old_r != max} T_old_r - sum_{old_r != max} (c_old_r / c_max) * T_old_r
        # = sum_{old_r != max} (1 - c_old_r / c_max) * T_old_r
        # 
        # So new_U2[row][new_r] = (1 - c[old_r] / c[max_idx]) * U[row][old_r]
        
        for old_r in range(R):
            if old_r == max_idx:
                continue
            new_r = new_idx_map[old_r]
            factor = 1.0 - c[old_r] / c[max_idx]
            for row in range(n * m):
                new_U2[row][new_r] = factor * U[row][old_r]
            for row in range(m * p):
                new_V2[row][new_r] = factor * V[row][old_r]
            for row in range(p * n):
                new_W2[row][new_r] = factor * W[row][old_r]
        
        # Verify the new decomposition
        # Convert to fractions for exact check
        # For simplicity, just check numerically
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    if not valid:
                        break
                    for ip in range(n):
                        if not valid:
                            break
                        for kp in range(p):
                            expected = 1 if (i == ip and k == kp) else 0
                            actual = 0.0
                            for r in range(new_rank):
                                actual += new_U2[i * m + j][r] * new_V2[j * p + k][r] * new_W2[kp * n + ip][r]
                            if abs(actual - expected) > 1e-9:
                                valid = False
                                break
        
        if valid:
            # Check if entries are integer or half-integer
            all_half_int = True
            for row in new_U2:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        all_half_int = False
                        break
                if not all_half_int:
                    break
            if all_half_int:
                for row in new_V2:
                    for val in row:
                        if abs(val * 2 - round(val * 2)) > 1e-9:
                            all_half_int = False
                            break
                    if not all_half_int:
                        break
            if all_half_int:
                for row in new_W2:
                    for val in row:
                        if abs(val * 2 - round(val * 2)) > 1e-9:
                            all_half_int = False
                            break
                    if not all_half_int:
                        break
            
            if all_half_int:
                # Round to half-integers
                def round_half(x):
                    return round(x * 2) / 2.0
                for row in new_U2:
                    for idx in range(len(row)):
                        row[idx] = round_half(row[idx])
                for row in new_V2:
                    for idx in range(len(row)):
                        row[idx] = round_half(row[idx])
                for row in new_W2:
                    for idx in range(len(row)):
                        row[idx] = round_half(row[idx])
                
                return {"U": new_U2, "V": new_V2, "W": new_W2}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Start with the known rank-56 solution
    base = _build_rank56()
    
    # Try to reduce the rank
    if time.time() < deadline - 10:
        reduced = _try_reduce_rank(base["U"], base["V"], base["W"], deadline=deadline - 10)
        if reduced is not None:
            return reduced
    
    # Fallback to rank-56
    return base
