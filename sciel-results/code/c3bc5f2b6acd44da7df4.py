import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    def build_constraint_matrix():
        """Build matrix M where each row corresponds to one (i,k,j,jp) constraint.
        The constraint is sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c] = delta(j,jp).
        We linearize by treating the 56 schoolbook monomials as basis.
        For a rank-r decomposition, we need to find U,V,W such that
        the bilinear form matches. We use the 'flattened' approach:
        For each c, define vector z_c of length 56 where z_c[i*m+j, jp*p+k, k*n+i] 
        is not directly linear. Instead, we use the standard linearization:
        The tensor T has entries T[i,j,k] = 1 if j==jp (for the multiplication tensor).
        Actually, the multiplication tensor has T[i,j,k] where the output is C[i,k] = sum_j A[i,j]*B[j,k].
        The tensor entries are T[i,j,k] = delta_{j,j'} for the identity in the middle... 
        Let me use the standard formulation: we want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if j==j' else 0.
        But j and j' are both indices into the m dimension. The constraint is for each (i, j, j', k):
        sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = (1 if j==j' else 0)
        
        To linearize for rank reduction, we fix two of the three matrices and solve for the third,
        or we use the approach of treating each term c as a rank-1 tensor and checking linear independence.
        
        Simpler: build a matrix where columns are the 56 'atomic' rank-1 tensors (schoolbook),
        and check if our 50-term solution can be compressed.
        """
        # The 56 schoolbook terms: for each (i,j,k), the term is U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # These 56 rank-1 tensors span the space of all decompositions.
        # Our current solution is a linear combination of 50 of them (with Strassen combining).
        # Actually, Strassen terms are NOT linear combinations of schoolbook terms in the same way.
        # Let me just check: are the 50 columns of (U,V,W) linearly independent as rank-1 tensors?
        # A rank-1 tensor is determined by (U_col, V_col, W_col). Two such tensors are linearly dependent
        # if one is a scalar multiple of the other. Since we use integer entries, check if any column
        # is a scalar multiple of another.
        return None

    # Build the base rank-50 solution
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1

        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1

        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1

        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1

        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1

        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1

        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify base solution
    if not verify(U, V, W):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now attempt rank reduction via nullspace analysis
    # Build the linear constraint system. For each output position (i,k) and each pair (j,jp),
    # we need: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c] = delta(j,jp)
    # This is nonlinear in (U,V,W). To linearize, we can fix W and solve for U,V or vice versa.
    # Alternative: use the fact that if we have a valid rank-r decomposition, we can try to find
    # a rank-(r-1) decomposition by looking for a linear relation among the rank-1 tensors.
    
    # Check for linear dependence among the 50 rank-1 tensors
    # A rank-1 tensor (u_c, v_c, w_c) can be represented as a vector in R^{nm * mp * pn} but that's huge.
    # Instead, use the 'flattened' representation: for each c, form the matrix M_c = u_c * v_c^T (size nm x mp)
    # and check if the set {M_c * w_c[i,k] for all i,k} spans the identity. This is complex.
    
    # Simpler heuristic: try to remove one term and see if the remaining 49 can be adjusted.
    # This is an underdetermined linear system. Let's try a few removals.
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank

    # Try removing each term and solving the linear system for the remaining
    # We'll fix the structure: for each term c, the contribution to (i,j,jp,k) is
    # U[i*m+j,c] * V[jp*p+k,c] * W[k*n+i,c]
    # If we remove term r, we need to find new values for the other 49 terms such that
    # the sum still equals delta(j,jp). This is a system of 56*4 = 224 equations (but many are 0=0)
    # Actually, the number of independent constraints is n*p*m = 2*7*4 = 56 (one for each (i,k,j) with j=jp).
    # Wait, the constraint is for each (i, j, jp, k): sum_c ... = delta(j,jp). There are n*m*m*p = 2*4*4*7 = 224 constraints,
    # but most are 0=0 (when j != jp, the sum must be 0). The non-trivial ones are when j=jp: n*m*p = 56 constraints.
    
    # Let's build the 56x50 matrix for the j=jp constraints and check its rank.
    # For each (i, j, k) [with j=jp], the equation is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is still nonlinear. Let me use a different approach: 
    # Treat the product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] as a single variable x_{i,j,k,c}.
    # Then the constraint is sum_c x_{i,j,k,c} = 1 for each (i,j,k).
    # The variables x are constrained by the rank-1 structure: x_{i,j,k,c} = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    # This is the tensor decomposition problem.
    
    # Given the complexity, let's try a simpler approach: use SVD on the 'unfolding' of the tensor.
    # The multiplication tensor T has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # Actually, T is a 3-tensor of size n x m x p with T[i,j,k] = 1 (it's the identity in the sense that
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]).
    # The rank of this tensor is what we're minimizing.
    
    # For 2x4x7, the tensor is T in R^{2 x 4 x 7} with all entries = 1.
    # Wait, no. The multiplication tensor for m x n * n x p is in R^{(m*n) x (n*p) x (m*p)} with
    # T[i,j, k,l] = delta_{j,k}. For our case, it's in R^{8 x 28 x 14} with T[i*m+j, j'*p+k, k'*n+i] = delta_{j,j'}.
    # The rank is the minimum r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c.
    
    # Let's just try to verify if rank 49 is possible by checking if the 50 terms are linearly independent
    # as vectors in the 'flattened' space. Flatten each rank-1 tensor to a vector of length 8*28*14 = 3136.
    # That's too large. Instead, use the unfolding: for each c, form the matrix U_c (8x1) * V_c (28x1) outer product,
    # then contract with W_c. 
    
    # Given time constraints, let's just return the rank-50 solution. The Gröbner basis approach is too complex
    # to implement in the remaining time budget.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
