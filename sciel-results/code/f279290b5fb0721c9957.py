"""Program for 2x2 * 2x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 110.0


def construct():
    n, m, p = DIMENSIONS
    
    # First, build the known rank-21 Strassen decomposition as a baseline
    U21, V21, W21 = _build_strassen_21(n, m, p)
    
    # Try to reduce rank by searching for a linear combination of the 21 rank-1 terms
    # that can be expressed as fewer terms, or by finding a rank-20 decomposition
    # via a local optimization / null-space search.
    
    # Strategy: The tensor T is decomposed as sum_{r=1}^{21} u_r \otimes v_r \otimes w_r.
    # We want to find if there exist vectors u', v', w' such that
    # T = sum_{r=1}^{20} u'_r \otimes v'_r \otimes w'_r.
    # This is equivalent to finding a rank-20 decomposition.
    
    # A practical approach: try to merge two of the 21 terms into one,
    # or find a linear dependence among the terms that allows reduction.
    
    # For 2x2x6, the tensor can be viewed as 3 blocks of 2x2x2.
    # The rank of 2x2x2 is 7 (Strassen). So 3*7 = 21.
    # Can we do better? The theoretical lower bound for 2x2x6 is unknown,
    # but it's at least the rank of 2x2x2 = 7, and at most 24 (schoolbook).
    # 21 is likely optimal or close to it, but let's try a quick search.
    
    # Quick search: try to find a rank-20 decomposition by solving a system of equations.
    # This is a non-convex problem, so we'll use a heuristic:
    # 1. Start with the 21-term decomposition.
    # 2. Try to express one term as a linear combination of the others.
    # 3. If successful, we have a rank-20 decomposition.
    
    # However, this is computationally expensive. Given the time constraint,
    # we'll stick with the known rank-21 decomposition unless we can quickly
    # verify a lower rank.
    
    # For now, return the rank-21 decomposition.
    # In a future iteration, we could implement a more sophisticated search.
    
    if time.time() > _DEADLINE:
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    
    # Try a quick rank reduction search
    U20, V20, W20 = _try_rank_reduction(U21, V21, W21, n, m, p)
    
    if U20 is not None:
        return {"U": U20.tolist(), "V": V20.tolist(), "W": W20.tolist()}
    
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}


def _build_strassen_21(n, m, p):
    """Build the rank-21 Strassen decomposition for 2x2 * 2x6."""
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*p + k
    # W indices: k*n + i
    
    # We process 3 blocks of 2x2
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V row indices for this block
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # W row indices for this block
        w_c00 = k0 * n + 0  # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n + 0  # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]
        
        # Strassen steps 0..6
        # Step 0: P1 = (a+d)(e+h)
        r = b * 7 + 0
        U[0, r] = 1; U[3, r] = 1
        V[v_e, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1; W[w_c11, r] = 1
        
        # Step 1: P2 = (c+d)e
        r = b * 7 + 1
        U[2, r] = 1; U[3, r] = 1
        V[v_e, r] = 1
        W[w_c10, r] = 1; W[w_c11, r] = -1
        
        # Step 2: P3 = a(f-h)
        r = b * 7 + 2
        U[0, r] = 1
        V[v_f, r] = 1; V[v_h, r] = -1
        W[w_c01, r] = 1; W[w_c11, r] = 1
        
        # Step 3: P4 = d(g-e)
        r = b * 7 + 3
        U[3, r] = 1
        V[v_g, r] = 1; V[v_e, r] = -1
        W[w_c00, r] = 1; W[w_c10, r] = 1
        
        # Step 4: P5 = (a+b)h
        r = b * 7 + 4
        U[0, r] = 1; U[1, r] = 1
        V[v_h, r] = 1
        W[w_c00, r] = -1; W[w_c01, r] = 1
        
        # Step 5: P6 = (c-a)(e+f)
        r = b * 7 + 5
        U[2, r] = 1; U[0, r] = -1
        V[v_e, r] = 1; V[v_f, r] = 1
        W[w_c11, r] = 1
        
        # Step 6: P7 = (b-d)(g+h)
        r = b * 7 + 6
        U[1, r] = 1; U[3, r] = -1
        V[v_g, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1
    
    return U, V, W


def _try_rank_reduction(U21, V21, W21, n, m, p):
    """Try to reduce the rank from 21 to 20 by finding a linear dependence."""
    rank = 21
    
    # The tensor T can be written as:
    # T = sum_{r=1}^{21} u_r \otimes v_r \otimes w_r
    # where u_r, v_r, w_r are the r-th columns of U, V, W.
    
    # We want to check if one of these terms can be expressed as a linear
    # combination of the others. This is equivalent to checking if the
    # 21 vectors (u_r \otimes v_r \otimes w_r) are linearly independent.
    
    # However, this is a high-dimensional space (2*2*6 = 24 dimensions),
    # so we can check if the 21 vectors span a 21-dimensional space.
    
    # Flatten each term into a 24-dimensional vector
    terms = []
    for r in range(rank):
        term = np.zeros(n * m * p)
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    # The term is u_r[i] * v_r[j] * w_r[k]
                    # But we need to map (i, j, k) to the correct tensor index.
                    # i = i_A * m + j_A, j = j_B * p + k_B, k = k_C * n + i_C
                    # The tensor index is (i_A, j_A, k_C) -> i_A * m * p + j_A * p + k_C
                    i_A = i // m
                    j_A = i % m
                    j_B = j // p
                    k_B = j % p
                    k_C = k // n
                    i_C = k % n
                    
                    # Check if this term contributes to the tensor at (i_A, j_A, k_C)
                    # The contribution is u_r[i] * v_r[j] * w_r[k]
                    # but only if j_A == j_B and k_B == k_C (for the matrix multiplication)
                    # Actually, the tensor T_{i_A, j_A, k_C} = sum_{j_B} A_{i_A, j_A} * B_{j_B, k_C}
                    # So the term u_r[i] * v_r[j] * w_r[k] contributes to T_{i_A, j_A, k_C}
                    # only if j_A == j_B and k_B == k_C.
                    if j_A == j_B and k_B == k_C:
                        term[i_A * m * p + j_A * p + k_C] += U21[i, r] * V21[j, r] * W21[k, r]
        terms.append(term)
    
    # Stack the terms into a matrix
    T_matrix = np.array(terms).T  # Shape: (24, 21)
    
    # Check the rank of this matrix
    try:
        matrix_rank = np.linalg.matrix_rank(T_matrix)
    except:
        return None, None, None
    
    # If the rank is less than 21, we can reduce the number of terms
    if matrix_rank < rank:
        # Find a basis for the column space
        # This is a complex step, so for now, we'll just return None
        # In a future iteration, we could implement this properly.
        pass
    
    return None, None, None
# EVOLVE-BLOCK-END
