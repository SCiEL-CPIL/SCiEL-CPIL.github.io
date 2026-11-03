import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 100.0  # 50s margin before 150s kill

    def check_identity(U, V, W, R):
        """
        Check if the factors satisfy the tensor identity.
        U: (nm, R)
        V: (mp, R)
        W: (pn, R)
        
        Identity:
        sum_r U[i'*m+j', r] V[j''*p+k', r] W[k*n+i, r] = delta_{i', i} delta_{k', k} delta_{j', j''}
        """
        max_err = 0.0
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for i_prime in range(n):
                    for k_prime in range(p):
                        for j_prime in range(m):
                            u_idx = i_prime * m + j_prime
                            for j_double_prime in range(m):
                                v_idx = j_double_prime * p + k_prime
                                target = 1.0 if (i == i_prime and k == k_prime and j_prime == j_double_prime) else 0.0
                                val = np.dot(U_arr[u_idx] * V_arr[v_idx], W_arr[w_idx])
                                err = abs(val - target)
                                if err > max_err:
                                    max_err = err
        return max_err

    def strassen_rank_28():
        """
        Constructs the known rank-28 Strassen-based solution.
        Returns U, V, W as numpy arrays.
        """
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        components = [
            # M1
            ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
            # M2
            ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
            # M3
            ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
            # M4
            ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
            # M5
            ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
            # M6
            ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
            # M7
            ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
        ]
        
        terms = [
            (0, 0, 0, 0), # A0, B00, C0
            (2, 2, 0, 0), # A1, B10, C0
            (0, 0, 2, 2), # A0, B01, C1
            (2, 2, 2, 2), # A1, B11, C1
        ]
        
        col = 0
        for (a_co, b_ro, b_co, c_co) in terms:
            for (u_c, v_c, w_c) in components:
                for (li, lj, val) in u_c:
                    gi = li
                    gj = a_co + lj
                    U[gi * m + gj, col] += val
                for (li, lj, val) in v_c:
                    gi = b_ro + li
                    gj = b_co + lj
                    V[gi * p + gj, col] += val
                for (li, lj, val) in w_c:
                    gi = li
                    gj = c_co + lj
                    W[(c_co + lj) * n + li, col] += val
                col += 1
        
        return U, V, W

    def check_strassen_dependencies(U, V, W, R):
        """
        Check if any column of U is a linear combination of others.
        If U[:, r1] = sum c_i U[:, r_i], check if corresponding V and W
        columns also satisfy the same relationship (up to scaling).
        If so, we can eliminate that column.
        """
        # For each column r1, check if it's in the span of the others
        for r1 in range(R):
            others = [r for r in range(R) if r != r1]
            U_others = U[:, others]
            U_r1 = U[:, r1]
            
            # Check if U_r1 is in span of U_others
            if np.linalg.matrix_rank(U_others) == np.linalg.matrix_rank(np.column_stack([U_others, U_r1])):
                # U_r1 is in span. Find coefficients
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(U_others, U_r1, rcond=None)
                    if np.allclose(U_others @ coeffs, U_r1, atol=1e-10):
                        # Check V and W
                        V_others = V[:, others]
                        V_r1 = V[:, r1]
                        W_others = W[:, others]
                        W_r1 = W[:, r1]
                        
                        # For the tensor identity to hold, we need:
                        # V_r1 = V_others @ coeffs (exactly)
                        # W_r1 = W_others @ coeffs (exactly)
                        # OR there exists a scalar s such that V_r1 = s * V_others @ coeffs and W_r1 = W_r1/s
                        # Actually, the condition is: U_r1 V_r1^T W_r1 = sum_i c_i U_i V_i^T W_i
                        # This means U_r1 = sum c_i U_i, V_r1 = V_others @ c, W_r1 = W_others @ c
                        # Wait, that's not right. Let me think again.
                        # If U_r1 = sum c_i U_i, then for the tensor to match:
                        # U_r1 * V_r1 * W_r1 = sum_i c_i U_i * V_r1 * W_r1
                        # = sum_i c_i U_i * (V_others @ c) * (W_others @ c)
                        # This only works if V_r1 and W_r1 are specific combinations.
                        # Actually the correct condition: if we remove column r1 and adjust,
                        # we need: sum_i c_i (U_i, V_i, W_i) to replace (U_r1, V_r1, W_r1)
                        # This requires U_r1 = sum c_i U_i AND V_r1 = V_others@c AND W_r1 = W_others@c
                        # But that's too restrictive. The actual condition for tensor rank reduction:
                        # The 3-tensor T = sum_r U_r ⊗ V_r ⊗ W_r
                        # If U_r1 = sum c_i U_i, we can write:
                        # U_r1 ⊗ V_r1 ⊗ W_r1 = sum_i c_i U_i ⊗ V_r1 ⊗ W_r1
                        # This is NOT equal to sum_i c_i (U_i ⊗ V_i ⊗ W_i) in general.
                        # So simple linear dependence in U alone doesn't help.
                        pass
                except:
                    pass
        return None

    def try_rectangular_winograd():
        """
        Rectangular Winograd approach:
        Model 2x4x4 as combination of row-multiplication tensors.
        
        Key insight: C = A*B where A is 2x4, B is 4x4.
        Row 0 of C: C[0,:] = A[0,:] @ B
        Row 1 of C: C[1,:] = A[1,:] @ B
        
        The B matrix is shared. We can use Winograd's approach for 1x4x4
        (which is just a linear combination of 4 dot products) but exploit
        that both rows use the same B.
        
        Winograd for 1x4x4:
        For a single row a = [a0, a1, a2, a3] and B (4x4):
        c_k = sum_j a_j * B[j,k]
        
        Using Winograd with transforms:
        Let G_a be the transform of a (size 4 -> size 6 for 1x4x4 with overlap 2)
        Actually for 1x4x4, the standard Winograd uses:
        G (4x6) for input a, F (6x4) for B columns, and element-wise multiply.
        
        But since n=2, we have TWO rows of A. Let's define:
        A = [a0; a1] where a0, a1 are 1x4 vectors.
        
        For Winograd 1x4x4 with transform size 6:
        g0 = a0, g1 = a1, g2 = a2, g3 = a3, g4 = a0+a1, g5 = a1+a2 (or similar)
        Actually the standard Winograd for 1x4x4:
        Input transform G (4x6):
        Row 0: [1, 0, 0, 0, 0, 0]  -> a0
        Row 1: [0, 1, 0, 0, 0, 0]  -> a1  
        Row 2: [0, 0, 1, 0, 0, 0]  -> a2
        Row 3: [0, 0, 0, 1, 0, 0]  -> a3
        Wait, that's just the identity. Winograd needs a different transform.
        
        For 1x4x4, the optimal Winograd uses a (4, 6) transform:
        Actually, let me use the known Winograd transform for 1x4x4:
        
        The 1x4x4 tensor has rank at most 4 (schoolbook).
        Winograd can reduce this: using a transform of size 6:
        G (4x6):
        [1,  0,  0,  0,  1,  0]
        [0,  1,  0,  0,  0,  1]
        [0,  0,  1,  0,  0,  0]
        [0,  0,  0,  1,  0,  0]
        
        F (6x4):
        [1,  0,  0,  0]
        [1,  1,  0,  0]
        [0,  1,  1,  0]
        [0,  0,  1,  1]
        [0,  0,  1,  0]
        [0,  0, 0,  1]
        
        Hmm, this is getting complicated. Let me use a simpler approach.
        
        Actually, for 2x4x4, let me try a direct structured approach:
        
        The tensor T_{i,j,k,i',j',k'} = delta_{i,i'} delta_{k,k'} delta_{j,j'}
        where i,i' in {0,1}, j,j' in {0..3}, k,k' in {0..3}.
        
        Let me try to find a rank-24 decomposition by exploiting the structure.
        
        Approach: Use the fact that for fixed (i,j), the contribution to C[i,k] 
        is A[i,j] * B[j,k]. So the tensor is:
        T = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        
        In tensor form: T_{(i,j),(j',k'),(k,i')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}
        
        Let me try a Winograd-style approach:
        For each "column" of B (fixed k'), we have a 2x4 subproblem.
        C[:,k'] = A @ B[:,k']
        
        This is a 2x4 times 4x1 multiplication, which has rank at most 8 (schoolbook).
        But we can do better using the structure.
        
        Actually, let me try a different approach: use the known rank bounds.
        For 2x4x4, the rank is at most 28 (Strassen). Lower bound is 24 (from 2x2x2).
        
        Let me try to construct a rank-24 solution using a structured approach.
        """
        R = 24
        # This is a placeholder - we need to actually construct the solution
        # For now, return None to fall back to Strassen
        return None

    # 1. Try to find rank 24 with rectangular Winograd
    result_24 = try_rectangular_winograd()
    if result_24 is not None:
        U, V, W = result_24
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # 2. Fallback to rank 28 (Strassen)
    U, V, W = strassen_rank_28()
    
    # Verify Strassen solution just in case
    err = check_identity(U, V, W, 28)
    if err > 1e-9:
        # Fallback to schoolbook
        R = n * m * p
        U_s = np.zeros((nm, R))
        V_s = np.zeros((mp, R))
        W_s = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
