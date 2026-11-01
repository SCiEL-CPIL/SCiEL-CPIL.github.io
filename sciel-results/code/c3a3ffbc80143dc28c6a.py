"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    Strategy:
    Split the 5x6 matrix B into two horizontal blocks: B1 (5x2) and B2 (5x4).
    Then C = A*B = [A*B1, A*B2].
    
    For A*B1 (2x5 * 5x2 -> 2x2):
      We can use Strassen's algorithm for 2x2*2x2, but A is 2x5 not 2x2.
      Actually, let's think differently.
      
    Alternative approach:
    Use the fact that rank(2x5x6) <= rank(2x2x6) + rank(2x3x6)
    where we split the middle dimension 5 = 2 + 3.
    
    For 2x2x6: we can further split as 2x2x2 + 2x2x2 + 2x2x2 (three 2x2x2 blocks)
    Each 2x2x2 can be done in rank 7 (Strassen).
    So rank(2x2x6) <= 3*7 = 21.
    
    For 2x3x6: split as 2x1x6 + 2x2x6
    2x1x6: rank = 1*6 = 6 (outer product style, each output element is a dot product of a row of A with a column of B)
    Actually 2x1x6: A is 2x1, B is 1x6, C is 2x6. Each C[i,k] = A[i,0]*B[0,k].
    This is rank 6: for each k, one term A[i,0]*B[0,k].
    Wait, that's not right. Let me reconsider.
    
    For 2x1x6: 
    U is 2x1 (2 rows, rank columns)
    V is 1x6 (6 rows)
    W is 6x2 (12 rows)
    
    C[i,k] = sum_r U[i,r] * V[r,k] * ... no wait.
    
    The tensor T_{i,j,k} = delta for multiplication: C[i,k] = sum_j A[i,j] B[j,k].
    
    For 2x1x6: j ranges over {0}. C[i,k] = A[i,0] B[0,k].
    This is an outer product of A (2x1) and B (1x6), giving C (2x6).
    The rank of this "matrix multiplication tensor" is 1*6 = 6? 
    Actually, T_{i,0,k} = 1 if we define it as the identity. No.
    
    The tensor for 2x1x6 has entries T_{i,j,k} where j is always 0.
    T_{i,0,k} = 1 for all i in {0,1}, k in {0,...,5}.
    
    We need: C[i,k] = sum_r U[i,r] * V[0,r] * W[k,r] ... no, that's not the convention.
    
    Convention: U has shape (n*m, rank) = (2*1, rank) = (2, rank)
    V has shape (m*p, rank) = (1*6, rank) = (6, rank)
    W has shape (p*n, rank) = (6*2, rank) = (12, rank)
    
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r]) * ... 
    Actually the standard convention is:
    C[i,k] = sum_j A[i,j] B[j,k]
    And the tensor decomposition is:
    T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, the W index is k*n+i. So W[k*n+i, r].
    
    For 2x1x6: i in {0,1}, j in {0}, k in {0,...,5}
    T_{i,0,k} = 1 (since C[i,k] = A[i,0]*B[0,k], the tensor entry is 1)
    
    We need: sum_r U[i,0,r] * V[0,k,r] * W[k*2+i, r] = 1 for all i,k
    
    Simplest: set rank=6. For r in 0..5:
    U[i,0,r] = 1 for all i (since we need it independent of i)
    V[0,k,r] = 1 if k==r, else 0
    W[k*2+i, r] = 1 if k==r, else 0
    
    Check: sum_r U[i,0,r] * V[0,k,r] * W[k*2+i, r] = sum_r 1 * delta(k,r) * delta(k,r) = delta(k,k) = 1. ✓
    
    So rank(2x1x6) = 6.
    
    For 2x2x6: split k into three groups of 2: k in {0,1}, {2,3}, {4,5}
    Each is a 2x2x2 problem. Use Strassen for each, rank 7 each.
    Total: 3*7 = 21.
    
    So rank(2x2x6) <= 21.
    
    Now for 2x3x6: split j into 1 + 2.
    j=0: 2x1x6, rank 6
    j=1,2: 2x2x6, rank 21
    Total: 6 + 21 = 27.
    
    So rank(2x3x6) <= 27.
    
    Total rank(2x5x6) <= rank(2x2x6) + rank(2x3x6) = 21 + 27 = 48.
    
    This is better than 60.
    """
    n, m, p = 2, 5, 6
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 10 x 48
    V = [[0] * rank for _ in range(m * p)]  # 30 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48
    
    def add_factors(start_col, u_indices, v_indices, w_indices, values):
        """
        Add one term to the decomposition.
        u_indices: list of (global_u_index, value)
        v_indices: list of (global_v_index, value)
        w_indices: list of (global_w_index, value)
        """
        for (ui, uv), (vi, vv), (wi, wv) in zip(u_indices, v_indices, w_indices):
            U[ui][start_col] += uv
            V[vi][start_col] += vv
            W[wi][start_col] += wv
    
    # Strassen 2x2x2 decomposition
    # For 2x2 * 2x2 -> 2x2
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U local: i*2 + j (4 entries)
    # V local: j*2 + k (4 entries)
    # W local: k*2 + i (4 entries)
    # 
    # Strassen's 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # 
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M3 + M6 + M7
    
    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # M1 = (A11+A22)*(B11+B22)
        # U: A11=(i0,j0), A22=(i1,j1). U[(i*2+j)] gets +1 for j such that...
        # Actually U[i*2+j, r] contributes to A[i,j]
        # For M1: we need U to pick out A11 and A22, V to pick out B11 and B22
        # But Strassen uses combinations. Let me define each product term.
        
        # Term 1: M1
        # U: A11 -> (0, +1), A22 -> (3, +1)  [indices: i*2+j: A11=0*2+0=0, A22=1*2+1=3]
        # V: B11 -> (0, +1), B22 -> (3, +1)  [indices: j*2+k: B11=0*2+0=0, B22=1*2+1=3]
        # W: C11 -> (1, +1), C22 -> (3, +1)  [indices: k*2+i: C11=0*2+0=0, C22=1*2+1=3]
        # Wait, W contributes to C. C11 gets +M1, C22 gets +M1.
        # But also C11 gets -M4, +M5, +M7, etc.
        # So for M1, W entries: C11 (k=0,i=0) -> index 0*2+0=0, coeff +1
        #                         C22 (k=1,i=1) -> index 1*2+1=3, coeff +1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        
        # Term 2: M2 = (A21+A22)*B11
        # U: A21 -> (2, +1), A22 -> (3, +1)
        # V: B11 -> (0, +1)
        # W: C21 -> (1, +1) [k=1,i=0 -> 1*2+0=2? No, k*2+i: k=1,i=0 -> 2+0=2]
        # Wait, C21 means i=1, k=0? No, C[i,k]. C21 is i=1,k=0? 
        # Standard: C is 2x2, C[i][k]. C11 is i=0,k=0. C21 is i=1,k=0.
        # W index: k*n+i = k*2+i. C21: k=0,i=1 -> 0*2+1=1.
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1)]),
        
        # Term 3: M3 = A11*(B12-B22)
        # U: A11 -> (0, +1)
        # V: B12 -> (1, +1), B22 -> (3, -1)
        # W: C12 -> (1, +1) [k=1,i=0 -> 1*2+0=2]
        # C12: i=0,k=1 -> W index 1*2+0=2
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1)]),
        
        # Term 4: M4 = A22*(B21-B11)
        # U: A22 -> (3, +1)
        # V: B21 -> (2, +1), B11 -> (0, -1)
        # W: C11 -> (0, -1), C22 -> (3, -1)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (3, -1)]),
        
        # Term 5: M5 = (A11+A12)*B22
        # U: A11 -> (0, +1), A12 -> (1, +1)
        # V: B22 -> (3, +1)
        # W: C11 -> (0, +1), C21 -> (1, +1)
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        
        # Term 6: M6 = (A21-A11)*(B11+B12)
        # U: A21 -> (2, +1), A11 -> (0, -1)
        # V: B11 -> (0, +1), B12 -> (1, +1)
        # W: C22 -> (3, +1)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        
        # Term 7: M7 = (A12-A22)*(B21+B22)
        # U: A12 -> (1, +1), A22 -> (3, -1)
        # V: B21 -> (2, +1), B22 -> (3, +1)
        # W: C11 -> (0, +1), C22 -> (3, +1)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)]),
    ]
    
    def add_strassen(col_offset, i_offset, j_offset, k_offset):
        """Add Strassen 2x2x2 factors for subproblem at given offsets."""
        for term in strassen_terms:
            u_entries, v_entries, w_entries = term
            for (local_ui, uv), (local_vi, vv), (local_wi, wv) in zip(u_entries, v_entries, w_entries):
                # Global indices
                # U: i*2+j, local i in {0,1}, j in {0,1}
                gi = (local_ui // 2) + i_offset
                gj = (local_ui % 2) + j_offset
                global_ui = gi * 2 + gj  # But wait, U is for the subproblem's n x m
                
                # Hmm, I need to be more careful. Let me restructure.
                
                # Actually, let me just build it directly.
                pass
    
    # Let me take a different, more direct approach.
    
    # Reset
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0,1} -> 2x2x6 subproblem
    # Split k into 3 groups of 2: k in {0,1}, {2,3}, {4,5}
    for k_group in range(3):
        k_start = k_group * 2
        # For each 2x2x2 block, use Strassen
        for term in strassen_terms:
            u_entries, v_entries, w_entries = term
            # u_entries are for local A (2x2), v_entries for local B (2x2), w_entries for local C (2x2)
            # Local A indices: (i,j) with i,j in {0,1}
            # Global A index: i * m + j, but j is local to this subproblem
            # For this subproblem, j_global = j_local + 0 (since j in {0,1})
            # i_global = i_local (since i in {0,1})
            # U global index: i_global * m + j_global = i_local * 5 + j_local
            # V global index: j_global * p + k_global = (j_local) * 6 + (k_local + k_start)
            # W global index: k_global * n + i_global = (k_local + k_start) * 2 + i_local
            
            for (local_ui, uv), (local_vi, vv), (local_wi, wv) in zip(u_entries, v_entries, w_entries):
                # local_ui is index into 2x2 A: i_local*2 + j_local
                i_local = local_ui // 2
                j_local = local_ui % 2
                global_ui = i_local * 5 + j_local  # since m=5, j_global = j_local
                
                # local_vi is index into 2x2 B: j_local*2 + k_local
                j_local_v = local_vi // 2
                k_local = local_vi % 2
                global_vi = j_local_v * 6 + (k_local + k_start)  # p=6
                
                # local_wi is index into 2x2 C: k_local*2 + i_local
                k_local_w = local_wi // 2
                i_local_w = local_wi % 2
                global_wi = (k_local_w + k_start) * 2 + i_local_w  # n=2
                
                U[global_ui][col] += uv
                V[global_vi][col] += vv
                W[global_wi][col] += wv
            
            col += 1
    
    # Part 2: j in {2,3,4} -> 2x3x6 subproblem
    # Split into j=2 (2x1x6) and j={3,4} (2x2x6)
    
    # 2x1x6: j=2
    # For each k in 0..5, one term
    for k in range(6):
        # U: A[i,2] for i in {0,1} -> global index i*5+2
        # V: B[2,k] -> global index 2*6+k = 12+k
        # W: C[i,k] -> global index k*2+i
        for i in range(2):
            U[i * 5 + 2][col] += 1
        V[2 * 6 + k][col] += 1
        for i in range(2):
            W[k * 2 + i][col] += 1
        col += 1
    
    # 2x2x6: j in {3,4}, k in 0..5
    # Split k into 3 groups of 2
    for k_group in range(3):
        k_start = k_group * 2
        for term in strassen_terms:
            u_entries, v_entries, w_entries = term
            for (local_ui, uv), (local_vi, vv), (local_wi, wv) in zip(u_entries, v_entries, w_entries):
                i_local = local_ui // 2
                j_local = local_ui % 2
                j_global = j_local + 3  # j in {3,4}
                global_ui = i_local * 5 + j_global
                
                j_local_v = local_vi // 2
                k_local = local_vi % 2
                j_global_v = j_local_v + 3
                global_vi = j_global_v * 6 + (k_local + k_start)
                
                k_local_w = local_wi // 2
                i_local_w = local_wi % 2
                global_wi = (k_local_w + k_start) * 2 + i_local_w
                
                U[global_ui][col] += uv
                V[global_vi][col] += vv
                W[global_wi][col] += wv
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

