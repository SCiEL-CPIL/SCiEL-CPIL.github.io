import numpy as np
import time

def construct():
    # Dimensions: A (2x3), B (3x9), C (2x9)
    # U shape: (2*3, rank) = (6, rank)
    # V shape: (3*9, rank) = (27, rank)
    # W shape: (9*2, rank) = (18, rank)
    
    # Target Rank: 50
    rank = 50
    
    # Initialize U, V, W as 2D lists of zeros
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(27)]
    W = [[0.0] * rank for _ in range(18)]
    
    # Helper to add a term to the decomposition
    # u_idx: index in U (0-5)
    # v_idx: index in V (0-26)
    # w_idx: index in W (0-17)
    # col: column index in the factor matrices
    # val: coefficient (1, -1, etc)
    def add_term(u_idx, v_idx, w_idx, col, val=1.0):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Part 1: 2x2x9 multiplication
    # A_sub = A[0:2, 0:2] -> indices 0, 1, 3, 4 in U
    # B_sub = B[0:2, :] -> indices 0-8, 9-17 in V
    # C_sub = C[:, :] -> indices 0-17 in W
    
    # We process 4 blocks of 2 columns of B_sub
    col = 0
    for block in range(4):
        # Column range for B: [2*block, 2*block+2)
        k_start = 2 * block
        
        # B indices for this block
        # B[0, k_start], B[0, k_start+1], B[1, k_start], B[1, k_start+1]
        v_b00 = 0 * 9 + k_start
        v_b01 = 0 * 9 + (k_start + 1)
        v_b10 = 1 * 9 + k_start
        v_b11 = 1 * 9 + (k_start + 1)
        
        # A indices (constant for all blocks)
        u_a00 = 0 # A[0,0]
        u_a01 = 1 # A[0,1]
        u_a10 = 3 # A[1,0]
        u_a11 = 4 # A[1,1]
        
        # W indices for this block (2x2 output block)
        # C[0, k_start], C[0, k_start+1], C[1, k_start], C[1, k_start+1]
        w_c00 = k_start * 2 + 0
        w_c01 = k_start * 2 + 1
        w_c10 = (k_start + 1) * 2 + 0 # Wait, W index is k*n + i. n=2.
        # W[k*n+i].
        # For k=k_start, i=0 -> k_start*2
        # For k=k_start, i=1 -> k_start*2 + 1
        # For k=k_start+1, i=0 -> (k_start+1)*2
        # For k=k_start+1, i=1 -> (k_start+1)*2 + 1
        w_c00 = k_start * 2
        w_c01 = k_start * 2 + 1
        w_c10 = (k_start + 1) * 2
        w_c11 = (k_start + 1) * 2 + 1
        
        # Strassen 7 terms
        # M1 = (a00 + a11) * (b00 + b11)
        add_term(u_a00, v_b00, w_c00, col, 1)
        add_term(u_a00, v_b11, w_c00, col, 1)
        add_term(u_a11, v_b00, w_c00, col, 1)
        add_term(u_a11, v_b11, w_c00, col, 1)
        # This naive addition creates 4 terms for M1. Strassen combines them.
        # Correct Strassen:
        # Term 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C11=1.
        # But W needs to distribute to C00 and C11.
        # In the tensor decomposition, a single rank-1 term is u * v * w.
        # We need to express the product sum.
        
        # Let's use the standard Strassen coefficients directly.
        # 1. M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        # We need 3 terms to realize this single multiplication if we can't factor W?
        # No, W is a vector.
        # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> contributes to C00 and C11?
        # If w has non-zeros at C00 and C11, it contributes to both.
        # Yes.
        
        # Let's define the 7 terms for a 2x2x2 block.
        # Indices in local block:
        # A: 0,1,2,3 (A00, A01, A10, A11)
        # B: 0,1,2,3 (B00, B01, B10, B11)
        # C: 0,1,2,3 (C00, C01, C10, C11)
        
        # T1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        # u: [1, 0, 0, 1]
        # v: [1, 0, 0, 1]
        # w: [1, 0, 0, 1]
        
        # T2: (A10+A11)B00 -> C10 +=, C11 -=
        # u: [0, 0, 1, 1]
        # v: [1, 0, 0, 0]
        # w: [0, 0, 1, -1]
        
        # T3: A00(B01-B11) -> C01 +=, C11 +=
        # u: [1, 0, 0, 0]
        # v: [0, 1, 0, -1]
        # w: [0, 1, 0, 1]
        
        # T4: A11(B10-B00) -> C00 +=, C10 +=
        # u: [0, 0, 0, 1]
        # v: [0, 0, 1, -1] # Wait, B10-B00 is [ -1, 0, 1, 0]
        # v: [-1, 0, 1, 0]
        # w: [1, 0, 1, 0]
        
        # T5: (A00+A01)B11 -> C00 -=, C01 +=
        # u: [1, 1, 0, 0]
        # v: [0, 0, 0, 1]
        # w: [-1, 1, 0, 0]
        
        # T6: (A10-A00)(B00+B01) -> C11 +=
        # u: [-1, 0, 1, 0]
        # v: [1, 1, 0, 0]
        # w: [0, 0, 0, 1]
        
        # T7: (A01-A11)(B10+B11) -> C00 +=
        # u: [0, 1, 0, -1]
        # v: [0, 0, 1, 1]
        # w: [1, 0, 0, 0]
        
        # Mapping local to global indices:
        # A: 0->0, 1->1, 2->3, 3->4
        # B: 0->v_b00, 1->v_b01, 2->v_b10, 3->v_b11
        # C: 0->w_c00, 1->w_c01, 2->w_c10, 3->w_c11
        
        # Apply T1
        U[u_a00][col] += 1; U[u_a11][col] += 1
        V[v_b00][col] += 1; V[v_b11][col] += 1
        W[w_c00][col] += 1; W[w_c11][col] += 1
        
        # Apply T2
        U[u_a10][col+1] += 1; U[u_a11][col+1] += 1
        V[v_b00][col+1] += 1
        W[w_c10][col+1] += 1; W[w_c11][col+1] -= 1
        
        # Apply T3
        U[u_a00][col+2] += 1
        V[v_b01][col+2] += 1; V[v_b11][col+2] -= 1
        W[w_c01][col+2] += 1; W[w_c11][col+2] += 1
        
        # Apply T4
        U[u_a11][col+3] += 1
        V[v_b10][col+3] += 1; V[v_b00][col+3] -= 1
        W[w_c00][col+3] += 1; W[w_c10][col+3] += 1
        
        # Apply T5
        U[u_a00][col+4] += 1; U[u_a01][col+4] += 1
        V[v_b11][col+4] += 1
        W[w_c00][col+4] -= 1; W[w_c01][col+4] += 1
        
        # Apply T6
        U[u_a10][col+5] += 1; U[u_a00][col+5] -= 1
        V[v_b00][col+5] += 1; V[v_b01][col+5] += 1
        W[w_c11][col+5] += 1
        
        # Apply T7
        U[u_a01][col+6] += 1; U[u_a11][col+6] -= 1
        V[v_b10][col+6] += 1; V[v_b11][col+6] += 1
        W[w_c00][col+6] += 1
        
        col += 7
        
    # Last block: 2x2x1 (B column 8)
    k_last = 8
    v_b08 = 0 * 9 + 8
    v_b18 = 1 * 9 + 8
    
    w_c08 = 8 * 2
    w_c18 = 8 * 2 + 1
    
    # Direct 4 terms
    # C08 = A00 B08 + A01 B18
    U[u_a00][col] += 1
    V[v_b08][col] += 1
    W[w_c08][col] += 1
    
    U[u_a01][col+1] += 1
    V[v_b18][col+1] += 1
    W[w_c08][col+1] += 1
    
    # C18 = A10 B08 + A11 B18
    U[u_a10][col+2] += 1
    V[v_b08][col+2] += 1
    W[w_c18][col+2] += 1
    
    U[u_a11][col+3] += 1
    V[v_b18][col+3] += 1
    W[w_c18][col+3] += 1
    
    col += 4
    
    # Part 2: 2x1x9 multiplication
    # A_sub2 = A[0:2, 2] -> indices 2, 5 in U
    # B_sub2 = B[2, :] -> indices 18-26 in V
    # C_sub2 = C[:, :] -> indices 0-17 in W (same C)
    
    u_a02 = 2 # A[0,2]
    u_a12 = 5 # A[1,2]
    
    for k in range(9):
        v_b2k = 2 * 9 + k
        
        # C[0, k] += A[0,2] B[2, k]
        U[u_a02][col] += 1
        V[v_b2k][col] += 1
        W[k * 2][col] += 1
        
        # C[1, k] += A[1,2] B[2, k]
        U[u_a12][col+1] += 1
        V[v_b2k][col+1] += 1
        W[k * 2 + 1][col+1] += 1
        
        col += 2
        
    # Verification
    def verify_decomp():
        # Sample check
        import random
        for _ in range(10):
            A = random.randint(0, 10, (2, 3))
            B = random.randint(0, 10, (3, 9))
            C = np.dot(A, B)
            
            # Compute C' using U, V, W
            # C'[i, k] = sum_r U[i*3+j, r] * V[j*9+l, r] * W[l*2+i, r] ?
            # The definition is:
            # sum_r U[row_U, r] * V[row_V, r] * W[row_W, r]
            # where row_U = i*m + j, row_V = j*p + k, row_W = k*n + i
            # This sum must equal C[i, k]
            
            C_check = np.zeros((2, 9))
            for i in range(2):
                for k in range(9):
                    s = 0.0
                    for r in range(rank):
                        # Need to sum over j?
                        # The tensor decomposition identity is:
                        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                        # Here T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
                        # So for each (i,k), sum over j? No, the tensor is indexed by (i,j,k).
                        # The multiplication tensor T has T_{i,j,k} = 1 if j matches, etc?
                        # Actually, the tensor for matrix multiplication M_{n,m,p} is defined such that
                        # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}
                        # where T_{i,j,k} = 1 if the indices match, 0 otherwise?
                        # No, T is the structure tensor.
                        # The decomposition says:
                        # T = sum_r u_r \otimes v_r \otimes w_r
                        # where u_r is a vector of size n*m, v_r of size m*p, w_r of size p*n.
                        # The element (i,j) of u_r is U[i*m+j, r].
                        # The element (j,k) of v_r is V[j*p+k, r].
                        # The element (k,i) of w_r is W[k*n+i, r].
                        # Then T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
                        # For matrix multiplication, T_{i,j,k} is 1 if i,j,k are valid, but specifically,
                        # the tensor is 1 only when the indices form a valid multiplication chain?
                        # Actually, T_{i,j,k} = 1 if j is the shared index?
                        # Yes, T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
                        # This is always true for the specific (i,j,k) tuple.
                        # So for a fixed (i,j,k), the sum should be 1.
                        # For other combinations of indices in the vectors?
                        # The vectors u_r, v_r, w_r are defined on the index sets.
                        # So we must check: for all i in [0,2), j in [0,3), k in [0,9):
                        # sum_r U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r] == 1
                        
                        pass # This loop is expensive.
                    
                    # Efficient check:
                    # Construct matrices U_mat (6xR), V_mat (27xR), W_mat (18xR)
                    # For each r, u_r is a 6x1, v_r is 27x1, w_r is 18x1?
                    # No, u_r is length 6.
                    # The tensor slice for rank 1 r is u_r (vector) * v_r (vector) * w_r (vector).
                    # We can't easily sum this for all (i,j,k) without looping.
                    # Given 10 samples, and 2*3*9 = 54 tuples, and rank 50.
                    # 10 * 54 * 50 = 27000 ops. Very fast.
                    
                    pass
            
            # Let's just do the full check for one random instance to be safe
            A = np.random.randint(0, 10, (2, 3))
            B = np.random.randint(0, 10, (3, 9))
            C_true = A @ B
            
            C_comp = np.zeros((2, 9))
            for i in range(2):
                for j in range(3):
                    for k in range(9):
                        # Coefficient of A[i,j] B[j,k] in C[i,k] is 1
                        # So sum_r U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r] must be 1
                        # But we are computing C[i,k] = sum_j A[i,j] B[j,k] sum_r ...
                        # So C[i,k] = sum_j A[i,j] B[j,k] * 1
                        # This confirms the condition.
                        pass
            
            # Direct computation of C_comp using the factors
            # C[i,k] = sum_r (sum_j U[i*3+j, r] A[i,j] ) * (sum_j V[j*9+k, r] B[j,k] ) * W[k*2+i, r] ?
            # No, that's not how it works.
            # The formula is:
            # C[i,k] = sum_{i',j,k'} T_{i',j,k'} A[i',j] B[j,k'] ... no.
            # Standard definition:
            # C = A B
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Decomposition:
            # A[i,j] = sum_r U[i*m+j, r] * (something) ? No.
            # The factors U, V, W are such that:
            # U[:, r] is a vector.
            # The rank-1 term is u_r \otimes v_r \otimes w_r.
            # The value of the tensor at (i,j,k) is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
            # This value must be 1 for all (i,j,k).
            
            # So we just check the identity:
            # for i in range(2):
            #   for j in range(3):
            #     for k in range(9):
            #       s = 0
            #       for r in range(rank):
            #         s += U[i*3+j][r] * V[j*9+k][r] * W[k*2+i][r]
            #       if s != 1: return False
            # return True
            
            is_valid = True
            for i in range(2):
                for j in range(3):
                    for k in range(9):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*3+j][r] * V[j*9+k][r] * W[k*2+i][r]
                        if abs(s - 1.0) > 1e-6:
                            is_valid = False
                            break
                    if not is_valid: break
                if not is_valid: break
            
            if not is_valid:
                raise ValueError("Decomposition invalid")
                
    verify_decomp()
    
    return {"U": U, "V": V, "W": W}
