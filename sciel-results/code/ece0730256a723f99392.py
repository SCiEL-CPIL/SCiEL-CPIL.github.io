"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 22.
    # Split m=3 into m1=2 and m2=1.
    # Split p=4 into p1=2 and p2=2.
    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] using Strassen (rank 7)
    # Part 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] using Strassen (rank 7)
    # Part 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4] using schoolbook (rank 8)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2 * 2x2
    # Inputs: A (2x2), B (2x2)
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def add_strassen_term(offset_u, offset_v, offset_w, coeff_u, coeff_v, coeff_w, u_idx, v_idx, w_idx):
        nonlocal col
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w
        col += 1

    # Part 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: (0,0), (0,1), (1,0), (1,1) -> U indices: 0, 1, 3, 4
    # B indices: (0,0), (0,1), (1,0), (1,1) -> V indices: 0, 1, 4, 5
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W indices: 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    add_strassen_term(0, 0, 0, 1, 1, 1, 0, 0, 0) # A00, B00 -> C00? No, this is just adding terms to the sum.
    # Wait, the Strassen algorithm computes intermediate products M1..M7, then combines them to get C.
    # The tensor decomposition corresponds to the bilinear complexity.
    # Each multiplication Mx corresponds to a rank-1 term in the tensor?
    # No, the rank of the tensor is the number of multiplications.
    # So we need to map the Strassen multiplications to the U, V, W factors.
    
    # Let's restart the construction logic properly.
    # A rank-r decomposition is a set of r triples (u_r, v_r, w_r) such that
    # T = sum_r u_r \otimes v_r \otimes w_r.
    # For matrix multiplication, T_{i,j,k} = delta_{i,j,k} is not correct.
    # T_{(i,j), (j,k), (i,k)} = 1.
    # So for each term r, we have vectors U_r, V_r, W_r.
    # The condition is: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 if (i,j,k) is a valid index triple?
    # No, the tensor T is defined by T_{i,j,k} = 1 if the indices match?
    # The matrix multiplication tensor has entries T_{i, j, k} where i in 1..n, j in 1..m, k in 1..p.
    # T_{i,j,k} = 1.
    # And the decomposition is sum_r U_r[i,j] V_r[j,k] W_r[k,i] = T_{i,j,k}.
    # Here U_r is a matrix of size n x m? No, U is a vector of size n*m.
    # So U_r[i*m+j] is the (i,j) entry.
    
    # So for each term r, we define:
    # U_r: n x m matrix
    # V_r: m x p matrix
    # W_r: p x n matrix
    # Such that sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k?
    # No, the tensor entry is 1 only for the specific (i,j,k) combination?
    # The tensor is in R^{n*m} x R^{m*p} x R^{p*n}.
    # The entry at ( (i,j), (j,k), (k,i) ) is 1.
    # All other entries are 0.
    # So we need sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 if the indices are consistent?
    # Actually, the indices i,j,k are not independent in the tensor definition?
    # The tensor T is defined by T_{i,j,k} = 1 for all i,j,k.
    # Wait, the matrix multiplication map is (A, B) -> C where C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k \otimes ...?
    # The tensor is in V^* \otimes W^* \otimes Z^*.
    # T(v, w, z) = z(C) where C = v w?
    # No, T(A, B, C) = trace(C^T A B)?
    # The standard tensor T_{m,n,p} is in R^{mn} \otimes R^{np} \otimes R^{pm}.
    # T_{(i,j), (j,k), (k,i)} = 1.
    # So for a fixed r, we have vectors u_r (len mn), v_r (len np), w_r (len pm).
    # The condition is: sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all i,j,k?
    # No, it must equal 1 for the specific (i,j,k) and 0 otherwise?
    # No, the sum must reproduce the tensor.
    # The tensor has 1s at positions ( (i,j), (j,k), (k,i) ).
    # So sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all i,j,k.
    
    # This is a system of equations.
    
    # Let's use the known Strassen decomposition for 2x2x2.
    # It provides 7 terms.
    # For 2x2x2, n=2, m=2, p=2.
    # U indices 0..3, V 0..3, W 0..3.
    # Terms:
    # 1. U: [[1,0],[0,1]], V: [[1,1],[1,1]], W: [[1,-1],[1,1]]? 
    # Let's use the explicit coefficients.
    
    # I will hardcode the rank 22 solution.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(len(u_vec)):
            U[i][col] += u_vec[i]
        for i in range(len(v_vec)):
            V[i][col] += v_vec[i]
        for i in range(len(w_vec)):
            W[i][col] += w_vec[i]
        col += 1

    # Strassen 2x2x2 terms
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # We need to express these as sum of rank-1 terms.
    # Each Mx is a product of a linear form in A and a linear form in B.
    # So each Mx corresponds to a term where U is the linear form in A, V is the linear form in B, and W is the coefficient in C.
    
    # For M1:
    # U1: a00+a11 -> [1,0,0,1]
    # V1: b00+b11 -> [1,0,0,1] (indices 0,1,2,3 for b00,b01,b10,b11)
    # W1: c00 coeff 1, c11 coeff 1 -> [1,0,0,1] (indices 0,1,2,3 for c00,c01,c10,c11)
    # Wait, W is p x n = 2x2. Indices k*n+i.
    # c00: k=0,i=0 -> 0
    # c01: k=1,i=0 -> 1
    # c10: k=0,i=1 -> 2
    # c11: k=1,i=1 -> 3
    # So W1: [1, 0, 0, 1]
    
    add_term([1,0,0,1], [1,0,0,1], [1,0,0,1])
    
    # M2:
    # U2: a10+a11 -> [0,0,1,1]
    # V2: b00 -> [1,0,0,0]
    # W2: c10 coeff 1 -> [0,0,1,0]
    add_term([0,0,1,1], [1,0,0,0], [0,0,1,0])
    
    # M3:
    # U3: a00 -> [1,0,0,0]
    # V3: b10-b11 -> [0,0,1,-1]
    # W3: c01 coeff 1, c11 coeff 1 -> [0,1,0,1]
    add_term([1,0,0,0], [0,0,1,-1], [0,1,0,1])
    
    # M4:
    # U4: a11 -> [0,0,0,1]
    # V4: b01-b00 -> [-1,1,0,0]
    # W4: c00 coeff 1, c10 coeff 1 -> [1,0,1,0]
    add_term([0,0,0,1], [-1,1,0,0], [1,0,1,0])
    
    # M5:
    # U5: a00+a01 -> [1,1,0,0]
    # V5: b11 -> [0,0,0,1]
    # W5: c00 coeff -1, c01 coeff 1 -> [-1,1,0,0]
    add_term([1,1,0,0], [0,0,0,1], [-1,1,0,0])
    
    # M6:
    # U6: a10-a00 -> [-1,0,1,0]
    # V6: b00+b01 -> [1,1,0,0]
    # W6: c11 coeff 1 -> [0,0,0,1]
    add_term([-1,0,1,0], [1,1,0,0], [0,0,0,1])
    
    # M7:
    # U7: a01-a11 -> [0,1,0,-1]
    # V7: b10+b11 -> [0,0,1,1]
    # W7: c00 coeff 1 -> [1,0,0,0]
    add_term([0,1,0,-1], [0,0,1,1], [1,0,0,0])
    
    # This handles A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # We need to map these to the full 2x3x4 indices.
    # The above used local indices. We need to shift them.
    
    # Let's redo with a function that takes offsets.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term_full(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        nonlocal col
        for i, c in zip(u_idx, coeff_u):
            U[i][col] += c
        for i, c in zip(v_idx, coeff_v):
            V[i][col] += c
        for i, c in zip(w_idx, coeff_w):
            W[i][col] += c
        col += 1

    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # U indices: i*3+j for i in 0,1 and j in 0,1 -> 0,1,3,4
    # V indices: j*4+k for j in 0,1 and k in 0,1 -> 0,1,4,5
    # W indices: k*2+i for k in 0,1 and i in 0,1 -> 0,1,2,3
    
    u_idx = [0, 1, 3, 4]
    v_idx = [0, 1, 4, 5]
    w_idx = [0, 1, 2, 3]
    
    # M1
    add_term_full(u_idx, v_idx, w_idx, [1,0,0,1], [1,0,0,1], [1,0,0,1])
    # M2
    add_term_full(u_idx, v_idx, w_idx, [0,0,1,1], [1,0,0,0], [0,0,1,0])
    # M3
    add_term_full(u_idx, v_idx, w_idx, [1,0,0,0], [0,0,1,-1], [0,1,0,1])
    # M4
    add_term_full(u_idx, v_idx, w_idx, [0,0,0,1], [-1,1,0,0], [1,0,1,0])
    # M5
    add_term_full(u_idx, v_idx, w_idx, [1,1,0,0], [0,0,0,1], [-1,1,0,0])
    # M6
    add_term_full(u_idx, v_idx, w_idx, [-1,0,1,0], [1,1,0,0], [0,0,0,1])
    # M7
    add_term_full(u_idx, v_idx, w_idx, [0,1,0,-1], [0,0,1,1], [1,0,0,0])
    
    # Part 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # U indices: same as above -> 0,1,3,4
    # V indices: j*4+k for j in 0,1 and k in 2,3 -> 2,3,6,7
    # W indices: k*2+i for k in 2,3 and i in 0,1 -> 4,5,6,7
    
    u_idx = [0, 1, 3, 4]
    v_idx = [2, 3, 6, 7]
    w_idx = [4, 5, 6, 7]
    
    # M1
    add_term_full(u_idx, v_idx, w_idx, [1,0,0,1], [1,0,0,1], [1,0,0,1])
    # M2
    add_term_full(u_idx, v_idx, w_idx, [0,0,1,1], [1,0,0,0], [0,0,1,0])
    # M3
    add_term_full(u_idx, v_idx, w_idx, [1,0,0,0], [0,0,1,-1], [0,1,0,1])
    # M4
    add_term_full(u_idx, v_idx, w_idx, [0,0,0,1], [-1,1,0,0], [1,0,1,0])
    # M5
    add_term_full(u_idx, v_idx, w_idx, [1,1,0,0], [0,0,0,1], [-1,1,0,0])
    # M6
    add_term_full(u_idx, v_idx, w_idx, [-1,0,1,0], [1,1,0,0], [0,0,0,1])
    # M7
    add_term_full(u_idx, v_idx, w_idx, [0,1,0,-1], [0,0,1,1], [1,0,0,0])
    
    # Part 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # A indices: i*3+2 for i in 0,1 -> 2, 5
    # B indices: 2*4+k for k in 0..3 -> 8,9,10,11
    # C indices: k*2+i for k in 0..3 and i in 0,1 -> 0,1,2,3,4,5,6,7
    
    # Schoolbook: 8 terms
    for i in range(2):
        for k in range(4):
            u = [2, 5]
            v = [8, 9, 10, 11]
            w = [0, 1, 2, 3, 4, 5, 6, 7]
            # U: 1 at i*3+2
            # V: 1 at 2*4+k
            # W: 1 at k*2+i
            u_coeff = [0, 0]
            u_coeff[i] = 1
            v_coeff = [0, 0, 0, 0]
            v_coeff[k] = 1
            w_coeff = [0, 0, 0, 0, 0, 0, 0, 0]
            w_coeff[k*2+i] = 1
            add_term_full(u, v, w, u_coeff, v_coeff, w_coeff)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
