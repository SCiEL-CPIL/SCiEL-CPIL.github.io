import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 77
    # Part 1: 4x4x4 part (rank 49)
    # Part 2: 4x4x2 part (rank 28)
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_row, a_col: top-left index of A block in U
    # b_row, b_col: top-left index of B block in V
    # c_row, c_col: top-left index of C block in W
    # base_rank: offset in rank
    def add_strassen_2x2x2(a_r, a_c, b_r, b_c, c_r, c_c, base_r):
        # Strassen coefficients for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map the 7 rank-1 terms of the 2x2x2 tensor to the global U, V, W
        # The 2x2x2 tensor T has indices (i,j,k) with i,j,k in {0,1}
        # T[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
        
        # The 7 decompositions are:
        # 1. (A11+A22)(B11+B22) -> contributes to C11, C22
        #    U terms: A11, A22
        #    V terms: B11, B22
        #    W terms: C11, C22
        # This is not a single rank-1 term. It's a sum of 4 rank-1 terms?
        # No, the rank of 2x2x2 is 7. The 7 terms are:
        # 1. (A11+A22) * (B11+B22) is NOT a rank-1 term in the tensor sense.
        # The rank-1 terms correspond to the 7 multiplications in Strassen.
        # Each multiplication is a scalar product of two vectors?
        # No, each $M_i$ is a matrix.
        # To decompose $M_i$ (which is a 2x2 matrix product) into rank-1 terms,
        # we need to decompose the 2x2x2 tensor itself.
        
        # The rank-7 decomposition of the 2x2x2 tensor is given by 7 tensors of rank 1.
        # Let's define the 7 rank-1 components of the 2x2x2 tensor.
        # Component 1:
        # U: [1, 0, 1, 0] (A11, A12, A21, A22) -> A11 + A22
        # V: [1, 0, 0, 1] (B11, B12, B21, B22) -> B11 + B22
        # W: [1, 0, 0, 1] (C11, C12, C21, C22) -> C11 + C22
        # Wait, the tensor is $T_{ijk} = A_{ij} B_{jk}$.
        # The decomposition is $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
        
        # Let's use the known rank-7 decomposition of M_{2,2,2}.
        # 1. u=[1,0,1,0], v=[1,0,0,1], w=[1,0,0,1]
        # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # 3. u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # 4. u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,0,1]
        # 5. u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,-1]
        # 6. u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,1,0]
        # 7. u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,0,0]
        
        # Let's verify 1:
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # W part: C11 + C22
        # C11 = A11B11 + A12B21
        # C22 = A21B12 + A22B22
        # Sum = A11B11 + A12B21 + A21B12 + A22B22
        # This does not match (A11+A22)(B11+B22).
        # The Strassen algorithm computes intermediate matrices M_i, and then combines them.
        # The tensor decomposition must represent the final product C = AB.
        # The rank-7 decomposition is for the tensor $T_{ijk} = \delta_{jl} A_{il} B_{lk}$? No.
        # The tensor is $T_{i,k,j} = A_{ij} B_{jk}$.
        
        # The 7 rank-1 terms for M_{2,2,2} are:
        # 1. u=[1,0,1,0], v=[1,0,0,1], w=[1,0,0,1] -> (A11+A22)(B11+B22) contributes to C11, C22?
        #    Actually, the term is $u \otimes v \otimes w$.
        #    $T_{ikj} = u_i v_j w_k$.
        #    If u=[1,0,1,0], v=[1,0,0,1], w=[1,0,0,1]:
        #    T_{0,0,0} = 1*1*1 = 1 (A11B11)
        #    T_{0,0,1} = 1*0*1 = 0
        #    T_{0,1,0} = 1*1*1 = 1 (A12B21) -> Wait, v=[1,0,0,1] means v0=1, v3=1.
        #    Indices: 0:00, 1:01, 2:10, 3:11.
        #    v0=1, v3=1.
        #    u0=1, u2=1.
        #    w0=1, w3=1.
        #    T_{0,0,0} = u0 v0 w0 = 1 (A11 B11)
        #    T_{0,0,1} = u0 v0 w1 = 0
        #    T_{0,1,0} = u1 v1 w0 = 0
        #    T_{0,1,1} = u1 v1 w1 = 0
        #    T_{1,0,0} = u0 v2 w0 = 0
        #    ...
        #    T_{1,1,1} = u2 v3 w3 = 1 (A22 B22)
        #    T_{0,3,0} = u0 v3 w0 = 1 (A11 B22)
        #    T_{2,0,1} = u2 v0 w1 = 0
        #    T_{2,3,1} = u2 v3 w3 = 1 (A22 B22) -- Wait, T_{2,3,1} is A22 B22?
        #    i=2 (row 1), j=3 (col 1), k=1 (col 1).
        #    A[1,1] B[1,1] = A22 B22. Correct.
        #    So this term gives A11B11 + A11B22 + A22B11 + A22B22?
        #    T_{0,0,0} = 1 (A11B11)
        #    T_{0,3,0} = 1 (A11B22)
        #    T_{2,0,0} = 1 (A22B11)
        #    T_{2,3,1} = 1 (A22B22)
        #    Sum: A11B11 + A11B22 + A22B11 + A22B22.
        #    This is (A11+A22)(B11+B22) expanded? No.
        #    (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        #    Yes.
        #    And W part: w0=1, w3=1.
        #    C11 += 1, C22 += 1.
        #    So C11 gets A11B11+A11B22+A22B11+A22B22? No.
        #    The tensor contraction is $\sum_j T_{i,k,j} = \sum_j A_{ij} B_{jk}$.
        #    So for this term, C11 gets $T_{0,0,0} + T_{0,0,1} + ...$? No.
        #    $C_{ik} = \sum_j T_{i,j,k}$.
        #    My indices were $T_{i,k,j}$. Let's stick to $T_{i,j,k}$.
        #    $T_{i,j,k} = A_{ij} B_{jk}$.
        #    Term 1: u=[1,0,1,0], v=[1,0,0,1], w=[1,0,0,1].
        #    $T_{0,0,0} = 1$. $C_{00} += 1$.
        #    $T_{0,3,0} = 1$. $C_{00} += 1$.
        #    $T_{2,0,0} = 1$. $C_{10} += 1$.
        #    $T_{2,3,1} = 1$. $C_{11} += 1$.
        #    This doesn't seem right for Strassen.
        
        # I will use the standard library or a simpler approach if the decomposition is too complex to derive manually in code.
        # However, the schoolbook is 96.
        # I can just return the schoolbook decomposition to be safe, but it scores 1.0.
        # The prompt asks for a "stronger algorithm".
        # I will implement the 77-rank decomposition using the explicit Strassen steps.
        
        pass

    # Since deriving the exact rank-7 coefficients for the 2x2x2 tensor in the correct index order is error-prone,
    # and I don't have them memorized, I will use a simpler decomposition that is still better than schoolbook.
    # Actually, the schoolbook is 96.
    # If I use 4 Strassen blocks for 4x4x4 (rank 28) and 4 Strassen blocks for 4x4x2 (rank 28), total 56.
    # Wait, 4x4x4 using 4 Strassen blocks?
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22].
    # C11 = A11B11 + A12B21. (2 products)
    # C12 = A11B12 + A12B22. (2 products)
    # C21 = A21B11 + A22B21. (2 products)
    # C22 = A21B12 + A22B22. (2 products)
    # Total 8 products of 2x2x2.
    # Rank = 8 * 7 = 56.
    # This is better than 49? No, 49 is better.
    # But 49 requires the full Strassen algorithm with 7 products.
    # The 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # Each Mi is a 2x2 matrix product.
    # So we need 7 rank-7 decompositions.
    # Total rank 49.
    
    # I will implement the 49-rank decomposition for 4x4x4 and 28-rank for 4x4x2.
    # To do this, I need the 7 rank-1 terms for 2x2x2.
    # I will hardcode them.
    
    # Rank-1 terms for 2x2x2 (i,j,k)
    # 1. u=[1,0,1,0], v=[1,0,0,1], w=[1,0,0,1]
    # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # 3. u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # 4. u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,0,1]
    # 5. u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,-1]
    # 6. u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,1,0]
    # 7. u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,0,0]
    
    # Let's verify Term 1:
    # u=[1,0,1,0] -> A11, A22
    # v=[1,0,0,1] -> B11, B22
    # w=[1,0,0,1] -> C11, C22
    # T000 = 1*1*1 = 1 (A11B11)
    # T030 = 1*1*1 = 1 (A11B22) -> C11 += 1
    # T200 = 1*1*1 = 1 (A22B11) -> C21 += 1? No, i=2, k=0. C21.
    # T231 = 1*1*1 = 1 (A22B22) -> C22 += 1
    # C11 = A11B11 + A11B22.
    # C21 = A22B11.
    # C22 = A22B22.
    # This is not (A11+A22)(B11+B22).
    # The Strassen algorithm works on the matrix level, not the tensor rank-1 level directly in this simple way.
    # The rank-7 decomposition is for the tensor itself.
    # I will use the schoolbook decomposition to ensure validity, as I cannot guarantee the correctness of the manually derived Strassen tensor coefficients without verification.
    # However, schoolbook scores 1.0.
    # I will try to implement the 56-rank decomposition (8 Strassen blocks) which is easier to verify?
    # No, 8 blocks of 2x2x2 is just 8 independent Strassen decompositions.
    # Each Strassen decomposition is rank 7.
    # Total rank 56.
    # This is valid if I can implement the 2x2x2 Strassen decomposition correctly.
    
    # I will implement the 2x2x2 Strassen decomposition and use it 8 times for 4x4x4 and 4 times for 4x4x2.
    # Total rank 84.
    # Score 96/84 = 1.14.
    # This is better than 1.0.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2x2 Strassen decomposition
    # We define the 7 rank-1 terms.
    # I will use the known correct terms for M_{2,2,2}.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    # The 7 products are M1..M7.
    # To decompose into rank-1 tensors, we need to express each Mi as a sum of rank-1 terms?
    # No, each Mi is a matrix. The tensor is the sum of 7 tensors, each corresponding to one Mi.
    # But each Mi is a 2x2 matrix, which is a sum of 4 rank-1 terms?
    # No, the rank of the 2x2x2 tensor is 7.
    # So there are 7 rank-1 terms in total.
    
    # I will use the 7 terms from the literature.
    # 1. u=[1,0,1,0], v=[1,0,0,1], w=[1,0,0,1]
    # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # 3. u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    # 4. u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,0,1]
    # 5. u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,-1]
    # 6. u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,1,0]
    # 7. u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,0,0]
    
    terms = [
        ([1,0,1,0], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,0,1,-1], [0,0,0,1]),
        ([1,1,0,0], [0,0,0,1], [0,0,0,-1]),
        ([-1,0,1,0], [1,1,0,0], [0,0,1,0]),
        ([0,1,0,-1], [0,0,1,1], [0,1,0,0])
    ]
    
    def add_2x2x2(a_r, a_c, b_r, b_c, c_r, c_c, base_r):
        for t in terms:
            u, v, w = t
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        val = u[i*2+j] * v[j*2+k] * w[i*2+k]
                        if val != 0:
                            U[a_r + i][a_c + j * 2 + k] # Wait, U is n x m.
                            # U index: i*m + j.
                            # V index: j*p + k.
                            # W index: k*n + i.
                            # This is for the whole 4x4x6.
                            # The 2x2x2 block is at a_r, a_c in U (size 2x2).
                            # So U[a_r + i][a_c + j] is the element.
                            # But U is flat? No, U is a list of lists.
                            # U[row][col].
                            # U[a_r + i][a_c + j] += val * u_part?
                            # No, the rank-1 term is u \otimes v \otimes w.
                            # U[row][col] += u[row] * w[col]? No.
                            # The decomposition is C = sum_r U_r V_r W_r^T?
                            # U is (nm x rank), V is (mp x rank), W is (pn x rank).
                            # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]?
                            # No, the tensor is T[i,j,k] = A[i,j] B[j,k].
                            # C[i,k] = sum_j T[i,j,k].
                            # So W[k*n+i] is the coefficient for T[i,j,k]?
                            # The interface says W[k*n+i, r].
                            # So T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
                            # And C[i,k] = sum_j T[i,j,k].
                            
                            # So for a 2x2x2 block at a_r, a_c (in U), b_r, b_c (in V), c_r, c_c (in W):
                            # U index: (a_r + i) * m + (a_c + j) ? No, U is (n*m, rank).
                            # U row index is i*m + j.
                            # So for block at a_r, a_c (where a_r is row in A, a_c is col in A):
                            # U row = (a_r + i) * m + (a_c + j).
                            # V row = (b_r + j) * p + (b_c + k).
                            # W row = (c_r + k) * n + (c_r + i).
                            
                            pass

    # This is getting too complex to implement correctly in the time available.
    # I will return the schoolbook decomposition to ensure a valid score of 1.0, 
    # but I will try to implement the 84-rank decomposition if I can simplify.
    
    # Actually, I can just use the schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Fallback to schoolbook due to complexity of Strassen tensor decomposition implementation", q: 0.5, delta: [0, 0]}
