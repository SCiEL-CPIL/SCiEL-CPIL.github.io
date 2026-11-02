"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-62 decomposition for 3x3 * 3x7 matrix multiplication.
    Uses a 26-rank block-Strassen decomposition for the first 3 columns (3x3x3)
    and a schoolbook 36-rank decomposition for the remaining 4 columns (3x3x4).
    Falls back to schoolbook rank 63 if validation fails.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def validate(U, V, W):
        """Check if the decomposition is exact using NumPy."""
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Reshape to tensors
            # U: (n*m, rank) -> (n, m, rank)
            # V: (m*p, rank) -> (m, p, rank)
            # W: (p*n, rank) -> (p, n, rank)
            
            # We need to check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == delta_{i,i'} delta_{j,j'} delta_{k,k'}?
            # No, the tensor T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k} is not right.
            # The multiplication tensor T has entries T[i, j, k] = 1 if the term A_ij B_jk contributes to C_ki.
            # Actually, the tensor is T[i, j, k] where i is row of A, j is col of A/row of B, k is col of B.
            # The output is C[k, i].
            # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
            
            n, m, p = DIMENSIONS
            rank = len(U[0])
            
            # Efficient check:
            # Construct the tensor via einsum or loops.
            # Given small size, loops are fine, but vectorized is faster.
            
            # For each i, j, k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Let's use a simple loop for clarity and safety, as rank is small (62)
            # and n*m*p is 63. 63 * 62 operations is trivial.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for r in range(rank):
                            s += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                        if abs(s - 1.0) > 1e-10:
                            return False
            return True
        except Exception:
            return False

    def construct_schoolbook():
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

    def construct_rank_62():
        # Total rank 62
        rank_3x3x3 = 26
        rank_3x3x4 = 36
        total_rank = rank_3x3x3 + rank_3x3x4
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # --- Part 1: 3x3x3 (k=0,1,2) using 26-rank block Strassen ---
        # Indices for A (3x3): i in 0..2, j in 0..2. u_idx = i*3+j
        # Indices for B (3x3): j in 0..2, k in 0..2. v_idx = j*3+k
        # Indices for C (3x3): k in 0..2, i in 0..2. w_idx = k*3+i
        
        col = 0
        
        # Helper to add a rank-1 component
        def add_component(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
            nonlocal col
            U[u_idx][col] += coeff_u
            V[v_idx][col] += coeff_v
            W[w_idx][col] += coeff_w
            col += 1

        # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A22 = [[A00, A01], [A10, A11]]
        # B22 = [[B00, B01], [B10, B11]]
        # C11 = [[C00, C01], [C10, C11]] (Note: C[k,i], so C[0,0], C[0,1], C[1,0], C[1,1])
        
        # P1 = (A00+A11)(B00+B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C11=1, C01=-1, C10=1  (Wait, Strassen: C00=P1+P4-P5+P7, C01=P3+P5, C10=P2+P4, C11=P1-P2+P3+P6)
        # So for P1, w has C00=1, C11=1.
        add_component(0, 0, 0, 1, 1, 1) # A00, B00, C00
        # We need to handle the linear combinations properly.
        # It is easier to define the 7 Strassen products explicitly with their w-vectors.
        
        # Let's restart the Strassen part with explicit vectors.
        # Rank 7 for 2x2x2.
        
        # P1
        u1 = {0: 1, 3: 1} # A00, A11 (idx 0, 3)
        v1 = {0: 1, 3: 1} # B00, B11 (idx 0, 3)
        w1 = {0: 1, 3: 1} # C00, C11 (idx 0, 3) -> C[k,i], C[0,0]=0, C[1,1]=3? 
        # C index: k*3+i.
        # C00 (k=0,i=0) -> 0
        # C01 (k=0,i=1) -> 1
        # C10 (k=1,i=0) -> 3
        # C11 (k=1,i=1) -> 4
        
        # Strassen Formulas:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # So:
        # P1 contributes to C00 (1) and C11 (1)
        # P2 contributes to C10 (1) and C11 (-1)
        # P3 contributes to C01 (1) and C11 (1)
        # P4 contributes to C00 (1) and C10 (1)
        # P5 contributes to C00 (-1) and C01 (1)
        # P6 contributes to C11 (1)
        # P7 contributes to C00 (1)
        
        # Definitions of P:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # u indices: A00=0, A01=1, A10=3, A11=4
        # v indices: B00=0, B01=1, B10=3, B11=4
        # w indices: C00=0, C01=1, C10=3, C11=4
        
        def add_strassen_product(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for u_idx, coeff_u in u_coeffs.items():
                U[u_idx][col] += coeff_u
            for v_idx, coeff_v in v_coeffs.items():
                V[v_idx][col] += coeff_v
            for w_idx, coeff_w in w_coeffs.items():
                W[w_idx][col] += coeff_w
            col += 1

        # P1
        add_strassen_product({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
        # P2
        add_strassen_product({3:1, 4:1}, {0:1}, {3:1, 4:-1})
        # P3
        add_strassen_product({0:1}, {3:1, 4:-1}, {1:1, 4:1})
        # P4
        add_strassen_product({4:1}, {1:1, 0:-1}, {0:1, 3:1})
        # P5
        add_strassen_product({0:1, 1:1}, {4:1}, {0:-1, 1:1})
        # P6
        add_strassen_product({3:1, 0:-1}, {0:1, 1:1}, {4:1})
        # P7
        add_strassen_product({1:1, 4:-1}, {3:1, 4:1}, {0:1})
        
        # 2. a1 b2^T (A col 2, B row 2) -> C[0:2, 0:2]
        # A02 (idx 2), A12 (idx 5)
        # B20 (idx 6), B21 (idx 7)
        # C00 (0), C01 (1), C10 (3), C11 (4)
        # C00 += A02 B20
        add_strassen_product({2:1}, {6:1}, {0:1})
        # C10 += A12 B20
        add_strassen_product({5:1}, {6:1}, {3:1})
        # C01 += A02 B21
        add_strassen_product({2:1}, {7:1}, {1:1})
        # C11 += A12 B21
        add_strassen_product({5:1}, {7:1}, {4:1})
        
        # 3. A22 b1 (A top-left, B col 2) -> C[2, 0:2]
        # A00(0), A01(1), A10(3), A11(4)
        # B02 (idx 2), B12 (idx 5)
        # C20 (idx 6), C21 (idx 7)
        # C20 += A00 B02 + A10 B12
        add_strassen_product({0:1}, {2:1}, {6:1})
        add_strassen_product({3:1}, {5:1}, {6:1})
        # C21 += A01 B02 + A11 B12
        add_strassen_product({1:1}, {2:1}, {7:1})
        add_strassen_product({4:1}, {5:1}, {7:1})
        
        # 4. a2^T B22 (A row 2, B top-left) -> C[0:2, 2]
        # A20 (idx 6), A21 (idx 7)
        # B00(0), B01(1), B10(3), B11(4)
        # C02 (idx 2), C12 (idx 5)
        # C02 += A20 B00 + A21 B10
        add_strassen_product({6:1}, {0:1}, {2:1})
        add_strassen_product({7:1}, {3:1}, {2:1})
        # C12 += A20 B01 + A21 B11
        add_strassen_product({6:1}, {1:1}, {5:1})
        add_strassen_product({7:1}, {4:1}, {5:1})
        
        # 5. a1 b33 (A col 2, B22) -> C[2, 0:2]
        # A02(2), A12(5)
        # B22 (idx 8)
        # C20 (6), C21 (7)
        # C20 += A02 B22
        add_strassen_product({2:1}, {8:1}, {6:1})
        # C21 += A12 B22
        add_strassen_product({5:1}, {8:1}, {7:1})
        
        # 6. a2^T b1 (A row 2, B col 2) -> C[2, 2]
        # A20(6), A21(7)
        # B02(2), B12(5)
        # C22 (idx 8)
        # C22 += A20 B02 + A21 B12
        add_strassen_product({6:1}, {2:1}, {8:1})
        add_strassen_product({7:1}, {5:1}, {8:1})
        
        # 7. a33 b2^T (A22, B row 2) -> C[2, 0:2]
        # A22 (idx 8)
        # B20(6), B21(7)
        # C20(6), C21(7)
        # C20 += A22 B20
        add_strassen_product({8:1}, {6:1}, {6:1})
        # C21 += A22 B21
        add_strassen_product({8:1}, {7:1}, {7:1})
        
        # 8. a33 b33 (A22, B22) -> C[2, 2]
        # A22(8), B22(8)
        # C22(8)
        add_strassen_product({8:1}, {8:1}, {8:1})
        
        # --- Part 2: 3x3x4 (k=3,4,5,6) Schoolbook ---
        # k in 3..6
        for k in range(3, 7):
            for i in range(3):
                for j in range(3):
                    u_idx = i * 3 + j
                    v_idx = j * 7 + k
                    w_idx = k * 3 + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}

    # Attempt rank 62
    try:
        result_62 = construct_rank_62()
        if validate(result_62["U"], result_62["V"], result_62["W"]):
            return result_62
    except Exception:
        pass

    # Fallback to schoolbook
    return construct_schoolbook()

# EVOLVE-BLOCK-END
