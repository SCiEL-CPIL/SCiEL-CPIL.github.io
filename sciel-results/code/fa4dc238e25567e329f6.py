import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # We'll build the decomposition using Strassen on 2x2 sub-blocks
    # Partition j (columns of A / rows of B) into pairs: (0,1), (2,3), and single 4
    # Partition k (columns of B / columns of C) into pairs: (0,1), (2,3), and single 4
    
    terms = []  # Each term is (U_dict, V_dict, W_dict)
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term with sparse entries."""
        terms.append((dict(u_entries), dict(v_entries), dict(w_entries)))
    
    # Strassen terms for 2x2 block with i in {0,1}, j in {a,a+1}, k in {b,b+1}
    # A[i,j], B[j,k], C[i,k]
    # M1 = (A[0,a]+A[1,a+1]) * (B[a,b]+B[a+1,b+1])
    # M2 = (A[1,a]+A[1,a+1]) * B[a,b]
    # M3 = A[0,a] * (B[a+1,b]-B[a+1,b+1])
    # M4 = A[1,a+1] * (B[a+1,b+1]-B[a,b])
    # M5 = (A[0,a]+A[0,a+1]) * B[a+1,b+1]
    # M6 = (A[1,a]-A[0,a]) * (B[a,b]+B[a,b+1])
    # M7 = (A[0,a+1]-A[1,a+1]) * (B[a+1,b]+B[a+1,b+1])
    #
    # C[0,b] = M1 - M2 + M4 - M5
    # C[0,b+1] = M3 + M5
    # C[1,b] = M2 + M4
    # C[1,b+1] = M1 - M3 + M4 - M6
    
    def add_strassen(a, b):
        """Add 7 Strassen terms for j-pair {a,a+1}, k-pair {b,b+1}."""
        # M1: (A[0,a]+A[1,a+1]) * (B[a,b]+B[a+1,b+1])
        add_term(
            {(0,a): 1, (1,a+1): 1},
            {(a,b): 1, (a+1,b+1): 1},
            {(b,0): 1, (b+1,1): 1}
        )
        # M2: (A[1,a]+A[1,a+1]) * B[a,b]
        add_term(
            {(1,a): 1, (1,a+1): 1},
            {(a,b): 1},
            {(b,1): 1}
        )
        # M3: A[0,a] * (B[a+1,b]-B[a+1,b+1])
        add_term(
            {(0,a): 1},
            {(a+1,b): 1, (a+1,b+1): -1},
            {(b,0): 1}
        )
        # M4: A[1,a+1] * (B[a+1,b+1]-B[a,b])
        add_term(
            {(1,a+1): 1},
            {(a+1,b+1): 1, (a,b): -1},
            {(b,0): 1, (b+1,1): 1}
        )
        # M5: (A[0,a]+A[0,a+1]) * B[a+1,b+1]
        add_term(
            {(0,a): 1, (0,a+1): 1},
            {(a+1,b+1): 1},
            {(b,0): -1, (b+1,1): 1}
        )
        # M6: (A[1,a]-A[0,a]) * (B[a,b]+B[a,b+1])
        add_term(
            {(1,a): 1, (0,a): -1},
            {(a,b): 1, (a,b+1): 1},
            {(b+1,1): -1}
        )
        # M7: (A[0,a+1]-A[1,a+1]) * (B[a+1,b]+B[a+1,b+1])
        add_term(
            {(0,a+1): 1, (1,a+1): -1},
            {(a+1,b): 1, (a+1,b+1): 1},
            {}  # M7 doesn't contribute to C? Let me check.
        )
    
    # Wait, I need to double-check the Strassen formula. Let me use the standard one.
    # Standard Strassen for C = AB, A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M2 + M4 - M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M4 - M6
    #
    # In our notation for block (i in {0,1}, j in {a,a+1}, k in {b,b+1}):
    # a = A[0,a], b = A[0,a+1], c = A[1,a], d = A[1,a+1]
    # e = B[a,b], f = B[a,b+1], g = B[a+1,b], h = B[a+1,b+1]
    # C[0,b] = C00, C[0,b+1] = C01, C[1,b] = C10, C[1,b+1] = C11
    
    # Let me redo this properly.
    
    # Clear terms
    terms = []
    
    def add_strassen_block(a, b):
        """Add 7 Strassen terms for j-pair {a,a+1}, k-pair {b,b+1}."""
        # M1 = (A[0,a]+A[1,a+1]) * (B[a,b]+B[a+1,b+1])
        add_term(
            {(0,a): 1, (1,a+1): 1},
            {(a,b): 1, (a+1,b+1): 1},
            {(b,0): 1, (b+1,1): 1}  # contributes +1 to C[0,b] and +1 to C[1,b+1]
        )
        # M2 = (A[1,a]+A[1,a+1]) * B[a,b]
        add_term(
            {(1,a): 1, (1,a+1): 1},
            {(a,b): 1},
            {(b,0): -1, (b,1): 1}  # contributes -1 to C[0,b] and +1 to C[1,b]
        )
        # M3 = A[0,a] * (B[a,b+1]-B[a+1,b+1])
        add_term(
            {(0,a): 1},
            {(a,b+1): 1, (a+1,b+1): -1},
            {(b+1,0): 1, (b+1,1): -1}  # contributes +1 to C[0,b+1] and -1 to C[1,b+1]
        )
        # M4 = A[1,a+1] * (B[a+1,b]-B[a,b])
        add_term(
            {(1,a+1): 1},
            {(a+1,b): 1, (a,b): -1},
            {(b,0): 1, (b,1): 1}  # contributes +1 to C[0,b] and +1 to C[1,b]
        )
        # M5 = (A[0,a]+A[0,a+1]) * B[a+1,b+1]
        add_term(
            {(0,a): 1, (0,a+1): 1},
            {(a+1,b+1): 1},
            {(b,0): -1, (b+1,1): 1}  # contributes -1 to C[0,b] and +1 to C[1,b+1]
        )
        # M6 = (A[1,a]-A[0,a]) * (B[a,b]+B[a,b+1])
        add_term(
            {(1,a): 1, (0,a): -1},
            {(a,b): 1, (a,b+1): 1},
            {(b,1): 0, (b+1,1): -1}  # contributes -1 to C[1,b+1]
        )
        # M7 = (A[0,a+1]-A[1,a+1]) * (B[a+1,b]+B[a+1,b+1])
        add_term(
            {(0,a+1): 1, (1,a+1): -1},
            {(a+1,b): 1, (a+1,b+1): 1},
            {}  # M7 contributes to C[1,b+1]? Let me check.
        )
    
    # I'm getting confused with the signs. Let me just implement it numerically and verify.
    
    # Actually, let me take a completely different approach. Let me just use the schoolbook method
    # but try to find linear dependencies to reduce the rank.
    
    # For now, let me implement the schoolbook method which gives rank 50, and then try to
    # improve it using Strassen.
    
    # Schoolbook: rank = n*m*p = 50
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
