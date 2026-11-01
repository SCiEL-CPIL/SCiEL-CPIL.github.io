# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 3
    
    # Rank 26 decomposition:
    # 7 Strassen terms for the 2x2 block (indices 0,1)
    # 19 schoolbook terms for the rest
    
    terms = []
    
    # Helper to add a term
    # U: dict (i,j) -> val
    # V: dict (j,k) -> val
    # W: dict (k,i) -> val  (Note: W index is k*n+i)
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # Strassen terms for 2x2 block (A[0:2,0:2] * B[0:2,0:2])
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    
    # M2 = (A10+A11)B00 -> C10
    add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}) # C10 is k=0, i=1 -> index 0*3+1=1? No.
    # W[k*n+i]. C[i,j]. 
    # C10 is i=1, j=0. Sum over k.
    # The term contributes to C[1,0].
    # In the tensor T[i,j,k] = A[i,k]B[k,j].
    # W matrix in the problem description: W[k*n+i, r].
    # This implies the output is indexed by (k, i)?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually, C[i,j] = sum_k A[i,k] B[k,j].
    # If W is indexed by k*n+i, it suggests the output is a vector of length 9, where index is k*n+i.
    # This corresponds to C[i,k]? Or C[k,i]?
    # Standard: C[i,j].
    # If W[k*n+i] is the coefficient for the product term, and the term is U*A * V*B,
    # then the product contributes to C[i,j] where i is from U, j is from V?
    # U is (n*m, rank). Index i*m+j.
    # V is (m*p, rank). Index j*p+k.
    # W is (p*n, rank). Index k*n+i.
    
    # Let's check the tensor identity:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta?
    # No, the condition is:
    # For all A, B:
    # sum_r U_r(A) V_r(B) = C = A*B
    
    # U_r(A) = sum_{i,j} U[i*m+j, r] A[i,j]
    # V_r(B) = sum_{j,k} V[j*p+k, r] B[j,k]
    
    # Product: sum_{i,j,k,l} U[i*m+j, r] V[j*p+k, r] A[i,j] B[j,k]
    # This sums to C[i,k] = sum_j A[i,j] B[j,k].
    # So the output C is indexed by (i,k).
    # The W matrix must encode which C[i,k] entries are affected.
    # W[k*n+i, r] suggests the index is k*n + i.
    # So W[k*n+i, r] is the coefficient for C[i,k].
    
    # Let's re-verify the Strassen terms with this indexing.
    # M1 = (A00+A11)(B00+B11)
    # U: A00, A11. (Indices 0, 1)
    # V: B00, B11. (Indices 0, 2)
    # Product contributes to:
    # A00*B00 -> C[0,0] (i=0, k=0). W[0*3+0, r] = W[0, r] = 1.
    # A00*B11 -> C[0,1] (i=0, k=1). W[1*3+0, r] = W[3, r] = 1.
    # A11*B00 -> C[1,0] (i=1, k=0). W[0*3+1, r] = W[1, r] = 1.
    # A11*B11 -> C[1,1] (i=1, k=1). W[1*3+1, r] = W[4, r] = 1.
    
    # So W for M1 has 1 at 0, 1, 3, 4.
    
    # Let's restart the term construction with correct W indices.
    
    terms = []
    
    # 1. M1
    add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {0: 1, 1: 1, 3: 1, 4: 1})
    
    # 2. M2 = (A10+A11)B00
    # A10*B00 -> C[1,0] (W[1])
    # A11*B00 -> C[1,1] (W[4])
    add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {1: 1, 4: 1})
    
    # 3. M3 = A00(B10-B11)
    # A00*B10 -> C[0,1] (W[3])
    # A00*B11 -> C[0,2] (W[6])
    add_term({(0,0): 1}, {(1,0): 1, (1,1): -1}, {3: 1, 6: -1})
    
    # 4. M4 = A11(B01-B00)
    # A11*B01 -> C[1,1] (W[4])
    # A11*B00 -> C[1,0] (W[1])
    add_term({(1,1): 1}, {(0,1): 1, (0,0): -1}, {4: 1, 1: -1})
    
    # 5. M5 = (A00+A01)B11
    # A00*B11 -> C[0,2] (W[6])
    # A01*B11 -> C[1,2] (W[7])
    add_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {6: 1, 7: 1})
    
    # 6. M6 = (A10-A00)(B00+B01)
    # A10*B00 -> C[1,0] (W[1])
    # A10*B01 -> C[1,1] (W[4])
    # A00*B00 -> C[0,0] (W[0])
    # A00*B01 -> C[0,1] (W[3])
    add_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {1: 1, 4: 1, 0: -1, 3: -1})
    
    # 7. M7 = (A01-A11)(B10+B11)
    # A01*B10 -> C[0,2] (W[6])
    # A01*B11 -> C[0,3]? No, B is 3x3. B[1,1] is k=1, j=1.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # A[0,1] B[1,1] -> C[0,1] (W[3])
    # A[1,1] B[1,0] -> C[1,0] (W[1])
    # A[1,1] B[1,1] -> C[1,1] (W[4])
    
    # Let's re-verify M7.
    # U: A01, -A11.
    # V: B10, B11.
    # A01*B10 -> C[0,1] (W[3])
    # A01*B11 -> C[0,2] (W[6])
    # A11*B10 -> C[1,1] (W[4])
    # A11*B11 -> C[1,2] (W[7])
    
    add_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {3: 1, 6: 1, 4: -1, 7: -1})
    
    # Now the schoolbook terms for the rest.
    # We need to cover all C[i,k] entries.
    # The Strassen terms cover C[0,0], C[0,1], C[1,0], C[1,1] partially?
    # No, they cover the contribution from j=0,1.
    # We need to add the contribution from j=2.
    
    # C[0,0] += A[0,2]B[2,0] -> W[0]
    add_term({(0,2): 1}, {(2,0): 1}, {0: 1})
    
    # C[0,1] += A[0,2]B[2,1] -> W[3]
    add_term({(0,2): 1}, {(2,1): 1}, {3: 1})
    
    # C[0,2] += A[0,2]B[2,2] -> W[6]
    add_term({(0,2): 1}, {(2,2): 1}, {6: 1})
    
    # C[1,0] += A[1,2]B[2,0] -> W[1]
    add_term({(1,2): 1}, {(2,0): 1}, {1: 1})
    
    # C[1,1] += A[1,2]B[2,1] -> W[4]
    add_term({(1,2): 1}, {(2,1): 1}, {4: 1})
    
    # C[1,2] += A[1,2]B[2,2] -> W[7]
    add_term({(1,2): 1}, {(2,2): 1}, {7: 1})
    
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    add_term({(2,0): 1}, {(0,0): 1}, {9: 1}) # W[2*3+0] = W[6]? No.
    # W index is k*n+i.
    # C[2,0] is i=2, k=0. Index 0*3+2 = 2.
    # C[2,1] is i=2, k=1. Index 1*3+2 = 5.
    # C[2,2] is i=2, k=2. Index 2*3+2 = 8.
    
    # Correct W indices for row i=2:
    # C[2,0] -> W[2]
    # C[2,1] -> W[5]
    # C[2,2] -> W[8]
    
    # Redo C[2,0] terms
    add_term({(2,0): 1}, {(0,0): 1}, {2: 1})
    add_term({(2,1): 1}, {(1,0): 1}, {2: 1})
    add_term({(2,2): 1}, {(2,0): 1}, {2: 1})
    
    # C[2,1] terms
    add_term({(2,0): 1}, {(0,1): 1}, {5: 1})
    add_term({(2,1): 1}, {(1,1): 1}, {5: 1})
    add_term({(2,2): 1}, {(2,1): 1}, {5: 1})
    
    # C[2,2] terms
    add_term({(2,0): 1}, {(0,2): 1}, {8: 1})
    add_term({(2,1): 1}, {(1,2): 1}, {8: 1})
    add_term({(2,2): 1}, {(2,2): 1}, {8: 1})
    
    # Total terms: 7 + 6 + 9 = 22?
    # 7 Strassen
    # 6 for C[0,2], C[1,2] with j=2? No.
    # I added 6 terms for C[0,0..2], C[1,0..2] with j=2.
    # I added 9 terms for C[2,0..2].
    # Total 22.
    
    # Verify rank 22.
    # If valid, score 27/22 = 1.22.
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] = val
        for (j, k), val in v.items():
            V[j * p + k][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
