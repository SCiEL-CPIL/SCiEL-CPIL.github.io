import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if this triple contributes to output
    # The multiplication tensor: (A * B)[i, k] = sum_j A[i,j] * B[j,k]
    # So tensor element T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # But we need to be careful: the output is indexed by (i, k), and the input indices are (i,j) and (j,k)
    
    # The matrix multiplication tensor T has entries T[i, j, k] = 1 for all i, j, k
    # because (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # We want to find a rank-r decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Note the W indexing: W[k*n + i, c] following the trace convention W[k*n+i, r]
    
    # Let's build the tensor as a 3D array
    T = np.ones((n, m, p), dtype=np.float64)
    
    # We'll try to find a low-rank decomposition using the approach of
    # exploiting the structure. For 2x2 times 2x11, we can potentially use
    # Strassen-like ideas for the 2x2 part and handle the 11 columns.
    
    # Key insight: A is 2x2, B is 2x11. So (A*B) is 2x11.
    # (A*B)[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # (A*B)[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # The inputs are: A has 4 elements, B has 22 elements, total 26 inputs.
    # The outputs are 22 elements.
    
    # For each output (i, k), we need a bilinear form in the inputs.
    
    # Let's think about this differently. We can group by the j index (which dimension of B's rows).
    # For a fixed k, the two outputs (0,k) and (1,k) depend on B[0,k] and B[1,k].
    
    # (A*B)[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # (A*B)[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # This is a 2x2 matrix A multiplied by a 2x1 column [B[0,k], B[1,k]]^T.
    
    # For each k, we need to compute a 2x2 matrix times a 2-vector, which has rank 2 in general
    # (since the 2x2 matrix is arbitrary). But we can use Strassen for the 2x2 part.
    
    # Strassen computes C = A*B for 2x2 matrices with 7 multiplications instead of 8.
    # Here, for each k, we're computing a 2x1 result from a 2x2 times 2x1.
    # That's essentially 2 separate dot products of length 2, which is rank 2.
    
    # But we can batch: think of it as A (2x2) times B (2x11).
    
    # Let me try a direct approach: enumerate possible rank-1 contributions and solve.
    
    # Actually, let's try a smarter approach. We'll use the fact that the problem is:
    # Compute M = A * B where A is 2x2 and B is 2x11.
    
    # We can write B as [B0; B1] where B0 is 1x11 and B1 is 1x11.
    # M[0, :] = A[0,0]*B0 + A[0,1]*B1
    # M[1, :] = A[1,0]*B0 + A[1,1]*B1
    
    # So M = [A[0,0]*B0 + A[0,1]*B1; A[1,0]*B0 + A[1,1]*B1]
    
    # The bilinear terms are:
    # For each k:
    #   M[0,k] = A[0,0]*B0[k] + A[0,1]*B1[k]
    #   M[1,k] = A[1,0]*B0[k] + A[1,1]*B1[k]
    
    # Total bilinear terms: 4 * 11 = 44 (schoolbook)
    
    # Can we do better? Let's think about it as a linear algebra problem.
    
    # The tensor T has shape (4, 22, 22) where:
    #   First index: (i, j) -> i*m+j for A element A[i,j]
    #   Second index: (j, k) -> j*p+k for B element B[j,k]
    #   Third index: (k, i) -> k*n+i for output M[i,k] (using trace convention W[k*n+i])
    
    # Wait, let me re-read the convention. The output factor W has shape (p*n, rank) with W[k*n+i, r].
    # So the output index is (k, i) but stored as k*n+i. This means output[i, k] is at position k*n+i.
    
    # Let me build the tensor properly.
    
    # Tensor T[a, b, c] where:
    #   a = i*m + j (index into U), i in [0,n), j in [0,m)
    #   b = j*p + k (index into V), j in [0,m), k in [0,p)
    #   c = k*n + i (index into W), k in [0,p), i in [0,n)
    
    # T[a, b, c] = 1 if the (i,j) element of A and (j,k) element of B contribute to output (i,k)
    # This happens when the indices are consistent: same j, and output is (i,k).
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k.
    
    # Let's verify: for each (i,j,k), we have exactly one nonzero entry in the tensor.
    # The total number of nonzero entries is n*m*p = 2*2*11 = 44.
    
    # We want to decompose T as sum of rank-1 tensors.
    
    # Approach: Use the known structure. For 2x2 matrices, Strassen reduces 8 to 7.
    # For 2x2 times 2xN, let's see if we can reduce below 2*2*N = 4N.
    
    # Actually, I recall that for multiplying an m x n matrix by an n x p matrix,
    # the tensor rank is at most m*n*p, and can be reduced.
    
    # Let me try a different approach: use the fact that we only need integer/half-integer
    # entries, and try small ranks with a linear system solver.
    
    # For rank r, we need to find U (4 x r), V (22 x r), W (22 x r) such that
    # for all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other (a, b, c): sum_c U[a, c] * V[b, c] * W[c, c] = 0
    
    # This is a system of bilinear equations, which is hard. But we can fix some structure.
    
    # Let's try rank 39 (from the Strassen-inspired approach mentioned in the knowledge).
    # The knowledge says "Using Strassen's algorithm for 5 pairs of columns reduces rank from 44 to 39".
    
    # Let me try to construct a rank-39 decomposition.
    
    # Strategy: Group the 11 columns into 5 pairs and 1 single column.
    # For each pair of columns (2k, 2k+1), use a Strassen-like 7-term computation.
    # For the single column, use 2 terms.
    # Total: 5*7 + 1*2 = 35 + 2 = 37? Or maybe 5*7 + 2 = 37?
    
    # Wait, the knowledge says 39. Let me think more carefully.
    
    # For a 2x2 matrix A and a 2x2 matrix B (one pair of columns), the product is 2x2.
    # Standard: 8 multiplications. Strassen: 7 multiplications.
    # So saving 1 per pair. 5 pairs: save 5. 44 - 5 = 39. Yes!
    
    # For the remaining 1 column (column 10, since 0-10 is 11 columns, pairs are (0,1), (2,3), (4,5), (6,7), (8,9), and single column 10):
    # Standard: 2 multiplications (one for each output row).
    # Total: 5*7 + 1*2 = 37? But knowledge says 39.
    
    # Hmm, maybe I'm miscounting. Let me reconsider.
    
    # Actually, for a 2x2 A times a 2x1 column (single column), we need 2 multiplications:
    # M[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]  -- this is 2 bilinear terms
    # M[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]  -- this is 2 bilinear terms
    # Total: 4 bilinear terms for a single column.
    
    # For a pair of columns (2x2 B), we have 8 bilinear terms standard, 7 with Strassen.
    # So: 5 pairs * 7 + 1 single * 4 = 35 + 4 = 39. Yes!
    
    # So rank 39 should be achievable.
    
    # Let me construct the Strassen decomposition for 2x2 times 2x2.
    
    # Strassen's 7 products for C = A * B where A, B are 2x2:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Now, for each pair of columns, we apply this.
    
    # Let me build the full decomposition.
    
    def build_strassen_decomposition():
        """Build a rank-39 decomposition using Strassen for 5 column pairs and schoolbook for 1 column."""
        n, m, p = DIMENSIONS
        rank = 39
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        col = 0
        
        # Process 5 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9)
        for pair_idx in range(5):
            k0 = 2 * pair_idx
            k1 = 2 * pair_idx + 1
            
            # For this pair, B columns are k0 and k1.
            # B[j, k0] and B[j, k1] for j in {0, 1}
            
            # Strassen products:
            # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
            # Wait, I need to be careful. The Strassen formula is for C = A * B where B is 2x2.
            # Here B is 2x2 with columns k0, k1.
            
            # Let me denote:
            # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
            # B0k0 = B[0,k0], B0k1 = B[0,k1], B1k0 = B[1,k0], B1k1 = B[1,k1]
            
            # M1 = (A00 + A11) * (B0k0 + B1k1)
            # M2 = (A10 + A11) * B0k0
            # M3 = A00 * (B0k1 - B1k1)
            # M4 = A11 * (B1k0 - B0k0)
            # M5 = (A00 + A01) * B1k1
            # M6 = (A10 - A00) * (B0k0 + B0k1)
            # M7 = (A01 - A11) * (B1k0 + B1k1)
            
            # C[0,k0] = M1 + M4 - M5 + M7
            # C[0,k1] = M3 + M5
            # C[1,k0] = M2 + M4
            # C[1,k1] = M1 - M2 + M3 + M6
            
            # Each Mc is a product of a linear form in A and a linear form in B.
            # We represent each as a rank-1 term.
            
            # For each Mc:
            #   U part: coefficients of (i*m+j) for the A linear form
            #   V part: coefficients of (j*p+k) for the B linear form
            #   W part: coefficients of (k*n+i) for the output linear form
            
            # M1: A part: A00 + A11 -> U[0*m+0, col] = 1, U[1*m+1, col] = 1
            #     B part: B0k0 + B1k1 -> V[0*p+k0, col] = 1, V[1*p+k1, col] = 1
            #     Output: C[0,k0] += 1, C[1,k1] += 1
            #     W[k0*n+0, col] = 1, W[k1*n+1, col] = 1
            
            U[0, col] = 1; U[3, col] = 1
            V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
            W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
            col += 1
            
            # M2: A part: A10 + A11 -> U[1*m+0, col] = 1, U[1*m+1, col] = 1
            #     B part: B0k0 -> V[0*p+k0, col] = 1
            #     Output: C[1,k0] += 1
            #     W[k0*n+1, col] = 1
            
            U[2, col] = 1; U[3, col] = 1
            V[0*p + k0, col] = 1
            W[k0*n + 1, col] = 1
            col += 1
            
            # M3: A part: A00 -> U[0*m+0, col] = 1
            #     B part: B0k1 - B1k1 -> V[0*p+k1, col] = 1, V[1*p+k1, col] = -1
            #     Output: C[0,k1] += 1
            #     W[k1*n+0, col] = 1
            
            U[0, col] = 1
            V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
            W[k1*n + 0, col] = 1
            col += 1
            
            # M4: A part: A11 -> U[1*m+1, col] = 1
            #     B part: B1k0 - B0k0 -> V[1*p+k0, col] = 1, V[0*p+k0, col] = -1
            #     Output: C[0,k0] += 1, C[1,k0] += 1
            #     W[k0*n+0, col] = 1, W[k0*n+1, col] = 1
            
            U[3, col] = 1
            V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
            W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
            col += 1
            
            # M5: A part: A00 + A01 -> U[0*m+0, col] = 1, U[0*m+1, col] = 1
            #     B part: B1k1 -> V[1*p+k1, col] = 1
            #     Output: C[0,k0] -= 1, C[0,k1] += 1
            #     W[k0*n+0, col] = -1, W[k1*n+0, col] = 1
            
            U[0, col] = 1; U[1, col] = 1
            V[1*p + k1, col] = 1
            W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
            col += 1
            
            # M6: A part: A10 - A00 -> U[1*m+0, col] = 1, U[0*m+0, col] = -1
            #     B part: B0k0 + B0k1 -> V[0*p+k0, col] = 1, V[0*p+k1, col] = 1
            #     Output: C[1,k1] += 1
            #     W[k1*n+1, col] = 1
            
            U[2, col] = 1; U[0, col] = -1
            V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
            W[k1*n + 1, col] = 1
            col += 1
            
            # M7: A part: A01 - A11 -> U[0*m+1, col] = 1, U[1*m+1, col] = -1
            #     B part: B1k0 + B1k1 -> V[1*p+k0, col] = 1, V[1*p+k1, col] = 1
            #     Output: C[0,k0] += 1
            #     W[k0*n+0, col] = 1
            
            U[1, col] = 1; U[3, col] = -1
            V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
            W[k0*n + 0, col] = 1
            col += 1
        
        # Process the single column k=10 with schoolbook (4 terms)
        k = 10
        # M[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # M[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        # Term 1: A[0,0] * B[0,k] -> M[0,k] += 1
        U[0, col] = 1
        V[0*p + k, col] = 1
        W[k*n + 0, col] = 1
        col += 1
        
        # Term 2: A[0,1] * B[1,k] -> M[0,k] += 1
        U[1, col] = 1
        V[1*p + k, col] = 1
        W[k*n + 0, col] = 1
        col += 1
        
        # Term 3: A[1,0] * B[0,k] -> M[1,k] += 1
        U[2, col] = 1
        V[0*p + k, col] = 1
        W[k*n + 1, col] = 1
        col += 1
        
        # Term 4: A[1,1] * B[1,k] -> M[1,k] += 1
        U[3, col] = 1
        V[1*p + k, col] = 1
        W[k*n + 1, col] = 1
        col += 1
        
        return U, V, W
    
    # Try the Strassen-based decomposition first
    U, V, W = build_strassen_decomposition()
    
    # Verify the decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        # Check that for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False, f"Nonzero entry ({i},{j},{k}): got {s}, expected 1"
        
        # Check that all other entries are 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Determine if this is a "valid" triple
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if i == i2 and j == j2 and k == k2:
                        continue  # Already checked
                    
                    s = 0.0
                    for cc in range(U.shape[1]):
                        s += U[a, cc] * V[b, cc] * W[c, cc]
                    if abs(s) > 1e-9:
                        return False, f"Zero entry ({a},{b},{c}): got {s}, expected 0"
        
        return True, "OK"
    
    valid, msg = verify(U, V, W)
    if valid:
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, column] = 1
                    V[j*p+k, column] = 1
                    W[k*n+i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
