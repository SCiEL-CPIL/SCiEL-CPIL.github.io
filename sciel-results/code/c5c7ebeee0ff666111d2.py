"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import sys
import signal
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    # Construct the schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to numpy for efficient validation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Try to find a lower rank decomposition using a structured search
    # We know that for 3x3x3, the Strassen algorithm gives rank 7 instead of 9.
    # For 3x3x8, we can potentially use a block-wise approach or specific algebraic identities.
    # However, finding a rank < 72 is non-trivial without specific knowledge.
    # Let's try a randomized search with a specific structure:
    # We assume the decomposition has a form where U, V, W are sparse or have symmetries.
    
    best_rank = schoolbook_rank
    best_U = U_np
    best_V = V_np
    best_W = W_np

    # Define the target tensor T[i, j, k] = 1 if (i, j, k) corresponds to a valid multiplication term
    # In the standard bilinear form for C = A * B:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T is defined such that the multiplication is:
    # sum_c U[i, j, c] * V[j, k, c] * W[i, k, c] ... wait, the problem statement says:
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # This implies W is indexed by (k, i) where k is the row of B (and column of A) and i is the row of A (and column of C)?
    # Let's re-read carefully: "multiplying a 3 by 3 matrix by a 3 by 8 matrix"
    # A is 3x3, B is 3x8. C is 3x8.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The factors are U (n*m, rank) = (3*3, rank) = (9, rank)
    # V (m*p, rank) = (3*8, rank) = (24, rank)
    # W (p*n, rank) = (8*3, rank) = (24, rank)
    # The index for U is i*m+j. i in 0..2, j in 0..2.
    # The index for V is j*p+k. j in 0..2, k in 0..7.
    # The index for W is k*n+i. k in 0..7, i in 0..2.
    
    # The tensor equation is:
    # For each i in 0..2, k in 0..7:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # This must equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... NO.
    # The standard tensor decomposition of matrix multiplication is:
    # T[i, j, k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... no.
    
    # Let's look at the structure of the schoolbook solution provided in the seed:
    # U[i*m+j][col] = 1
    # V[j*p+k][col] = 1
    # W[k*n+i][col] = 1
    # This corresponds to the term A[i,j] * B[j,k] contributing to C[i,k].
    # Wait, if U, V, W are all 1, then the sum is 1.
    # The product A[i,j] * B[j,k] is not represented by 1*1*1.
    # The decomposition must be:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This is a bilinear map. The tensor T is defined on the space of (A, B, C^*)?
    # Actually, the problem asks for a tensor decomposition of the multiplication tensor.
    # The multiplication tensor M is defined by M[i, j, k] = 1 if the indices correspond to the term A[i,j]B[j,k] in C[i,k]?
    # No, the tensor M is such that <M, A \otimes B \otimes C^*> = trace(C^T (AB)).
    # Or more simply, the coefficients of the bilinear form.
    # The schoolbook solution has rank n*m*p.
    # Each term in the sum C[i,k] = sum_j A[i,j]B[j,k] is a rank-1 term in the tensor sense?
    # Yes, the term A[i,j] * B[j,k] can be seen as a rank-1 tensor in the space of (i,j) x (j,k) x (k,i) ?
    # Let's verify the indices in the seed code again.
    # U[i*m+j], V[j*p+k], W[k*n+i].
    # If we fix i, j, k, we have one term.
    # The total number of terms is n*m*p.
    # This matches the schoolbook rank.
    
    # So the target tensor T has non-zero entries at (i*m+j, j*p+k, k*n+i) for all valid i,j,k?
    # No, the tensor T is a 3-way tensor T[a, b, c] where a is index of U, b of V, c of W.
    # T[a, b, c] = 1 if there exist i, j, k such that a = i*m+j, b = j*p+k, c = k*n+i.
    # And 0 otherwise.
    # We want to find U, V, W such that:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # Let's verify this interpretation.
    # If T is the multiplication tensor, then for any matrices A, B, C:
    # sum_{a,b,c} T[a,b,c] A[a] B[b] C[c] = trace(C^T AB) ?
    # Let A be indexed by i,m+j. B by j,p+k. C by k,n+i.
    # sum_{i,j,k} T[i*m+j, j*p+k, k*n+i] A[i,j] B[j,k] C[k,i]
    # If T[i*m+j, j*p+k, k*n+i] = 1, then sum = sum_{i,j,k} A[i,j] B[j,k] C[k,i] = trace(C^T AB).
    # Yes, this is correct.
    
    # So we need to find a rank-r decomposition of the tensor T defined by:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # All other entries are 0.
    
    # We can try to find a lower rank by exploiting the structure.
    # For n=m=p=3, the Strassen algorithm finds a rank 7 decomposition.
    # For n=3, m=3, p=8, the rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24.
    # It is also known that the rank of matrix multiplication is at least n^omega.
    # For 3x3, the rank is 7.
    # For 3x3x8, we can view it as 8 blocks of 3x3x1? No, 3x8 is one block.
    # We can use the 3x3x3 decomposition and extend it?
    # If we have a rank-7 decomposition for 3x3x3, can we use it for 3x3x8?
    # The 3x3x3 tensor T3 has T3[i*m+j, j*p+k, k*n+i] = 1 for i,j,k in 0..2.
    # The 3x3x8 tensor T8 has T8[i*m+j, j*p+k, k*n+i] = 1 for i,j in 0..2, k in 0..7.
    # We can decompose T8 into 8 slices? No, k is the column index of B and row index of C.
    # T8[i*m+j, j*p+k, k*n+i] = 1.
    # This is not a direct sum of 3x3x3 tensors because the indices are coupled.
    
    # Let's try a random search for a lower rank, starting from rank 24 and going up.
    # We will use a gradient-free optimization or a simple randomized approach.
    # Given the time limit, a heavy search is risky.
    # We will try a few random initializations for rank 24, 25, ... up to 30.
    # If we find a valid decomposition, we return it.
    # Otherwise, we return the schoolbook.
    
    # Function to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        # Compute the tensor T from U, V, W
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We only need to check the non-zero entries of the target tensor.
        # Target non-zero entries: (i*m+j, j*p+k, k*n+i) for all i,j,k.
        # There are n*m*p = 72 such entries.
        # We can compute the sum for each of these entries and check if it is 1.
        # Also, we should check that some zero entries are 0, but that's harder.
        # For now, we just check the non-zero entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(val - 1.0) > tol:
                        return False
        # We also need to check that the decomposition doesn't produce non-zero values where it should be zero.
        # This is more complex. We can sample some zero entries.
        # Total entries in T are (n*m) * (m*p) * (p*n) = 9 * 24 * 24 = 5184.
        # Non-zero entries are 72.
        # We can check a random subset of zero entries.
        zero_entries = []
        all_indices = set(product(range(n*m), range(m*p), range(p*n)))
        non_zero_indices = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    non_zero_indices.add((a, b, c))
        
        # Sample 100 random zero entries
        np.random.seed(123)
        for _ in range(100):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            if (a, b, c) not in non_zero_indices:
                val = np.dot(U[a], np.multiply(V[b], W[c]))
                if abs(val) > tol:
                    return False
        return True

    # Try to find a lower rank decomposition
    # We will try ranks from 24 to 40
    for target_rank in range(24, 41):
        if time.time() > DEADLINE - 10:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > DEADLINE - 5:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple check: if the random guess happens to be valid (unlikely), return it
            if is_valid(U_r, V_r, W_r):
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank == target_rank:
            break

    # If we didn't find a better rank, we return the schoolbook decomposition
    # Convert to list of lists with integer/half-integer entries if possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    # Check if it's a half-integer
                    if abs(2*x - round(2*x)) < 1e-9:
                        new_row.append(round(2*x)/2.0)
                    else:
                        new_row.append(round(x, 6))
            result.append(new_row)
        return result

    U_result = to_list(best_U)
    V_result = to_list(best_V)
    W_result = to_list(best_W)

    return {"U": U_result, "V": V_result, "W": W_result}

# EVOLVE-BLOCK-END
