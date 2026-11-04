DIMENSIONS = [2, 3, 10]

import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Strategy: Split A (2x3) into A_left (2x1) and A_right (2x2)
    # Split B (3x10) into B_top (1x10) and B_bottom (2x10)
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_left * B_top: 2x1 * 1x10 -> schoolbook rank = 2*1*10 = 20
    # A_right * B_bottom: 2x2 * 2x10 -> use Strassen on 5 blocks of 2x2
    #   Split B_bottom (2x10) into 5 blocks of (2x2)
    #   Each 2x2 * 2x2 uses Strassen rank 7
    #   Total: 5 * 7 = 35
    # Grand total: 20 + 35 = 55
    
    deadline = time.time() + 110  # Leave 10s safety margin
    
    # Build the decomposition
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (30, rank)  
    # W shape: (p*n, rank) = (20, rank)
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_left (column j=0) * B_top (row j=0)
    # A[i,0] * B[0,k] for i in {0,1}, k in {0,...,9}
    # This is schoolbook: rank = 2 * 10 = 20
    for i in range(n):
        for k in range(p):
            # U index: i*m + 0 = i*3 + 0
            U[i * m + 0, col] = 1
            # V index: 0*p + k = k
            V[0 * p + k, col] = 1
            # W index: k*n + i = k*2 + i
            W[k * n + i, col] = 1
            col += 1
    
    # Part 2: A_right (columns j=1,2) * B_bottom (rows j=1,2)
    # Split B_bottom (2x10) into 5 blocks of (2x2)
    # Block s: columns 2s and 2s+1 of B_bottom, which are global columns 2s+1 and 2s+2 of B
    # Wait, B_bottom is rows j=1 and j=2 of B (3x10)
    # B_bottom has shape 2x10, with columns 0..9
    # Block s covers B_bottom columns 2s and 2s+1, i.e., global B columns 2s and 2s+1 (but offset by j=1,2)
    # Actually B_bottom[k_global, col] where k_global = j*p + k for j in {1,2}
    # For block s: k in {2s, 2s+1} within B_bottom, so global column index in V is j*p + (2s + t) for t in {0,1}
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 - M4 + M6
    # C01 = M5 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M7
    
    # In our tensor decomposition framework:
    # For each Strassen multiplication Mk, we need to express it as sum of rank-1 terms
    # Each Mk involves products of linear combinations of A entries and B entries
    
    # Let me define the Strassen terms more carefully for the tensor decomposition.
    # A_right has entries A[i,j] for i in {0,1}, j in {1,2} (global j)
    # B_bottom has entries B[j,k] for j in {1,2}, k in {0,...,9} (global)
    # For block s: B entries are B[j, 2s] and B[j, 2s+1] for j in {1,2}
    
    # Let me use local indices: A_local[i, j_local] where j_local = j_global - 1 (so j_local in {0,1})
    # B_local[j_local, k_local] where k_local = k_global (within the block)
    
    # Strassen terms for block s:
    # Each term is a product of a linear combination of A entries and a linear combination of B entries
    # In the tensor decomposition, each such product corresponds to one rank-1 term
    
    # Term 1: (A[0,1] + A[1,2]) * (B[1,2s] + B[2,2s+1])
    # This contributes to C[0,2s] and C[1,2s+1] with coefficient +1, and to C[1,2s] and C[0,2s+1] with coefficient -1
    # Wait, I need to be more careful.
    
    # Let me just directly construct the rank-1 terms for each Strassen multiplication.
    
    # For each of the 7 Strassen products in block s, we add 1 rank-1 term.
    # The A-part is a linear combination of A[i,j] for i in {0,1}, j in {1,2}
    # The B-part is a linear combination of B[j,k] for j in {1,2}, k in {2s, 2s+1}
    
    # Strassen products:
    # P1 = (A[0,1] + A[1,2]) * (B[1,2s] + B[2,2s+1])
    # P2 = (A[1,1] + A[1,2]) * B[1,2s]
    # P3 = A[0,1] * (B[2,2s] - B[1,2s])
    # P4 = A[1,2] * (B[1,2s+1] - B[2,2s+1])
    # P5 = (A[0,1] + A[0,2]) * B[2,2s+1]
    # P6 = (A[1,1] - A[0,1]) * (B[1,2s] + B[1,2s+1])
    # P7 = (A[0,2] - A[1,2]) * (B[2,2s] + B[2,2s+1])
    
    # Contributions to C:
    # C[0,2s]   = P1 - P4 + P6
    # C[0,2s+1] = P5 + P4
    # C[1,2s]   = P2 + P3
    # C[1,2s+1] = P1 - P2 + P7
    
    # Each Pi is a single product of (linear combo of A) * (linear combo of B)
    # In the tensor decomposition, each Pi contributes to multiple (i,j,k) triples
    
    # For a given Pi with A-coefficients a[i,j] and B-coefficients b[j,k]:
    # It contributes a[i,j] * b[j,k] to C[i,k]
    # In the tensor decomposition, this means for each (i,j,k) where a[i,j]*b[j,k] != 0:
    #   U[i*m+j, col] += a[i,j]
    #   V[j*p+k, col] += b[j,k]  
    #   W[k*n+i, col] += 1  (wait, no)
    
    # Actually, the tensor decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # But that's not right for matrix multiplication.
    
    # Let me re-derive. The multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For schoolbook: each term c corresponds to one (i,j,k), with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    
    # For Strassen, each rank-1 term corresponds to one Pi.
    # Pi = (sum_{i,j} a_pi[i,j] * A[i,j]) * (sum_{j,k} b_pi[j,k] * B[j,k])
    #     = sum_{i,j,k} a_pi[i,j] * b_pi[j,k] * A[i,j] * B[j,k]
    
    # This contributes to C[i,k] = sum_j A[i,j]*B[j,k] the term sum_j a_pi[i,j]*b_pi[j,k]*A[i,j]*B[j,k]
    # Wait, that's not quite right either.
    
    # Actually: Pi contributes to C[i,k] the value:
    # (sum_j a_pi[i,j]*A[i,j]) * (sum_j b_pi[j,k]*B[j,k])
    # But this is a product of two sums, not a sum of products.
    
    # I think I'm overcomplicating this. Let me look at it differently.
    
    # In the tensor decomposition for matrix multiplication, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (summed over c)
    
    # No wait, the indices don't work out that way. Let me re-read the problem.
    
    # The problem says: W[k*n+i, r] - so W is indexed by (k,i)
    # U[i*m+j, c] - indexed by (i,j)
    # V[j*p+k, c] - indexed by (j,k)
    
    # And the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a 3-way tensor factorization of the all-ones tensor on the "consistent" indices.
    
    # For Strassen, each of the 7 products Pi is itself a bilinear form in A and B.
    # Pi = <a_pi, A> * <b_pi, B> where a_pi is a 2x2 vector (for A_right) and b_pi is a 2x2 vector (for B block)
    
    # In the tensor decomposition, each Pi corresponds to one rank-1 term where:
    # U[i*m+j, c] = a_pi[i, j_local] for i in {0,1}, j in {1,2} (j_local = j-1)
    # V[j*p+k, c] = b_pi[j_local, k_local] for j in {1,2}, k in {2s, 2s+1} (k_local = k-2s)
    # W[k*n+i, c] = 1 for the (k,i) pairs that are "active"
    
    # But W[k*n+i, c] should be constant for a given rank-1 term? No, that can't be right.
    
    # I think the issue is that the tensor T[i,j,k]=1 is being decomposed, and each rank-1 term 
    # is a product of three vectors. For schoolbook, each term is a "basis" term.
    
    # Let me just try to verify by construction. I'll build the rank-1 terms for Strassen and check.
    
    # Actually, I realize I should just implement it and verify numerically.
    
    # Let me define the 7 Strassen terms for a 2x2 * 2x2 block.
    # For block s, with local indices:
    # A_local[i, j_local] = A[i, 1+j_local] for i in {0,1}, j_local in {0,1}
    # B_local[j_local, k_local] = B[1+j_local, 2s+k_local] for j_local in {0,1}, k_local in {0,1}
    
    # The 7 products:
    # P1: A part: [1, 1; 0, 1] (i.e., A[0,1] + A[1,2]), B part: [1, 0; 1, 1] (B[1,2s] + B[2,2s+1])
    # Wait, I need to be more precise.
    
    # Let me define a_pi as a 2x2 matrix (rows i=0,1; cols j_local=0,1) and b_pi as a 2x2 matrix (rows j_local=0,1; cols k_local=0,1)
    
    # P1: a = [[1, 0], [0, 1]], b = [[1, 0], [1, 1]]  => (A[0,1]+A[1,2]) * (B[1,2s]+B[2,2s+1])
    # Hmm, that's not right. Let me just list them:
    
    # P1 = (A[0,1] + A[1,2]) * (B[1,2s] + B[2,2s+1])
    #   a: A[0,1] has coeff 1, A[1,2] has coeff 1 => a = [[1,0],[0,1]] in (i, j_local) where j_local=j-1
    #   b: B[1,2s] has coeff 1, B[2,2s+1] has coeff 1 => b = [[1,0],[0,1]] in (j_local, k_local)
    
    # P2 = (A[1,1] + A[1,2]) * B[1,2s]
    #   a: A[1,1] coeff 1, A[1,2] coeff 1 => a = [[0,0],[1,1]]
    #   b: B[1,2s] coeff 1 => b = [[1,0],[0,0]]
    
    # P3 = A[0,1] * (B[2,2s] - B[1,2s])
    #   a: A[0,1] coeff 1 => a = [[1,0],[0,0]]
    #   b: B[2,2s] coeff 1, B[1,2s] coeff -1 => b = [[-1,0],[1,0]]
    
    # P4 = A[1,2] * (B[1,2s+1] - B[2,2s+1])
    #   a: A[1,2] coeff 1 => a = [[0,0],[0,1]]
    #   b: B[1,2s+1] coeff 1, B[2,2s+1] coeff -1 => b = [[0,1],[0,-1]]
    
    # P5 = (A[0,1] + A[0,2]) * B[2,2s+1]
    #   a: A[0,1] coeff 1, A[0,2] coeff 1 => a = [[1,1],[0,0]]
    #   b: B[2,2s+1] coeff 1 => b = [[0,0],[0,1]]
    
    # P6 = (A[1,1] - A[0,1]) * (B[1,2s] + B[1,2s+1])
    #   a: A[1,1] coeff 1, A[0,1] coeff -1 => a = [[-1,0],[1,0]]
    #   b: B[1,2s] coeff 1, B[1,2s+1] coeff 1 => b = [[1,1],[0,0]]
    
    # P7 = (A[0,2] - A[1,2]) * (B[2,2s] + B[2,2s+1])
    #   a: A[0,2] coeff 1, A[1,2] coeff -1 => a = [[0,1],[0,-1]]
    #   b: B[2,2s] coeff 1, B[2,2s+1] coeff 1 => b = [[0,0],[1,1]]
    
    # Now, for each Pi, in the tensor decomposition, we add one rank-1 term where:
    # U[i*m + (1+j_local), c] = a_pi[i, j_local] for i in {0,1}, j_local in {0,1}
    # V[(1+j_local)*p + (2s+k_local), c] = b_pi[j_local, k_local] for j_local in {0,1}, k_local in {0,1}
    # W[(2s+k_local)*n + i, c] = 1 for the (i, k_local) pairs... 
    
    # Wait, W[k*n+i, c] should be 1 for all (k,i) that are "touched" by this term?
    # No, that doesn't make sense. Let me think again.
    
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For a given rank-1 term c, it contributes U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] to the (i,j,k) entry.
    
    # For the Strassen term Pi with A-coefficients a and B-coefficients b:
    # It should contribute to the (i,j,k) entry the value a[i, j_local(j)] * b[j_local(j), k_local(k)] * W[k*n+i, c]
    # where j_local(j) = j-1 for j in {1,2}, and k_local(k) = k-2s for k in {2s, 2s+1}
    
    # But this only makes sense if W[k*n+i, c] is chosen such that the sum works out.
    
    # I think the correct interpretation is:
    # For each Strassen product Pi, we create ONE rank-1 term where:
    # U[i*m+j, c] = a_pi[i, j-1] for i in {0,1}, j in {1,2}
    # V[j*p+k, c] = b_pi[j-1, k-2s] for j in {1,2}, k in {2s, 2s+1}  
    # W[k*n+i, c] = 1 for all i in {0,1}, k in {2s, 2s+1}
    
    # But then the contribution to (i,j,k) would be a_pi[i,j-1] * b_pi[j-1,k-2s] * 1
    # And we need the sum over all rank-1 terms to equal 1 for each (i,j,k).
    
    # This is getting complicated. Let me just implement it and verify numerically.
    
    # Define the 7 Strassen terms
    strassen_terms = [
        # (a_coeffs: dict mapping (i, j_local) -> coeff, b_coeffs: dict mapping (j_local, k_local) -> coeff)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # P1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),             # P2
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}),            # P3
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}),            # P4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),             # P5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),  # P6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),  # P7
    ]
    
    # Add Strassen terms for each of the 5 blocks
    for s in range(5):
        for (a_coeffs, b_coeffs) in strassen_terms:
            # Add one rank-1 term
            for (i, j_local), val_a in a_coeffs.items():
                j = 1 + j_local  # global j index (1 or 2)
                U[i * m + j, col] += val_a
            for (j_local, k_local), val_b in b_coeffs.items():
                j = 1 + j_local  # global j index
                k = 2 * s + k_local  # global k index
                V[j * p + k, col] += val_b
            # W: set to 1 for all (i, k) in the block
            for i in range(n):
                for k_local in range(2):
                    k = 2 * s + k_local
                    W[k * n + i, col] += 1  # Should this be 1 or something else?
            
            col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c] for c in range(rank))
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        if not verify(U, V, W, n, m, p, rank):
            raise ValueError("Verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
