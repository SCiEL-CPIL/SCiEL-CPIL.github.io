"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # We'll build the decomposition using a block approach:
    # Split m=4 into two blocks of size 2: m_blocks = [2, 2]
    # For each m-block, split p=7 into blocks: [4, 3]
    # For each (m_block, p_block), we have a sub-problem of size n x m_i x p_j
    # Apply Strassen where the sub-problem is 2x2x2, otherwise schoolbook.
    
    # Actually, let's try a cleaner approach:
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # We decompose into rank-1 terms.
    
    # Strategy: Use Strassen for 2x2x2 sub-blocks within the 3x4x7 structure.
    # Split i in {0,1,2}, j in {0,1,2,3}, k in {0,...,6}
    # Group j into pairs: (0,1) and (2,3)
    # Group k into: (0,1), (2,3), (4,5,6)
    # Group i into: (0,1) and (2) -- but Strassen needs 2x2
    
    # Let me try a different approach: enumerate all possible 2x2x2 sub-cubes
    # and apply Strassen to them, then handle the rest with schoolbook.
    
    # Actually, the cleanest approach for a guaranteed improvement:
    # Use the fact that for any 2x2x2 sub-tensor, we can replace 8 schoolbook
    # terms with 7 Strassen terms.
    
    # Let's build the full schoolbook first, then try to merge.
    # But merging is complex. Instead, let's directly construct using Strassen
    # for all 2x2x2 blocks where they fit.
    
    # For 3x4x7:
    # i can be 0,1,2 -> pairs: (0,1) only for 2x2
    # j can be 0,1,2,3 -> pairs: (0,1), (2,3)
    # k can be 0,...,6 -> pairs: (0,1), (2,3), (4,5) and single (6)
    
    # This gets complicated. Let me use a simpler but effective approach:
    # Directly construct using known Strassen decomposition for 2x2x2 blocks
    # embedded in the larger tensor.
    
    # Alternative: Use a computational approach to find a low-rank decomposition.
    # Given time constraints, let's try a heuristic: start with schoolbook and
    # try to find linear dependencies to reduce rank.
    
    # Actually, let me try the block decomposition properly.
    
    # The key insight: if we split m into two groups of 2, we get two independent
    # 3x2x7 sub-tensors. Each can be solved independently.
    # For 3x2x7, split p into groups: we can use Strassen-like for 2x2 sub-blocks.
    
    # Let me just implement a verified Strassen-based decomposition for specific
    # sub-blocks and schoolbook for the rest.
    
    # Start with schoolbook
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
    
    # Now try to reduce rank by finding Strassen patterns.
    # For each 2x2x2 sub-block (i0,i1) x (j0,j1) x (k0,k1), we can replace
    # 8 schoolbook terms with 7 Strassen terms.
    
    # Identify all 2x2x2 sub-blocks:
    # i: (0,1) only since n=3
    # j: (0,1), (2,3)
    # k: (0,1), (2,3), (4,5)
    
    # For each such sub-block, apply Strassen.
    # But the terms overlap with schoolbook, so we need to be careful.
    
    # Let's take a different approach: construct directly using Strassen for
    # the 2x2x2 blocks and schoolbook for the remaining terms.
    
    # Reset
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen 2x2x2 decomposition
    # For sub-block with i in {i0,i1}, j in {j0,j1}, k in {k0,k1}
    # The 7 Strassen products:
    # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
    # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
    # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    
    # C[i0,k0] = M1 + M4 - M5 + M7
    # C[i0,k1] = M3 + M5
    # C[i1,k0] = M2 + M4
    # C[i1,k1] = M1 - M2 + M3 + M6
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        """Add Strassen 7-term decomposition for 2x2x2 sub-block."""
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U1 = np.zeros(n*m)
        V1 = np.zeros(m*p)
        W1 = np.zeros(p*n)
        U1[i0*m+j0] = 1
        U1[i1*m+j1] = 1
        V1[j0*p+k0] = 1
        V1[j1*p+k1] = 1
        W1[k0*n+i0] = 1
        W1[k1*n+i1] = 1
        U_list.append(U1)
        V_list.append(V1)
        W_list.append(W1)
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U2 = np.zeros(n*m)
        V2 = np.zeros(m*p)
        W2 = np.zeros(p*n)
        U2[i1*m+j0] = 1
        U2[i1*m+j1] = 1
        V2[j0*p+k0] = 1
        W2[k0*n+i1] = 1
        W2[k1*n+i1] = -1
        U_list.append(U2)
        V_list.append(V2)
        W_list.append(W2)
        
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U3 = np.zeros(n*m)
        V3 = np.zeros(m*p)
        W3 = np.zeros(p*n)
        U3[i0*m+j0] = 1
        V3[j0*p+k1] = 1
        V3[j1*p+k1] = -1
        W3[k1*n+i0] = 1
        W3[k1*n+i1] = 1
        U_list.append(U3)
        V_list.append(V3)
        W_list.append(W3)
        
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U4 = np.zeros(n*m)
        V4 = np.zeros(m*p)
        W4 = np.zeros(p*n)
        U4[i1*m+j1] = 1
        V4[j1*p+k0] = 1
        V4[j0*p+k0] = -1
        W4[k0*n+i0] = 1
        W4[k0*n+i1] = 1
        U_list.append(U4)
        V_list.append(V4)
        W_list.append(W4)
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U5 = np.zeros(n*m)
        V5 = np.zeros(m*p)
        W5 = np.zeros(p*n)
        U5[i0*m+j0] = 1
        U5[i0*m+j1] = 1
        V5[j1*p+k1] = 1
        W5[k1*n+i0] = -1
        W5[k1*n+i1] = 1
        U_list.append(U5)
        V_list.append(V5)
        W_list.append(W5)
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U6 = np.zeros(n*m)
        V6 = np.zeros(m*p)
        W6 = np.zeros(p*n)
        U6[i1*m+j0] = 1
        U6[i0*m+j0] = -1
        V6[j0*p+k0] = 1
        V6[j0*p+k1] = 1
        W6[k0*n+i1] = 1
        W6[k1*n+i1] = 1
        U_list.append(U6)
        V_list.append(V6)
        W_list.append(W6)
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U7 = np.zeros(n*m)
        V7 = np.zeros(m*p)
        W7 = np.zeros(p*n)
        U7[i0*m+j1] = 1
        U7[i1*m+j1] = -1
        V7[j1*p+k0] = 1
        V7[j1*p+k1] = 1
        W7[k0*n+i0] = 1
        W7[k1*n+i0] = 1
        U_list.append(U7)
        V_list.append(V7)
        W_list.append(W7)
    
    # Schoolbook term for single (i,j,k)
    def add_schoolbook(i, j, k):
        U_s = np.zeros(n*m)
        V_s = np.zeros(m*p)
        W_s = np.zeros(p*n)
        U_s[i*m+j] = 1
        V_s[j*p+k] = 1
        W_s[k*n+i] = 1
        U_list.append(U_s)
        V_list.append(V_s)
        W_list.append(W_s)
    
    # Apply Strassen to 2x2x2 blocks:
    # i pairs: (0,1)
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5)
    
    strassen_applied = set()
    for i0, i1 in [(0, 1)]:
        for j0, j1 in [(0, 1), (2, 3)]:
            for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
                add_strassen(i0, i1, j0, j1, k0, k1)
                # Mark all 8 (i,j,k) in this block as covered
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            strassen_applied.add((i, j, k))
    
    # Add schoolbook for remaining (i,j,k) not covered by Strassen
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_applied:
                    add_schoolbook(i, j, k)
    
    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    # Verify the decomposition
    T_computed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_computed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    if not np.allclose(T_computed, T_expected, atol=1e-10):
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
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
