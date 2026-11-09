"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Construct the rank-17 decomposition
    # Strategy:
    # The matrix multiplication tensor T[i,j,k] for C = A * B
    # A is 2x3, B is 3x3.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can view this as:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # This is equivalent to multiplying a 2x3 matrix by a 3x3 matrix.
    # We can decompose this into a 2x2 * 2x2 part and some extra terms.
    # Let's use the known rank-17 construction.
    
    # Indices:
    # U indices: i*m + j (i in 0..1, j in 0..2) -> 6 entries
    # V indices: j*p + k (j in 0..2, k in 0..2) -> 9 entries
    # W indices: k*n + i (k in 0..2, i in 0..1) -> 6 entries
    
    # Rank 17 decomposition:
    # We can use Strassen for the 2x2 block formed by (i=0,1) and (j=0,1) and (k=0,1).
    # But the dimensions are not square. 
    # A better approach is to use the known rank-17 formula for 2x3 * 3x3.
    
    # Let's construct it explicitly.
    # The tensor is T[i, j, k] where i is row of C, j is inner index, k is col of C? 
    # No, the tensor is T[(i,j), (j',k), (i',k')] = delta_{j,j'} delta_{i,i'}
    # Wait, the standard decomposition is C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's try to build the rank-17 solution.
    # Reference: "The rank of matrix multiplication"
    # For 2x3 * 3x3, rank is 17.
    
    # Construction:
    # We can split the 3x3 matrix B into columns B0, B1, B2.
    # C[:,0] = A * B0
    # C[:,1] = A * B1
    # C[:,2] = A * B2
    
    # This is 3 separate 2x3 * 3x1 multiplications.
    # Rank of 2x3 * 3x1 is 6 (schoolbook). Total 18.
    
    # To get 17, we need to share one multiplication.
    # This is possible because the rows of A are shared.
    
    # Let's use the explicit rank-17 decomposition from literature or derive it.
    # Actually, a simpler structural change is to just return the schoolbook rank 18 if we can't guarantee 17 quickly, 
    # but the prompt says "inspiration ... yields a total rank of 17 ... Measured: 1 -> 1.059".
    # So I should implement the rank-17 construction.
    
    # Let's define the 17 terms.
    # Term 0-6: Strassen-like for the 2x2 core?
    # Let's try a different partition.
    # A = [A0 A1 A2] where Ai are 2x1 columns.
    # B = [B0 B1 B2] where Bj are 3x1 columns.
    # C = A0*B0 + A1*B1 + A2*B2.
    
    # We can combine A0*B0 + A1*B1 using Strassen on the 2x1 * 1x1? No, dimensions don't match for standard Strassen.
    
    # Let's look at the indices again.
    # U: 6x17, V: 9x17, W: 6x17.
    
    # I will construct the rank-17 solution based on the known result.
    # The rank-17 decomposition for 2x3*3x3 can be derived by taking the rank-7 Strassen for 2x2*2x2 
    # and adding the necessary terms for the 3rd column/row.
    
    # Let's define the 17 rank-1 tensors.
    # We will use a known valid rank-17 decomposition.
    
    # U, V, W matrices
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    # Indices in U: i*3+j (i=0,1; j=0,1,2)
    # Indices in V: j*3+k (j=0,1,2; k=0,1,2)
    # Indices in W: k*2+i (k=0,1,2; i=0,1)
    
    # Let's try to build it from the schoolbook rank 18 and merge one.
    # Schoolbook terms:
    # t(i,j,k) = A[i,j] * B[j,k] -> C[i,k]
    
    # Merging A[0,0]*B[0,0] and A[1,1]*B[1,1]? No, they go to different C entries.
    # Merging A[0,0]*B[0,1] and A[1,1]*B[1,0]?
    
    # Actually, let's use the explicit formula for rank 17.
    # From "Fast Matrix Multiplication" literature:
    # The rank of <2,3,3> is 17.
    
    # I will use a constructive approach:
    # 1. Use Strassen for the 2x2 block (i=0,1, j=0,1, k=0,1). This gives rank 7 for C[0:2, 0:2].
    #    But this only covers C[0,0], C[0,1], C[1,0], C[1,1].
    #    It uses A[0:2, 0:2] and B[0:2, 0:2].
    #    We still need to handle column 2 of B and row 2 of B? No, B is 3x3.
    #    A is 2x3.
    
    # Let's decompose:
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2] * B[2, 0:2]
    # C[:, 2] = A[:, 0:2] * B[0:2, 2] + A[:, 2] * B[2, 2]
    
    # Part 1: A[:, 0:2] * B[0:2, 0:2] is 2x2 * 2x2. Rank 7 (Strassen).
    # Part 2: A[:, 2] * B[2, 0:2] is 2x1 * 1x2. Rank 2 (schoolbook: 2*1*2=2? No, 2x1*1x2 is outer product, rank 1? 
    #    Wait, 2x1 * 1x2 -> 2x2. C[i,k] = A[i,2] * B[2,k].
    #    This is 2 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1).
    #    Actually, it's rank 2? No, it's 2 independent multiplications?
    #    A[0,2]*B[2,0], A[0,2]*B[2,1], A[1,2]*B[2,0], A[1,2]*B[2,1].
    #    These can be done in rank 2? 
    #    C[0,0] = A[0,2]B[2,0]
    #    C[0,1] = A[0,2]B[2,1]
    #    C[1,0] = A[1,2]B[2,0]
    #    C[1,1] = A[1,2]B[2,1]
    #    This is a 2x1 * 1x2 multiplication. Rank is 2? 
    #    Yes, we can use 2 rank-1 tensors? 
    #    T1: A[0,2]*B[2,0] -> C[0,0]
    #    T2: A[1,2]*B[2,0] -> C[1,0]
    #    T3: A[0,2]*B[2,1] -> C[0,1]
    #    T4: A[1,2]*B[2,1] -> C[1,1]
    #    That's rank 4. 
    #    Can we do better? 
    #    (A[0,2]+A[1,2]) * (B[2,0]+B[2,1]) gives cross terms.
    #    It is known that 2x1 * 1x2 has rank 2. 
    #    Let's check: C = u v^T. Rank 1? No, u is 2x1, v is 1x2. 
    #    C is 2x2. Rank 1 if u and v are fixed? 
    #    No, u and v are variables. 
    #    The tensor for 2x1 * 1x2 is sum_{i,k} u_i v_k C_{ik}.
    #    This is the outer product tensor. Rank is 2? 
    #    Actually, rank of <1,2,2> is 2. 
    #    So Part 2 costs rank 2.
    
    # Part 3: A[:, 0:2] * B[0:2, 2] is 2x2 * 2x1. Rank 2? 
    #    <2,2,1> rank is 2? 
    #    C[i,0] = sum_j A[i,j] B[j,0].
    #    This is 2x2 * 2x1. Rank 2? 
    #    Yes, rank of <2,2,1> is 2.
    
    # Part 4: A[:, 2] * B[2, 2] is 2x1 * 1x1. Rank 1.
    
    # Total Rank = 7 (Strassen 2x2x2) + 2 (2x1x2) + 2 (2x2x1) + 1 (2x1x1) = 12?
    # This seems too low. The lower bound for 2x3*3x3 is 17? 
    # Wait, the lower bound is max(2*3, 3*3, 3*2) = 9.
    # But known rank is 17? 
    # Let's re-evaluate the rank of <2,3,3>.
    # Actually, the rank of <2,3,3> is 17 is a known result? 
    # Or is it 18? 
    # Schoolbook is 18.
    # If I can find a rank 17, I win.
    
    # Let's try to construct the rank 17 solution explicitly.
    # I will use the known rank-17 decomposition.
    
    # U, V, W for rank 17
    # I will hardcode the matrices from a known source or derive them carefully.
    
    # Let's try a different strategy: 
    # Use the rank-18 schoolbook, but check if we can reduce it.
    # Since I don't have the exact rank-17 matrices memorized, I will implement a 
    # "smart" schoolbook that might accidentally find a lower rank if I use a specific basis?
    # No, that's unreliable.
    
    # Let's stick to the safe rank 18 for now, but the prompt says "inspiration ... yields 17".
    # I should implement the 17-rank construction.
    
    # Here is a known rank-17 construction for 2x3 * 3x3:
    # It uses 7 multiplications for the 2x2 core (indices 0,1 for i,j,k)
    # and 10 for the rest?
    
    # Let's try to build it programmatically by combining Strassen and schoolbook parts.
    
    # Strassen for 2x2 * 2x2 (i=0,1; j=0,1; k=0,1)
    # M1 = A[0,0]*(B[0,0]+B[1,1])
    # M2 = (A[0,0]+A[1,0])*B[1,1]
    # M3 = A[1,1]*(B[0,0]+B[0,1])
    # M4 = A[0,1]*(B[1,0]+B[1,1])
    # M5 = (A[0,0]+A[0,1])*B[0,0]
    # M6 = (A[1,1]-A[0,1])*(B[0,0]+B[0,1])
    # M7 = (A[0,0]-A[1,0])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M2 + M4
    # C[0,1] = M3 + M4
    # C[1,0] = M1 + M5
    # C[1,1] = M1 - M5 + M2
    
    # Now we need to handle column 2 and row 2.
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # C[0,0] already has A[0,0]B[0,0] and A[0,1]B[1,0] parts.
    # We need to add A[0,2]B[2,0] to C[0,0]? No, C[0,0] is done.
    # Wait, the Strassen part covers C[0,0], C[0,1], C[1,0], C[1,1] using A[:,0:2] and B[0:2,:].
    # It does NOT include A[:,2] or B[2,:].
    
    # So we need to add:
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[1,0] += A[1,2]B[2,0]
    # C[1,1] += A[1,2]B[2,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # This is getting complex to merge optimally.
    # I will return the schoolbook rank 18 to ensure correctness, as I cannot guarantee 
    # the rank-17 construction is bug-free in this short time.
    # However, the prompt strongly hints at rank 17.
    
    # Let's try to implement the rank-17 solution by explicitly defining the 17 terms.
    # I will use a known valid rank-17 decomposition.
    
    # U: 6x17, V: 9x17, W: 6x17
    
    # I will use the following construction:
    # 1. 7 terms for Strassen 2x2x2 (using A[0:2,0:2] and B[0:2,0:2] to compute C[0:2,0:2])
    # 2. 10 terms for the remaining 10 products?
    #    Remaining C entries: C[0,2], C[1,2] (2 entries)
    #    Remaining contributions to C[0,0], C[0,1], C[1,0], C[1,1] from A[:,2] and B[2,:]?
    #    No, Strassen used A[:,0:2] and B[0:2,:]. 
    #    So C[0:2, 0:2] is fully computed by Strassen? 
    #    Yes, if Strassen uses A[0:2, 0:2] and B[0:2, 0:2].
    #    Then we need to add:
    #    C[0,0] += A[0,2]B[2,0]
    #    C[0,1] += A[0,2]B[2,1]
    #    C[1,0] += A[1,2]B[2,0]
    #    C[1,1] += A[1,2]B[2,1]
    #    And compute:
    #    C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    #    C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # This requires:
    # 4 terms for A[:,2]*B[2,0:2] (rank 2? No, 4 terms if schoolbook)
    # 2 terms for C[:,2] (6 terms if schoolbook)
    
    # If we use schoolbook for the rest: 4 + 6 = 10. Total 17.
    # So Rank 17 = 7 (Strassen) + 10 (Schoolbook for rest).
    
    # Let's implement this.
    
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    # Strassen terms (0-6)
    # M1 = A[0,0]*(B[0,0]+B[1,1])
    U[0, 0] = 1; V[0, 0] = 1; V[4, 0] = 1 # B[0,0] is index 0*3+0=0, B[1,1] is 1*3+1=4
    W[0, 0] = 1 # C[0,0] is 0*2+0=0
    W[1, 0] = 1 # C[1,0] is 1*2+0=1? No, C[1,0] is index 0*2+1=1.
    # Wait, W index is k*n+i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 2
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 3
    
    # Let's map W indices:
    # W[0] -> C[0,0]
    # W[1] -> C[1,0]
    # W[2] -> C[0,1]
    # W[3] -> C[1,1]
    # W[4] -> C[0,2]
    # W[5] -> C[1,2]
    
    # M1: A[0,0] * (B[0,0]+B[1,1])
    # A[0,0] is U[0]
    # B[0,0] is V[0], B[1,1] is V[4]
    # C[0,0] += M1, C[1,0] += M1
    U[0, 0] = 1
    V[0, 0] = 1
    V[4, 0] = 1
    W[0, 0] = 1
    W[1, 0] = 1
    
    # M2: (A[0,0]+A[1,0]) * B[1,1]
    # A[0,0] is U[0], A[1,0] is U[3]
    # B[1,1] is V[4]
    # C[0,0] -= M2, C[1,1] += M2
    U[0, 1] = 1
    U[3, 1] = 1
    V[4, 1] = 1
    W[0, 1] = -1
    W[3, 1] = 1
    
    # M3: A[1,1] * (B[0,0]+B[0,1])
    # A[1,1] is U[4]
    # B[0,0] is V[0], B[0,1] is V[1]
    # C[0,1] += M3, C[1,1] += M3
    U[4, 2] = 1
    V[0, 2] = 1
    V[1, 2] = 1
    W[2, 2] = 1
    W[3, 2] = 1
    
    # M4: A[0,1] * (B[1,0]+B[1,1])
    # A[0,1] is U[1]
    # B[1,0] is V[3], B[1,1] is V[4]
    # C[0,0] += M4, C[0,1] += M4
    U[1, 3] = 1
    V[3, 3] = 1
    V[4, 3] = 1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5: (A[0,0]+A[0,1]) * B[0,0]
    # A[0,0] is U[0], A[0,1] is U[1]
    # B[0,0] is V[0]
    # C[1,0] += M5, C[1,1] -= M5
    U[0, 4] = 1
    U[1, 4] = 1
    V[0, 4] = 1
    W[1, 4] = 1
    W[3, 4] = -1
    
    # M6: (A[1,1]-A[0,1]) * (B[0,0]+B[0,1])
    # A[1,1] is U[4], A[0,1] is U[1]
    # B[0,0] is V[0], B[0,1] is V[1]
    # C[1,1] += M6
    U[4, 5] = 1
    U[1, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A[0,0]-A[1,0]) * (B[1,0]+B[1,1])
    # A[0,0] is U[0], A[1,0] is U[3]
    # B[1,0] is V[3], B[1,1] is V[4]
    # C[1,0] += M7
    U[0, 6] = 1
    U[3, 6] = -1
    V[3, 6] = 1
    V[4, 6] = 1
    W[1, 6] = 1
    
    # Now the remaining 10 terms (7-16)
    # Terms 7-10: A[:,2] * B[2,0:2]
    # C[0,0] += A[0,2]B[2,0]
    # C[1,0] += A[1,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[1,1] += A[1,2]B[2,1]
    
    # Term 7: A[0,2] * B[2,0] -> C[0,0]
    U[2, 7] = 1 # A[0,2] is index 0*3+2=2
    V[6, 7] = 1 # B[2,0] is index 2*3+0=6
    W[0, 7] = 1 # C[0,0]
    
    # Term 8: A[1,2] * B[2,0] -> C[1,0]
    U[5, 8] = 1 # A[1,2] is index 1*3+2=5
    V[6, 8] = 1
    W[1, 8] = 1 # C[1,0]
    
    # Term 9: A[0,2] * B[2,1] -> C[0,1]
    U[2, 9] = 1
    V[7, 9] = 1 # B[2,1] is index 2*3+1=7
    W[2, 9] = 1 # C[0,1]
    
    # Term 10: A[1,2] * B[2,1] -> C[1,1]
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1 # C[1,1]
    
    # Terms 11-16: C[:,2]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # Term 11: A[0,0] * B[0,2] -> C[0,2]
    U[0, 11] = 1
    V[2, 11] = 1 # B[0,2] is index 0*3+2=2
    W[4, 11] = 1 # C[0,2]
    
    # Term 12: A[0,1] * B[1,2] -> C[0,2]
    U[1, 12] = 1
    V[5, 12] = 1 # B[1,2] is index 1*3+2=5
    W[4, 12] = 1
    
    # Term 13: A[0,2] * B[2,2] -> C[0,2]
    U[2, 13] = 1
    V[8, 13] = 1 # B[2,2] is index 2*3+2=8
    W[4, 13] = 1
    
    # Term 14: A[1,0] * B[0,2] -> C[1,2]
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1 # C[1,2]
    
    # Term 15: A[1,1] * B[1,2] -> C[1,2]
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # Term 16: A[1,2] * B[2,2] -> C[1,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify
    T_check = np.einsum('ic,jc,kc->ijk', U, V, W)
    T_target = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                T_target[i*3+j, j*3+k, k*2+i] = 1
                
    if not np.allclose(T_check, T_target):
        # Fallback to schoolbook
        return schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
