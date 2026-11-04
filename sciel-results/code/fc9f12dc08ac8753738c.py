<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.array_equal(T_recon, T.astype(np.int64))
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find a lower rank decomposition using structured approaches
    # Key idea: exploit the structure of 8 = 2*4 or 8 = 2*2*2
    
    # Approach: Try to use Strassen-like tricks on sub-blocks
    # For 2x2x2, Strassen gives rank 7 instead of 8
    # We can try to embed this into the 3x4x8 problem
    
    # Let's try a block decomposition where we split the p-dimension
    # p = 8 can be split into two groups of 4, or four groups of 2
    
    # For each pair of 2x2 sub-blocks in the p-dimension, we could potentially
    # use Strassen's algorithm to save one multiplication
    
    # However, the full 3x4x8 tensor doesn't decompose neatly into independent
    # 2x2x2 blocks because the i and j indices couple across all k values
    
    # Let's try a different approach: random search for linear combinations
    # that can replace multiple schoolbook terms
    
    # Given the complexity, let's try a few specific structured decompositions
    # and verify them
    
    # For now, return the schoolbook decomposition as it's guaranteed correct
    # Future work: implement proper Strassen embedding or other low-rank methods
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.array_equal(T_recon, T.astype(np.int64))
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Strategy: Use Strassen's algorithm on 2x2x2 sub-blocks
    # Split dimensions to create independent 2x2x2 problems where possible
    
    # Key insight: We can split the k-dimension (size 8) into 4 pairs of size 2
    # For each pair (k, k+1), we have a sub-problem involving those two k values
    # But the i and j dimensions couple across all k values, so we need to be careful
    
    # Alternative: Use the fact that rank(M_{n,m,p}) <= rank(M_{n1,m,p}) + rank(M_{n2,m,p})
    # where n = n1 + n2, by splitting the output matrix
    
    # Let's try: Split n=3 into n1=1, n2=2
    # rank(M_{3,4,8}) <= rank(M_{1,4,8}) + rank(M_{2,4,8})
    
    # For M_{1,4,8}: This is just a linear combination, rank = 4*8 = 32? No...
    # Actually M_{1,m,p} is trivial: C[k] = sum_j A[j]*B[j,k], rank = m*p
    
    # Better approach: Use known results for small cases and combine
    
    # Let me try a direct construction using Strassen on 2x2 blocks
    
    # For 2x2x2, Strassen's decomposition has rank 7 with the following structure:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For our 3x4x8 problem, let's try to decompose it into smaller sub-problems
    
    # Split m=4 into two groups of 2: m1=2, m2=2
    # rank(M_{3,4,8}) <= rank(M_{3,2,8}) + rank(M_{3,2,8}) = 2 * rank(M_{3,2,8})
    
    # For M_{3,2,8}: Split k=8 into four groups of 2
    # rank(M_{3,2,8}) <= 4 * rank(M_{3,2,2})
    
    # For M_{3,2,2}: This is multiplying a 3x2 matrix by a 2x2 matrix
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This can be seen as two independent 3x1x2 multiplications
    # rank(M_{3,2,2}) <= 2 * rank(M_{3,1,2}) = 2 * 3 * 2 = 12? 
    # But M_{3,1,2} is just C[i,k] = A[i]*B[k], which has rank 3*2=6 (outer product)
    # So rank(M_{3,2,2}) <= 2 * 6 = 12
    
    # Total: rank(M_{3,4,8}) <= 2 * 4 * 12 = 96 (same as schoolbook, not helpful)
    
    # Let me try a different split. Use Strassen directly on 2x2x2 sub-problems.
    
    # Actually, let's try a random search approach: start from schoolbook and try to
    # find linear dependencies that allow us to reduce the rank
    
    # The idea: if we can find a linear combination of the rank-1 tensors that equals zero,
    # we can remove one term from the decomposition
    
    # For now, let's try a structured approach using known low-rank decompositions
    
    # For 2x2x2, we know rank = 7. Let's embed this into our problem.
    
    # Consider splitting the 4x8 part into 2x2 blocks:
    # We have 2*4 = 8 blocks of size 2x2 in the (j,k) plane
    # For each i (3 values) and each (j_block, k_block) pair, we could potentially
    # use a Strassen-like decomposition
    
    # But this is getting complex. Let me try a simpler approach:
    # Use the fact that we can share computations across different i values
    
    # Practical approach: Try to find a decomposition with rank < 96 using
    # integer/half-integer entries
    
    # Let's try rank = 80, 72, 64, etc. and see if we can construct valid decompositions
    
    # For now, return schoolbook as a safe fallback
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I'm not making progress with this approach. Let me think more carefully about known results.

For matrix multiplication M_{n,m,p}, the rank is known for small cases:
- M_{2,2,2}: rank 7 (Strassen)
- M_{2,2,3}: rank 11 (best known)
- M_{2,2,4}: rank 14? or less?

Actually, let me look at this from a different angle. The key insight from the inspiration is that a program improved island 5 with a score of 1.091, meaning it found a rank of 96/1.091 ≈ 88. So there IS a better decomposition.

Let me try a systematic approach: use the structure of 8 = 4*2 and apply Strassen to 2x2x2 sub-problems.

Here's a concrete plan:
1. Split the k-dimension (size 8) into two groups of 4: K1 = {0,1,2,3}, K2 = {4,5,6,7}
2. For each group, we have a 3×4×4 sub-problem
3. For each 3×4×4, split m=4 into two groups of 2: M1 = {0,1}, M2 = {2,3}
4. This gives four 3×2×4 sub-problems
5. For each 3×2×4, split k=4 into two groups of 2: this gives eight 3×2×2 sub-problems
6. For each 3×2×2, we can use a rank-12 decomposition (schoolbook)

Total: 8 * 12 = 96 (same as schoolbook)

This doesn't help. I need to actually use Strassen's trick.

The key is that for 2×2×2, we can save one multiplication. So if I can identify 2×2×2 sub-problems within the 3×4×8 problem, I can save one rank per such sub-problem.

Let me try: identify all possible 2×2×2 sub-tensors and see if I can apply Strassen to them.

Actually, I think the right approach is to use a more sophisticated construction. Let me try to directly construct a lower-rank decomposition.

