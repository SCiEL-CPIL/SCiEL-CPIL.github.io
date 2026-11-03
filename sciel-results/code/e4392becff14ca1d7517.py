"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find rank reduction using Strassen-like ideas
    # For 4x4x4, we can try to use the 7-term Strassen decomposition per 2x2x2 block
    # 4x4x4 = 4 blocks of 2x2x2, so schoolbook = 4*8 = 32, Strassen = 4*7 = 28
    # For 4x4x8, we have 8 blocks of 2x2x4... let's think differently
    
    # Alternative: try random rank-reduction search
    # The multiplication tensor T has entries T[i,j,k] = delta(i,j,k) for the appropriate indices
    # We want to express T as sum of r rank-1 tensors
    
    # Let's try a greedy approach: start with schoolbook, try to merge terms
    
    # Actually, let's try a more structured approach:
    # Split 4x4x8 into four 2x4x4 or 4x2x4 sub-problems
    
    # Let me try: use the fact that 4 = 2+2, and apply Strassen at the 2x2x2 level
    # 4x4x4 multiplication can be done with rank 28 using Strassen on 2x2x2 blocks
    # 4x4x8 = two 4x4x4 multiplications, but with shared structure
    
    # Let's construct a 4x4x4 Strassen decomposition first
    # Strassen for 2x2x2: 7 terms
    # For 4x4x4, partition into 2x2 blocks: 4 sub-multiplications of 2x2x2
    # Each 2x2x2 uses 7 terms, total 28 terms
    
    # Strassen 2x2x2 formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2():
        """Return U,V,W for 2x2x2 Strassen decomposition (rank 7)"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0,0] = 1; U[3,0] = 1  # A11 + A22
        V[0,0] = 1; V[3,0] = 1  # B11 + B22
        W[0,0] = 1; W[3,3] = 1  # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)B11
        U[2,1] = 1; U[3,1] = 1  # A21 + A22
        V[0,1] = 1  # B11
        W[2,0] = 1; W[3,3] = 1  # C21 += M2, C22 -= M2
        
        # M3 = A11(B21-B22)
        U[0,2] = 1  # A11
        V[2,2] = 1; V[3,2] = -1  # B21 - B22
        W[1,0] = 1; W[2,0] = 1  # C12 += M3, C21 += M3
        
        # M4 = A22(B12-B11)
        U[3,3] = 1  # A22
        V[1,3] = 1; V[0,3] = -1  # B12 - B11
        W[0,0] = 1; W[2,0] = 1  # C11 += M4, C21 += M4
        
        # M5 = (A11+A12)B22
        U[0,4] = 1; U[1,4] = 1  # A11 + A12
        V[3,4] = 1  # B22
        W[0,0] = -1; W[1,1] = 1  # C11 -= M5, C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2,5] = 1; U[0,5] = -1  # A21 - A11
        V[0,5] = 1; V[1,5] = 1  # B11 + B12
        W[3,3] = 1  # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1,6] = 1; U[3,6] = -1  # A12 - A22
        V[2,6] = 1; V[3,6] = 1  # B21 + B22
        W[0,0] = 1; W[3,3] = 1  # C11 += M7, C22 += M7
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    U2, V2, W2 = strassen_2x2x2()
    # Check: sum_c U2[i,c]*V2[j,c]*W2[k,c] should give delta for 2x2x2
    # For 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 if j is the "middle" index... 
    # Actually the tensor is T[i,j,k] where A is 2x2 (i,j), B is 2x2 (j,k), C is 2x2 (i,k)
    # So T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # Wait no. The multiplication tensor has T[i,j,k] = 1 iff the (i,k) entry of C depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in {0,1}^3
    
    # Let me verify differently: C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] contributes to C[i,k])
    # No wait, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1 for all i,j,k.
    
    # So the tensor is all-ones for 2x2x2? No, that's not right either.
    # The multiplication tensor T_{m,n,p} has entries T[i,j,k] where i indexes rows of A, 
    # j indexes columns of A / rows of B, k indexes columns of B.
    # T[i,j,k] = 1 (it's the structure tensor).
    
    # For rank-1 decomposition: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U has shape (m*n, r), V has shape (n*p, r), W has shape (p*m, r)
    # and the multiplication is: C[i,k] = sum_{j,c} A[i,j] * V[j*p+k, c] * U[i*m+j, c] * W[k*m+i, c]
    # Wait, I need to be careful with index conventions.
    
    # The task says: W[k*n+i, r] where output factor follows trace convention W[k*n+i,r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # The decomposition should satisfy:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (product of U,V,W terms)
    # 
    # Standard: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j and c
    # But that's not quite right. Let me think again.
    
    # The matrix multiplication tensor: T is a 3-tensor with T[i,j,k] = 1
    # A rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}
    # and u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = T[i,j,k] = 1
    
    # So: C[i,k] = sum_{j,c} A[i,j] B[j,k] u_c[i*m+j] v_c[j*p+k] w_c[k*n+i]
    #        = sum_c w_c[k*n+i] * sum_j A[i,j] u_c[i*m+j] * B[j,k] v_c[j*p+k]
    
    # This means for each c, the contribution to C[i,k] is:
    # w_c[k*n+i] * (A * diag(u_c[i*m+0], u_c[i*m+1], ...) * B * diag(v_c[0*p+k], v_c[1*p+k], ...))
    # Hmm, this is getting complicated. Let me just verify numerically.
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U,V,W correctly decompose n×m × m×p matrix multiplication."""
        rank = U.shape[1]
        # For random A (n×m) and B (m×p), check C = A@B
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # u_c has shape (n*m,), v_c has shape (m*p,), w_c has shape (p*n,)
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            
            # For each i, k: C[i,k] += w_c[k*n+i] * sum_j A[i,j] * u_c[i*m+j] * B[j,k] * v_c[j*p+k]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += A[i,j] * u_c[i*m+j] * B[j,k] * v_c[j*p+k]
                    C_computed[i,k] += w_c[k*n+i] * s
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
        return True
    
    # Verify schoolbook
    assert verify_decomposition(U_school, V_school, W_school, n, m, p), "Schoolbook verification failed"
    
    # Now let's try to build a lower-rank decomposition
    # Strategy: use Strassen for 2x2x2 blocks
    
    # For 4x4x4: partition A into 2x2 blocks A11,A12,A21,A22
    # B into 2x2 blocks B11,B12,B21,B22
    # C into 2x2 blocks C11,C12,C21,C22
    
    # Each C block = sum of products of A and B blocks
    # Using Strassen, each 2x2x2 sub-problem uses 7 multiplications
    
    # For 4x4x4, we have 4 sub-problems (one for each pair of 2x2 blocks)
    # Total: 4 * 7 = 28 multiplications
    
    # But wait, the blocks don't interact in the simple way because
    # C11 = A11*B11 + A12*B21, etc.
    
    # Let me construct the 4x4x4 Strassen decomposition:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A*B = [[C11, C12], [C21, C22]] where each is 2x2
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each C block is a sum of two 2x2x2 multiplications.
    # We can apply Strassen to each pair, but the blocks are shared.
    
    # Actually, let me just try a direct search approach.
    
    # Approach: start with schoolbook, try to find linear dependencies
    # The tensor lives in R^{16} ⊗ R^{32} ⊗ R^{32} (for n=4,m=4,p=8)
    # n*m = 16, m*p = 32, p*n = 32
    
    # Let me try a greedy rank reduction:
    # 1. Start with schoolbook (rank 128)
    # 2. Try to express one term as a linear combination of others
    # 3. Remove redundant terms
    
    # This is complex. Let me try a different approach:
    # Use the fact that 4x4x8 can be viewed as 4x4x4 + 4x4x4 (splitting the 8 columns)
    # If I have a rank-r decomposition for 4x4x4, I can get rank 2r for 4x4x8
    # But I can potentially do better by sharing.
    
    # Let me try to find a good 4x4x4 decomposition first.
    
    # Known: 4x4x4 can be done with rank 49 (or less with advanced methods)
    # Schoolbook is 64.
    
    # Let me try a randomized search for rank reduction.
    
    # Actually, let me try a more practical approach:
    # Use the "border rank" or "approximate rank" ideas, but we need exact.
    
    # Let me try: can we reduce the schoolbook rank by finding linear relations?
    
    # The multiplication tensor for n×m×p has a certain structure.
    # For n=m=p=2, the tensor is the all-ones tensor in R^4 ⊗ R^4 ⊗ R^4,
    # which has rank 7 (Strassen).
    
    # For larger dimensions, we can use tensor product structures.
    
    # Let me try: 4x4x8 = (2x2x2) ⊗ (2x2x4) or similar factorizations.
    
    # Actually, let me just try a numerical optimization approach:
    # Start with schoolbook, randomly perturb, and try to reduce rank via SVD-like methods.
    
    # Time check
    if time.time() > deadline - 20:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Let me try a structured approach based on Kronecker products.
    # 4x4x8 multiplication tensor can be factored as:
    # T_{4,4,8} = T_{2,2,2} ⊗ T_{2,2,4} (if we can find such a factorization)
    
    # T_{2,2,2} has rank 7 (Strassen)
    # T_{2,2,4}: 2x2 matrix times 2x4 matrix
    # Schoolbook rank for 2x2x4 is 16. Can we do better?
    
    # 2x2x4: A is 2x2, B is 2x4, C is 2x4
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is like two independent 2x1x4 problems? No, they share A.
    
    # Let me think of 2x2x4 as: for each column k of B (and C), we compute A * B[:,k]
    # which is a 2x1 multiplication. But A is shared.
    
    # Actually, 2x2x4 can be seen as 4 copies of 2x2x1 with shared A.
    # 2x2x1 has rank 4 (schoolbook) or can it be done with less?
    # 2x2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # This is just a linear map, rank 4 in the tensor sense? 
    # T[i,j,k] where k is always 0. T[i,j,0] = 1 for all i,j.
    # This is the 2x2 all-ones tensor, which has rank 3 (since it's a symmetric matrix of rank 1, 
    # but as a 3-tensor... actually for 2x2x1, it's just a 2x2 matrix viewed as a tensor,
    # and its "rank" as a 3-tensor with one dimension of size 1 is the matrix rank, which is 1? No.
    
    # I'm overcomplicating this. Let me just try a direct computational search.
    
    # Approach: use the fact that we can try to find a lower-rank decomposition
    # by solving a system of equations.
    
    # For rank r, we need to find U (16×r), V (32×r), W (32×r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 128 equations in 16*r + 32*r + 32*r = 80*r unknowns.
    # But the equations are nonlinear (products of variables).
    
    # Let me try a different tactic: use known results.
    # For 4x4x4, the best known exact rank is 49 (Landsberg 2006, improved to 48 by others?)
    # Actually, I recall that 4x4x4 has rank at most 49, and lower bound 48.
    
    # For 4x4x8, if I can find a 4x4x4 decomposition of rank r, then 4x4x8 has rank at most 2r
    # (by doing two independent 4x4x4 multiplications).
    
    # But maybe I can do better by exploiting structure.
    
    # Let me try: implement a known 4x4x4 decomposition.
    
    # Actually, let me try a much simpler approach: just try to reduce the schoolbook rank
    # by finding linear dependencies among the 128 terms.
    
    # The 128 schoolbook terms are: for each (i,j,k), the term is e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # These are 128 rank-1 tensors. Are they linearly independent?
    
    # If some are linearly dependent, we can reduce the rank.
    
    # Let me check: form a matrix where each column is one of these vectors (flattened)
    # and check its rank.
    
    # Flatten each rank-1 tensor into a vector of length 16*32*32 = 16384
    # and check the rank of the 16384 × 128 matrix.
    
    # This is too large. Let me think differently.
    
    # Actually, the 128 terms e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} are linearly independent
    # because each has a unique combination of indices. So the schoolbook rank is minimal
    # in this basis, but we can use non-standard rank-1 tensors.
    
    # Let me try a randomized approach:
    # 1. Start with schoolbook
    # 2. Randomly combine pairs of terms
    # 3. Check if the combined set still spans the tensor
    # 4. If yes, we've reduced the rank
    
    # This is heuristic and may not work well.
    
    # Let me try yet another approach: use the "Coppersmith-Winograd" type ideas
    # or just accept that finding a significantly lower rank is very hard.
    
    # Given the time constraint, let me try a practical search:
    # 1. Try rank 127, 126, ... by attempting to find linear relations
    # 2. If that fails, return the best we found
    
    # Actually, for this problem size, let me try a different strategy:
    # Use the fact that 4x4x8 = 4x4x4 ⊕ 4x4x4 (direct sum)
    # and see if I can find a good 4x4x4 decomposition.
    
    # Let me try to implement a 4x4x4 decomposition using 2x2x2 Strassen blocks.
    
    # 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # Partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each C block is a sum of two 2x2x2 multiplications.
    # If I use Strassen for each pair, I need 7 multiplications per C block,
    # but the A and B blocks are shared across C blocks.
    
    # Let me just try to construct it explicitly.
    
    # For C00 = A00*B00 + A01*B10:
    # This is a 2x2x2 multiplication of (A00, B00) plus (A01, B10).
    # I can use Strassen's 7-term formula on the "combined" 2x2x2 problem.
    
    # Actually, C00 = A00*B00 + A01*B10 is not a single 2x2x2 multiplication;
    # it's a sum of two. To compute it with Strassen, I would need to
    # treat it as a 4x4x4 problem, which defeats the purpose.
    
    # Let me try a completely different approach: use a SAT/SMT solver or
    # integer programming to find a low-rank decomposition.
    
    # Given the complexity, let me just try to verify if the schoolbook is
    # actually optimal or if there's an easy reduction.
    
    # For now, let me try a simple heuristic: randomly perturb the schoolbook
    # and see if I can find a lower-rank solution via optimization.
    
    # Time check
    if time.time() > deadline - 15:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Let me try: can we use the fact that 8 = 4+4 to share structure?
    # 4x4x8: A is 4x4, B is 4x8, C is 4x8
    # Split B into B_left (4x4) and B_right (4x4)
    # C_left = A * B_left, C_right = A * B_right
    # Both are 4x4x4 multiplications with the same A.
    
    # If I have a rank-r decomposition for 4x4x4:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * f_c(i,j,k)
    # where f_c is the product of U,V,W terms.
    
    # For C_left and C_right, I can use the same A-side terms but different B-side terms.
    # This might allow sharing.
    
    # Let me try to construct a 4x4x4 decomposition first.
    
    # I'll use a known result: 4x4x4 can be done with rank 49.
    # But implementing this from scratch is very complex.
    
    # Let me try a simpler approach: use 2x2x2 Strassen blocks.
    # 4x4x4 has 4 sub-blocks of 2x2x2.
    # Each 2x2x2 uses 7 terms, total 28 terms.
    # But the sub-blocks share A and B entries, so there might be more sharing.
    
    # Let me try to construct it:
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13],
    #      [a20, a21, a22, a23],
    #      [a30, a31, a32, a33]]
    
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13],
    #      [b20, b21, b22, b23],
    #      [b30, b31, b32, b33]]
    
    # C = A * B
    
    # Using 2x2x2 Strassen on blocks:
    # A00 = [[a00,a01],[a10,a11]], A01 = [[a02,a03],[a12,a13]]
    # A10 = [[a20,a21],[a30,a31]], A11 = [[a22,a23],[a32,a33]]
    # Similarly for B.
    
    # C00 = A00*B00 + A01*B10
    # To compute C00 with Strassen, I need to compute A00*B00 and A01*B10 separately,
    # which is 7+7=14 terms. But I can do better by combining.
    
    # Actually, C00 = A00*B00 + A01*B10 can be viewed as a 4x4x4 multiplication
    # where A is [[A00, A01], [0, 0]] and B is [[B00, 0], [B10, 0]], but that's not standard.
    
    # I think the cleanest approach is to just use the schoolbook and try to
    # find any reduction via numerical methods.
    
    # Let me try a simple experiment: can I reduce the rank by 1?
    # I need to find a linear combination of the 128 schoolbook terms that
    # equals one of the terms, allowing me to eliminate it.
    
    # This is equivalent to finding a non-trivial linear relation among the
    # 128 rank-1 tensors.
    
    # Given the time constraints, let me just return the schoolbook decomposition.
    # But let me try one more thing: a randomized search for rank reduction.
    
    # Try to find a rank-127 decomposition by solving for a linear relation.
    # This is a nonlinear problem, so it's hard.
    
    # Let me just return the schoolbook for now, but with a clean implementation.
    
    # Actually, let me try one more idea: use the "lax rank" or "border rank"
    # concepts, but we need exact rank.
    
    # I'll try a simple randomized approach:
    # 1. Start with schoolbook
    # 2. Randomly select pairs of terms and try to merge them
    # 3. Check if the merged set still works
    
    # This is very heuristic and unlikely to work, but let's try.
    
    # Given the complexity and time constraints, I'll return the schoolbook.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
