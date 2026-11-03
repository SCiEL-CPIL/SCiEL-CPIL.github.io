"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = n * m * p       # 18
    
    # Build the matrix multiplication tensor T[i,j,k] = delta_{i,k} for A[i,j] * B[j,k] -> C[i,k]
    # In the standard tensor notation: T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the multiplication tensor M_{i,j,k} = 1 where A[i,j] and B[j,k] contribute to C[i,k]
    # So M[i,j,k] = 1 for all valid (i,j,k) - wait no.
    # 
    # The matrix multiplication tensor is: M[i,j,k] = delta_{i,k}... no.
    # 
    # Let me think carefully. We have A (n x m) * B (m x p) = C (n x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T is a 3-way tensor of dimensions (n, m, p) where T[i,j,k] = 1 
    # (since A[i,j]*B[j,k] contributes to C[i,k]).
    # 
    # The tensor decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where U is (n*m, rank)... wait, the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    #
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And "output factor follows the official trace convention W[k*n+i,r]"
    #
    # So U has rows indexed by (i,j) pairs: row index = i*m + j, for i in [0,n), j in [0,m)
    # V has rows indexed by (j,k) pairs: row index = j*p + k, for j in [0,m), k in [0,p)
    # W has rows indexed by (k,i) pairs: row index = k*n + i, for k in [0,p), i in [0,n)
    #
    # The decomposition condition: For all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Wait, that's not right either. Let me think about what tensor we're decomposing.
    #
    # The matrix multiplication tensor T has T[i,j,k] = 1 for the "multiplication" operation.
    # Actually, the standard formulation: the tensor for matrix mult is T[i,j,k] where
    # T[i,j,k] = 1 means the product A[i,j]*B[j,k] goes into C[i,k].
    # 
    # But actually every (i,j,k) triple has T[i,j,k] = 1 because A[i,j]*B[j,k] always 
    # contributes to C[i,k]. So T is the all-ones tensor of size n x m x p? No that's wrong.
    #
    # Let me reconsider. The bilinear map is: (A, B) -> C where C = A*B.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # As a trilinear form: <T, (A,B,C)> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # For this to equal the "inner product" that captures C = A*B, we need:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k] = sum_{i,k} C[i,k] * (sum_j A[i,j]*B[j,k])
    # = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    #
    # So T[i,j,k] = 1 for all i,j,k! The tensor is the all-ones tensor.
    #
    # Wait, that can't be right because then the rank would just be 1 (all-ones = 1*1*1).
    #
    # I'm confusing myself. Let me look at the schoolbook decomposition in the code:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    #
    # So for each (i,j,k) triple, there's a rank-1 term where:
    # U has 1 at row (i*m+j), V has 1 at row (j*p+k), W has 1 at row (k*n+i)
    #
    # The condition must be: for all (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    #
    # And T[i,j,k] = 1 for all i,j,k (the all-ones tensor).
    #
    # Hmm, but that means the rank-1 decomposition of the all-ones tensor... 
    # The all-ones tensor of size 2x3x3 has rank 1! Because T[i,j,k] = 1 = 1*1*1.
    #
    # I must be misunderstanding the tensor structure. Let me look more carefully.
    #
    # Actually, I think the issue is that the tensor is NOT all-ones. Let me reconsider.
    #
    # In the standard matrix multiplication tensor formulation:
    # The tensor M has M[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to the (i,k) entry of C.
    #
    # But actually, I think the correct formulation is different. The multiplication tensor
    # is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # This can be written as a tensor equation. The tensor T of size (n x p) x (n x m) x (m x p)
    # where T[(i,k), (i',j), (j',k')] = 1 if i=i', k=k', j=j'.
    #
    # So T[(i,k),(i',j),(j',k')] = delta_{i,i'} * delta_{k,k'} * delta_{j,j'}
    #
    # In the flattened indices:
    # Row of U: (i,j) -> index i*m+j
    # Row of V: (j,k) -> index j*p+k  
    # Row of W: (k,i) -> index k*n+i
    #
    # T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{i,i'}
    #
    # Wait, let me be more careful. We need:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{i,i'}
    #
    # Hmm, but the indices need to match properly. Let me think about it as:
    # The bilinear map takes (A, B) -> C.
    # A is a vector in R^{n*m} with A[i*m+j] = A[i,j]
    # B is a vector in R^{m*p} with B[j*p+k] = B[j,k]
    # C is a vector in R^{n*p} with C[k*n+i] = C[i,k] (trace convention: W[k*n+i,r])
    #
    # The map is: C[i,k] = sum_j A[i,j] * B[j,k]
    # In vector form: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    #
    # So the tensor T has T[k*n+i, i*m+j, j*p+k] = 1 (for the specific j that's summed over)
    #
    # More precisely: T[c, a, b] = 1 where c = k*n+i, a = i*m+j, b = j*p+k
    # And the condition is that for each (i,j,k): T[k*n+i, i*m+j, j*p+k] = 1
    # All other entries of T are 0.
    #
    # So T is a tensor of size (p*n) x (n*m) x (m*p) = 6 x 6 x 9
    # with T[k*n+i, i*m+j, j*p+k] = 1 for all valid i,j,k.
    #
    # The decomposition: T[c,a,b] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    #
    # For the schoolbook: for each (i,j,k), we have a rank-1 term:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # This gives T[k*n+i, i*m+j, j*p+k] += 1, which is correct.
    #
    # Now I understand the structure. Let me verify: the tensor T has exactly n*m*p = 18
    # nonzero entries, one for each (i,j,k) triple.
    
    # Let me verify the schoolbook works:
    # For a given (i,j,k): T[k*n+i, i*m+j, j*p+k] should equal 1.
    # In schoolbook, the rank-1 term for (i,j,k) contributes:
    #   U[i*m+j, r_{i,j,k}] * V[j*p+k, r_{i,j,k}] * W[k*n+i, r_{i,j,k}] = 1*1*1 = 1
    # And no other rank-1 term contributes to this (a,b,c) = (i*m+j, j*p+k, k*n+i) position
    # because each rank-1 term has exactly one nonzero in each of U, V, W.
    #
    # What about other positions? For example, T[k*n+i, i*m+j', j*p+k] where j' != j:
    # No rank-1 term has U[i*m+j', r] = 1 AND V[j*p+k, r] = 1 simultaneously in schoolbook,
    # because each rank-1 term fixes a specific (i,j,k) triple.
    # So T[k*n+i, i*m+j', j*p+k] = 0 for j' != j. Correct!
    
    # Now, can we do better than rank 18?
    # The known lower bound for 2x3x3 matrix multiplication is 17.
    # There are known constructions achieving rank 17.
    
    # Strategy: Use Strassen's algorithm on a 2x2x2 sub-block.
    # 
    # The 2x3x3 multiplication: A is 2x3, B is 3x3, C is 2x3.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # One approach: group the 3 columns of A into (2+1) and use Strassen on a 2x2 part.
    # 
    # Actually, let me think about a different decomposition.
    # 
    # Consider: C[i,k] = A[i,0]*B[0,k] + (A[i,1]+A[i,2])*(B[1,k]+B[2,k]) - A[i,1]*B[2,k] - A[i,2]*B[1,k]
    #
    # Let's verify: A[i,1]*(B[1,k]+B[2,k]) + A[i,2]*(B[1,k]+B[2,k]) - A[i,1]*B[2,k] - A[i,2]*B[1,k]
    # = A[i,1]*B[1,k] + A[i,1]*B[2,k] + A[i,2]*B[1,k] + A[i,2]*B[2,k] - A[i,1]*B[2,k] - A[i,2]*B[1,k]
    # = A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # So: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #            = A[i,0]*B[0,k] + (A[i,1]+A[i,2])*(B[1,k]+B[2,k]) - A[i,1]*B[2,k] - A[i,2]*B[1,k]
    #
    # This uses: 1 + 1 + 1 + 1 = 4 products per (i,k) pair, but the products are shared across i and k.
    #
    # Let me count the rank-1 terms:
    # 1. A[i,0] * B[0,k] -> contributes to C[i,k]: U[i*m+0], V[0*p+k], W[k*n+i]
    # 2. (A[i,1]+A[i,2]) * (B[1,k]+B[2,k]) -> contributes to C[i,k]: 
    #    U[i*m+1] + U[i*m+2] (both = 1), V[1*p+k] + V[2*p+k] (both = 1), W[k*n+i]
    # 3. A[i,1] * B[2,k] with negative sign: U[i*m+1] = -1, V[2*p+k] = 1, W[k*n+i] = -1
    # 4. A[i,2] * B[1,k] with negative sign: U[i*m+2] = -1, V[1*p+k] = 1, W[k*n+i] = -1
    #
    # Wait, but these are for each (i,k) pair. The rank-1 terms are indexed by (i,k) essentially.
    # For n=2, p=3: there are 2*3 = 6 such (i,k) pairs.
    # Each (i,k) pair uses 4 rank-1 terms, giving 6*4 = 24 terms. That's worse!
    #
    # The issue is that the products need to be shared. Let me reconsider.
    #
    # Actually, in a tensor decomposition, a rank-1 term is U[:,r] * V[:,r] * W[:,r] (outer product).
    # So for term r: U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    #
    # The key insight for Strassen: we can combine multiple (i,j,k) contributions into one rank-1 term.
    #
    # Let me think about this differently. The standard Strassen algorithm for 2x2x2 uses 7 products
    # instead of 8. The 7 rank-1 terms are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # And: C00 = M1 + M4 - M5 + M7
    #     C01 = M3 + M5
    #     C10 = M2 + M4
    #     C11 = M1 - M2 + M3 + M6
    #
    # Now for 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # 
    # I can partition A as [A_left | A_right] where A_left is 2x2 and A_right is 2x1.
    # And B as [B_top; B_bottom] where B_top is 2x3 and B_bottom is 1x3.
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_left * B_top is a 2x2 times 2x3 multiplication.
    # A_right * B_bottom is a 2x1 times 1x3 multiplication (rank 1, needs 2 products).
    #
    # For 2x2 times 2x3: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for k=0,1,2.
    # This is like three 2x2x2 multiplications sharing the A matrix.
    #
    # Hmm, this is getting complex. Let me try a more systematic approach.
    #
    # Actually, let me try a numerical search for rank 17.
    #
    # The tensor T has 18 nonzero entries. We want to find rank r such that:
    # T[c,a,b] = sum_{r=1}^{R} U[a,r] * V[b,r] * W[c,r]
    #
    # This is a system of equations. For rank 17, we have:
    # U: 6 x 17, V: 9 x 17, W: 6 x 17
    # Total unknowns: 6*17 + 9*17 + 6*17 = 425
    # Constraints: 6*6*9 = 324 (but only 18 are nonzero, and the rest must be 0)
    #
    # Actually, the constraints are:
    # For each (c,a,b) in the 6x6x9 grid: sum_r U[a,r]*V[b,r]*W[c,r] = T[c,a,b]
    # That's 324 equations with 425 unknowns. Should be solvable.
    #
    # But it's a nonlinear system. Let me try a numerical approach.
    
    # Build the tensor T
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    # Try rank 17 with numerical optimization
    best_result = None
    deadline = time.time() + 140  # safety margin
    
    def try_rank(R, max_iters=200):
        """Try to find a rank-R decomposition using numerical optimization."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        
        # Use the schoolbook as initialization and try to merge two terms
        # Start with rank R+1 schoolbook-like and try to optimize
        
        # Actually, let's use a random initialization
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        def objective(params):
            U_flat = params[:n * m * R].reshape(n * m, R)
            V_flat = params[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
            W_flat = params[n * m * R + m * p * R :].reshape(p * n, R)
            
            # Compute T_approx[c,a,b] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Use einsum
            T_approx = np.einsum('ar,br,cr->cab', U_flat, V_flat, W_flat)
            
            # Residual
            residual = T_approx - T
            return np.sum(residual ** 2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, x0, method='BFGS', 
                             options={'maxiter': max_iters, 'gtol': 1e-12})
            
            if result.fun < 1e-8:
                U_opt = result.x[:n * m * R].reshape(n * m, R)
                V_opt = result.x[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                W_opt = result.x[n * m * R + m * p * R :].reshape(p * n, R)
                
                # Verify
                T_check = np.einsum('ar,br,cr->cab', U_opt, V_opt, W_opt)
                if np.max(np.abs(T_check - T)) < 1e-6:
                    return (U_opt, V_opt, W_opt)
        except Exception:
            pass
        
        return None
    
    # Try rank 17
    result17 = try_rank(17)
    if result17 is not None:
        U_opt, V_opt, W_opt = result17
        # Convert to list format
        U_list = [[float(val) for val in row] for row in U_opt]
        V_list = [[float(val) for val in row] for row in V_opt]
        W_list = [[float(val) for val in row] for row in W_opt]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If rank 17 numerical search fails, try a constructive approach
    # using Strassen on a 2x2x2 sub-block
    
    if time.time() < deadline - 10:
        # Constructive rank-17 decomposition using Strassen
        # 
        # Partition: A (2x3) = [A0 | A1] where A0 is 2x2, A1 is 2x1
        # B (3x3) = [B0; B1] where B0 is 2x3, B1 is 1x3
        # C (2x3) = A0*B0 + A1*B1
        #
        # A0*B0: 2x2 * 2x3. This can be done with Strassen-like approach.
        # For each column k of B0, we have a 2x2 * 2x1 multiplication.
        # But we can share Strassen products across columns.
        #
        # Let me use a different partition. 
        # 
        # Actually, let me try: partition the middle dimension (j) into 2+1.
        # C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
        #        = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
        #
        # Use the identity: x*y + u*v = (x+u)(y+v) - x*v - u*y
        # So: A[i,1]B[1,k] + A[i,2]B[2,k] = (A[i,1]+A[i,2])(B[1,k]+B[2,k]) - A[i,1]B[2,k] - A[i,2]B[1,k]
        #
        # C[i,k] = A[i,0]B[0,k] + (A[i,1]+A[i,2])(B[1,k]+B[2,k]) - A[i,1]B[2,k] - A[i,2]B[1,k]
        #
        # Now, the products needed:
        # For each (i,k): 
        #   P1(i,k) = A[i,0] * B[0,k]
        #   P2(i,k) = (A[i,1]+A[i,2]) * (B[1,k]+B[2,k])
        #   P3(i,k) = A[i,1] * B[2,k]
        #   P4(i,k) = A[i,2] * B[1,k]
        #
        # But P1, P2, P3, P4 are bilinear forms that can be shared.
        #
        # P1(i,k) = A[i,0] * B[0,k]: This is a rank-1 bilinear form in (A,B).
        #   As a tensor term: U[i*m+0, r] = 1, V[0*p+k, r] = 1, W[k*n+i, r] = 1
        #   But this is for specific (i,k). We need to express it as a sum of rank-1 tensor terms.
        #
        # Actually, P1(i,k) = A[i,0] * B[0,k] is already a product of a linear form in A and a linear form in B.
        # As a rank-1 term in the tensor decomposition:
        #   U[a, r] = 1 if a = i*m+0, else 0  (for each i)
        #   V[b, r] = 1 if b = 0*p+k, else 0  (for each k)
        #   W[c, r] = 1 if c = k*n+i, else 0
        #
        # But this requires separate rank-1 terms for each (i,k) pair! That's n*p = 6 terms just for P1.
        #
        # The key is that we can combine terms. A rank-1 tensor term U[:,r]*V[:,r]*W[:,r] 
        # has U, V, W as full vectors, not just single entries.
        #
        # So for P1: the bilinear form is sum_{i,k} A[i,0]*B[0,k]*C[i,k]
        # This can be written as: (sum_i A[i,0]*e_i^T) * (sum_k B[0,k]*f_k^T) * (sum_{i,k} e_i^T*C*f_k)
        #
        # Hmm, I think the issue is that P1(i,k) = A[i,0]*B[0,k] as a function of (i,k) 
        # is a rank-1 matrix in (i,k) space. So it corresponds to ONE rank-1 tensor term!
        #
        # Let me verify: Define U[:,r1] such that U[i*m+0, r1] = 1 for all i, and 0 elsewhere.
        # V[:,r1] such that V[0*p+k, r1] = 1 for all k, and 0 elsewhere.
        # W[:,r1] such that W[k*n+i, r1] = 1 for all (i,k), and 0 elsewhere.
        #
        # Then sum_c U[a,c]*V[b,c]*W[c',c] for c'=k*n+i:
        # = U[i*m+0, r1] * V[0*p+k, r1] * W[k*n+i, r1] = 1*1*1 = 1
        #
        # And for other positions: if a != i*m+0 or b != 0*p+k or c' != k*n+i, the product is 0.
        #
        # Wait, but W[k*n+i, r1] = 1 for ALL (i,k). So for a fixed r1:
        # T_approx[c',a,b] = sum_r U[a,r]*V[b,r]*W[c',r]
        # For r = r1: U[a,r1]*V[b,r1]*W[c',r1]
        # This is nonzero only if a is of the form i*m+0 (i.e., j=0), b is of the form 0*p+k (i.e., j=0),
        # and c' is any value (since W[c',r1]=1 for all c').
        #
        # So T_approx[c', a, b] for r=r1 is:
        # = 1 if a = i*m+0 for some i, b = 0*p+k for some k, and c' = k'*n+i' for some (i',k')
        # But W[c',r1] = 1 for all c', so it's 1 for all c' when a and b are in the right positions.
        #
        # That means T_approx[c', i*m+0, 0*p+k] = 1 for all c', but we only want T[k*n+i, i*m+0, 0*p+k] = 1.
        # For c' != k*n+i, we'd have T_approx = 1 but T = 0. That's wrong!
        #
        # So P1(i,k) = A[i,0]*B[0,k] does NOT correspond to a single rank-1 tensor term.
        # It requires n*p = 6 separate rank-1 terms (one for each (i,k) pair).
        #
        # This makes sense because the tensor T has the constraint that the (i,k) index in W
        # must match the (i,k) indices in U and V.
    
        # OK so my earlier understanding was correct: each (i,j,k) triple gives a separate
        # position in the tensor, and the schoolbook uses one rank-1 term per triple.
        #
        # To get rank < 18, we need to find rank-1 terms that cover multiple tensor positions.
        # A rank-1 term U[:,r]*V[:,r]*W[:,r] contributes to T[c,a,b] for ALL (c,a,b) where
        # U[a,r] != 0, V[b,r] != 0, and W[c,r] != 0.
        #
        # So if U[:,r] has support on multiple (i,j) pairs, V[:,r] on multiple (j',k) pairs,
        # and W[:,r] on multiple (k',i') pairs, then this one rank-1 term contributes to many
        # tensor positions simultaneously.
        #
        # The constraint is that for positions where T=0, the sum of contributions must be 0.
        #
        # This is the key to Strassen: you design rank-1 terms where the supports overlap
        # in a way that cancels at unwanted positions.
    
        # Let me try a specific construction.
        # 
        # For 2x3x3, let's try to achieve rank 17 by modifying the schoolbook.
        # 
        # Idea: Take the schoolbook (18 terms) and try to merge two terms by finding
        # a rank-1 term that covers the positions of two schoolbook terms and cancels
        # at the unwanted positions, while adjusting the other terms.
        #
        # Alternatively, use known results: the rank of 2x3x3 matrix multiplication is 17.
        # A known construction uses the following approach:
        #
        # Write C = A*B where A is 2x3, B is 3x3.
        # Let A = [A_1 | A_2] where A_1 is 2x2 and A_2 is 2x1.
        # Let B = [B_1; B_2] where B_1 is 2x3 and B_2 is 1x3.
        # C = A_1 * B_1 + A_2 * B_2
        #
        # A_1 * B_1 is 2x2 * 2x3. We can compute this using Strassen on the 2x2 part
        # with the 3 columns of B_1.
        #
        # For 2x2 * 2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2.
        # 
        # Using Strassen on the 2x2 block:
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])  -- but B[0,0] and B[1,0] are scalars
        #
        # Hmm, Strassen is for 2x2 * 2x2. For 2x2 * 2x3, I need to adapt it.
        #
        # Let me just use the numerical approach with better initialization.
        #
        # Actually, let me try a different numerical approach: use the schoolbook 
        # initialization and then optimize to reduce the rank.
        
        # Try multiple random initializations for rank 17
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            R = 17
            
            # Initialize close to schoolbook but with two terms merged
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Use first 16 schoolbook terms
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < R - 1:
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
            
            # For the last two schoolbook terms, use a combined approach
            # The 17th and 18th terms in schoolbook correspond to specific (i,j,k) triples
            # Let's add small random perturbations to encourage merging
            U[:, -1] = np.random.randn(n * m) * 0.3
            V[:, -1] = np.random.randn(m * p) * 0.3
            W[:, -1] = np.random.randn(p * n) * 0.3
            
            # Also perturb the last few schoolbook terms slightly
            for c in range(R - 3, R - 1):
                U[:, c] += np.random.randn(n * m) * 0.1
                V[:, c] += np.random.randn(m * p) * 0.1
                W[:, c] += np.random.randn(p * n) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_f = params[:n * m * R].reshape(n * m, R)
                V_f = params[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                W_f = params[n * m * R + m * p * R :].reshape(p * n, R)
                T_approx = np.einsum('ar,br,cr->cab', U_f, V_f, W_f)
                residual = T_approx - T
                return np.sum(residual ** 2)
            
            try:
                result = minimize(objective, x0, method='BFGS',
                                 options={'maxiter': 100, 'gtol': 1e-10})
                
                if result.fun < 1e-6:
                    U_opt = result.x[:n * m * R].reshape(n * m, R)
                    V_opt = result.x[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                    W_opt = result.x[n * m * R + m * p * R :].reshape(p * n, R)
                    
                    T_check = np.einsum('ar,br,cr->cab', U_opt, V_opt, W_opt)
                    if np.max(np.abs(T_check - T)) < 1e-5:
                        # Round to nearest half-integer if possible
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        
                        T_rounded = np.einsum('ar,br,cr->cab', U_rounded, V_rounded, W_rounded)
                        if np.max(np.abs(T_rounded - T)) < 1e-5:
                            U_opt, V_opt, W_opt = U_rounded, V_rounded, W_rounded
                        
                        U_list = [[float(val) for val in row] for row in U_opt]
                        V_list = [[float(val) for val in row] for row in V_opt]
                        W_list = [[float(val) for val in row] for row in W_opt]
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
    
    # Fallback: return schoolbook
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
# EVOLVE-BLOCK-END
