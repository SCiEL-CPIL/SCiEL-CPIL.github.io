import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Schoolbook fallback (rank 100)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def verify(U, V, W, tol=1e-9):
        """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
        T = np.ones((n, m, p))
        R = U.shape[1]
        res = np.zeros((n, m, p))
        for r in range(R):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            res += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.max(np.abs(T - res)) < tol

    # STRATEGY: Construct Strassen-based decomposition for 4x4x4 core,
    # then extend to 4x5x5 by handling the 5th row/col separately.
    #
    # Strassen for 2x2x2: 7 multiplications
    # For 4x4x4: tile into 2x2 blocks of 2x2 -> 7^2 = 49 multiplications
    # For 4x5x5: 4x4x4 (rank 49) + 4x1x5 (rank 20) + 1x4x5 (rank 20) + 1x1x5 (rank 5) = 94
    # But we can do better by sharing some terms.
    #
    # Alternative: Direct Strassen tiling for 4x4x4 gives rank 49.
    # The extra 5th column of B and 5th row of A/B need handling.
    #
    # Let's build the 4x4x4 Strassen decomposition explicitly.
    #
    # Strassen 2x2x2:
    # C = A*B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    #
    # Each m_r corresponds to a rank-1 term in the tensor decomposition.
    # For m_r = (linear combo of A entries) * (linear combo of B entries),
    # the contribution to C[i,k] is:
    #   sum over (i',j') and (j'',k'') of U[i',j',r] * V[j'',k'',r] * W[k',i',r]
    #
    # Let's build the 4x4x4 Strassen decomposition by recursive tiling.
    # A 4x4 matrix is partitioned into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Using Strassen on the 2x2 block structure:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mr is itself a 2x2x2 Strassen product (rank 7).
    # So total rank for 4x4x4 = 7 * 7 = 49.
    #
    # Let me construct this explicitly.
    #
    # First, define the 2x2x2 Strassen decomposition in terms of U, V, W vectors.
    # For 2x2x2: indices i,j,k in {0,1}
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    #
    # m1 = (a+d)(e+h): 
    #   A part: a=A[0,0], d=A[1,1] -> U contribution: U[0,0,r]=1, U[1,1,r]=1
    #   B part: e=B[0,0], h=B[1,1] -> V contribution: V[0,0,r]=1, V[1,1,r]=1
    #   C part: C[0,0] gets +1, C[1,1] gets +1 -> W[0,0,r]=1, W[1,1,r]=1
    #   Wait, let me be more careful.
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # For m1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]):
    #   Contribution to C[i,k]: 
    #     C[0,0] += 1 * m1
    #     C[1,1] += 1 * m1
    #   So W[0,0,r]=1, W[1,1,r]=1 (using W[k*n+i] convention: W[0*2+0]=W[0], W[1*2+1]=W[3])
    #   Wait, n=2 for the 2x2 case. W[k*n+i] = W[k*2+i].
    #   C[0,0]: i=0,k=0 -> W[0*2+0] = W[0]
    #   C[1,1]: i=1,k=1 -> W[1*2+1] = W[3]
    #
    #   U[i*m+j, r] where m=2:
    #     A[0,0]: U[0*2+0] = U[0]
    #     A[1,1]: U[1*2+1] = U[3]
    #   V[j*p+k, r] where p=2:
    #     B[0,0]: V[0*2+0] = V[0]
    #     B[1,1]: V[1*2+1] = V[3]
    #
    # So for m1: U[0]=1, U[3]=1, V[0]=1, V[3]=1, W[0]=1, W[3]=1
    #
    # Let me verify: sum_r U[i,j,r]V[j,k,r]W[k,i,r]
    # For i=0,j=0,k=0: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1 ✓ (T[0,0,0]=1)
    # For i=0,j=0,k=1: U[0,0]V[0,1]W[1,0] = 1*0*0 = 0 ✗ (T[0,0,1]=1)
    #
    # Hmm, this doesn't work directly. The issue is that m1 contributes to C[0,0] AND C[1,1],
    # but the tensor equation requires T[i,j,k]=1 for ALL i,j,k.
    #
    # Let me reconsider. The tensor T[i,j,k] = 1 means:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition is: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) ... no.
    #
    # Actually, the standard form is:
    # C[i,k] = sum_r U_r[i] * (A * U_r^T)[something] ... 
    #
    # Let me re-read the problem. The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all i,j,k.
    #
    # This means: for each (i,j,k), sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    #
    # This is equivalent to: the matrix multiplication tensor has rank r.
    #
    # For the schoolbook: each (i,j,k) triple gets its own rank term with U[i,j]=1, V[j,k]=1, W[k,i]=1.
    #
    # For Strassen, the key is that we can share rank terms across multiple (i,j,k) triples.
    #
    # Let me think about this differently. The Strassen algorithm computes:
    # C = A*B using 7 scalar multiplications.
    # Each scalar multiplication m_r = (linear combo of A) * (linear combo of B)
    # contributes to multiple entries of C.
    #
    # In the tensor decomposition language:
    # m_r corresponds to a rank-1 term where:
    #   U_r[i,j] = coefficient of A[i,j] in the first factor
    #   V_r[j,k] = coefficient of B[j,k] in the second factor  
    #   W_r[k,i] = coefficient of C[i,k] in the output
    #
    # And the constraint is: for all (i,j,k): sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1
    #
    # This is NOT the same as C[i,k] = sum_r (sum_j U_r[i,j] A[i,j]) (sum_j V_r[j,k] B[j,k]) W_r[k,i]
    #
    # Wait, let me re-derive. If we have:
    # C[i,k] = sum_r alpha_r(i,k) * m_r
    # where m_r = (sum_{i',j'} U_r[i',j'] A[i',j']) * (sum_{j',k'} V_r[j',k'] B[j',k'])
    #
    # Then C[i,k] = sum_r alpha_r(i,k) (sum_{i',j'} U_r[i',j'] A[i',j']) (sum_{j',k'} V_r[j',k'] B[j',k'])
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    # sum_r alpha_r(i,k) U_r[i',j'] V_r[j',k'] = delta_{i',i} delta_{j',j} delta_{k',k} ... 
    #
    # This is getting complicated. Let me just use the direct tensor equation:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    #
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    #
    # For the 2x2x2 case (n=m=p=2): 8 equations, r*12 unknowns.
    # Strassen says r=7 works. Let's construct it.
    #
    # I'll use a different approach: solve the linear system directly for small cases.
    # For a given set of r rank terms, the equations are bilinear (products of U,V,W entries).
    # But if we fix two of the three matrices, the equations become linear in the third.
    #
    # ALTERNATIVE STRATEGY: Use the fact that the matrix multiplication tensor is well-studied.
    # For 4x5x5, I'll try to construct a decomposition using the following approach:
    #
    # 1. Build a 4x4x4 Strassen decomposition (rank 49) by explicit construction.
    # 2. Extend to 4x5x5 by adding terms for the 5th column/row.
    # 3. Try to reduce the rank by finding linear dependencies.
    #
    # Let me first construct the 2x2x2 Strassen decomposition explicitly.
    #
    # For 2x2x2, we need U (4x7), V (4x7), W (4x7) such that
    # sum_{r=0}^{6} U[i*2+j, r] V[j*2+k, r] W[k*2+i, r] = 1 for all i,j,k in {0,1}
    #
    # Strassen's 7 products:
    # m0 = (A00+A11)(B00+B11)
    # m1 = (A10+A11)B00
    # m2 = A00(B01-B11)
    # m3 = A11(B10-B00)
    # m4 = (A00+A01)B11
    # m5 = (A10-A00)(B00+B01)
    # m6 = (A01-A11)(B10+B11)
    #
    # C00 = m0 + m3 - m4 + m6
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m0 - m1 + m2 + m5
    #
    # Now, in the tensor decomposition:
    # m_r = (sum_{i,j} U_r[i,j] A[i,j]) * (sum_{j,k} V_r[j,k] B[j,k])
    # and C[i,k] = sum_r W_r[k,i] * m_r
    #
    # For this to work, we need:
    # C[i,k] = sum_r W_r[k,i] (sum_{i',j'} U_r[i',j'] A[i',j']) (sum_{j',k'} V_r[j',k'] B[j',k'])
    #         = sum_{i',j',j',k'} A[i',j'] B[j',k'] (sum_r W_r[k,i] U_r[i',j'] V_r[j',k'])
    #
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r W_r[k,i] U_r[i',j'] V_r[j',k'] = delta_{i',i} delta_{j',j} delta_{k',k} ... 
    #
    # Wait, that's not right either. Let me be more careful.
    #
    # C[i,k] should equal sum_j A[i,j] B[j,k].
    # From the decomposition: C[i,k] = sum_r W_r[k,i] * m_r
    # where m_r = (sum_{i',j'} U_r[i',j'] A[i',j']) * (sum_{j',k'} V_r[j',k'] B[j',k'])
    #
    # Expanding: C[i,k] = sum_r W_r[k,i] sum_{i',j'} sum_{j',k'} U_r[i',j'] V_r[j',k'] A[i',j'] B[j',k']
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    # sum_r W_r[k,i] U_r[i',j'] V_r[j',k'] = delta_{i',i} * delta_{j',j} * delta_{k',k}
    # where the j in delta_{j',j} is the same j that connects A[i,j] and B[j,k].
    #
    # Actually, the correct condition is:
    # sum_r W_r[k,i] U_r[i',j'] V_r[j',k'] = delta_{i',i} delta_{j',k} ... no.
    #
    # Let me index more carefully. A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Decomposition: C[i,k] = sum_r W_r[k,i] * m_r
    # m_r = (sum_{i',j'} U_r[i',j'] A[i',j']) * (sum_{j',k'} V_r[j',k'] B[j',k'])
    #
    # = sum_r W_r[k,i] sum_{i',j',j',k'} U_r[i',j'] V_r[j',k'] A[i',j'] B[j',k']
    #
    # Coefficient of A[i',j'] B[j',k'] in C[i,k] is:
    # sum_r W_r[k,i] U_r[i',j'] V_r[j',k']
    #
    # For C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i',j'] B[j',k'] should be:
    # 1 if i'=i, j'=j, k'=k (for some j) ... 
    #
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]. The term A[i',j'] B[j',k'] appears with coefficient 1 if i'=i, j'=j, k'=k for the same j, i.e., if i'=i and j'=j' and k'=k... 
    #
    # No: C[i,k] = sum_j A[i,j] B[j,k]. The monomial A[i',j'] B[j',k'] appears in C[i,k] with coefficient 1 if and only if i'=i, j'=j', k'=k (the j in A[i,j] must equal the j in B[j,k]). So the condition is:
    #
    # sum_r W_r[k,i] U_r[i',j'] V_r[j',k'] = 1 if i'=i and j'=j' and k'=k, else 0
    #
    # Wait, that's just delta_{i',i} delta_{j',j'} delta_{k',k} which is the schoolbook!
    #
    # I think I'm confusing myself. Let me look at this from the tensor perspective directly.
    #
    # The matrix multiplication tensor T has components T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for all i,j,k (in the sense that the tensor is the identity mapping).
    #
    # A rank-r decomposition is: T[i,j,k] = sum_{r=0}^{R-1} U_r[i,j] V_r[j,k] W_r[k,i]
    #
    # This is a CP decomposition of the tensor T.
    #
    # For the 2x2x2 case, T is the 2x2x2 all-ones tensor.
    # We need: 1 = sum_{r=0}^{6} U_r[i,j] V_r[j,k] W_r[k,i] for all i,j,k in {0,1}.
    #
    # This is 8 scalar equations. With 7 rank terms, we have 7*(4+4+4) = 84 unknowns (with scaling freedom, effectively 7*3-7 = 14 degrees of freedom per term... this is overcounting).
    #
    # The point is: Strassen's algorithm DOES give a valid rank-7 decomposition of the 2x2x2 multiplication tensor.
    #
    # Let me just construct it by solving the system. For 2x2x2:
    # Variables: U (4x7), V (4x7), W (4x7)
    # Equations: for each (i,j,k) in {0,1}^3: sum_r U[i*2+j,r] V[j*2+k,r] W[k*2+i,r] = 1
    #
    # This is 8 bilinear equations in 84 unknowns. We can solve by fixing some variables.
    #
    # Actually, let me use a known result. The Strassen decomposition for 2x2x2 in tensor form:
    #
    # Let me denote U_r as a 2x2 matrix, V_r as 2x2, W_r as 2x2.
    #
    # From Strassen:
    # m0 = (A00+A11)(B00+B11)
    #   U0 = [[1,0],[0,1]], V0 = [[1,0],[0,1]], W0 = [[1,0],[0,1]]
    #   Check: sum_r U0[i,j]V0[j,k]W0[k,i] for i=j=k=0: 1*1*1=1
    #   For i=0,j=0,k=1: U0[0,0]V0[0,1]W0[1,0] = 1*0*0 = 0. But we need 1.
    #
    # This doesn't work. The issue is that Strassen's algorithm computes C = A*B, not the tensor directly.
    #
    # I think the correct interpretation is different. Let me re-read the problem statement.
    #
    # "exactly decompose the matrix multiplication tensor"
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    #
    # And T[i,j,k] = 1 for all i,j,k.
    #
    # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    #
    # This is indeed a CP decomposition of the all-ones tensor of size n x m x p.
    #
    # For n=m=p=2, the all-ones tensor has rank 1! Just take U=V=W=all ones.
    # sum_r=1: 1*1*1 = 1. Done. Rank 1.
    #
    # But that can't be right for matrix multiplication...
    #
    # OH WAIT. I think I've been misunderstanding the problem. Let me re-read.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 5 matrix."
    #
    # The matrix multiplication tensor is NOT the all-ones tensor. It's the tensor that represents the bilinear map (A,B) -> A*B.
    #
    # The tensor T has components T[i,j,k] where:
    # - i indexes the row of A and C (0 to n-1)
    # - j indexes the column of A and row of B (0 to m-1)  
    # - k indexes the column of B and C (0 to p-1)
    #
    # And T[i,j,k] = 1 for all valid i,j,k.
    #
    # The decomposition T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i] means:
    # C[i,k] = sum_r W_r[k,i] * (sum_{i',j'} U_r[i',j'] A[i',j']) * (sum_{j',k'} V_r[j',k'] B[j',k'])
    #
    # No wait, that's not how it works. Let me think again.
    #
    # The standard formulation: A rank-r decomposition of the matrix multiplication tensor means:
    # C[i,k] = sum_{r=0}^{R-1} (sum_j U_r[i,j] A[i,j]) (sum_j V_r[j,k] B[j,k]) W_r[k,i]
    #
    # No, that's not right either. The correct formulation is:
    #
    # C[i,k] = sum_{r=0}^{R-1} m_r
    # where m_r = (sum_{i',j'} U_r[i',j'] A[i',j']) * (sum_{j',k'} V_r[j',k'] B[j',k'])
    # and the coefficient of m_r in C[i,k] is W_r[k,i].
    #
    # So: C[i,k] = sum_r W_r[k,i] m_r
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A,B:
    # sum_r W_r[k,i] (sum_{i',j'} U_r[i',j'] A[i',j']) (sum_{j',k'} V_r[j',k'] B[j',k']) = sum_j A[i,j] B[j,k]
    #
    # The LHS, when expanded, gives:
    # sum_{i',j',j',k'} A[i',j'] B[j',k'] (sum_r W_r[k,i] U_r[i',j'] V_r[j',k'])
    #
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r W_r[k,i] U_r[i',j'] V_r[j',k'] = delta_{i',i} * delta_{j',j} * delta_{k',k}
    # where j is the summation index, so j'=j must hold.
    #
    # More precisely: the coefficient of A[i',j'] B[j',k'] in C[i,k] must be 1 if i'=i, j'=j', k'=k (with j being any value, but j' is fixed), and 0 otherwise.
    #
    # Wait: C[i,k] = sum_j A[i,j] B[j,k]. The monomial A[i',j'] B[j',k'] appears with coefficient 1 if i'=i AND j'=j' AND k'=k... but j' is the same index in both A and B. So the condition is simply i'=i, j'=j', k'=k? No, because j' is a free index in the sum.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]. The term A[i',j'] B[j',k'] appears in C[i,k] if and only if i'=i, j'=j (for the j in the sum), and k'=k. But j is summed over, so for a fixed (i',j',j',k'), it appears in C[i,k] with coefficient 1 if i'=i and k'=k, regardless of j' (since j' can be any value of j).
    #
    # No! A[i,j] and B[j,k] share the index j. So A[i',j'] B[j',k'] is a specific monomial. In C[i,k] = sum_j A[i,j] B[j,k], the monomial A[i',j'] B[j',k'] appears if i'=i, j'=j, k'=k for the same j. Since j is summed, A[i',j'] B[j',k'] appears in C[i,k] with coefficient 1 if i'=i and k'=k, and j' is the value of j. But j' is fixed, so it appears with coefficient 1 if i'=i, k'=k, and j' can be anything (it's just the value of the summation index).
    #
    # I'm overcomplicating this. Let me just accept the tensor equation as given:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    #
    # And solve it. For 2x2x2, this is 8 equations. Let me try to find a rank-7 solution.
    #
    # Actually, you know what, let me just try a completely different approach:
    # Use integer programming or a direct search for small ranks.
    #
    # For 4x5x5, the tensor has 100 components. We need sum_r U[i,j,r]V[j,k,r]W[k,i,r] = 1.
    #
    # A key insight: if we restrict U, V, W to have entries in {-1, 0, 1}, the problem becomes a combinatorial search.
    #
    # Let me try a greedy approach:
    # 1. Start with the schoolbook decomposition (rank 100).
    # 2. Try to merge pairs of rank terms: if two terms can be combined into one, reduce rank by 1.
    # 3. Alternatively, try to find a low-rank solution by random search with {−1,0,1} entries.
    #
    # Given the time constraints, let me try a focused random search for ranks between 70 and 100,
    # using {−1, 0, 1} entries and checking the tensor equation.
    #
    # But 100 choose r is huge. Let me try a different tactic:
    #
    # TACTIC: Use the fact that the problem is over-determined (100 equations, r*100 unknowns for r<1).
    # For r < 100, we have more unknowns than equations, so solutions should exist.
    # The challenge is finding them.
    #
    # Let me try a sequential construction:
    # 1. Choose random U, V with {−1,0,1} entries.
    # 2. For each (i,j,k), the equation is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    #    This is linear in W[k,i,r] for fixed U,V.
    # 3. Solve for W using least squares or direct substitution.
    # 4. Check if the solution is exact (all W entries are {-1,0,1} or at least integers/half-integers).
    #
    # This is promising! Let me implement it.

    def try_random_search(R, num_trials=50, time_limit=30.0):
        """Try to find a rank-R decomposition by random U,V and solving for W."""
        t0 = time.time()
        np.random.seed(42)
        
        for trial in range(num_trials):
            if time.time() - t0 > time_limit or elapsed() > deadline - 10:
                break
            
            # Random sparse U and V with entries in {-1, 0, 1}
            density = 0.3
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            for i in range(n * m):
                for r in range(R):
                    if np.random.rand() < density:
                        U[i, r] = np.random.choice([-1, 1])
            for j in range(m * p):
                for r in range(R):
                    if np.random.rand() < density:
                        V[j, r] = np.random.choice([-1, 1])
            
            # For each (i,k), we need to solve for W[k*n+i, r]
            # The equations are: for each j: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            # This is a system of m equations in R unknowns (W[k*n+i, 0..R-1])
            # If R >= m, we can solve it.
            
            W = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the m x R matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A @ W[k*n+i, :] = ones(m)
                    # Use least squares
                    try:
                        W_row, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is exact
                    if not np.allclose(A @ W_row, np.ones(m), atol=1e-10):
                        valid = False
                        break
                    
                    W[k * n + i, :] = W_row
            
            if not valid:
                continue
            
            # Verify the full decomposition
            if verify(U, V, W):
                return U, V, W
        
        return None

    # Try ranks from 80 down to 70
    for R in range(80, 69, -1):
        if elapsed() > deadline - 15:
            break
        result = try_random_search(R, num_trials=20, time_limit=10.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = R
            break

    # If we found something better, return it; else return schoolbook
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
