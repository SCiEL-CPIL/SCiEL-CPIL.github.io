"""
Matrix multiplication 2x4 @ 4x6: rank-minimal exact tensor decomposition.

Strategy (DIFFERENT from the current program):
  The current program builds a 2304-equation linear system in U for each random (V,W) pair and
  runs numpy lstsq. This is slow and the search space is effectively random.

  New strategy: exploit the *block structure* of the tensor.
  The tensor T has the property that T[i*m+j, j*p+k, k*n+i] = 1.
  This is a "matrix multiplication tensor" with a very specific sparsity pattern.
  
  Key insight: we can use the "slice rank" / "border rank" approach more directly.
  For matrix multiplication <n,m,p>, the tensor rank is at most n*m*p (schoolbook).
  For small cases, we can use known constructions.
  
  A more effective approach: use the "Strassen" algorithm generalized to rectangular matrices.
  For 2x4 * 4x6, we can partition as:
    A = [A1; A2] where A1 is 2x4, A2 is 2x4  (actually A is 2x4, so no vertical partition)
    Wait, A is 2x4. Let's partition A into two 2x2 blocks: A = [A11 A12]
    B is 4x6. Partition B into [B11; B21] (top 2x6, bottom 2x6)
    C = A*B is 2x6. C = [C11 C12] where C11 = A11*B11, C12 = A12*B21... no.
    
    Actually: A (2x4) * B (4x6) = C (2x6)
    Partition A = [A1 A2] where A1 is 2x2, A2 is 2x2
    Partition B = [B1; B2] where B1 is 2x6, B2 is 2x6
    Then C = A1*B1 + A2*B2
    
    Each A1*B1 is 2x2 * 2x6 = 2x6. We can use Strassen for 2x2*2x2 if we partition B1 further.
    B1 is 2x6 = [B1a B1b] where B1a is 2x3, B1b is 2x3.
    A1 is 2x2.
    A1*B1a is 2x2 * 2x3. This is not square.
    
    Let's use a different approach: direct enumeration of small-rank decompositions.
    The known lower bound for <2,4,6> is not well-established, but let's try to find
    a decomposition of rank < 48 (schoolbook) by using a structured approach.
    
    Actually, the most promising approach for small rectangular matrix multiplication
    is to use the "border rank" or to directly search for a decomposition using the
    fact that the tensor has a very specific structure.
    
    Let's try a completely different approach: use the "rank profile" method.
    The matrix multiplication tensor <n,m,p> can be written as a sum of rank-1 tensors.
    The minimum number is the tensor rank.
    
    For <2,4,6>, let's try to use a "Strassen-like" construction with fewer multiplications.
    Standard Strassen for 2x2*2x2 uses 7 multiplications.
    For 2x4 * 4x6, we can think of it as 2 blocks of 2x2 * 2x6, but 2x6 is not square.
    
    Alternative: use the "Laderman" algorithm or similar for rectangular matrices.
    Laderman's algorithm for 2x2 * 2x2 uses 7 multiplications (same as Strassen).
    
    Let's try a direct search for a decomposition of rank r where r < 48.
    We'll use a randomized approach but with a much better structure:
    - Fix the W matrix to have a specific sparse pattern (e.g., each column has exactly 2 nonzeros)
    - Then solve for U and V using a linear system that is much smaller.
    
    Actually, let's try the simplest possible improvement: the schoolbook rank is 48.
    Can we do better? The tensor has 48 nonzero entries. A rank-1 tensor can contribute
    to at most one entry if it's "diagonal" in some sense, but in general a rank-1 tensor
    U⊗V⊗W has U[i,j]*V[j,k]*W[k,i] which is 1 for at most... well, it's a product of three vectors.
    
    Let's just try to find ANY decomposition with rank < 48 using a smart search.
    We'll use the fact that for matrix multiplication, there are known lower bounds.
    For <2,4,6>, the lower bound is at least 2*4*6/2 = 24 (from the "slice rank" lower bound).
    
    Let's try a structured search:
    1. Start with the schoolbook decomposition (rank 48).
    2. Try to merge pairs of rank-1 terms if their sum is also rank-1 or can be replaced by fewer terms.
    3. Use a randomized search with verification.
    
    Given the time constraints, let's implement a more efficient search:
    - Use the "slice space" method: fix a set of W vectors, then solve for U and V.
    - The key is to choose W vectors that are "compatible" with the tensor structure.
    
    For <2,4,6>, the output dimension is 12. Let's try W matrices where each column
    has exactly 2 nonzeros (corresponding to the two input rows i=0,1 for a fixed k).
"""

import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 4, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm = n * m   # 8
    mp = m * p   # 24
    pn = p * n   # 12

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition."""
        rank = len(U[0]) if U and U[0] else 0
        # Check all entries of the tensor
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += float(U[a][r]) * float(V[b][r]) * float(W[c][r])
                    if abs(s - expected) > tol:
                        return False
        return True

    # Schoolbook decomposition (rank 48)
    def schoolbook():
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()

    # Try to find a better decomposition using a structured search.
    # Key idea: the tensor T[i*m+j, j*p+k, k*n+i] = 1.
    # For a rank-r decomposition, we need r triples (u_r, v_r, w_r) such that
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all valid (i,j,k).
    
    # Structured approach: use the fact that the tensor is a "matrix multiplication tensor"
    # and try to use known algorithms or constructions.
    
    # For now, try a randomized search with a better structure:
    # - Fix W to have a specific pattern (e.g., each column supports a specific (i,k) pair)
    # - Then solve for U and V.
    
    # Method 1: Try to find a decomposition where each W column has exactly 2 nonzeros
    # (one for i=0 and one for i=1, for a fixed k). This is motivated by the structure
    # of the tensor: for a fixed k, the output coordinates are k*n+0 and k*n+1.
    
    def try_structured_search(r, num_trials=500):
        """Try to find a rank-r decomposition with structured W."""
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate W with structure: each column has at most 2 nonzeros
            W = [[0] * r for _ in range(pn)]
            for col in range(r):
                # Choose which (i,k) pairs this column supports
                # For simplicity, assign each column to a specific k
                k = col % p
                # Choose i values (0 and/or 1)
                if random.random() < 0.5:
                    # Support both i=0 and i=1
                    W[k * n + 0][col] = random.choice([-1, 1])
                    W[k * n + 1][col] = random.choice([-1, 1])
                else:
                    # Support only one i
                    i = random.choice([0, 1])
                    W[k * n + i][col] = random.choice([-1, 1])
            
            # Now solve for U and V.
            # For each (i,k), the equation is:
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (for valid j)
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0  (for invalid j)
            
            # This is a bilinear system. Let's fix V randomly and solve for U.
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
            
            # Build the linear system for U.
            # Unknowns: U[a, r] for a in 0..nm-1, r in 0..r-1. Total: nm*r unknowns.
            # Equations: for each (a, b, c), sum_r U[a, r] * V[b, r] * W[c, r] = T[a,b,c]
            # Total equations: nm * mp * pn = 2304.
            
            # Use numpy to solve the least squares problem.
            A_mat = np.zeros((nm * mp * pn, nm * r))
            b_vec = np.zeros(nm * mp * pn)
            
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        eq_idx = a * mp * pn + b * pn + c
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if i == i2 and j == j2 and k == k2:
                            b_vec[eq_idx] = 1.0
                        else:
                            b_vec[eq_idx] = 0.0
                        for r_idx in range(r):
                            A_mat[eq_idx, a * r + r_idx] = float(V[b][r_idx]) * float(W[c][r_idx])
            
            # Solve least squares
            try:
                U_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            U_check = U_sol.reshape(nm, r)
            
            # Verify
            max_err = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        s = 0.0
                        for r_idx in range(r):
                            s += U_check[a][r_idx] * float(V[b][r_idx]) * float(W[c][r_idx])
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                        max_err = max(max_err, abs(s - expected))
            
            if max_err < 1e-9:
                # Found a valid decomposition!
                U_int = [[int(round(U_check[a][r_idx])) for r_idx in range(r)] for a in range(nm)]
                return {"U": U_int, "V": V, "W": W}, r
        
        return None

    # Try ranks from best_rank - 1 down to 24
    for r in range(best_rank - 1, 23, -1):
        if time.time() > deadline:
            break
        result = try_structured_search(r, num_trials=200)
        if result is not None:
            best_result, best_rank = result
            break

    return best_result
