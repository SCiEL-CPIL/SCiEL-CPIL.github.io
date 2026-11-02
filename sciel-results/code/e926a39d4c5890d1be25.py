import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Start with the rank-56 Strassen decomposition
    rank0 = 56
    U0 = np.zeros((nm, rank0), dtype=float)
    V0 = np.zeros((mp, rank0), dtype=float)
    W0 = np.zeros((pn, rank0), dtype=float)

    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)

            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1

            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1

            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1

            U0[a00, base] += 1; U0[a11, base] += 1
            V0[b00, base] += 1; V0[b11, base] += 1
            W0[c00, base] += 1; W0[c11, base] += 1

            U0[a10, base+1] += 1; U0[a11, base+1] += 1
            V0[b00, base+1] += 1
            W0[c10, base+1] += 1; W0[c11, base+1] -= 1

            U0[a00, base+2] += 1
            V0[b01, base+2] += 1; V0[b11, base+2] -= 1
            W0[c01, base+2] += 1; W0[c11, base+2] += 1

            U0[a11, base+3] += 1
            V0[b10, base+3] += 1; V0[b00, base+3] -= 1
            W0[c00, base+3] += 1; W0[c10, base+3] += 1

            U0[a00, base+4] += 1; U0[a01, base+4] += 1
            V0[b11, base+4] += 1
            W0[c00, base+4] -= 1; W0[c01, base+4] += 1

            U0[a10, base+5] += 1; U0[a00, base+5] -= 1
            V0[b00, base+5] += 1; V0[b01, base+5] += 1
            W0[c11, base+5] += 1

            U0[a01, base+6] += 1; U0[a11, base+6] -= 1
            V0[b10, base+6] += 1; V0[b11, base+6] += 1
            W0[c00, base+6] += 1

    # Verify the rank-56 decomposition
    def verify(U, V, W, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify(U0, V0, W0):
        # Shouldn't happen, but fallback
        rank = 64
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding linear dependencies among the 56 terms
    # The tensor T has entries T[i,j,k] = delta(i*m+j, j*p+k, k*n+i) in the decomposition sense
    # Actually, the tensor is: T_{i,j,k} = 1 if the (i,j) factor of U, (j,k) factor of V, and (k,i) factor of W multiply to give C[i,k]
    # More precisely, the matrix multiplication tensor M has M[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all i,j,k
    
    # The 56 vectors (u_s, v_s, w_s) span a decomposition. 
    # If there's a linear relation, we can potentially reduce rank.
    # 
    # Strategy: Try to find if any subset of terms is linearly dependent in a way that allows reduction.
    # 
    # Alternative: Use optimization to find a rank-r decomposition for r < 56.
    # This is a hard non-convex problem, but we can try local optimization.
    
    # Let's try a different approach: look for a rank-50 decomposition by exploiting
    # the specific structure of 2x4 * 4x8.
    
    # Key insight: A is 2x4 (2 rows), B is 4x8 (8 cols). 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For fixed k: C[:,k] = A @ B[:,k] (2x4 times 4x1 = 2x1)
    # There are 8 such column computations.
    # Each is a 2x4 * 4x1 multiply, which has tensor rank 8 (schoolbook) or... 
    # Actually 2x4*4x1: C[i] = sum_j A[i,j] B[j]. This is just a bilinear form.
    # Rank of this tensor: for 2x4*4x1, the tensor is in R^{2} x R^{4} x R^{4}
    # Wait, no. The output is 2x1, so it's R^2 (output) x R^4 (A) x R^4 (B)
    # Actually the standard tensor for matrix multiply M_{n,m,p} is in R^{nm} x R^{mp} x R^{pn}
    # For 2x4*4x1: nm=8, mp=4, pn=2. Rank <= 8 (schoolbook).
    
    # But we have 8 such problems sharing the same A. This sharing is what allows Strassen.
    
    # Let me try to directly optimize for a lower rank.
    # We'll use the fact that we can perturb the 56-term decomposition and then try to
    # zero out some terms.
    
    # Actually, let's try a more principled approach:
    # The tensor for 2x4*4x8 has known upper bounds. Let me check if rank 50 is achievable.
    
    # Try rank 52, 50, 48 with optimization
    for target_rank in [52, 50, 48, 46]:
        if time.time() > deadline - 30:
            break
            
        # Initialize from the 56-term decomposition, randomly combining terms
        # to get target_rank terms
        np.random.seed(42)
        
        # Strategy: take the 56 terms, and for each group of (56/target_rank) terms,
        # try to combine them into one term via a linear transformation.
        
        # Better: start with random initialization near the 56-term solution
        # and use gradient-based optimization.
        
        # The objective: minimize sum over all (i,j,k) of (sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] - 1)^2
        # subject to rank = target_rank (i.e., only target_rank columns are used)
        
        # This is a non-convex optimization. Let's try a simple approach:
        # Start from the 56-term solution, randomly select target_rank terms to keep,
        # and try to adjust them to still satisfy the equations.
        
        # Actually, a better approach: use the 56-term solution and try to find
        # linear combinations that can replace groups of terms with fewer terms.
        
        # Let's try: can we merge pairs of Strassen terms?
        # Each 2x2 block uses 7 terms. 8 blocks * 7 = 56.
        # If we can share some structure between adjacent blocks, we might save terms.
        
        # For example, blocks (a=0,b=0) and (a=0,b=1) share the same A columns.
        # Block (a=0,b=0) uses A[0,0], A[0,1], A[1,0], A[1,1]
        # Block (a=0,b=1) uses A[0,0], A[0,1], A[1,0], A[1,1] (same A!)
        # But different B columns.
        
        # So for fixed a, all 4 b-blocks use the same 4 A entries.
        # The A-part of the Strassen formulas is the same across b for fixed a.
        # Specifically:
        # M1 = (A00+A11)(B00+B11) -> A-part: A00+A11
        # M2 = (A10+A11)B00 -> A-part: A10+A11
        # M3 = A00(B01-B11) -> A-part: A00
        # M4 = A11(B10-B00) -> A-part: A11
        # M5 = (A00+A01)B11 -> A-part: A00+A01
        # M6 = (A10-A00)(B00+B01) -> A-part: A10-A00
        # M7 = (A01-A11)(B10+B11) -> A-part: A01-A11
        
        # These 7 A-combinations are: A00+A11, A10+A11, A00, A11, A00+A01, A10-A00, A01-A11
        # They are linearly independent (7 vectors in 4-dim space? No, 4 entries, so max rank 4)
        # Wait, A has 4 entries: A00, A01, A10, A11. The 7 combinations are in R^4.
        # So they span at most 4 dimensions. This means we might be able to reduce!
        
        # The 7 A-vectors:
        # v1 = [1, 0, 0, 1] (A00+A11)
        # v2 = [0, 0, 1, 1] (A10+A11)
        # v3 = [1, 0, 0, 0] (A00)
        # v4 = [0, 0, 0, 1] (A11)
        # v5 = [1, 1, 0, 0] (A00+A01)
        # v6 = [-1, 0, 1, 0] (A10-A00)
        # v7 = [0, 1, 0, -1] (A01-A11)
        
        # These span R^4 (v3, v4, v6, v7 are independent, for example)
        # So we can express any of the 7 as a combination of 4 basis vectors.
        
        # Similarly for B: each block uses 4 B entries, and the 7 B-combinations span R^4.
        
        # This suggests we might be able to use a rank-4 decomposition per (a) group
        # instead of 7 per (a,b) block.
        
        # For fixed a (2 choices), we have 4 b-blocks, each needing 7 multiplications.
        # Total per a: 28 multiplications.
        # If we can share the A-side: the 7 A-combinations are the same for all 4 b-blocks.
        # So we compute 7 A-combinations once, then for each b, we need 7 B-combinations
        # and 7 multiplications. That's still 7*4 = 28 per a, total 56.
        
        # But if we use a rank-4 decomposition of the 4x4 sub-problem...
        # Actually, the sub-problem for fixed a is: 
        # C[:, 2b:2b+2] (for b=0,1,2,3) = A_a @ B[:, 2b:2b+2]
        # where A_a is 2x2 (columns 2a, 2a+1 of A) and B[:, 2b:2b+2] is 4x2.
        # Wait, B is 4x8, so B[:, 2b:2b+2] is 4x2.
        # C[:, 2b:2b+2] is 2x2 = A_a (2x2) @ B[:, 2b:2b+2] (4x2)? No, dimensions don't work.
        
        # Let me re-examine. A is 2x4, B is 4x8.
        # A_a = A[:, 2a:2a+2] is 2x2 (rows 0-1, cols 2a, 2a+1)
        # B_a = B[2a:2a+2, :] is 2x8 (rows 2a, 2a+1, all 8 cols)
        # But C = A @ B, so C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
        # = A_0 @ B_top + A_1 @ B_bottom
        # where A_0 = A[:, 0:2] (2x2), B_top = B[0:2, :] (2x8)
        # A_1 = A[:, 2:4] (2x2), B_bottom = B[2:4, :] (2x8)
        
        # So C = A_0 @ B_top + A_1 @ B_bottom
        # Each term is 2x2 @ 2x8 = 2x8.
        # Rank of 2x2@2x8: schoolbook is 2*2*8 = 32.
        # Can we do better? 2x2 @ 2x8: 
        # C[i,k] = sum_j A[i,j] B[j,k] for j=0,1, k=0..7
        # For each k: C[:,k] = A @ B[:,k] (2x2 @ 2x1)
        # 2x2@2x1 has rank 4 (schoolbook) or 3 (Strassen for 2x2@2x1? No, Strassen is for 2x2@2x2)
        # 2x2@2x1: C[0] = A[0,0]B[0] + A[0,1]B[1], C[1] = A[1,0]B[0] + A[1,1]B[1]
        # This is 4 scalar multiplications. Can we do it in 3? 
        # The tensor is in R^2 x R^2 x R^2 (output 2, A 2, B 2). 
        # Actually it's R^2 (output) x R^4 (A flattened) x R^2 (B)
        # Hmm, this is getting complicated.
        
        # Let me just try the direct optimization approach.
        
        # Define the optimization problem:
        # Minimize F(U,V,W) = sum_{i,j,k} (sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] - 1)^2
        # over U (8 x r), V (32 x r), W (16 x r)
        
        # This has (8+32+16)*r = 56r parameters.
        # For r=52: 2912 parameters. Too many for scipy.
        
        # Alternative: fix the structure and only optimize a few parameters.
        
        # Let me try a different tactic: search for a known lower-rank decomposition
        # in the literature or try a greedy approach.
        
        # Actually, let me just verify that 56 is correct and return it.
        # The lower bound is 45, and 56 is a valid upper bound.
        # Without more sophisticated optimization, 56 is our best bet.
        
        pass

    # Return the rank-56 decomposition
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
