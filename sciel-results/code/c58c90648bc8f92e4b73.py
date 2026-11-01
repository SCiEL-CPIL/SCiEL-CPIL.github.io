"""Rank-minimal tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Uses a structural approach: model the multiplication as a tensor contraction
over the group ring Z[C2 x C3], exploiting character orthogonality to find
a minimal set of group homomorphisms that separate the support of the
multiplication tensor.

The key insight: for 2x4x6, we can decompose the problem into:
- 2 blocks of A (each 2x2)
- 3 blocks of B (each 2x2)  
- 3 blocks of C (each 2x2)

Using Strassen's 7-term algorithm for each 2x2x2 block gives rank 42.
But we can do better by exploiting the group structure C2 x C3.

The group C2 x C3 has 6 elements and 6 irreducible characters (since it's abelian).
Each character gives a rank-1 term in the group ring representation.

However, the actual matrix multiplication tensor for 2x4x6 has rank at most 48
(schoolbook), and Strassen gives 42. The group-theoretic approach may allow
us to find a decomposition with rank < 42 by using the character basis.

Let's try: the tensor T_{i,j,k} = delta_{i,k} (for the multiplication pattern).
Actually, for 2x4 @ 4x6 -> 2x6:
  C[i,k] = sum_j A[i,j] * B[j,k]

The tensor is T[i,j,k] = 1 if the indices are consistent, else 0.
This is a "partial identity" tensor.

For the group ring approach over C2 x C3:
- Map i in {0,1} to C2 element
- Map j in {0,1,2,3} to... hmm, 4 doesn't divide 6 cleanly.

Let me think differently. The structural change: instead of fixed Strassen blocks,
use a global optimization approach. We know the rank is at most 42. Let's try to
find a rank-1 term that can be added/removed to reduce the total.

Actually, let's try a different structural approach: use the fact that the
multiplication tensor for 2x4x6 can be written as a sum of Kronecker products
of smaller tensors. 

2x4x6 = (2x2x2) * (2x2x3) ... no, that doesn't work cleanly.

Let me try: express the 2x4x6 multiplication as a combination of:
- 2 independent 2x2x2 multiplications (for the "even" and "odd" columns of B)
- Plus correction terms

Actually, the cleanest structural approach: use the bilinear algorithm for
each 2x2x2 block, but try to share terms across blocks where the A and B
patterns overlap.

For 2x4 @ 4x6:
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
  C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
  ...
  C[1,5] = A[1,0]*B[0,5] + A[1,1]*B[1,5] + A[1,2]*B[2,5] + A[1,3]*B[3,5]

Group B columns into 3 pairs: (0,1), (2,3), (4,5).
Group A rows into 2 groups: {0,1} (only 2 rows, so 1 group of 2).

Wait, A is 2x4, so 2 rows, 4 cols. B is 4x6, so 4 rows, 6 cols.

The 4 rows of B / 4 cols of A form the "middle" dimension m=4.
We can split the middle dimension into 2 blocks of 2: m1=2 (cols 0-1 of A, rows 0-1 of B), m2=2 (cols 2-3 of A, rows 2-3 of B).

Then C[i,k] = sum_{j in block1} A[i,j]*B[j,k] + sum_{j in block2} A[i,j]*B[j,k]

For each middle block, we have a 2x2x6 subproblem: A_block is 2x2, B_block is 2x6, C is 2x6.

For 2x2x6: split B into 3 blocks of 2x2. Each 2x2x2 uses 7 Strassen terms.
Total: 2 middle blocks * 3 B-blocks * 7 = 42.

Can we do better? The 2x2x6 problem: A is 2x2, B is 2x6, C is 2x6.
This is equivalent to multiplying a 2x2 matrix by a 2x6 matrix.

The rank of 2x2x6 matrix multiplication: this is the rank of the tensor
T[i,j,k] where i in {0,1}, j in {0,1}, k in {0..5}.

This can be viewed as: for each output column k, we compute C[:,k] = A @ B[:,k].
So it's 3 independent 2x2x2 multiplications (one for each pair of output columns),
giving 7*3 = 21 terms for the 2x2x6 problem.

So total rank = 2 (middle blocks) * 21 = 42. Same as before.

The structural change: try to use a different block decomposition or a global
search for rank reduction. One approach: use the fact that some Strassen terms
across different B-blocks might be linearly dependent or combinable.

Let me try a rank-reduction search: start with the 42-term decomposition and try
to find linear dependencies among the columns of U, V, W that allow merging.

Actually, a cleaner structural approach: use the "laser method" or "group algebra"
method from algebraic complexity theory. But for this small case, let me just try
a local search: start with rank 42 and try to reduce by 1, 2, ... terms.

Given time constraints, let me implement a structured approach:
1. Build the 42-term Strassen decomposition.
2. Try to find if any two terms can be merged (their U, V, W vectors are linearly
   dependent in a way that allows combining into one term).
3. Return the best found.

For the merge check: two terms r1, r2 can be merged if there exists a scalar c such that
U_r1 = c * U_r2, V_r1 = c * V_r2, W_r1 = (1/c) * W_r2 (up to the product being preserved).
Actually, for Kronecker product terms, term r = u_r ⊗ v_r ⊗ w_r.
We can merge r1 and r2 into one term if u1⊗v1⊗w1 + u2⊗v2⊗w2 is itself a rank-1 tensor.
This happens iff there exist scalars such that the sum is a Kronecker product.

A simpler sufficient condition: if u1 = a*u2, v1 = b*v2, w1 = c*w2 with a*b*c = 1,
then the sum is (a*u2⊗b*v2⊗c*w2) + (u2⊗v2⊗w2) = (a*b*c + 1) * (u2⊗v2⊗w2) = 2*(u2⊗v2⊗w2),
which is rank-1. But this requires a*b*c=1 and a specific ratio.

In general, this is hard to check. Let me just try a numerical approach: for each pair
of columns, check if the 48-dim (for U) * 24-dim (for V) * 12-dim (for W) combined
tensor is rank-1 when summed.

Given the time budget, let me implement this carefully.
"""

import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    
    def make_strassen_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 += M2
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): 1}
        ))
        # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        terms.append((
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ))
        # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
        terms.append((
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ))
        # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ))
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        return terms
    
    strassen_terms = make_strassen_terms()
    
    # Build the 42-term decomposition
    rank_init = 42
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    col = 0
    for a_block in range(2):
        for b_block in range(3):
            b_row_start = a_block * 2
            b_col_start = b_block * 2
            
            for u_dict, v_dict, w_dict in strassen_terms:
                for (i_local, j_local), val in u_dict.items():
                    i_global = i_local
                    j_global = a_block * 2 + j_local
                    U[i_global * m + j_global, col] += val
                
                for (i_local, j_local), val in v_dict.items():
                    i_global = b_row_start + i_local
                    j_global = b_col_start + j_local
                    V[i_global * p + j_global, col] += val
                
                for (i_local, j_local), val in w_dict.items():
                    i_global = i_local
                    j_global = b_col_start + j_local
                    W[j_global * n + i_global, col] += val
                
                col += 1
    
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding mergeable pairs
    def try_merge(U, V, W, r1, r2):
        """Check if terms r1 and r2 can be merged into one term."""
        # The sum of two Kronecker terms u1⊗v1⊗w1 + u2⊗v2⊗w2
        # is rank-1 iff the combined tensor has rank 1.
        # We can check this by forming the matrix U_combined[:, :2] * V_combined[:, :2]^T * ... 
        # Actually, let's just check if the 2-term sum is expressible as a single Kronecker product.
        
        # Form the 3-mode tensor: T[a,b,c] = U[a,r1]*V[b,r1]*W[c,r1] + U[a,r2]*V[b,r2]*W[c,r2]
        # Check if this tensor is rank-1: i.e., T[a,b,c] = x[a]*y[b]*z[c] for some x,y,z.
        # This is equivalent to: the matricizations T_(1), T_(2), T_(3) all have rank 1.
        
        ua, va, wa = U[:, r1], V[:, r1], W[:, r1]
        ub, vb, wb = U[:, r2], V[:, r2], W[:, r2]
        
        # Check if there exist scalars alpha, beta such that:
        # ua + alpha*ub = x * scale_u
        # va + alpha*vb = y * scale_v  
        # wa + beta*wb = z * scale_w
        # and x*y*z = 1 (up to the product)
        
        # Simpler: check if ua is proportional to ub, va to vb, wa to wb
        if np.allclose(ub, 0):
            return None
        if np.allclose(ua, 0):
            # Try swapping
            ua, ub = ub, ua
            va, vb = vb, va
            wa, wb = wb, wa
        
        # Check proportionality
        try:
            cu = np.dot(ua, ub) / np.dot(ub, ub)
            if not np.allclose(ua, cu * ub, atol=1e-9):
                return None
        except:
            return None
        
        try:
            cv = np.dot(va, vb) / np.dot(vb, vb)
            if not np.allclose(va, cv * vb, atol=1e-9):
                return None
        except:
            return None
        
        try:
            cw = np.dot(wa, wb) / np.dot(wb, wb)
            if not np.allclose(wa, cw * wb, atol=1e-9):
                return None
        except:
            return None
        
        # If all proportional, the sum is rank-1
        # New term: (ua+ub) ⊗ (va+vb) ⊗ (wa+wb) is proportional to ub⊗vb⊗wb
        # We can just keep one term and scale
        # Actually, the sum ua⊗va⊗wa + ub⊗vb⊗wb = (cu+1)*ub⊗vb⊗wb if va=cv*vb, etc.
        # Wait, ua = cu*ub, va = cv*vb, wa = cw*wb
        # Sum = cu*ub⊗cv*vb⊗cw*wb + ub⊗vb⊗wb = (cu*cv*cw + 1) * (ub⊗vb⊗wb)
        # So the new single term has coefficient (cu*cv*cw + 1)
        # But we need the new term to still satisfy the decomposition.
        # The new U column = ua + ub = (cu+1)*ub
        # The new V column = va + vb = (cv+1)*vb
        # The new W column = wa + wb = (cw+1)*wb
        # Product: (cu+1)*(cv+1)*(cw+1) * ub⊗vb⊗wb
        # Original: cu*cv*cw * ub⊗vb⊗wb + ub⊗vb⊗wb = (cu*cv*cw+1) * ub⊗vb⊗wb
        # These are equal iff (cu+1)(cv+1)(cw+1) = cu*cv*cw + 1
        # Expanding LHS: cu*cv*cw + cu*cv + cu*cw + cv*cw + cu + cv + cw + 1
        # RHS: cu*cv*cw + 1
        # So we need: cu*cv + cu*cw + cv*cw + cu + cv + cw = 0
        # This is a specific condition. Let's check it.
        
        cond = cu*cv + cu*cw + cv*cw + cu + cv + cw
        if abs(cond) > 1e-6:
            return None
        
        # Merge possible: new term is (ua+ub, va+vb, wa+wb)
        new_u = ua + ub
        new_v = va + vb
        new_w = wa + wb
        return (new_u, new_v, new_w)
    
    # Try all pairs
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank_init
    merged_pairs = set()
    
    # Sort pairs by "closeness" to prioritize likely merges
    pair_scores = []
    for r1 in range(current_rank):
        for r2 in range(r1+1, current_rank):
            # Score: how proportional are the U, V, W vectors?
            ua, ub = current_U[:, r1], current_U[:, r2]
            va, vb = current_V[:, r1], current_V[:, r2]
            wa, wb = current_W[:, r1], current_W[:, r2]
            
            nu = np.linalg.norm(ua) * np.linalg.norm(ub)
            nv = np.linalg.norm(va) * np.linalg.norm(vb)
            nw = np.linalg.norm(wa) * np.linalg.norm(wb)
            
            if nu < 1e-12 or nv < 1e-12 or nw < 1e-12:
                continue
            
            cu = np.dot(ua, ub) / (nu + 1e-12)
            cv = np.dot(va, vb) / (nv + 1e-12)
            cw = np.dot(wa, wb) / (nw + 1e-12)
            
            # Score: how close to being proportional
            score = -abs(cu) - abs(cv) - abs(cw)  # More negative = more proportional
            pair_scores.append((score, r1, r2))
    
    pair_scores.sort()
    
    # Try merging in order of best score
    for score, r1, r2 in pair_scores:
        if time.time() > deadline - 5:
            break
        
        result = try_merge(current_U, current_V, current_W, r1, r2)
        if result is not None:
            new_u, new_v, new_w = result
            # Replace r1 with the merged term, remove r2
            current_U[:, r1] = new_u
            current_V[:, r1] = new_v
            current_W[:, r1] = new_w
            # Remove r2 by shifting columns
            idx = list(range(current_rank))
            idx.remove(r2)
            current_U = current_U[:, idx]
            current_V = current_V[:, idx]
            current_W = current_W[:, idx]
            current_rank -= 1
            merged_pairs.add((min(r1,r2), max(r1,r2)))
    
    # Verify the reduced decomposition
    if verify(current_U, current_V, current_W):
        return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
    else:
        # Fallback to original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
