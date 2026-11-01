"""Rewritten program for matrix multiplication 2,4,5 rank-minimal decomposition."""
import numpy as np
from scipy.optimize import linprog
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition using the structure of matrix multiplication
    # The tensor T_{i,j,k} = delta_{i,j,k} for matrix multiplication C = A * B
    # where A is n x m, B is m x p, C is n x p
    # T_{i,j,k} = 1 if (i,j,k) corresponds to a valid multiplication term, else 0
    # More precisely: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T_{i,k,j} = delta_{i,k} * something... 
    # Actually, the multiplication tensor has components:
    # For each (i, j, k): T[i*m+j, k*n+i] ... no, let me use the standard form.
    #
    # The matrix multiplication tensor T has T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # for some i in [0,n), j in [0,m), k in [0,p)
    #
    # We want to find rank r such that T = sum_{r=1}^{rank} U[:,r] * V[:,r] * W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Start with a reasonable rank bound. For 2x4x5, schoolbook is 40.
    # Let's try to find a decomposition with lower rank.
    
    # The tensor entries: T[a,b,c] = 1 if there exist i,j,k such that:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # Note: for a given a=i*m+j, we have i = a//m, j = a%m
    # For a given b=j*p+k, we have j = b//p, k = b%p
    # For a given c=k*n+i, we have k = c//n, i = c%n
    # So T[a,b,c] = 1 iff (a//m == c%n) and (a%m == b//p) and (b%p == c//n)
    
    # Let's build the tensor
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # We want to find the minimum rank decomposition.
    # This is NP-hard in general, but for small tensors we can try various ranks.
    # 
    # For a rank-r decomposition:
    # T[a,b,c] = sum_{s=1}^{r} U[a,s] * V[b,s] * W[c,s]
    #
    # This is a bilinear system in (U,V,W) for fixed r.
    # 
    # One approach: fix U and V, then W is determined linearly.
    # Or use the fact that for each c, the matrix T[:,:,c] = sum_s U*V_s * W[c,s]
    # where V_s is the s-th column of V.
    
    # Let's try a direct approach for small ranks.
    # For rank r, we have r * (nm + mp + pn) unknowns.
    # The constraints are nm * mp * pn equations, but only nm*mp*pn are non-trivial.
    
    # Actually, let's try a different approach. Since the tensor is sparse (only n*m*p = 40 non-zero entries),
    # maybe we can exploit this.
    
    # For each non-zero entry T[a,b,c] = 1, we need:
    # sum_s U[a,s] * V[b,s] * W[c,s] = 1
    # For all other entries (a,b,c) not in the support, we need:
    # sum_s U[a,s] * V[b,s] * W[c,s] = 0
    
    # This is a system of polynomial equations. Let's try to solve it for small r.
    
    # Strategy: try r from some lower bound up to 40 (schoolbook)
    # Lower bound: at least max(nm, mp, pn) = max(8, 20, 10) = 20? No, that's not right.
    # The rank is at least the matrix rank of any unfolding.
    
    # Let's compute the rank of the matricizations to get a lower bound.
    T_01_2 = T.reshape(nm, mp * pn)
    T_0_12 = T.reshape(nm * mp, pn)
    T_10_2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    
    rank_01_2 = np.linalg.matrix_rank(T_01_2)
    rank_0_12 = np.linalg.matrix_rank(T_0_12)
    rank_10_2 = np.linalg.matrix_rank(T_10_2)
    
    lower_bound = max(rank_01_2, rank_0_12, rank_10_2)
    
    # Try ranks from lower_bound to 40
    for r in range(lower_bound, 41):
        # For each rank r, try to find a decomposition.
        # We'll use a randomized approach: initialize U and V randomly, then solve for W.
        # Then check if the reconstruction is exact.
        
        # Actually, let's use a more systematic approach.
        # Fix U to be a matrix with specific structure.
        
        # For rank r, we can set U to be the identity-like matrix or use a random matrix.
        # Then the equations become linear in V and W given U, but they're still bilinear.
        
        # Let's try: set U[a,s] = 1 if s corresponds to a "basis" element, else 0.
        # Specifically, if r >= nm, we can set U to be the identity padded with zeros.
        # Then T[a,b,c] = V[b,a] * W[c,a] for a < nm, and 0 otherwise.
        # This requires V[b,a] * W[c,a] = T[a,b,c] for all a,b,c.
        # For a given a, T[a,b,c] is non-zero only for specific (b,c).
        # If T[a,b,c] = 1, we need V[b,a] * W[c,a] = 1.
        # If T[a,b,c] = 0, we need V[b,a] * W[c,a] = 0.
        
        if r >= nm:
            # Set U to be [I_nm | 0]
            U = np.zeros((nm, r))
            for a in range(nm):
                U[a, a] = 1
            
            # Now for each a, we need to find V[:,a] and W[:,a] such that:
            # V[b,a] * W[c,a] = T[a,b,c] for all b,c.
            # Let's collect the constraints for each a.
            
            success = True
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            for a in range(nm):
                # Get the slice T[a,:,:]
                Ta = T[a, :, :]
                # Find non-zero entries
                nonzero = np.nonzero(Ta)
                if len(nonzero[0]) == 0:
                    # All zeros, so V[:,a] and W[:,a] can be anything (as long as products are 0)
                    # Set them to 0
                    continue
                
                b_vals = nonzero[0]
                c_vals = nonzero[1]
                
                # We need V[b,a] * W[c,a] = 1 for (b,c) in support
                # and V[b,a] * W[c,a] = 0 for all other (b,c)
                
                # Let's try: set V[b,a] = 1 for b in b_vals, 0 otherwise
                # Then W[c,a] must be 1 for c in c_vals, 0 otherwise
                # But we need to check consistency: if (b1,c1) and (b2,c2) are both in support,
                # and b1 != b2 or c1 != c2, this might not work directly.
                
                # Actually, for matrix multiplication, for a fixed a = i*m+j:
                # T[a,b,c] = 1 iff b = j*p+k and c = k*n+i for some k.
                # So for fixed a, the support is {(j*p+k, k*n+i) : k in [0,p)}
                # These are p distinct (b,c) pairs.
                
                # Let's set V[b,a] = 1 if b is in the support's b-values, else 0
                # And W[c,a] = 1 if c is in the support's c-values, else 0
                # But we need V[b,a]*W[c,a] = 1 only when (b,c) is in support.
                # If we set V[b,a]=1 for all b in support_b and W[c,a]=1 for all c in support_c,
                # then V[b,a]*W[c,a] = 1 for all (b,c) in support_b x support_c, which is larger than the support.
                
                # So this doesn't work directly. We need a more careful assignment.
                
                # For fixed a, the support is a "matching" between b-values and c-values.
                # Specifically, it's a permutation-like structure.
                
                # Let's try: for each (b,c) in support, assign a unique value.
                # Set V[b,a] = 1/sqrt(p) and W[c,a] = sqrt(p) for all in support? No.
                
                # Actually, since the support for fixed a has p entries, and they form a "graph"
                # where each b is connected to exactly one c and vice versa (it's a perfect matching),
                # we can set V[b,a] = 1 and W[c,a] = 1 for the matched pairs, and 0 otherwise.
                
                # Let's verify: for fixed a=i*m+j, the support is:
                # k=0: b=j*p+0, c=0*n+i = i
                # k=1: b=j*p+1, c=1*n+i = n+i
                # ...
                # k=p-1: b=j*p+p-1, c=(p-1)*n+i
                # So each b is distinct and each c is distinct. It's indeed a perfect matching.
                
                # So we can set V[b,a] = 1 for b in support_b, 0 otherwise
                # and W[c,a] = 1 for c in support_c, 0 otherwise
                # Then V[b,a]*W[c,a] = 1 iff b in support_b AND c in support_c
                # But we need it to be 1 only when (b,c) is the matched pair.
                # Since support_b has p elements and support_c has p elements,
                # and the support is a perfect matching, the product V[b,a]*W[c,a] will be 1
                # for all p*p combinations, not just the p matched ones.
                
                # So this approach fails. We need a different strategy.
                
                # Alternative: use different values for different k.
                # Set V[j*p+k, a] = alpha_k and W[k*n+i, a] = beta_k such that alpha_k * beta_k = 1.
                # And for b not in support_b, V[b,a] = 0, and for c not in support_c, W[c,a] = 0.
                # This works! Because then V[b,a]*W[c,a] = alpha_k * beta_k = 1 only when
                # b = j*p+k and c = k*n+i for the same k.
                
                # So let's do this.
                for idx in range(len(b_vals)):
                    b_val = b_vals[idx]
                    c_val = c_vals[idx]
                    V[b_val, a] = 1
                    W[c_val, a] = 1
                
                # Check: for b in support_b and c in support_c, if (b,c) is not a matched pair,
                # V[b,a]*W[c,a] = 1*1 = 1, but T[a,b,c] = 0. So this still fails!
                
                # I see the issue. The support is a matching, but the Cartesian product of the
                # b-values and c-values is larger than the matching.
                
                # We need V[b,a]*W[c,a] to be 1 only for matched pairs.
                # One way: use orthogonal vectors. But V[:,a] and W[:,a] are scalars per entry.
                
                # Actually, for a rank-r decomposition, each "rank-1" term is U[:,s] * V[:,s] * W[:,s].
                # For the a-th slice (fixing a), we have T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s].
                # If U[a,s] = delta_{s,a} (i.e., U is identity), then T[a,b,c] = V[b,a] * W[c,a].
                # This requires V[b,a]*W[c,a] to match T[a,b,c] exactly.
                
                # The problem is that T[a,:,:] has a "cross" structure, not a rank-1 structure.
                # So we can't represent it as a single outer product V[:,a] * W[:,a].
                
                # We need multiple rank-1 terms for each a.
                
                success = False
                break
            
            if success:
                break
        
        # If r < nm or the above didn't work, try a different approach.
        # For now, let's just return the schoolbook if we can't find a better one.
    
    # Fallback: return schoolbook decomposition
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
# EVOLVE-BLOCK-END
