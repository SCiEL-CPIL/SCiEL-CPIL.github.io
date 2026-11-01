"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses structural insight: each rank component is tied to a single j-index.
    For each j, decompose the 2x4 sub-tensor, then combine.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1) > 1e-9:
                        return False
        
        # Check that T[a,b,c] = 0 when (a,b,c) doesn't correspond to valid multiplication
        for a in range(nm):
            i, j = divmod(a, m)
            for b in range(mp):
                j2, k = divmod(b, p)
                if j2 != j:
                    for c in range(pn):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    valid_c = k * n + i
                    for c in range(pn):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if c == valid_c:
                            if abs(val - 1) > 1e-9:
                                return False
                        else:
                            if abs(val) > 1e-9:
                                return False
        return True
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if all entries are integer or half-integer within tolerance."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val) < 1e-12:
                        continue
                    two_val = 2 * val
                    nearest = round(two_val)
                    if abs(two_val - nearest) > tol:
                        return False
        return True
    
    # Schoolbook solution of rank 24
    rank_schoolbook = n * m * p  # 24
    U_sb = [[0] * rank_schoolbook for _ in range(nm)]
    V_sb = [[0] * rank_schoolbook for _ in range(mp)]
    W_sb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Structural insight: each rank component is tied to a single j-index.
    # For each j, we need to decompose the 2x4 sub-tensor where T[i,k] = 1 for all i,k.
    # The sub-tensor for fixed j has entries at positions (i, j) in U, (j, k) in V, (k, i) in W.
    # 
    # Key: for fixed j, the constraint is that for each (i,k), the sum over rank components
    # assigned to j of U[i,j,r] * V[j,k,r] * W[k,i,r] = 1.
    # And for (i',k') != (i,k), the sum must be 0.
    # 
    # This is equivalent to: the 2x4 matrix of ones must be decomposed as a sum of rank-1
    # matrices (with an extra scalar from W). Specifically, if we fix W[k,i,r] = w_{k,i,r},
    # then U[i,j,r] * V[j,k,r] * w_{k,i,r} must sum to 1 for all (i,k).
    # 
    # Let's define for each rank component r assigned to j:
    #   u_i = U[i*m+j, r], v_k = V[j*p+k, r], w_{k,i} = W[k*n+i, r]
    # Then we need: sum_r u_i * v_k * w_{k,i} = 1 for all i,k
    # And: sum_r u_i * v_k * w_{k',i'} = 0 for (k',i') != (k,i)
    #
    # The second condition means: for each r, the matrix w_r (indexed by k,i) must be such that
    # w_{k',i'} = 0 unless (k',i') is in the support of u * v.
    # 
    # Simplest case: w_{k,i,r} = delta_{k,i} * c_r for some scalar. But that doesn't work
    # because we need w to be nonzero at all (k,i) pairs.
    #
    # Alternative: use the fact that for fixed j, the problem is to write the all-ones 2x4
    # matrix as a sum of rank-1 matrices. The minimum rank is 2 (since the matrix has rank 2).
    # But we also have the W factor...
    #
    # Let me think differently. For fixed j, define:
    #   A_j[i,k] = 1 for all i in [0,2), k in [0,4)
    # We need: A_j[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # If we set W[k,i,r] = 1 for all k,i,r (within this j's components), then we need:
    #   1 = sum_r U[i,j,r] * V[j,k,r] for all i,k
    # This means the 2x4 all-ones matrix = sum_r u_r * v_r^T (outer products)
    # The minimum rank for this is 2 (since the all-ones matrix has rank 1, actually!).
    # Wait, the all-ones 2x4 matrix has rank 1. So we can write it as a single outer product:
    #   ones_2x4 = [1,1]^T * [1,1,1,1]
    # So U[i,j,r] = 1 for all i, V[j,k,r] = 1 for all k, W[k,i,r] = 1 for all k,i.
    # That's rank 1 per j!
    #
    # But wait, does this satisfy the zero constraints? For (i',k') != (i,k):
    #   sum_r U[i,j,r] * V[j,k,r] * W[k',i',r] = U[i,j,r] * V[j,k,r] * W[k',i',r]
    # If W[k',i',r] = 1 for all k',i',r, then this equals U[i,j,r]*V[j,k,r] which is 1*1 = 1,
    # NOT 0. So this doesn't work!
    #
    # The issue is that W is shared across different (i,k) pairs. We need W to be zero at
    # positions (k',i') that don't correspond to the "active" (i,k) for that rank component.
    #
    # So for each rank component r, the support of W must be restricted. If U[i,j,r] and
    # V[j,k,r] are both nonzero, then W[k,i,r] should be nonzero, but W[k',i',r] should be 0
    # for other (k',i') pairs.
    #
    # This means each rank component effectively handles a subset of (i,k) pairs. If U and V
    # have full support (all i and all k nonzero), then W must be zero everywhere except...
    # but we need W[k,i,r] to be nonzero for ALL (k,i) to get the sum to 1.
    #
    # Contradiction! So we can't have full support in U and V simultaneously.
    #
    # Resolution: each rank component handles a specific subset. For example:
    #   Rank component r: U[i,j,r] nonzero only for i in I_r, V[j,k,r] nonzero only for k in K_r
    #   Then W[k,i,r] can be nonzero only for (k,i) in K_r x I_r
    #   And we need: for each (i,k), sum over r where i in I_r and k in K_r of
    #     U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    #   And for (i,k) not in any I_r x K_r... but we need coverage of all (i,k).
    #
    # So we need the sets I_r x K_r to cover all (i,k) pairs, with appropriate coefficients.
    #
    # For n=2, p=4: we can use 4 rank components per j, each handling one k value:
    #   r=0: I={0,1}, K={0}, U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1
    #   r=1: I={0,1}, K={1}, U[i,j,1]=1, V[j,1,1]=1, W[1,i,1]=1
    #   etc.
    # Total: 3 j * 4 k = 12 rank components.
    #
    # Or 2 rank components per j, each handling both k values:
    #   r=0: I={0}, K={0,1,2,3}, U[0,j,0]=1, V[j,k,0]=1, W[k,0,0]=1
    #   r=1: I={1}, K={0,1,2,3}, U[1,j,1]=1, V[j,k,1]=1, W[k,1,1]=1
    # Total: 3 j * 2 i = 6 rank components.
    #
    # Let me verify the second option:
    # For (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    #   = U[i,j,0]*V[j,k,0]*W[k,i,0] + U[i,j,1]*V[j,k,1]*W[k,i,1]
    #   = (1 if i=0 else 0)*1*1 + (1 if i=1 else 0)*1*1
    #   = 1 if i=0 or i=1 = 1. ✓
    #
    # For (i,j,k) with wrong i': W[k,i',r] is nonzero only if i'=i (for the r where i matches).
    #   sum_r U[i,j,r]*V[j,k,r]*W[k,i',r]
    #   = (1 if i=0 else 0)*1*(1 if i'=0 else 0) + (1 if i=1 else 0)*1*(1 if i'=1 else 0)
    #   = 1 if i'=i, 0 otherwise. ✓
    #
    # For wrong k': V[j,k',r] is 1 for all k', so:
    #   sum_r U[i,j,r]*V[j,k',r]*W[k',i,r]
    #   = (1 if i=0 else 0)*1*(1 if i'=0 else 0) + ... wait, W[k',i,r] depends on k'
    #   For r=0: W[k',0,0] = 1 for all k'. So U[0,j,0]*V[j,k',0]*W[k',0,0] = 1*1*1 = 1
    #   For r=1: W[k',1,1] = 1 for all k'. So U[1,j,1]*V[j,k',1]*W[k',1,1] = 1*1*1 = 1 (if i=1)
    #   So sum = 1 (if i=0) + 1 (if i=1) = 1, NOT 0! ✗
    #
    # So this doesn't work either. The problem is that W[k',i,r] is nonzero for all k'.
    #
    # I need W to be zero at (k',i') for k' != k. But W is indexed by (k,i), so I can set
    # W[k',i,r] = 0 for k' != k_r where k_r is the "active" k for this component.
    #
    # Revised: each rank component r is associated with a specific (i_r, k_r) pair:
    #   U[i,j,r] = 1 if i = i_r, else 0
    #   V[j,k,r] = 1 if k = k_r, else 0  
    #   W[k,i,r] = 1 if k = k_r and i = i_r, else 0
    #
    # This gives rank 24 (one per (i,j,k) triple), which is schoolbook.
    #
    # To reduce rank, I need to share components. The constraint is that W must be zero
    # at positions that would create spurious terms. 
    #
    # Let me try: for each j, use 4 components (one per k), with full i-support:
    #   r_j,k: U[i,j,r] = 1 for all i, V[j,k,r] = 1, V[j,k',r] = 0 for k' != k
    #          W[k,i,r] = 1 for all i, W[k',i,r] = 0 for k' != k
    #
    # Check: for (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    #   Only r = r_j,k contributes: 1*1*1 = 1. ✓
    # For (i,j,k') with k' != k: V[j,k',r_j,k] = 0, so sum = 0. ✓
    # For (i',j,k) with i' != i: W[k,i',r_j,k] = 1 (since W is 1 for all i), so
    #   sum = U[i',j,r]*V[j,k,r]*W[k,i',r] = 1*1*1 = 1, NOT 0! ✗
    #
    # Still doesn't work. W[k,i,r] = 1 for all i causes problems.
    #
    # I think the fundamental issue is that for the 2x4 sub-tensor, the minimum rank is 8
    # (you need one component per (i,k) pair because of the zero constraints).
    #
    # Let me just implement the schoolbook solution and a smarter search.
    
    random.seed(42)
    
    # Try structured patterns with the constraint that each rank component is tied to a j
    # For each j, try to find a minimal decomposition of the 2x4 all-ones pattern
    
    candidates = [0.5, -0.5, 1, -1]
    
    # Try ranks from 12 to 23 (since 24 is schoolbook)
    for target_rank in range(12, 24):
        if time.time() > deadline - 5:
            break
        
        num_attempts = 300 if target_rank <= 15 else 150
        found = False
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
            
            U = [[0.0] * target_rank for _ in range(nm)]
            V = [[0.0] * target_rank for _ in range(mp)]
            W = [[0.0] * target_rank for _ in range(pn)]
            
            # Assign rank components to j-values
            # Each component is tied to one j
            j_assignments = []
            for r in range(target_rank):
                j_assignments.append(random.randint(0, m - 1))
            
            for r in range(target_rank):
                hub_j = j_assignments[r]
                
                # U: touch 1-2 (i, hub_j) positions
                num_u = random.randint(1, 2)
                u_is = random.sample(range(n), num_u)
                for i in u_is:
                    U[i * m + hub_j][r] = random.choice(candidates)
                
                # V: touch 1-2 (hub_j, k) positions
                num_v = random.randint(1, 2)
                v_ks = random.sample(range(p), num_v)
                for k in v_ks:
                    V[hub_j * p + k][r] = random.choice(candidates)
                
                # W: touch positions consistent with U and V supports
                # W[k,i,r] should be nonzero only if i is in u_is and k is in v_ks
                w_positions = [(k, i) for k in v_ks for i in u_is]
                if w_positions:
                    num_w = random.randint(1, min(2, len(w_positions)))
                    chosen = random.sample(w_positions, num_w)
                    for (k, i) in chosen:
                        W[k * n + i][r] = random.choice(candidates)
            
            if validate(U, V, W, target_rank):
                if check_half_integer(U, V, W):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
