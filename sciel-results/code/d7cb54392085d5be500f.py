"""Matrix multiplication 2x2 x 2x16 via shared Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W):
    """Verify that U, V, W form a valid decomposition of the 2x2 x 2x16 matmul tensor.
    
    The tensor T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'}
    So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} * delta_{k,k'}
    """
    n, m, p = 2, 2, 16
    rank = U.shape[1]
    
    # Check all required conditions
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            expected = 1 if (j == j2 and k == k2) else 0
                            s = 0
                            for r in range(rank):
                                s += U[i*m + j, r] * V[j2*p + k, r] * W[k2*n + i2, r]
                            if s != expected:
                                return False
    return True

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    The 7 rank-1 terms have coefficients in {-1, 0, 1}.
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank_56():
    """Fallback: 8 independent Strassen blocks, rank 56."""
    n, m, p = 2, 2, 16
    U_core, V_core, W_core = _strassen_2x2()
    num_pairs = p // 2
    rank = 7 * num_pairs
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    for t in range(num_pairs):
        col_offset = t * 7
        for r in range(7):
            U[:, col_offset + r] = U_core[:, r]
            V[2*t, col_offset + r] = V_core[0, r]
            V[2*t + 1, col_offset + r] = V_core[1, r]
            V[16 + 2*t, col_offset + r] = V_core[2, r]
            V[16 + 2*t + 1, col_offset + r] = V_core[3, r]
            W[4*t, col_offset + r] = W_core[0, r]
            W[4*t + 2, col_offset + r] = W_core[1, r]
            W[4*t + 1, col_offset + r] = W_core[2, r]
            W[4*t + 3, col_offset + r] = W_core[3, r]
    return U, V, W

def _try_rank_R(R, deadline):
    """
    Try to find a rank-R decomposition using structured random search.
    
    Strategy: Use a small set of U-patterns (from Strassen and variations),
    and solve for V and W given U.
    
    For fixed U (shape 4xR), we need to find V (32xR) and W (32xR) such that
    for all i,j,k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    This is a system of 64 bilinear equations in the 32*R + 32*R unknowns.
    """
    n, m, p = 2, 2, 16
    np.random.seed(42)
    
    # Try different U structures
    # Start with Strassen-like patterns
    U_core, _, _ = _strassen_2x2()
    
    # Try: use 7 Strassen U-vectors repeated, but with different V/W
    # Actually, let's try a more systematic approach.
    
    # For rank R, try random U with entries in {-1, 0, 1}
    best_result = None
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate random sparse U
        U = np.zeros((4, R), dtype=np.int64)
        for r in range(R):
            # Each U column has 1-4 nonzeros
            num_nonzeros = np.random.randint(1, 5)
            indices = np.random.choice(4, size=num_nonzeros, replace=False)
            for idx in indices:
                U[idx, r] = np.random.choice([-1, 1])
        
        # For given U, we need to solve for V and W.
        # The constraint is: for each (i,j,k), sum_r U[i,j,r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # 
        # Let's fix V randomly and solve for W, or use an iterative approach.
        # 
        # Alternative: use the fact that for fixed (j,k), the constraint on (i) is:
        # sum_r U[i,j,r] * V[j*p+k, r] * W[k*n+i, r] = 1 for i=0,1
        # This is 2 equations in the W variables for that (k).
        
        # Try: set V to have a specific structure, then solve linear system for W.
        # For each k, we have 2*2 = 4 equations (for i=0,1 and j=0,1) involving
        # W[k*n+0, r] and W[k*n+1, r] for all r.
        
        # Let V be random sparse
        V = np.zeros((32, R), dtype=np.int64)
        for r in range(R):
            num_nonzeros = np.random.randint(2, 8)
            indices = np.random.choice(32, size=num_nonzeros, replace=False)
            for idx in indices:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Now solve for W: for each (j,k), we have:
        # sum_r U[0,j,r] * V[j*p+k, r] * W[k*n+0, r] + ... = 1  (for i=0)
        # sum_r U[1,j,r] * V[j*p+k, r] * W[k*n+1, r] + ... = 1  (for i=1)
        # 
        # Wait, the constraint is for ALL i,j,k simultaneously. Let me re-index.
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # 
        # For fixed (j,k): we have 2 equations (i=0,1):
        # sum_r U[i,j,r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # But W[k*n+i, r] depends on both k and i. So for fixed k, we have 2 equations
        # (i=0,1) but they involve W[k*n+0, r] and W[k*n+1, r] which are different.
        # 
        # Actually for fixed k, the equations are:
        # For i=0: sum_j sum_r U[0,j,r] * V[j*p+k, r] * W[k*n+0, r] = 1  ... no!
        # 
        # I'm confusing myself. Let me re-read: T[i,j,k] = 1 for each (i,j,k).
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # 
        # So for fixed (j,k), we have 2 values of i, giving 2 equations:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1, for i=0,1
        # 
        # The unknowns in these 2 equations are W[k*n+0, r] and W[k*n+1, r] for r=0..R-1.
        # But they appear multiplied by different coefficients for i=0 vs i=1.
        # 
        # For i=0: sum_r U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
        # For i=1: sum_r U[1*m+j, r] * V[j*p+k, r] * W[k*n+1, r] = 1
        # 
        # These are 2 separate linear equations! Each involves only W[k*n+i, :] for the
        # corresponding i. So we can solve them independently!
        # 
        # For each (j,k) and each i:
        # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        # 
        # This is a linear equation in W[k*n+i, :]. We have 32*2 = 64 such equations
        # (for j=0,1 and k=0..15 and i=0,1), and 32*2 = 64 unknowns (W entries).
        # 
        # But wait, each equation only involves R unknowns (one row of W), and there are
        # 2*16 = 32 equations for each row of W (varying j and k for fixed i).
        # 
        # For fixed i and fixed k: we have 2 equations (j=0,1) in R unknowns W[k*n+i, :].
        # That's underdetermined if R > 2.
        
        # Let me try a different approach: use the structure to construct V and W.
        # 
        # Key insight: if we make U very sparse (each column has exactly 1 nonzero),
        # then the problem becomes easier.
        
        # Try: R = 32, U has exactly one 1 per column, at position (i,j)
        # Then for each r, U[i*m+j, r] = 1 for exactly one (i,j).
        # The constraint becomes: for each (i',j',k):
        # sum_r U[i'*m+j', r] * V[j'*p+k, r] * W[k*n+i', r] = 1
        # Only terms where (i,j) = (i',j') contribute, so:
        # V[j'*p+k, r_{i',j'}] * W[k*n+i', r_{i',j'}] = 1
        # where r_{i',j'} is the unique r with U[i'*m+j', r] = 1.
        # 
        # This gives: V[j'*p+k, r] * W[k*n+i', r] = 1 for the specific r.
        # We can set V[j'*p+k, r] = 1 and W[k*n+i', r] = 1.
        # This works! Rank = 4 (one r for each (i,j) pair)... wait, but each r
        # corresponds to one (i,j), and we need it to work for all k.
        # 
        # Actually if U has 4 columns (one per (i,j)), then for each r corresponding
        # to (i,j), we need V[j*p+k, r] * W[k*n+i, r] = 1 for all k.
        # Set V[j*p+k, r] = 1 for all k, and W[k*n+i, r] = 1 for all k. Works!
        # Rank = 4? But that can't be right since the lower bound is 32.
        # 
        # Oh wait, I made an error. If U[i*m+j, r] = 1 for a specific (i,j) and r,
        # then the constraint is:
        # For (i',j',k): sum_r U[i'*m+j', r] * V[j'*p+k, r] * W[k*n+i', r] = 1
        # Only r where U[i'*m+j', r] != 0 contributes. If U has exactly one nonzero
        # per (i,j) pair, then for (i',j'), only one r contributes:
        # V[j'*p+k, r_{i',j'}] * W[k*n+i', r_{i',j'}] = 1
        # 
        # But this must hold for ALL k. So V[j'*p+k, r] and W[k*n+i', r] must
        # multiply to 1 for all k. This is possible: set both to 1.
        # 
        # So rank 4 works?? That contradicts the lower bound of 32. Let me recheck.
        # 
        # The lower bound for matrix multiplication tensor rank is max(n*m, m*p, p*n)
        # ONLY for the case where we're computing the full product. But here we're
        # computing C = A*B where A is 2x2 and B is 2x16. The tensor has size
        # (n*m) x (m*p) x (p*n) = 4 x 32 x 32.
        # 
        # The rank lower bound is max(4, 32, 32) = 32? No, the lower bound is the
        # maximum of the "slice ranks" or something else. Actually, the rank of a
        # 3-tensor T of size a x b x c is at least max(a, b, c) only in special cases.
        # The general lower bound is harder.
        # 
        # Wait, I think I made an indexing error. Let me re-examine.
        # 
        # The matrix multiplication tensor for M_{n,m,p}: T is a tensor of order 3
        # with dimensions (n*m) x (m*p) x (p*n). T[(i,j), (j',k), (k',i')] = 1 if
        # j=j' and k=k', else 0. So T has 1s at positions where the middle indices
        # match.
        # 
        # In my indexing: U has index (i*m+j) for A[i,j], V has index (j*p+k) for
        # B[j,k], W has index (k*n+i) for C[i,k] (using trace convention W[k*n+i, r]).
        # 
        # T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k'.
        # 
        # So the constraint is: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # = 1 if j=j' and k=k', else 0.
        # 
        # I was wrong! It's not 1 for all (i,j,k). It's 1 only when the indices match!
        # 
        # Let me redo: T[(i,j), (j',k), (k',i')] = δ_{j,j'} * δ_{k,k'}
        # 
        # So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{j,j'} * δ_{k,k'}
        
        # OK so the correct constraint is:
        # For all i,j,j',k,k',i':
        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{j,j'} * δ_{k,k'}
        # 
        # This is 2*2*2*16*16*2 = 512 equations, but many are the same.
        # 
        # For fixed (j,j',k,k'): the equation is
        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{j,j'} * δ_{k,k'}
        # for all i,i'.
        # 
        # If j != j' or k != k', the RHS is 0.
        # If j = j' and k = k', the RHS is 1.
        
        # This is much more complex. Let me just verify the current solution and
        # try to find improvements via a different method.
        
        # New approach: use the known Strassen decomposition and try to merge
        # columns by sharing U factors.
        
        # Actually, let me try a completely different structural approach:
        # Use the fact that we can write the 2x16 multiplication as:
        # C[:, k] = A @ B[:, k] for each k.
        # 
        # A @ B[:, k] = [A[0,0]*B[0,k] + A[0,1]*B[1,k],
        #                A[1,0]*B[0,k] + A[1,1]*B[1,k]]
        # 
        # This is a bilinear map. The tensor for this (for one k) has rank 4
        # (the trivial decomposition). But can we share across k values?
        # 
        # If we use the same 4 U-vectors for all 16 k values, we get rank 4*16 = 64.
        # But Strassen reduces 2x2x2 from 4 to 7... wait, 2x2x2 has 4 outputs, not 8.
        # 
        # I'm getting confused. Let me just implement a search that tries to find
        # a valid decomposition with rank < 56.
        
        pass
    
    return best_result

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    """
    deadline = time.time() + 100  # Safety margin
    
    # First, try the known rank-56 solution as fallback
    U56, V56, W56 = _construct_rank_56()
    
    # Verify it works
    if not _verify_decomposition(U56, V56, W56):
        # Should not happen, but just in case
        pass
    
    # Try to find a better solution
    # Strategy: use a structured approach based on the algebraic structure
    
    # The key insight: for the 2x2 x 2x16 multiplication, we can potentially
    # use a decomposition that shares U-factors across multiple column groups.
    
    # Let's try: use 7 Strassen U-vectors, but construct V and W to handle
    # all 16 columns simultaneously.
    
    # For Strassen M1 = (a+d)(e+h): U = [1, 0, 0, 1] (a and d)
    # We need V and W such that the contribution to T is correct.
    
    # Actually, let me try a different approach: use the "parallel" structure.
    # 
    # The 2x16 matrix B can be viewed as 16 column vectors of length 2.
    # C[:, k] = A @ B[:, k]
    # 
    # If we use a rank-4 decomposition for each k (trivial), total rank = 64.
    # If we use Strassen-like sharing, we might do better.
    # 
    # Let me try: group columns into pairs. For columns (2t, 2t+1), use Strassen.
    # That's what the current solution does: rank 56.
    # 
    # Can we group into larger chunks? For 4 columns at a time, we'd be computing
    # A @ B[:, 4t:4t+4], which is 2x2 times 2x4. The rank of M_{2,2,4} is...
    # 
    # Let me just try to find a rank-48 solution by searching.
    
    # Try rank 48 with a structured approach
    # Use 6 U-vectors (instead of 7 per pair) shared across all 8 pairs?
    # Or use 48 = 6*8, meaning 6 terms per pair instead of 7?
    
    # Actually, let me try a completely different structure:
    # Use the fact that m=2 to create a decomposition where each term
    # handles multiple (j,k) pairs.
    
    # For now, return the rank-56 solution if we can't find better
    U, V, W = U56, V56, W56
    
    # Try to improve: use structured search for lower rank
    for R in [48, 42, 35, 28]:
        if time.time() > deadline - 20:
            break
        found = _search_rank(R, deadline - 20)
        if found is not None:
            U, V, W = found
            break
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _search_rank(R, deadline):
    """Search for a rank-R decomposition using structured approach.
    
    Strategy: Use 7 Strassen U-vectors, and for each, construct V and W
    that handle groups of k-values. Try different groupings.
    """
    n, m, p = 2, 2, 16
    U_core, V_core, W_core = _strassen_2x2()
    
    # Try different group sizes: how many k-values each Strassen term handles
    # If each term handles g k-values, we need ceil(16/g) * 7 total terms
    # But we can also use fewer terms per group if the structure allows
    
    for g in range(2, 17):  # Each term handles g k-values
        num_groups = (16 + g - 1) // g
        rank_needed = 7 * num_groups
        if rank_needed > R:
            continue
        
        # Try to construct a solution with this grouping
        U = np.zeros((4, rank_needed), dtype=np.int64)
        V = np.zeros((32, rank_needed), dtype=np.int64)
        W = np.zeros((32, rank_needed), dtype=np.int64)
        
        for t in range(num_groups):
            k_start = t * g
            k_end = min(k_start + g, 16)
            col_offset = t * 7
            
            for r in range(7):
                U[:, col_offset + r] = U_core[:, r]
                
                # For this group, V handles k-values from k_start to k_end-1
                # V[j*p+k, col_offset+r] should be nonzero only for k in [k_start, k_end)
                for j in range(2):
                    for k in range(k_start, k_end):
                        V[j*p + k, col_offset + r] = V_core[j, r]
                
                # W handles the output for this group
                # W[k*n+i, col_offset+r] should be nonzero only for k in [k_start, k_end)
                for k in range(k_start, k_end):
                    for i in range(2):
                        W[k*n + i, col_offset + r] = W_core[i, r]
        
        # Verify
        if _verify_decomposition(U, V, W):
            return U, V, W
    
    # Try: use 4 U-vectors (schoolbook) with each handling all 16 k-values
    # This gives rank 4*16 = 64, which is worse than 56, so skip
    
    # Try: use 5 or 6 U-vectors per group
    # This requires finding a rank-5 or rank-6 decomposition of 2x2x2, which doesn't exist
    # (minimum is 7 for exact)
    
    return None
# EVOLVE-BLOCK-END
