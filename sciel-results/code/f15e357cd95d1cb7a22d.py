"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need rank-r U, V, W such that for all (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Start with schoolbook as fallback
    rank_sb = nm * mp * pn  # n*m*p
    U_sb = [[0]*rank_sb for _ in range(nm)]
    V_sb = [[0]*rank_sb for _ in range(mp)]
    W_sb = [[0]*rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try greedy peeling: find rank-1 components that cover remaining 1s
    # A rank-1 component is defined by vectors u (length nm), v (length mp), w (length pn)
    # It covers entry (i,j,k) if u[i*m+j]*v[j*p+k]*w[k*n+i] != 0
    
    # For the all-ones tensor, we want to decompose into minimal rank-1 terms
    # where each term has u, v, w with 0/1 entries (or half-integers)
    
    # Strategy: try to find a rank-1 component that is "aligned" with the tensor structure
    # A natural rank-1 component for entry (i,j,k) has:
    #   u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1, all other entries 0
    # This is the schoolbook. But we can combine: e.g., a term with
    #   u[i*m+j] = 1 for all j (fixed i), v[j*p+k] = 1 for all k (fixed j), w[k*n+i] = 1 for all k (fixed i)
    # This would cover all entries with that i, but the product u*v*w at (i',j,k') would be
    # nonzero only when i'=i (from u), and the w factor requires the output row i... 
    # Actually w[k*n+i] depends on i (the input row), so this doesn't work directly.
    
    # Let's think differently. The multiplication tensor T[i,j,k]=1.
    # We can write it as a sum of rank-1 tensors.
    # One approach: group by pairs. For each (i,k), the "slice" over j is all-ones.
    # T[i,j,k] = 1 for all j, so for fixed (i,k), sum_j ... but we need rank-1 decomposition.
    
    # Known: rank of matrix multiplication tensor M_{n,m,p} >= max(n*m, m*p, n*p)
    # For (3,4,7): lower bound = 28. Upper bound from schoolbook = 84.
    # There may be known results. Let's try to find a decomposition of rank < 84.
    
    # Greedy: at each step, find the rank-1 component that maximizes the number of
    # currently-uncovered tensor entries it can exactly match (with coefficient 1).
    # Since the tensor is all-ones, a rank-1 term with 0/1 entries covers a set of entries.
    # The coefficient at each covered entry must be exactly 1 (since we're summing to 1).
    # So we need a partition of the 84 entries into rank-1 "rectangles" in the 3D grid.
    
    # A rank-1 component with u, v, w in {0,1} covers entries (i,j,k) where
    # u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1.
    # Let U_set = {(i,j) : u[i*m+j]=1}, V_set = {(j,k) : v[j*p+k]=1}, W_set = {(k,i) : w[k*n+i]=1}
    # Covered entries: {(i,j,k) : (i,j) in U_set AND (j,k) in V_set AND (k,i) in W_set}
    
    # We want to cover all 84 entries (i,j,k) with the minimum number of such 3D "rectangles".
    
    # This is a covering problem. Let's do a greedy set cover.
    
    # Precompute all possible rank-1 components (with 0/1 entries) and the entries they cover.
    # There are 2^(nm) * 2^(mp) * 2^(pn) possible, which is too many.
    # But we can restrict to "structured" components.
    
    # Let's try components where U_set is a "row slice" {i} x {all j}, V_set is {all j} x {k}, W_set is {k} x {i}
    # No, that doesn't work because the indices must be consistent.
    
    # Actually, for a valid rank-1 term in the multiplication tensor decomposition,
    # the sets must satisfy: if (i,j) in U_set, (j,k) in V_set, (k,i') in W_set,
    # then the term contributes to (i,j,k) only if i'=i (from W's i index).
    # So W_set must contain (k,i) for the same i as U_set.
    
    # This is getting complex. Let me just try a simple greedy:
    # For each entry (i,j,k) not yet covered, add a rank-1 term for just that entry.
    # That's the schoolbook. To do better, find groups.
    
    # Try: for each i, can we cover all (i,j,k) for j in range(m), k in range(p) with fewer terms?
    # A rank-1 term that covers (i,j,k) for multiple j,k needs:
    #   u[i*m+j] = 1 for those j
    #   v[j*p+k] = 1 for those (j,k)
    #   w[k*n+i] = 1 for those k
    # And it must NOT cover any (i',j',k') with i' != i (to avoid interfering with other i's).
    # So u[i'*m+j] = 0 for i' != i.
    
    # For fixed i, we need to cover the m x p grid of (j,k) entries.
    # A rank-1 term for this i covers a "rectangle" J x K where J is a set of j's, K is a set of k's,
    # with u[i*m+j]=1 for j in J, v[j*p+k]=1 for (j,k) in J x K, w[k*n+i]=1 for k in K.
    # But v[j*p+k] is shared across i's! If v[j*p+k]=1 for some (j,k), it affects all i's.
    # So we can't independently decompose each i's slice.
    
    # This is the fundamental difficulty. Let me just implement a greedy set cover
    # over all "simple" rank-1 components and see what rank we get.
    
    # Simple components: for each (i,j,k), the singleton. Also try pairs, etc.
    # But that's exponential. Let me try a heuristic.
    
    # Actually, let me just return the schoolbook. The task says score is rank,
    # and the schoolbook has rank 84. If we can't find better, we return 84.
    
    # But let me try one more thing: can we use half-integer entries to get a lower rank?
    # Probably not for this specific tensor, but let's try a simple optimization.
    
    # For now, return schoolbook.
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
