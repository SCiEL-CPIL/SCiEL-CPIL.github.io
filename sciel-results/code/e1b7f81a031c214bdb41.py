"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Input dimensions: A is n×m (8 elements), B is m×p (20 elements), C is n×p (10 elements)
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)  
    # W shape: (p*n, rank) = (10, rank)
    
    # Build the multiplication tensor T[i,j,k] where:
    # i indexes C (n*p = 10 elements), j indexes A (n*m = 8), k indexes B (m*p = 20)
    # C[i] = sum_{j,k} T[i,j,k] * A[j] * B[k]
    # 
    # C is n×p matrix, indexed by (row, col) = (r, c) where r in [0,n), c in [0,p)
    # C[r,c] = sum_{j=0}^{m-1} A[r,j] * B[j,c]
    # 
    # A is indexed as A[r,j] -> flat index j_A = r*m + j (0 to n*m-1)
    # B is indexed as B[j,c] -> flat index j_B = j*p + c (0 to m*p-1)
    # C is indexed as C[r,c] -> flat index j_C = c*n + r (following W[k*n+i,r] convention)
    #   where k=c, i=r, so W index = c*n + r
    
    # T[i_C, j_A, j_B] = 1 if C[i_C] gets contribution from A[j_A]*B[j_B]
    # C[r,c] = sum_j A[r,j]*B[j,c]
    # i_C = c*n + r, j_A = r*m + j, j_B = j*p + c
    
    num_C = n * p  # 10
    num_A = n * m  # 8
    num_B = m * p  # 20
    
    # Build tensor as list of non-zero entries
    # T[i_C, j_A, j_B] = 1 when:
    # i_C = c*n + r, j_A = r*m + j, j_B = j*p + c
    # for valid r, j, c
    
    tensor = {}
    for r in range(n):
        for j in range(m):
            for c in range(p):
                i_C = c * n + r
                j_A = r * m + j
                j_B = j * p + c
                tensor[(i_C, j_A, j_B)] = Fraction(1)
    
    # Greedy rank-1 extraction
    # We want to decompose T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^{num_C}, v_t in R^{num_A}, w_t in R^{num_B}
    
    remaining = dict(tensor)
    U_list = []  # Each element is a dict {i_C: value}
    V_list = []  # Each element is a dict {j_A: value}
    W_list = []  # Each element is a dict {j_B: value}
    
    def extract_rank1(rem):
        """Extract a rank-1 component from the remaining tensor."""
        if not rem:
            return None, None, None
        
        # Find the entry with maximum absolute value (all are 1 initially, but may change)
        # Actually, let's pick any non-zero entry and try to find a rank-1 decomposition
        # that includes it.
        
        # Strategy: Pick a non-zero entry (i0, j0, k0) with value val
        # We want u[i0]*v[j0]*w[k0] = val
        # Set v[j0] = 1, w[k0] = 1, u[i0] = val
        # Then for all other entries (i, j, k) in rem where we can set u[i]*v[j]*w[k] to match:
        # If rem has (i, j, k), we need u[i]*v[j]*w[k] to equal rem[(i,j,k)]
        # This is constrained. Let's use a simpler approach:
        
        # Pick a non-zero entry
        i0, j0, k0 = next(iter(rem.keys()))
        val0 = rem[(i0, j0, k0)]
        
        # Initialize u, v, w with only the anchor
        u = {i0: val0}
        v = {j0: Fraction(1)}
        w = {k0: Fraction(1)}
        
        # Now try to extend: for each other entry (i, j, k) in rem,
        # if u[i] is set and v[j] is set, then w[k] should be rem[(i,j,k)]/(u[i]*v[j])
        # if w[k] is set and v[j] is set, then u[i] should be rem[(i,j,k)]/(w[k]*v[j])
        # etc.
        
        # This is complex. Let's use a different approach:
        # Greedily pick rank-1 terms by finding a 2D slice that is rank 1.
        
        # Simpler: just extract one entry at a time (gives schoolbook rank)
        # But let's try to be smarter.
        
        # For now, extract the single entry
        return {i0: val0}, {j0: Fraction(1)}, {k0: Fraction(1)}
    
    # Actually, let me think more carefully. The tensor is sparse and structured.
    # Let's try a different approach: use the fact that the tensor has a specific structure.
    # 
    # C[r,c] = sum_j A[r,j] * B[j,c]
    # 
    # This is essentially a bilinear map. The rank is at most n*m*p but could be lower.
    # For 2×4 × 4×5, the known upper bound is 40, but there might be better.
    #
    # Let me try a systematic greedy approach where I try to find rank-1 terms
    # that cancel multiple entries.
    
    # Reset
    remaining = dict(tensor)
    U_list = []
    V_list = []
    W_list = []
    
    max_rank = n * m * p  # 40
    
    for rank_attempt in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-1 decomposition of the remaining tensor
        # using greedy extraction with backtracking if needed
        
        # For simplicity, use greedy: pick the entry with most "connections"
        # An entry (i, j, k) has connections to all other entries sharing i, j, or k
        
        if not remaining:
            break
        
        # Score each entry by how many other entries it shares indices with
        best_entry = None
        best_score = -1
        for (i, j, k) in list(remaining.keys()):
            score = 0
            for (i2, j2, k2) in remaining.keys():
                if (i2, j2, k2) == (i, j, k):
                    continue
                if i2 == i or j2 == j or k2 == k:
                    score += 1
            if score > best_score:
                best_score = score
                best_entry = (i, j, k)
        
        if best_entry is None:
            break
        
        i0, j0, k0 = best_entry
        val0 = remaining[(i0, j0, k0)]
        
        # Build rank-1 term anchored at (i0, j0, k0)
        u = {i0: val0}
        v = {j0: Fraction(1)}
        w = {k0: Fraction(1)}
        
        # Try to extend this rank-1 term to cover more entries
        # For each entry (i, j, k) in remaining that shares at least one index with (i0, j0, k0):
        # If it shares i and j, then w[k] = rem[(i,j,k)] / (u[i]*v[j]) if u[i] and v[j] are set
        # If it shares i and k, then v[j] = rem[(i,j,k)] / (u[i]*w[k]) if u[i] and w[k] are set
        # If it shares j and k, then u[i] = rem[(i,j,k)] / (v[j]*w[k]) if v[j] and w[k] are set
        
        changed = True
        while changed:
            changed = False
            for (i, j, k) in list(remaining.keys()):
                if (i, j, k) == (i0, j0, k0):
                    continue
                if (i, j, k) not in remaining:
                    continue
                
                # Try to determine if this entry can be covered by the current rank-1 term
                # Case 1: i == i0 and j == j0, need to set w[k]
                if i == i0 and j == j0 and k != k0:
                    if i0 in u and j0 in v and u[i0] != 0 and v[j0] != 0:
                        w_val = remaining[(i, j, k)] / (u[i0] * v[j0])
                        if k in w:
                            if w[k] != w_val:
                                # Conflict, can't cover this entry
                                pass
                        else:
                            w[k] = w_val
                            changed = True
                # Case 2: i == i0 and k == k0, need to set v[j]
                elif i == i0 and k == k0 and j != j0:
                    if i0 in u and k0 in w and u[i0] != 0 and w[k0] != 0:
                        v_val = remaining[(i, j, k)] / (u[i0] * w[k0])
                        if j in v:
                            if v[j] != v_val:
                                pass
                        else:
                            v[j] = v_val
                            changed = True
                # Case 3: j == j0 and k == k0, need to set u[i]
                elif j == j0 and k == k0 and i != i0:
                    if j0 in v and k0 in w and v[j0] != 0 and w[k0] != 0:
                        u_val = remaining[(i, j, k)] / (v[j0] * w[k0])
                        if i in u:
                            if u[i] != u_val:
                                pass
                        else:
                            u[i] = u_val
                            changed = True
                # Case 4: i == i0, j != j0, k != k0 - need both v[j] and w[k]
                # This is harder, skip for now
                # Case 5: j == j0, i != i0, k != k0 - need both u[i] and w[k]
                # Case 6: k == k0, i != i0, j != j0 - need both u[i] and v[j]
        
        # Now subtract this rank-1 term from remaining
        for (i, j, k) in list(remaining.keys()):
            if i in u and j in v and k in w:
                term_val = u[i] * v[j] * w[k]
                new_val = remaining[(i, j, k)] - term_val
                if new_val == 0:
                    del remaining[(i, j, k)]
                else:
                    remaining[(i, j, k)] = new_val
        
        # Also need to check if we introduced new non-zero entries
        # (this shouldn't happen since we're subtracting from existing entries)
        
        # Add to decomposition
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        if not remaining:
            break
    
    # Build final matrices
    if not remaining:
        rank = len(U_list)
        U_mat = [[Fraction(0)] * rank for _ in range(num_A)]
        V_mat = [[Fraction(0)] * rank for _ in range(num_B)]
        W_mat = [[Fraction(0)] * rank for _ in range(num_C)]
        
        for t in range(rank):
            for i, val in U_list[t].items():
                U_mat[i][t] = val
            for j, val in V_list[t].items():
                V_mat[j][t] = val
            for k, val in W_list[t].items():
                W_mat[k][t] = val
        
        # Verify
        if time.time() < deadline - 5:
            # Quick verification
            for (i_C, j_A, j_B) in tensor:
                expected = tensor[(i_C, j_A, j_B)]
                actual = Fraction(0)
                for t in range(rank):
                    if U_list[t] and j_A in V_list[t] and j_B in W_list[t] and i_C in U_list[t]:
                        actual += U_list[t][i_C] * V_list[t][j_A] * W_list[t][j_B]
                if actual != expected:
                    # Verification failed, fall back to schoolbook
                    break
            else:
                # Convert to lists of floats or keep as fractions
                # The problem says integer/half-integer entries, so fractions with denominator 1 or 2
                U_out = [[float(x) for x in row] for row in U_mat]
                V_out = [[float(x) for x in row] for row in V_mat]
                W_out = [[float(x) for x in row] for row in W_mat]
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0.0] * rank for _ in range(num_A)]
    V = [[0.0] * rank for _ in range(num_B)]
    W = [[0.0] * rank for _ in range(num_C)]
    column = 0
    for r in range(n):
        for j in range(m):
            for c in range(p):
                i_A = r * m + j
                i_B = j * p + c
                i_C = c * n + r
                U[i_A][column] = 1.0
                V[i_B][column] = 1.0
                W[i_C][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

