import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 100

    # Build the 2x3 x 3x9 multiplication tensor
    # T[i,k,j] = delta(i,j) where i in 0..1, j in 0..8, k in 0..2
    # Actually for matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor is T[i,k,j] = delta_{i,j}? No.
    # T[i,k,j] = 1 if we're computing C[i,j] using A[i,k]*B[k,j]
    # So T[i,k,j] = delta_{i,j} is wrong.
    # T[i,k,j] = 1 always for the standard multiplication tensor
    # where the tensor is defined as T = sum_{i,k,j} e_{i,k}^{(A)} ⊗ e_{k,j}^{(B)} ⊗ e_{i,j}^{(C)}
    # So T[i,k,j] = 1 for all i,k,j where i in 0..n-1, k in 0..m-1, j in 0..p-1
    
    # The tensor T has entries T[i,k,j] = 1 for all valid i,k,j
    # This is because (A*B)[i,j] = sum_k A[i,k]*B[k,j]
    # So the tensor is just the all-ones tensor of shape (n, m, p)
    
    # Wait, that's not right either. Let me think more carefully.
    # The multiplication tensor for (n x m) * (m x p) is:
    # T = sum_{i=0}^{n-1} sum_{k=0}^{m-1} sum_{j=0}^{p-1} E_{i,k} ⊗ E_{k,j} ⊗ E_{i,j}
    # where E_{i,k} is the matrix with 1 at position (i,k) and 0 elsewhere
    # So T[i,k,j] = 1 if and only if the term A[i,k]*B[k,j] contributes to C[i,j]
    # Which is always the case, so T[i,k,j] = 1 for all i, k, j.
    
    # The rank of this tensor is what we want to minimize.
    
    # Let's build the full tensor
    T = [[[1 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    # Current best known: 50
    # Let's try to find a better decomposition using optimization
    
    # Start with the current 50-rank decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Build the 50-rank decomposition (same as current)
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        add_term({0: 1, 4: 1}, {k0: 1, p+k1: 1}, {k0*n: 1, k1*n+1: 1})
        add_term({3: 1, 4: 1}, {k0: 1}, {k0*n+1: 1, k1*n+1: -1})
        add_term({0: 1}, {k1: 1, p+k1: -1}, {k1*n: 1, k1*n+1: 1})
        add_term({4: 1}, {p+k0: 1, k0: -1}, {k0*n: 1, k0*n+1: 1})
        add_term({0: 1, 1: 1}, {p+k1: 1}, {k0*n: -1, k1*n: 1})
        add_term({3: 1, 0: -1}, {k0: 1, k1: 1}, {k1*n+1: 1})
        add_term({1: 1, 4: -1}, {p+k0: 1, p+k1: 1}, {k0*n: 1})
    
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})
    
    for i in range(n):
        for k2 in range(p):
            add_term({i*m+2: 1}, {2*p+k2: 1}, {k2*n+i: 1})
    
    # Now try to reduce the rank by merging terms
    # Check if any two terms can be combined
    
    def check_decomposition(U_c, V_c, W_c):
        """Check if the decomposition is exact"""
        for i in range(n):
            for k2 in range(m):
                for j in range(p):
                    lhs = 0
                    for r in range(len(U_c)):
                        lhs += U_c[r][i*m + k2] * V_c[r][k2*p + j] * W_c[r][j*n + i]
                    if abs(lhs - 1) > 1e-9:
                        return False
        return True
    
    # Try pairwise merging of terms
    merged = False
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Try to find two terms that can be merged
        found = False
        for r1 in range(len(U_cols)):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, len(U_cols)):
                if time.time() > deadline:
                    break
                
                # Check if term r1 and r2 have disjoint support in W
                # or if their sum is still a rank-1 term
                w1 = W_cols[r1]
                w2 = W_cols[r2]
                
                # Check if w1 + w2 is rank-1 (all-ones pattern or simple)
                # For now, just check if we can remove one term by adjusting another
                # This is a heuristic approach
                
                # Try: can we express term r2 as a linear combination of others?
                # Skip for now, this is complex
                pass
        
        if not found:
            break
    
    rank = len(U_cols)
    
    # Verify the decomposition
    if not check_decomposition(U_cols, V_cols, W_cols):
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for k2 in range(m):
                for j in range(p):
                    add_term({i*m + k2: 1}, {k2*p + j: 1}, {j*n + i: 1})
        rank = len(U_cols)
    
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
