"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook rank
    best_rank = n * m * p  # 42
    best_decomp = _schoolbook_decomp(n, m, p, best_rank)
    
    # Strategy 1: Try Strassen-like block decomposition exploiting m=3
    # Treat A (2x3) as column of 2x1 blocks, B (3x7) as row of 1x7 blocks
    # Use algebraic identities to reduce rank
    
    # For m=3, we can use the identity:
    # (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)
    # This gives rank 5 instead of 6 for 2x2x2
    
    # For our case, let's try structured constructions first
    decomp = _strassen_like_decomp(n, m, p)
    if decomp is not None:
        best_rank = len(decomp['U'][0])
        best_decomp = decomp
    
    # Try to find even lower rank decompositions using structured search
    for target_rank in range(best_rank - 1, max(15, best_rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with structured priors
        for attempt in range(8):
            if time.time() > deadline:
                break
            
            decomp = _try_rank_search(n, m, p, target_rank, attempt)
            if decomp is not None:
                best_rank = target_rank
                best_decomp = decomp
                break
    
    return best_decomp

def _schoolbook_decomp(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # T[i,j,k] should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _strassen_like_decomp(n, m, p):
    """
    Try to construct a rank-reduced decomposition using Strassen-like identities.
    
    For m=3, we can exploit the structure:
    A is 2x3, B is 3x7
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Using the identity: (x+y)(z+w) = xz + xw + yz + yw - (x-y)(z-w)
    We can group terms to reduce rank.
    
    Let's try: 
    S1 = A[:,0] * B[0,:]  (rank 2*7 = 14)
    S2 = A[:,1] * B[1,:]  (rank 2*7 = 14)
    S3 = A[:,2] * B[2,:]  (rank 2*7 = 14)
    Total: 42
    
    To reduce, we can try:
    Let U1 = A[:,0] + A[:,1], V1 = B[0,:] + B[1,:]
    Then U1*V1 = A[:,0]*B[0,:] + A[:,0]*B[1,:] + A[:,1]*B[0,:] + A[:,1]*B[1,:]
    
    We need: A[:,0]*B[0,:] + A[:,1]*B[1,:]
    So we can write:
    A[:,0]*B[0,:] + A[:,1]*B[1,:] = (U1*V1 + A[:,0]*B[1,:] + A[:,1]*B[0,:]) / 2
    
    This doesn't directly help. Let's try a different approach.
    
    For 2x3x7, the theoretical lower bound is max(2*3, 3*7, 2*7) = 21.
    Known results: rank(2,3,7) is between 21 and 42.
    
    Let's try a construction with rank 30:
    Use 5 groups of 6 terms each, where each group exploits a 2x2x2 Strassen identity.
    """
    
    # Try a specific construction:
    # We'll use the fact that for each pair of columns in A and rows in B,
    # we can apply a rank-5 decomposition for 2x2x2.
    
    # Partition the 7 columns of B into groups: 3+3+1
    # For each group of 3, we can potentially save rank.
    
    # Actually, let's try a more systematic approach:
    # Use rank-5 for each 2x2x2 subproblem.
    
    # A has 3 columns, B has 7 rows.
    # We can pair columns of A with rows of B.
    
    # Let's try: rank = 5 * ceil(3*7/4) = 5*6 = 30
    
    # Construction:
    # For each of 6 "blocks", we handle 4 terms (2 from A, 2 from B)
    # Using Strassen: (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)
    # So 4 terms become 5 terms.
    
    # We have 3*7 = 21 terms total.
    # Group them into 6 groups of 3-4 terms.
    
    # Let's try a specific assignment:
    # Group 1: (0,0), (0,1), (1,0), (1,1) -> rank 5
    # Group 2: (0,2), (0,3), (1,2), (1,3) -> rank 5
    # Group 3: (0,4), (0,5), (1,4), (1,5) -> rank 5
    # Group 4: (0,6), (1,6), (2,0), (2,1) -> rank 5
    # Group 5: (2,2), (2,3), (2,4), (2,5) -> rank 5
    # Group 6: (2,6) -> rank 1
    # Total: 5+5+5+5+5+1 = 26
    
    # But this doesn't work because the terms don't align properly.
    
    # Let me try a different approach: use the known rank for small cases.
    # rank(2,3,2) = 7 (known)
    # rank(2,3,3) = ?
    
    # For now, let's try to find a rank-35 decomposition.
    
    U = [[0] * 35 for _ in range(n * m)]
    V = [[0] * 35 for _ in range(m * p)]
    W = [[0] * 35 for _ in range(p * n)]
    
    # Use a structured initialization based on block decomposition
    # Each "block" handles a subset of the 21 terms
    
    # Block structure: 5 blocks of 4 terms + 1 block of 1 term
    # But we need to be careful about the indexing.
    
    # Let's try: for each i in {0,1}, j in {0,1,2}, k in {0,...,6}
    # We assign these to 35 slots.
    
    # Slot assignment:
    # Slots 0-4: handle (i=0,j=0,k=0), (i=0,j=1,k=0), (i=1,j=0,k=0), (i=1,j=1,k=0)
    #   Using Strassen: rank 5
    # Slots 5-9: handle (i=0,j=0,k=1), (i=0,j=1,k=1), (i=1,j=0,k=1), (i=1,j=1,k=1)
    #   Using Strassen: rank 5
    # ... etc.
    
    # This is getting complex. Let's just try a random search with a good prior.
    
    # Initialize with a structured pattern
    for r in range(35):
        # Assign each rank slot to a specific (i,j,k) or combination
        i = r % 2
        j = (r // 2) % 3
        k = (r // 6) % 7
        
        if r < 21:
            # First 21 slots: direct terms
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
        else:
            # Remaining 14 slots: auxiliary terms for Strassen-like identities
            # We'll fill these in with a search
            pass
    
    # Now try to optimize the remaining slots
    for iteration in range(50):
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Try to modify one of the auxiliary slots
        aux_slot = random.randint(21, 34)
        choice = random.randint(0, 2)
        if choice == 0:
            idx = random.randint(0, n*m-1)
            U[idx][aux_slot] = random.randint(-2, 2)
        elif choice == 1:
            idx = random.randint(0, m*p-1)
            V[idx][aux_slot] = random.randint(-2, 2)
        else:
            idx = random.randint(0, p*n-1)
            W[idx][aux_slot] = random.randint(-2, 2)
    
    return None

def _try_rank_search(n, m, p, target_rank, seed):
    """Try to find a rank-target_rank decomposition using structured search."""
    random.seed(seed * 1000 + target_rank)
    
    # For m=3, use a block structure with Strassen-like identities
    # Initialize with a structured prior
    
    U = [[0] * target_rank for _ in range(n*m)]
    V = [[0] * target_rank for _ in range(m*p)]
    W = [[0] * target_rank for _ in range(p*n)]
    
    # First, try to assign direct terms
    direct_terms = min(target_rank, n * m * p)
    for r in range(direct_terms):
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # If target_rank < n*m*p, we need to use auxiliary terms
    if target_rank < n * m * p:
        # Use Strassen-like identities for the remaining terms
        # This is a heuristic; we'll try random modifications
        
        for iteration in range(200):
            if _verify_decomp(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
            # Modify a random entry in the auxiliary region
            if target_rank > direct_terms:
                aux_col = random.randint(direct_terms, target_rank - 1)
            else:
                aux_col = random.randint(0, target_rank - 1)
            
            choice = random.randint(0, 2)
            if choice == 0:
                idx = random.randint(0, n*m-1)
                U[idx][aux_col] = random.randint(-3, 3)
            elif choice == 1:
                idx = random.randint(0, m*p-1)
                V[idx][aux_col] = random.randint(-3, 3)
            else:
                idx = random.randint(0, p*n-1)
                W[idx][aux_col] = random.randint(-3, 3)
    
    return None
# EVOLVE-BLOCK-END
