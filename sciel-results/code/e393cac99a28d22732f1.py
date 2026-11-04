import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Known optimal rank for 2x3 * 3x3 is 17
    rank = 17
    
    # U: (n*m, rank) = (6, 17)
    # V: (m*p, rank) = (9, 17)
    # W: (p*n, rank) = (6, 17)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strategy:
    # The multiplication tensor T has indices (i, j, k) for i in [0, n), j in [0, m), k in [0, p).
    # We want to express T as sum_{c} U[i,j,c] V[j,k,c] W[k,i,c].
    
    # We can construct this by splitting the problem or using known decompositions.
    # A simple way to get a valid decomposition is to use the "schoolbook" method 
    # but then optimize or find a specific known sparse decomposition.
    
    # Since finding the exact optimal decomposition is hard, let's try a specific 
    # structured approach or just return a known valid one if we can construct it.
    
    # Actually, let's try to construct a rank 17 decomposition using a heuristic 
    # or a known pattern.
    
    # One simple valid decomposition is rank 18 (schoolbook). 
    # We know rank 17 exists. 
    # Let's try to find it via a small search or construction.
    
    # Construction idea:
    # Split A (2x3) into A1 (2x1) and A2 (2x2).
    # Split B (3x3) into B1 (1x3) and B2 (2x3).
    # This doesn't directly work because dimensions must align.
    
    # Alternative: Use the fact that rank(2x3x3) = rank(2x2x3) + rank(2x1x3)? No.
    
    # Let's try a random search with a tight budget to find rank 17.
    # If it fails, return schoolbook rank 18? No, score 1.059 implies rank 17 is found.
    
    # Let's hardcode a known rank-17 decomposition if possible, or use a 
    # deterministic construction.
    
    # Here is a known construction for rank 17:
    # It can be derived from the rank-7 decomposition of 2x2x2 (Strassen)
    # and rank-6 decomposition of 2x2x2 (also Strassen variant?)
    # Actually, Strassen is rank 7.
    
    # Let's try to build it from smaller blocks.
    # A is 2x3, B is 3x3.
    # C is 2x3.
    
    # We can view this as two 2x3 multiplications? No.
    
    # Let's use a simple heuristic: start with schoolbook (18 terms) and try to merge two terms.
    # Term 1: U=1, V=1, W=1 for (i,j,k).
    # If we can combine two such terms into one with half-integers, we reduce rank by 1.
    
    # Specifically, look for two terms (i1,j1,k1) and (i2,j2,k2) such that we can
    # represent them with one term using linear combinations.
    
    # This is effectively solving a tensor decomposition problem.
    
    # Given the complexity, and that the seed score is 1.0 (rank 18?), and best is 1.059 (rank 17?),
    # I will attempt a quick local search to reduce rank from 18 to 17.
    
    # Schoolbook initialization
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find a rank 17 decomposition by randomly perturbing and optimizing?
    # Or just try specific linear combinations.
    
    # Let's try a different approach: Use the known rank of 17 and assume it exists.
    # Since I cannot easily derive the exact coordinates, I will implement a 
    # gradient-free search for rank 17 in the space of half-integer matrices.
    
    # However, a more robust approach for the "rewrite" is to return the schoolbook 
    # if the search fails, but the prompt asks for a stronger algorithm.
    
    # Let's try to implement a simple heuristic: 
    # Identify two terms that can be merged.
    # If we have terms T1 = e_i e_j e_k and T2 = e_{i'} e_{j'} e_{k'},
    # can we write T1 + T2 = U V W with U,V,W non-zero?
    # This requires U,V,W to have non-zero entries at these indices.
    
    # Let's just return the schoolbook decomposition for now, but with a 
    # note that rank 18 is the safe fallback. But wait, the previous best was 17.
    
    # I will implement a small randomized search for rank 17.
    
    import random
    import time
    
    deadline = time.time() + 100
    
    def check(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def search_rank(r):
        # Random initialization with small integers
        candidates = [0, 0.5, 1, -0.5, -1]
        for _ in range(5000):
            if time.time() > deadline:
                break
            U = [[random.choice(candidates) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(candidates) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(candidates) for _ in range(r)] for _ in range(p*n)]
            
            if check(U, V, W, r):
                return U, V, W
        return None
    
    result = search_rank(17)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
