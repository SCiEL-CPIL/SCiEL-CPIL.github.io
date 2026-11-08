import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def check_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def to_half_integers(U, V, W):
        """Round to nearest half-integer."""
        def round_half(x):
            return int(round(x * 2)) / 2.0
        U2 = [[round_half(U[i][r]) for r in range(len(U[0]))] for i in range(len(U))]
        V2 = [[round_half(V[i][r]) for r in range(len(V[0]))] for i in range(len(V))]
        W2 = [[round_half(W[i][r]) for r in range(len(W[0]))] for i in range(len(W))]
        return U2, V2, W2
    
    # Try structured construction using Kronecker products
    def try_kronecker():
        """
        Try to build solution using Kronecker product structure.
        4x5x5 can be seen as combinations of smaller problems.
        
        Approach: Use the fact that we can write 4 = 2*2, 5 = 5, 5 = 5
        and try to use rank-2 solutions for 2x2x2 (rank 7) combined with 1x5x5 etc.
        
        Actually, let's try a block approach:
        Split the 4x5 matrix into 2x(2x5) blocks, and 5x5 into (5x5).
        This doesn't directly help.
        
        Better: Use the "decomposition" where we write:
        4x5x5 = sum of terms
        Try rank = 2*2*5 + 2*5*5 = 20 + 50 = 70? No, that's not how it works.
        
        Let me try a different structural approach: 
        Use the identity that if we have a rank-r decomposition of a x b x c,
        we can use it to build a (a*d) x (b*e) x (c*f) decomposition with rank r*d*e
        via Kronecker products.
        
        For 4x5x5, we could use:
        - 2x1x1 (rank 1) Kronecker 2x5x5 -> but 2x5x5 is what we're solving
        - This is circular.
        
        Let me try a direct structured search with sparsity.
        """
        return None
    
    # Randomized search with structured initialization
    def randomized_search(rank, max_iters=500):
        """
        Try to find a rank-r decomposition using randomized local search.
        Key idea: start with random sparse initialization and use coordinate descent.
        """
        if time.time() > deadline:
            return None
        
        # Initialize with sparse random values
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # For each rank component, assign random non-zero entries
        for r in range(rank):
            # Choose which (i,j) pairs get non-zero U entries
            num_u = random.randint(1, min(3, nm))
            indices_u = random.sample(range(nm), num_u)
            for idx in indices_u:
                U[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
            
            num_v = random.randint(1, min(3, mp))
            indices_v = random.sample(range(mp), num_v)
            for idx in indices_v:
                V[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
            
            num_w = random.randint(1, min(3, pn))
            indices_w = random.sample(range(pn), num_w)
            for idx in indices_w:
                W[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
        
        # Coordinate descent: try to fix one component at a time
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1) ** 2
            
            if err < 1e-10:
                return U, V, W, err
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Pick a random rank component to modify
            r = random.randint(0, rank - 1)
            
            # Try modifying U entries for this r
            # For fixed r, the contribution to (i,j,k) is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # We want to adjust U[:, r] to minimize error
            # This is complex, so let's just try random perturbations
            
            # Perturb a random entry in U, V, or W for component r
            choice = random.randint(0, 2)
            if choice == 0:
                idx = random.randint(0, nm - 1)
                U[idx, r] += random.choice([-0.5, 0.5])
            elif choice == 1:
                idx = random.randint(0, mp - 1)
                V[idx, r] += random.choice([-0.5, 0.5])
            else:
                idx = random.randint(0, pn - 1)
                W[idx, r] += random.choice([-0.5, 0.5])
            
            # Occasionally reset a component
            if random.random() < 0.05:
                U[:, r] = 0
                V[:, r] = 0
                W[:, r] = 0
                num_u = random.randint(1, 2)
                for idx in random.sample(range(nm), num_u):
                    U[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
                num_v = random.randint(1, 2)
                for idx in random.sample(range(mp), num_v):
                    V[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
                num_w = random.randint(1, 2)
                for idx in random.sample(range(pn), num_w):
                    W[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
        
        # Check best solution
        if best_err < 1e-8:
            return best_U, best_V, best_W, best_err
        return None
    
    # Try to build solution using block structure
    def try_block_decomposition():
        """
        Try to decompose 4x5x5 using block structure.
        
        Idea: 4 = 2 + 2, so we can think of this as two 2x5x5 problems.
        If we can solve 2x5x5 with rank r, then 4x5x5 has rank 2r.
        
        But we need a better approach. Let's try:
        4x5x5 where we split the 5 dimension: 5 = 4 + 1
        Then 4x5x5 = 4x(4+1)x(4+1) = 4x4x4 + 4x4x1 + 4x1x4 + 4x1x1
        
        The ranks would be: rank(4x4x4) + rank(4x4x1) + rank(4x1x4) + rank(4x1x1)
        = 49 + 16 + 16 + 4 = 85
        
        This gives rank 85! Let's try to construct this.
        """
        if time.time() > deadline:
            return None
        
        # We need rank-optimal decompositions of:
        # 4x4x4: rank 49
        # 4x4x1: rank 16 (each of 4*4=16 terms needs rank 1)
        # 4x1x4: rank 16
        # 4x1x1: rank 4
        
        # For 4x4x4, we use a known rank-49 construction
        # This is complex, so let's use a simpler approach:
        # Use the schoolbook for the 4x4x4 part (rank 64) and see if we can do better.
        
        # Actually, let's just try to find ANY solution with rank < 100 using the block idea.
        # For now, let's construct the block solution with schoolbook for each block.
        
        rank_444 = 4 * 4 * 4  # schoolbook for 4x4x4
        rank_441 = 4 * 4 * 1  # schoolbook for 4x4x1
        rank_414 = 4 * 1 * 4  # schoolbook for 4x1x4
        rank_411 = 4 * 1 * 1  # schoolbook for 4x1x1
        
        total_rank = rank_444 + rank_441 + rank_414 + rank_411
        # = 64 + 16 + 16 + 4 = 100, which is just schoolbook
        
        # To get rank 85, we need rank(4x4x4) = 49
        # Let's try to find a rank-49 decomposition of 4x4x4 using known methods.
        
        return None
    
    # Try various ranks with randomized search
    for rank in range(70, 100):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            
            result = randomized_search(rank, max_iters=200)
            if result is not None:
                U, V, W, err = result
                U_list = [[float(U[i][r]) for r in range(rank)] for i in range(nm)]
                V_list = [[float(V[i][r]) for r in range(rank)] for i in range(mp)]
                W_list = [[float(W[i][r]) for r in range(rank)] for i in range(pn)]
                U_h, V_h, W_h = to_half_integers(U_list, V_list, W_list)
                
                if check_solution(U_h, V_h, W_h, rank):
                    return {"U": U_h, "V": V_h, "W": W_h}
    
    return best
