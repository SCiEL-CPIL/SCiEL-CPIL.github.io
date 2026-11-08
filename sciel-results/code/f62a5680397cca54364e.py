import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # We use a simple random search for R from 40 to 63
    # To keep it fast, we limit the number of trials per R.
    
    deadline = time.time() + 140 # Safety margin
    
    def check(U, V, W, R):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check sum_c U[i,j,c] V[j',k,c] W[k,i,c] == delta(j,j')
        # We check a subset of (i, j, j', k) to speed up, then verify fully if promising.
        # Actually, just check all. 2*4*4*8 = 256 checks.
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(R):
                            s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                        if j == jp:
                            if s != 1:
                                return False
                        else:
                            if s != 0:
                                return False
        return True

    def random_search(R, trials):
        for t in range(trials):
            if time.time() > deadline:
                return None
            U = [[0]*R for _ in range(n*m)]
            V = [[0]*R for _ in range(m*p)]
            W = [[0]*R for _ in range(p*n)]
            
            # Initialize with random -1, 0, 1
            for i in range(n*m):
                for c in range(R):
                    U[i][c] = np.random.choice([-1, 0, 1])
            for i in range(m*p):
                for c in range(R):
                    V[i][c] = np.random.choice([-1, 0, 1])
            for i in range(p*n):
                for c in range(R):
                    W[i][c] = np.random.choice([-1, 0, 1])
            
            # Simple local search / hill climbing
            # This might be too slow. Let's just check random initializations first.
            if check(U, V, W, R):
                return U, V, W
                
            # If not found, maybe do a few steps of gradient descent?
            # Or just return None.
            
        return None

    # Try ranks from 40 to 63
    # We can be optimistic and try lower ranks first, but 40 is a safe bet for a quick win?
    # Actually, let's try 50, 55, 60 to be safe and fast.
    for R in [50, 55, 60, 63]:
        if time.time() > deadline:
            break
        result = random_search(R, 100)
        if result:
            best_U, best_V, best_W = result
            best_rank = R
            break # Found a lower rank, return it
    
    return {"U": best_U, "V": best_V, "W": best_W}
