import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
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
    
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Precompute index mappings for efficiency
    # alpha = i*m + j, beta = j*p + k, gamma = k*n + i
    # Valid triples: (i, j, k) -> (i*m+j, j*p+k, k*n+i)
    # Invalid: (i, j, j', k) with j != j' -> (i*m+j, j'*p+k, k*n+i)
    
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_triples.append((alpha, beta, gamma))
    
    invalid_triples = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    alpha = i * m + j
                    beta = jp * p + k
                    gamma = k * n + i
                    invalid_triples.append((alpha, beta, gamma))
    
    def validate(U, V, W):
        """Check if (U, V, W) is a valid decomposition. Returns True if valid."""
        # Check valid triples: sum_r U[alpha,r]*V[beta,r]*W[gamma,r] == 1
        for (alpha, beta, gamma) in valid_triples:
            val = np.dot(U[alpha] * V[beta], W[gamma])
            if val != 1:
                return False
        # Check invalid triples: sum_r U[alpha,r]*V[beta,r]*W[gamma,r] == 0
        for (alpha, beta, gamma) in invalid_triples:
            val = np.dot(U[alpha] * V[beta], W[gamma])
            if val != 0:
                return False
        return True
    
    def count_violations(U, V, W):
        """Count how many constraints are violated."""
        violations = 0
        for (alpha, beta, gamma) in valid_triples:
            val = np.dot(U[alpha] * V[beta], W[gamma])
            if val != 1:
                violations += 1
        for (alpha, beta, gamma) in invalid_triples:
            val = np.dot(U[alpha] * V[beta], W[gamma])
            if val != 0:
                violations += 1
        return violations
    
    # Try to find a decomposition of rank R < 50
    # Use structured random initialization with local search
    
    for R in range(40, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Structured random initialization
            # Use sparse matrices with small integer entries
            U = np.zeros((n * m, R), dtype=int)
            V = np.zeros((m * p, R), dtype=int)
            W = np.zeros((p * n, R), dtype=int)
            
            # Initialize with random sparse pattern
            # Each column has a few non-zero entries
            for r in range(R):
                # Choose a random (i,j,k) triple to "target"
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                
                # Set U[alpha, r], V[beta, r], W[gamma, r] to 1
                U[alpha, r] = 1
                V[beta, r] = 1
                W[gamma, r] = 1
                
                # Add some random perturbation to other entries
                if np.random.rand() < 0.3:
                    other_alpha = np.random.randint(n * m)
                    U[other_alpha, r] = np.random.choice([-1, 1])
                if np.random.rand() < 0.3:
                    other_beta = np.random.randint(m * p)
                    V[other_beta, r] = np.random.choice([-1, 1])
                if np.random.rand() < 0.3:
                    other_gamma = np.random.randint(p * n)
                    W[other_gamma, r] = np.random.choice([-1, 1])
            
            # Local search: try to reduce violations
            for step in range(500):
                if time.time() > deadline:
                    break
                
                viol = count_violations(U, V, W)
                if viol == 0:
                    found = True
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
                
                # Try flipping a random entry
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = np.random.randint(n * m)
                    col = np.random.randint(R)
                    old_val = U[row, col]
                    new_val = -old_val if old_val != 0 else np.random.choice([-1, 1])
                    if new_val == old_val:
                        continue
                    U[row, col] = new_val
                    new_viol = count_violations(U, V, W)
                    if new_viol >= viol:
                        U[row, col] = old_val
                    # Keep the change if it reduces violations or is equal (to explore)
                elif mat == 'V':
                    row = np.random.randint(m * p)
                    col = np.random.randint(R)
                    old_val = V[row, col]
                    new_val = -old_val if old_val != 0 else np.random.choice([-1, 1])
                    if new_val == old_val:
                        continue
                    V[row, col] = new_val
                    new_viol = count_violations(U, V, W)
                    if new_viol >= viol:
                        V[row, col] = old_val
                else:
                    row = np.random.randint(p * n)
                    col = np.random.randint(R)
                    old_val = W[row, col]
                    new_val = -old_val if old_val != 0 else np.random.choice([-1, 1])
                    if new_val == old_val:
                        continue
                    W[row, col] = new_val
                    new_viol = count_violations(U, V, W)
                    if new_viol >= viol:
                        W[row, col] = old_val
            
            if found:
                break
        
        if found:
            break  # Found the best possible for this R, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
