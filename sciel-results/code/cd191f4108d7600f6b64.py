"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Continuous Relaxation (Projected Gradient Descent)
    # We want to minimize Tr(X^T A X B) where X is a doubly stochastic matrix.
    # Gradient: 2 * (A @ X @ B)
    # We use a fixed step size or adaptive.
    
    # Initialize X as a random doubly stochastic matrix (or simple identity permutation)
    # Better initialization: Rank matching
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    heavy_order = np.argsort(-a_sum, kind="stable")
    central_order = np.argsort(b_sum, kind="stable")
    
    # Start with the permutation matrix of the rank matching
    X = np.zeros((n, n))
    for i, loc in zip(heavy_order, central_order):
        X[i, loc] = 1.0
        
    # Projected Gradient Descent
    # The projection onto the Birkhoff polytope is the Hungarian algorithm (linear sum assignment).
    # However, doing Hungarian every step is expensive.
    # Instead, we can use a soft projection or just run PGD for a few steps and then project.
    # Actually, for QAP, the gradient is G = 2 * (A @ X @ B).
    # Let's try a simple approach:
    # Iterate:
    # 1. Compute gradient G = A @ X @ B
    # 2. Update X = X - lr * G
    # 3. Project X to nearest doubly stochastic matrix (via Hungarian on -X)
    
    # Since n=150, Hungarian is O(n^3) ~ 3.3M ops. In Python/SciPy, this might take ~1-5ms.
    # 110s / 5ms = 22000 iterations. That's a lot.
    # But the convergence might be slow.
    
    # Alternative: Use a simpler relaxation or just use the discrete search but with a better init?
    # The prompt asks for a MATERIALLY DIFFERENT strategy.
    # Continuous relaxation is one.
    
    # Let's implement a simplified version:
    # We will run PGD for a fixed number of steps (e.g., 500) to get a "good" fractional solution,
    # then recover a permutation, then run a fast local search.
    
    # Actually, a known trick is to use the gradient to guide the swaps in the discrete search,
    # or to use the relaxed solution as a starting point.
    
    # Let's try the PGD loop.
    lr = 0.01
    max_pgd_steps = 1000
    
    # Precompute A and B for faster matmul? They are already numpy arrays.
    A = a
    B = b
    
    for k in range(max_pgd_steps):
        if time.time() > deadline - 20: # Leave time for polish
            break
            
        # Gradient: 2 * (A @ X @ B)
        # Note: The cost is sum_{i,j} a_{ij} b_{p(i), p(j)} = Tr(X^T A X B)
        # d/dX Tr(X^T A X B) = A X B + A^T X B^T.
        # If A and B are symmetric (which they often are in QAP, but not always), it's 2 A X B.
        # Let's assume general case: G = A @ X @ B + A.T @ X @ B.T
        
        G = A @ X @ B + A.T @ X @ B.T
        
        # Update
        X = X - lr * G
        
        # Projection: Find permutation matrix closest to X (maximizing sum X_{ij} P_{ij})
        # This is equivalent to minimizing sum (-X_{ij}) P_{ij}
        # Use Hungarian on -X
        row_ind, col_ind = linear_sum_assignment(-X)
        
        # Check if we've converged (permutation didn't change)
        # We don't store previous permutation explicitly, but we can check cost change or just run for fixed steps.
        # To save time, we only project every 10 steps? No, projection is what keeps it on the manifold.
        # But Hungarian is the expensive part.
        
        # Let's build the new permutation matrix
        X_new = np.zeros((n, n))
        X_new[row_ind, col_ind] = 1.0
        
        # Check convergence: if X_new is same as X, we might be stuck.
        # For now, just update X.
        X = X_new
        
        # Optional: Reduce learning rate
        # lr *= 0.999
        
    # Recover permutation from final X
    p = col_ind.copy()
    
    # 2. Discrete Polish (Local Search)
    # Use the fast local search from the seed, but start from this better initialization.
    
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    tabu_size = 30
    tabu_list = {}
    current_step = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            tabu_list.clear()
            continue
            
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        Bp = b[:, p]
        
        for i in range(n):
            u = p[i]
            ai = a[i]
            
            for j in range(i + 1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                v = p[j]
                
                delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_i == -1:
            tabu_list.clear()
            continue
            
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        tabu_list[(best_i, best_j)] = current_step + tabu_size
        tabu_list[(best_j, best_i)] = current_step + tabu_size
        current_step += 1
        
        if current_step % 100 == 0:
            keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()
# EVOLVE-BLOCK-END
