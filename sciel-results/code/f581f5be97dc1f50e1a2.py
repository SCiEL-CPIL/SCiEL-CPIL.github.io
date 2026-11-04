"""
Strategy: Continuous Graph Embedding for QAP.
Reframes the QAP as a geometric problem where facilities and locations are points in a high-dimensional space.
We minimize the Frobenius norm ||A - P B P^T||_F^2 by optimizing the permutation matrix P in a relaxed continuous
space (or by optimizing coordinates directly). This decouples the combinatorial hardness from the geometric
structure, allowing gradient-based methods to find a "soft" assignment which is then "hardened" via nearest-neighbor
matching (Hungarian algorithm).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym=None):
    """Compute the change in cost when swapping facilities at positions i and j."""
    if a_sym is None:
        a_sym = a + a.T
    x = p[i]
    y = p[j]
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    temp = diff_a * diff_b[p]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    return delta

def _local_search_2opt(a, b, p, time_limit, a_sym=None, max_passes=10):
    """Perform 2-opt local search on permutation p."""
    n = len(p)
    if a_sym is None:
        a_sym = a + a.T
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Randomize the order of checks to escape local minima faster
        indices = np.arange(n)
        for i in indices:
            if time.time() > time_limit:
                break
            for j in indices[i+1:]:
                if time.time() > time_limit:
                    break
                delta = _swap_delta(a, b, best_p, i, j, a_sym)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
    
    return best_p, best_cost

def _continuous_embedding_solver(a, b, time_limit):
    """
    Solve QAP using continuous embedding.
    1. Embed facilities and locations in R^d.
    2. Minimize the cost by moving points to reduce the interaction energy.
    3. Extract permutation via matching.
    """
    n = len(a)
    a_sym = a + a.T
    
    # Choose dimensionality. n is 256. 2D is too small for 256 points to be distinct in a useful way for QAP structure?
    # Actually, QAP structure is often high-dimensional. But for "embedding" intuition, let's try a moderate dimension.
    # However, the "Frobenius norm" approach usually works on the matrices directly.
    # Let's use the matrix factorization approach: A ~ P B P^T.
    # This is equivalent to finding a permutation P that aligns A and B.
    
    # A robust method for "continuous" relaxation of permutation matrices is the Sinkhorn-Knopp algorithm 
    # or simply gradient descent on a doubly stochastic matrix.
    
    # Let's try optimizing a doubly stochastic matrix X (n x n) where X[i, j] is the probability that facility i is at location j.
    # Objective: sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    # This is a quadratic assignment on the relaxed space.
    
    # Initialize X with noise
    X = np.ones((n, n)) / n
    # Add random perturbation
    X += np.random.randn(n, n) * 0.01
    X = np.clip(X, 0, None)
    X /= X.sum(axis=1, keepdims=True)
    
    # Gradient Descent with Sinkhorn normalization
    # Gradient of Cost w.r.t X:
    # C(X) = tr(X^T A X B^T) ... wait, the cost is sum a[i,j] b[p[i], p[j]]
    # In matrix form with permutation P (rows are facilities, cols are locations):
    # Cost = sum_{i,j} a[i,j] b[P[i], P[j]]
    # If X is the assignment matrix, Cost = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    # Let M = a (n x n), N = b (n x n)
    # Cost = sum_{i,k} X[i,k] (sum_j a[i,j] sum_l X[j,l] b[k,l])
    # Let Y = X @ b (n x n). Y[i, l] = sum_k X[i,k] b[k,l] ... no, indices are tricky.
    
    # Let's use the form: Cost = tr( (X^T A X) B^T )? No.
    # Cost = sum_{i,j} A_{ij} B_{P_i, P_j}
    # = sum_{i,j,k,l} A_{ij} B_{kl} X_{ik} X_{jl}
    # = sum_{i,k} X_{ik} ( sum_{j,l} A_{ij} X_{jl} B_{kl} )
    # Let Z = X @ B (n x n). Z_{j,l} = sum_k X_{j,k} B_{k,l} ... wait.
    # Z_{j, k} = sum_l X_{j,l} B_{l,k}?
    
    # Let's just compute the gradient numerically or via vectorization.
    # G_{ik} = dC/dX_{ik} = sum_{j,l} A_{ij} X_{jl} B_{kl} + sum_{j,l} A_{ji} X_{jl} B_{kl} (due to symmetry in i,j if we consider full sum)
    # Actually, since we sum over all i,j, the term for X_{ik} appears in A_{ij}B_{kl}X_{ik}X_{jl} AND A_{ji}B_{lk}X_{ji}X_{lk} (if i!=j).
    # If we assume A and B are symmetric (or symmetrize them), the gradient is:
    # G = (A @ (X @ B)) + (A.T @ (X @ B.T)) ... something like that.
    
    # Let's simplify: Symmetrize A and B.
    A_sym = a + a.T
    B_sym = b + b.T
    
    # Cost = sum_{i,j,k,l} A_sym[i,j] B_sym[k,l] X[i,k] X[j,l] / 2  (factor 1/2 for double counting if symmetric)
    # Actually, standard QAP cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    # If we use symmetric matrices, sum_{i,j} A_sym[i,j] B_sym[p[i], p[j]] = 2 * sum_{i<j} ... + sum_i ...
    # It's safer to stick to the original definition or ensure consistency.
    
    # Let's use the gradient formula for C = sum_{i,j,k,l} A_{ij} B_{kl} X_{ik} X_{jl}
    # dC/dX_{ik} = sum_{j,l} A_{ij} B_{kl} X_{jl} + sum_{j,l} A_{ji} B_{lk} X_{jl}
    # Term 1: For fixed i, k. Sum over j, l.
    # Let M1 = A @ (X @ B) -> M1[i, l] = sum_j A[i,j] sum_k' X[j,k'] B[k', l] ... no.
    # (X @ B)[j, l] = sum_k X[j,k] B[k,l].
    # So (A @ (X @ B))[i, l] = sum_j A[i,j] (X @ B)[j, l] = sum_{j,k} A[i,j] X[j,k] B[k,l].
    # We need sum_{j,l} A[i,j] B[k,l] X[j,l].
    # Let Y = X (n x n).
    # Sum_l B[k,l] X[j,l] = (B @ X.T)[k, j]
    # So Term 1 = sum_j A[i,j] (B @ X.T)[k, j] = (A @ (B @ X.T))[i, k]
    
    # Term 2: sum_{j,l} A[j,i] B[l,k] X[j,l]
    # Sum_l B[l,k] X[j,l] = (B.T @ X)[l? no. (B.T @ X) is not right dimensions.
    # Sum_l X[j,l] B[l,k] = (X @ B)[j, k]
    # So Term 2 = sum_j A[j,i] (X @ B)[j, k] = (A.T @ (X @ B))[i, k]
    
    # Total Gradient G = A @ (B @ X.T) + A.T @ (X @ B)
    
    # Let's verify with small case.
    # If X is permutation P, X.T = P.T = P^-1.
    
    # Optimization Loop
    # We will use gradient descent with Sinkhorn normalization to keep X doubly stochastic.
    
    X = np.random.rand(n, n)
    X /= X.sum(axis=1, keepdims=True)
    
    # Precompute constants
    A = a
    B = b
    
    # Learning rate schedule
    lr = 0.01
    time_start = time.time()
    
    # We need to be careful with the scale of the gradient.
    # A and B can have large values.
    # Normalize A and B for the optimization, then scale back? 
    # Or just use a small lr.
    
    for step in range(1000):
        if time.time() > time_limit:
            break
        
        # Compute Gradient
        # G = A @ (B @ X.T) + A.T @ (X @ B)
        # Note: X is (n,n). B is (n,n).
        # B @ X.T is (n,n). A @ (B @ X.T) is (n,n).
        # X @ B is (n,n). A.T @ (X @ B) is (n,n).
        
        # Optimization: 
        # T1 = A @ (B @ X.T)
        # T2 = A.T @ (X @ B)
        
        # To save time, we can compute intermediate matrices.
        # BXt = B @ X.T
        # X B = X @ B
        
        # However, for n=256, matrix multiplication is O(n^3) ~ 16M ops. 1000 steps is 16B ops. Might be too slow in Python/NumPy without optimization?
        # NumPy BLAS is fast. 16M ops is ~10ms. 1000 steps is 10s. This is feasible.
        
        BXt = B @ X.T
        XB = X @ B
        
        G = A @ BXt + A.T @ XB
        
        # Gradient Descent Step
        X -= lr * G
        
        # Sinkhorn Normalization to project back to doubly stochastic polytope
        # Iterate a few times
        for _ in range(5):
            X /= X.sum(axis=1, keepdims=True)
            X /= X.sum(axis=0, keepdims=True)
        
        # Decay learning rate
        lr *= 0.99
        
    # Extract Permutation
    # Use Hungarian Algorithm on the "cost" matrix derived from X?
    # Or just take the argmax of X?
    # Since X is doubly stochastic, it's a convex combination of permutations.
    # The "nearest" permutation in terms of cost is not necessarily the argmax of X.
    # However, we can use the linear assignment problem to find the permutation P that minimizes the "distance" to X?
    # Or we can just take the row-wise argmax and resolve conflicts?
    # Better: Use the Hungarian algorithm to minimize the cost of the permutation directly, but initialized near X?
    # Actually, we can just compute the cost of the permutation derived from X (e.g., argmax) and then polish it.
    
    # Let's extract a permutation from X.
    # A common heuristic: take the permutation that maximizes the sum of X[i, p[i]].
    # This is the Linear Sum Assignment Problem (LSAP) with weights X.
    # We want to maximize sum X[i, p[i]].
    # scipy.optimize.linear_sum_assignment minimizes. So we use -X.
    
    row_ind, col_ind = linear_sum_assignment(-X)
    p = col_ind.astype(np.int64)
    
    return p

def _solve_tai256c(a, b, time_limit=110.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # Phase 1: Continuous Embedding
    # Run multiple times with different random initializations to escape local minima
    best_p = None
    best_cost = float('inf')
    
    num_runs = 3
    time_per_run = (time_limit - 30) / num_runs
    
    for run in range(num_runs):
        if time.time() - start_time > time_limit - 30:
            break
        
        p_candidate = _continuous_embedding_solver(a, b, start_time + time_per_run * (run + 1))
        cost_candidate = _qap_cost(a, b, p_candidate)
        
        if cost_candidate < best_cost:
            best_cost = cost_candidate
            best_p = p_candidate.copy()
    
    # Fallback if continuous embedding failed or didn't produce a valid permutation
    if best_p is None:
        # Fallback to rank matching
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
        best_cost = _qap_cost(a, b, best_p)
    
    # Phase 2: Polish with 2-opt local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        best_p, best_cost = _local_search_2opt(
            a, b, best_p, time_limit - 5.0, a_sym, max_passes=20
        )
    
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
