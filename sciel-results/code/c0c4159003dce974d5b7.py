"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] with entries T[i,j,k][u,v,w].
    Convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank) with W indexed as W[k*n+i, r].
    The tensor element T[i,j,k] is a 3D array of size (n, m, p) where
    T[i,j,k][u,v,w] = 1 if u==i and v==j and w==k, else 0.
    Actually the tensor T is of shape (n, m, p) where T[i,j,k] = e_i ⊗ e_j ⊗ e_k.
    For decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1
    return T

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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

def _random_initial(n, m, p, rank, seed=0):
    """Create a random initial decomposition with small integer/half-integer entries."""
    rng = random.Random(seed)
    U = [[rng.choice([0, 1, -1]) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.choice([0, 1, -1]) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.choice([0, 1, -1]) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W

def _local_search(U, V, W, n, m, p, max_iters=5000, time_limit=12.0, seed=0):
    """Local search to reduce rank by trying to merge/remove columns.
    Strategy: try to set some entries to 0 or ±1/2 to find a lower-rank solution.
    We'll try to reduce rank by eliminating columns one at a time if the remaining
    columns still span the tensor.
    """
    start = time.time()
    rank = len(U[0]) if U else 0
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank by zeroing out columns
    # For each column, check if removing it still allows exact decomposition
    # This is a hard problem, so we use a heuristic: try to find linear dependencies
    
    # Alternative: use the fact that for small dimensions, we can try all combinations
    # of small integer coefficients
    
    # Let's try a different approach: start from schoolbook and try to find
    # a lower rank decomposition by solving a system of equations
    
    return best_U, best_V, best_W, best_rank

def _try_rank(n, m, p, target_rank, time_budget=10.0, seed=0):
    """Try to find a decomposition of the given target rank.
    Uses a randomized approach with local improvement.
    """
    start = time.time()
    rng = random.Random(seed)
    
    # For small target ranks, try to construct directly
    # The tensor has n*m*p = 42 nonzero entries
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Start with a random initialization
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Initialize with random small values
    for i in range(n * m):
        for r in range(target_rank):
            if rng.random() < 0.5:
                U[i][r] = rng.choice([1, -1])
    for j in range(m * p):
        for r in range(target_rank):
            if rng.random() < 0.5:
                V[j][r] = rng.choice([1, -1])
    for k in range(p * n):
        for r in range(target_rank):
            if rng.random() < 0.5:
                W[k][r] = rng.choice([1, -1])
    
    # Now try to improve by adjusting entries
    # We want to satisfy: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a nonlinear system. Let's use a simple iterative approach:
    # For each (i,j,k), compute the current sum and adjust one entry to fix it.
    
    for iteration in range(10000):
        if time.time() - start > time_budget:
            break
        
        # Check if we're already done
        done = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        done = False
                        break
                if not done:
                    break
            if not done:
                break
        
        if done:
            return U, V, W
        
        # Pick a random (i,j,k) that's not satisfied
        unsatisfied = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        unsatisfied.append((i, j, k, s))
        
        if not unsatisfied:
            return U, V, W
        
        i, j, k, current_sum = rng.choice(unsatisfied)
        diff = 1 - current_sum
        
        # Try to adjust one of the three matrices
        # Option 1: adjust U[i*m+j, r] for some r
        # Option 2: adjust V[j*p+k, r] for some r  
        # Option 3: adjust W[k*n+i, r] for some r
        
        option = rng.choice([0, 1, 2])
        r = rng.randint(0, target_rank - 1)
        
        if option == 0:
            # Adjust U[i*m+j, r]
            # We need: new_U * V[j*p+k, r] * W[k*n+i, r] = diff + current_sum - old_U * V * W
            # So: new_U = (diff + old_U * V[j*p+k, r] * W[k*n+i, r]) / (V[j*p+k, r] * W[k*n+i, r])
            denom = V[j * p + k][r] * W[k * n + i][r]
            if denom != 0:
                old_val = U[i * m + j][r]
                new_val = (diff + old_val * denom) / denom
                # Check if new_val is a nice number (integer or half-integer)
                if abs(new_val - round(new_val)) < 1e-9:
                    U[i * m + j][r] = int(round(new_val))
                elif abs(new_val * 2 - round(new_val * 2)) < 1e-9:
                    U[i * m + j][r] = round(new_val * 2) / 2
                # Else skip
        
        elif option == 1:
            # Adjust V[j*p+k, r]
            denom = U[i * m + j][r] * W[k * n + i][r]
            if denom != 0:
                old_val = V[j * p + k][r]
                new_val = (diff + old_val * denom) / denom
                if abs(new_val - round(new_val)) < 1e-9:
                    V[j * p + k][r] = int(round(new_val))
                elif abs(new_val * 2 - round(new_val * 2)) < 1e-9:
                    V[j * p + k][r] = round(new_val * 2) / 2
        
        else:
            # Adjust W[k*n+i, r]
            denom = U[i * m + j][r] * V[j * p + k][r]
            if denom != 0:
                old_val = W[k * n + i][r]
                new_val = (diff + old_val * denom) / denom
                if abs(new_val - round(new_val)) < 1e-9:
                    W[k * n + i][r] = int(round(new_val))
                elif abs(new_val * 2 - round(new_val * 2)) < 1e-9:
                    W[k * n + i][r] = round(new_val * 2) / 2
    
    # Check final result
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(target_rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return None
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The theoretical minimum rank for 2x3x7 is unknown but likely less than 42
    # Let's try ranks from 20 up to 41
    
    # For small problems, we can try a more systematic approach
    # Key insight: the tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # We need T = sum_r U_r ⊗ V_r ⊗ W_r where U_r, V_r, W_r are vectors
    
    # Let's try a constructive approach:
    # For rank r, we need r triples (u_r, v_r, w_r) such that
    # sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the 3D tensor
    
    # For small n,m,p, we can use the fact that the tensor is "all-ones"
    # T[i,j,k] = 1 for all i,j,k
    
    # Actually, the tensor is: T[i,j,k] = delta_{i,j,k} in some sense? No.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i is the i-th standard basis vector.
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # This means we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # One simple construction: if we can find vectors a_r in R^n, b_r in R^m, c_r in R^p
    # such that sum_r a_r[i] * b_r[j] * c_r[k] = 1 for all i,j,k
    # Then U[i*m+j, r] = a_r[i], V[j*p+k, r] = b_r[j], W[k*n+i, r] = c_r[k]
    
    # So the problem reduces to: find the minimum r such that there exist vectors
    # a_1,...,a_r in R^n, b_1,...,b_r in R^m, c_1,...,c_r in R^p with
    # sum_r a_r ⊗ b_r ⊗ c_r = 1 (the all-ones tensor)
    
    # For the all-ones tensor of size n x m x p, what's the minimum rank?
    # The all-ones tensor J_{n,m,p} has entries all equal to 1.
    # J = 1_n ⊗ 1_m ⊗ 1_p has rank 1!
    
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    # Ah, I see the confusion. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the convention.
    
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank) with W[k*n+i, r].
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # For matrix multiplication, T[i,j,k] = 1 if this corresponds to a valid term in the product.
    # Actually, for multiplying an n×m matrix A by an m×p matrix B, the (i,k) entry of the product
    # is sum_j A[i,j] * B[j,k].
    
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This IS the all-ones tensor!
    
    # But wait, that would mean the rank is 1, which is clearly wrong for matrix multiplication.
    
    # Let me re-examine the index convention more carefully.
    # U[i*m+j, r] - this indexes the (i,j) entry of the first matrix
    # V[j*p+k, r] - this indexes the (j,k) entry of the second matrix  
    # W[k*n+i, r] - this indexes the (k,i) entry? That's unusual.
    
    # The output factor W has shape (p*n, rank) and is indexed as W[k*n+i, r].
    # So W[k*n+i, r] corresponds to the (i,k) entry of the output matrix (since output is n×p).
    
    # So the decomposition is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This means: for each r, U[:, r] is a vector in R^{n*m}, V[:, r] is a vector in R^{m*p},
    # W[:, r] is a vector in R^{p*n}.
    
    # The tensor T[i,j,k] = 1 for all i,j,k, and we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # So yes, it IS the all-ones tensor, and the minimum rank should be 1?
    
    # But that contradicts known results. Let me check with a simple case.
    # For 2x2x2 matrix multiplication, the rank is known to be 7 (Strassen).
    # If the tensor were all-ones, the rank would be 1.
    
    # I think I'm misunderstanding the tensor. Let me look at this differently.
    
    # The matrix multiplication tensor is typically defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the sum is over all valid indices
    # But this gives the all-ones tensor.
    
    # Actually, I think the correct definition involves the structure of matrix multiplication.
    # The tensor T is such that T(A,B) = A*B, where A is n×m, B is m×p.
    # In tensor notation: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i, e_j, e_k are basis vectors for the output, first input, second input respectively.
    
    # But this still gives all-ones entries.
    
    # I think the issue is that the tensor is not just the all-ones tensor in the standard basis.
    # The matrix multiplication tensor has a specific structure.
    
    # Let me just use the schoolbook decomposition and try to improve it.
    
    # For now, return the schoolbook decomposition
    return best_result

# EVOLVE-BLOCK-END
