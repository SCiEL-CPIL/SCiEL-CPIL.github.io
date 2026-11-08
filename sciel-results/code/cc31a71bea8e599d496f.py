"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] for all i,j,k
    Wait, the convention is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    The tensor for matrix multiplication C = A B is T[i,j,k] = 1 if A is n x m, B is m x p, C is p x n?
    No, standard is A (n x m) * B (m x p) = C (n x p).
    Tensor T has indices i (row of A/C), j (col of A / row of B), k (col of B/C).
    T[i,j,k] = 1.
    Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ?
    Let's check the shapes given in the prompt:
    U: (n*m, rank) -> indexed by (i, j)
    V: (m*p, rank) -> indexed by (j, k)
    W: (p*n, rank) -> indexed by (k, i)
    
    So the equation is:
    1 = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    This is a system of non-linear equations.
    n=2, m=3, p=4.
    Total unknowns: 2*3*R + 3*4*R + 4*2*R = 6R + 12R + 8R = 26R.
    Constraints: 2*3*4 = 24 equations.
    
    For R=1: 26 unknowns, 24 equations. Likely no solution (or very sparse).
    For R=2: 52 unknowns, 24 equations.
    
    We can try to find a solution with small rank.
    Known lower bounds:
    For 2x2x2, rank is 7.
    For 2x3x4, what is the rank?
    
    Let's use a numerical approach to find a decomposition.
    We can fix one of the matrices (e.g., W) and solve for U and V, or use iterative methods.
    Or, we can use the fact that we just need *an* exact decomposition.
    
    A simple approach:
    Start with the schoolbook decomposition (Rank 24).
    Try to reduce the rank by merging terms or finding linear dependencies.
    
    However, a more direct approach is to use numerical optimization or algebraic geometry tools if available, but we only have NumPy/SciPy.
    
    Let's try a heuristic search for low rank decompositions.
    We can parameterize U, V, W with small integers or half-integers.
    
    Actually, there is a known result for 2x3x4 matrix multiplication.
    The rank is 14? 15?
    
    Let's try to construct a decomposition of rank 14 or similar.
    
    Alternative strategy:
    Use the "border rank" or "tensor rank" algorithms.
    Since we need an *exact* decomposition, we can use a randomized algorithm to find a low-rank decomposition.
    
    Algorithm:
    1. Initialize U, V, W with random small values.
    2. Define the error tensor E[i,j,k] = 1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    3. Minimize ||E||_F^2.
    4. If the minimum is 0 (or very close), we have a solution.
    5. We can use gradient descent or least squares.
    
    However, the problem is non-convex.
    
    Let's try a specific known decomposition.
    For 2x2x2, the rank 7 decomposition is well-known.
    For 2x3x4, we can try to extend it.
    
    Let's try to use a numerical solver to find a rank R decomposition.
    We can fix W to be random, and then for each (i,j), we have a linear system for U[i,j,:] and V[:, :, :].
    This is still non-linear.
    
    Let's try a simpler approach:
    The tensor T is the matrix multiplication tensor.
    We can use the "subspace" method.
    
    Actually, let's just try to find *any* decomposition with rank < 24.
    If we find rank 20, that's an improvement.
    
    We can use the following trick:
    If we have a decomposition of rank R, we can try to reduce it by checking if one of the terms can be removed or combined.
    
    Let's try to use a numerical optimization with a time limit.
    
    Start with Rank = 14 (a guess).
    Use random initialization and gradient descent.
    
    But wait, the prompt says "integer/half-integer entries".
    This suggests that the optimal solution might have rational entries.
    
    Let's try to find a solution with small integer entries.
    
    We can use a search over small integers for U, V, W.
    But the space is too large.
    
    Let's try a different approach:
    Use the fact that the tensor is invariant under certain transformations.
    
    Actually, I recall that the rank of 2x3x4 matrix multiplication is 14.
    Let's try to construct a rank 14 decomposition.
    
    I will use a numerical method to find a rank 14 decomposition, then round to integers/half-integers and verify.
    
    Steps:
    1. Set R = 14.
    2. Initialize U, V, W with random values.
    3. Use an iterative algorithm to minimize the error.
    4. Check if the error is close to zero.
    5. If so, round to the nearest half-integer and verify exactness.
    
    If R=14 doesn't work, try R=15, etc.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try ranks from 14 up to 24
    best_result = None
    
    for R in range(14, 25):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers
            U = [[random.randint(-2, 2) for _ in range(R)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(R)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(R)] for _ in range(p * n)]
            
            # Check if this random guess works (unlikely, but possible for small R)
            if _check_decomposition(U, V, W, n, m, p):
                best_result = (R, U, V, W)
                found = True
                break
            
            # Try a simple iterative improvement
            # This is a placeholder for a more sophisticated optimizer
            # For now, we'll just try a few random perturbations
            
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Pick a random element to perturb
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, R - 1)
                    U[row][col] += random.choice([-1, 1])
                elif mat == 'V':
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, R - 1)
                    V[row][col] += random.choice([-1, 1])
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, R - 1)
                    W[row][col] += random.choice([-1, 1])
                
                if _check_decomposition(U, V, W, n, m, p):
                    best_result = (R, U, V, W)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    # If no low-rank solution found, fall back to schoolbook
    if best_result is None:
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    R, U, V, W = best_result
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition."""
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
