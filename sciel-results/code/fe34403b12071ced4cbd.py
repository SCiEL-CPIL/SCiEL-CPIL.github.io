<<<<<<< SEARCH
import random
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 42
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
    
    # Try to find a lower rank decomposition
    # Strategy: Use a structured approach based on the 2x3 x 3x1 rank-7 construction
    
    # For 2x3 x 3x1, rank is 7. For 2x3 x 3x7, we can potentially do better than 42.
    # Let's try to find a decomposition with rank < 42.
    
    # Approach: Set up the tensor equation and solve for low-rank factors.
    # The equation is: for all (i, j, k) with i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for inconsistent combinations, the sum = 0.
    
    # We'll try ranks from 21 down to 14, and if we find a solution, return it.
    
    # First, let's try a structured ansatz: assume the decomposition has a block structure
    # where for each output column k, we have a "local" decomposition.
    
    # Actually, let's try a direct numerical approach.
    # We'll parameterize U, V, W as random matrices and use gradient descent
    # to minimize the error in the tensor equation.
    
    def _try_find_decomposition(target_rank, deadline):
        """
        Try to find a rank-target_rank decomposition using numerical optimization.
        Returns (U, V, W) as lists of lists, or None if not found.
        """
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        R = target_rank
        num_u = n * m  # 6
        num_v = m * p  # 21
        num_w = p * n  # 14
        
        # Initialize with small random values
        np.random.seed(42)
        U = np.random.randn(num_u, R) * 0.1
        V = np.random.randn(num_v, R) * 0.1
        W = np.random.randn(num_w, R) * 0.1
        
        # Define the tensor equation residual
        def _residual(U, V, W):
            """Compute the residual of the tensor equation."""
            residuals = []
            # For valid (i, j, k): target = 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        residuals.append(val - 1.0)
            # For invalid combinations: target = 0
            # We need to check all (a, b, c) where a in [0, num_u), b in [0, num_v), c in [0, num_w)
            # such that the combination is "inconsistent"
            # a = i*m+j, b = j'*p+k', c = k''*n+i''
            # Inconsistent if j != j' or i != i'' or k != k''
            # But we can't iterate over all (a,b,c) as that's 6*21*14 = 1764, which is fine.
            for a in range(num_u):
                i_u = a // m
                j_u = a % m
                for b in range(num_v):
                    j_v = b // p
                    k_v = b % p
                    for c in range(num_w):
                        k_w = c // n
                        i_w = c % n
                        # Check if consistent: j_u == j_v and i_u == i_w and k_v == k_w
                        if not (j_u == j_v and i_u == i_w and k_v == k_w):
                            val = 0.0
                            for r in range(R):
                                val += U[a, r] * V[b, r] * W[c, r]
                            residuals.append(val - 0.0)
            return residuals
        
        # Use gradient descent with a few iterations
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        num_iters = 2000
        for it in range(num_iters):
            if time.time() > deadline - 5:
                break
            
            res = _residual(U, V, W)
            loss = sum(x*x for x in res)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Compute gradients (approximate)
            # This is expensive, so we'll use a simpler approach: random perturbation
            # Actually, let's use a proper gradient computation for small problems
            
            # For efficiency, let's use a different approach: 
            # We'll use the fact that the problem is small and do a local search.
            
            # Actually, gradient computation for this problem is complex. Let's try a different approach.
            # We'll use a structured ansatz instead.
            break
        
        # If loss is small enough, verify and return
        if best_loss < 1e-10:
            # Convert to exact fractions and verify
            U_frac = [[Fraction(int(round(best_U[i][r] * 1000000)), 1000000) for r in range(R)] for i in range(num_u)]
            V_frac = [[Fraction(int(round(best_V[i][r] * 1000000)), 1000000) for r in range(R)] for i in range(num_v)]
            W_frac = [[Fraction(int(round(best_W[i][r] * 1000000)), 1000000) for r in range(R)] for i in range(num_w)]
            
            # Verify with exact arithmetic
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
        
        return None
    
    def _verify_exact(U, V, W, n, m, p):
        """Verify the decomposition with exact arithmetic."""
        R = len(U[0]) if U else 0
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        # Check valid combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(R):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        
        # Check invalid combinations
        for a in range(num_u):
            i_u = a // m
            j_u = a % m
            for b in range(num_v):
                j_v = b // p
                k_v = b % p
                for c in range(num_w):
                    k_w = c // n
                    i_w = c % n
                    if not (j_u == j_v and i_u == i_w and k_v == k_w):
                        val = Fraction(0)
                        for r in range(R):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if val != 0:
                            return False
        
        return True
    
    # Try ranks from 21 down to 14
    for target_rank in [21, 20, 19, 18, 17, 16, 15, 14]:
        if time.time() > deadline - 10:
            break
        
        result = _try_find_decomposition(target_rank, deadline)
        if result is not None:
            U_frac, V_frac, W_frac = result
            return {
                "U": U_frac,
                "V": V_frac,
                "W": W_frac
            }
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
=======
import random
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 42
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
    
    # Strategy: Construct a rank-21 decomposition using Kronecker product structure.
    # The 2x3 x 3x7 multiplication can be viewed as two independent 1x3 x 3x7 multiplications
    # (one for each row of the 2x3 matrix), each of which has rank 7 (schoolbook for 1x3x3x7).
    # Total rank = 2 * 7 = 14? No, let's think more carefully.
    
    # Actually, for 1x3 x 3x7, the schoolbook rank is 3*7 = 21.
    # So two independent ones would be 42, which is the same as the full schoolbook.
    
    # Let me reconsider. The tensor T_{i,j,k} = A_{i,j} * B_{j,k} where:
    # i in [0,2), j in [0,3), k in [0,7)
    # This is a 2x3x7 tensor.
    
    # The schoolbook decomposition has rank 42.
    # Can we do better?
    
    # For 2x1 x 1x7 (vector times vector), rank is 7 (outer product).
    # For 1x3 x 3x1 (vector times vector), rank is 3 (inner product).
    
    # Actually, let's try a direct approach: set up the system of equations and solve.
    
    def _verify_exact(U, V, W, n, m, p):
        """Verify the decomposition with exact arithmetic."""
        R = len(U[0]) if U else 0
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        # Check valid combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(R):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        
        # Check invalid combinations
        for a in range(num_u):
            i_u = a // m
            j_u = a % m
            for b in range(num_v):
                j_v = b // p
                k_v = b % p
                for c in range(num_w):
                    k_w = c // n
                    i_w = c % n
                    if not (j_u == j_v and i_u == i_w and k_v == k_w):
                        val = Fraction(0)
                        for r in range(R):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if val != 0:
                            return False
        
        return True
    
    # Try to find a rank-21 decomposition using a structured approach.
    # We'll use the fact that the problem has a natural block structure.
    
    # Approach: Use random initialization and Newton's method to solve the system.
    # The equations are:
    # For each (i,j,k) with i in [0,2), j in [0,3), k in [0,7):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For each inconsistent (a,b,c):
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # Total number of equations: 2*3*7 + (6*21*14 - 2*3*7) = 42 + (1764 - 42) = 1764
    # Unknowns: 6*R + 21*R + 14*R = 41*R
    # For R=21: 41*21 = 861 unknowns, 1764 equations (overdetermined)
    
    # Let's try R=21 with a structured initialization.
    
    def _try_rank_decomposition(target_rank, deadline):
        """Try to find a decomposition with the given rank."""
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        R = target_rank
        num_u = n * m  # 6
        num_v = m * p  # 21
        num_w = p * n  # 14
        
        # Initialize with a structured pattern
        # For rank-21, we can try to use the schoolbook structure but with some sharing.
        
        # Let's try a simple approach: initialize with random values and use
        # a local search method.
        
        np.random.seed(42)
        
        # Initialize U, V, W as numpy arrays
        U = np.random.randn(num_u, R) * 0.1
        V = np.random.randn(num_v, R) * 0.1
        W = np.random.randn(num_w, R) * 0.1
        
        # Define the loss function
        def _loss(U, V, W):
            loss = 0.0
            # Valid combinations
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        loss += (val - 1.0) ** 2
            # Invalid combinations
            for a in range(num_u):
                i_u = a // m
                j_u = a % m
                for b in range(num_v):
                    j_v = b // p
                    k_v = b % p
                    for c in range(num_w):
                        k_w = c // n
                        i_w = c % n
                        if not (j_u == j_v and i_u == i_w and k_v == k_w):
                            val = 0.0
                            for r in range(R):
                                val += U[a, r] * V[b, r] * W[c, r]
                            loss += val ** 2
            return loss
        
        # Use a simple gradient descent approach
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try a few random restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(num_u, R) * 0.1
            V = np.random.randn(num_v, R) * 0.1
            W = np.random.randn(num_w, R) * 0.1
            
            # Gradient descent
            lr = 0.001
            for it in range(1000):
                if time.time() > deadline - 5:
                    break
                
                loss = _loss(U, V, W)
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-15:
                    break
                
                # Compute gradients numerically (expensive but simple)
                # For efficiency, let's skip this and just try random perturbations
                # Actually, let's use a simpler approach: accept if loss decreases
                
                # Random perturbation
                U_pert = U + np.random.randn(num_u, R) * 0.01
                V_pert = V + np.random.randn(num_v, R) * 0.01
                W_pert = W + np.random.randn(num_w, R) * 0.01
                
                loss_pert = _loss(U_pert, V_pert, W_pert)
                if loss_pert < loss:
                    U, V, W = U_pert, V_pert, W_pert
        
        # Check if we found a good solution
        if best_loss < 1e-10:
            # Convert to fractions
            U_frac = [[Fraction(int(round(best_U[i][r] * 1000000)), 1000000) for r in range(R)] for i in range(num_u)]
            V_frac = [[Fraction(int(round(best_V[i][r] * 1000000)), 1000000) for r in range(R)] for i in range(num_v)]
            W_frac = [[Fraction(int(round(best_W[i][r] * 1000000)), 1000000) for r in range(R)] for i in range(num_w)]
            
            # Verify with exact arithmetic
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
        
        return None
    
    # Try ranks from 21 down to 14
    for target_rank in [21, 20, 19, 18, 17, 16, 15, 14]:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition(target_rank, deadline)
        if result is not None:
            U_frac, V_frac, W_frac = result
            return {
                "U": U_frac,
                "V": V_frac,
                "W": W_frac
            }
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

