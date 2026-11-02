"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build schoolbook rank-36 decomposition
    def build_schoolbook():
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
        return U, V, W
    
    U0, V0, W0 = build_schoolbook()
    best_result = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p
    
    # Verify decomposition: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 if i,j,k all match, else 0
    def verify_decomposition(U, V, W, tol=1e-10):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if True else 0  # Always 1 for valid (i,j,k)
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - expected) > tol:
                        return False
        # Also check that incorrect combinations give 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check a few incorrect index combinations
                    for i2 in range(n):
                        if i2 != i:
                            total = 0
                            for r in range(rank):
                                total += U[i2*m+j][r] * V[j*p+k][r] * W[k*n+i2][r]
                            if abs(total) > tol:
                                return False
        return True
    
    # Try to find lower rank via random perturbations and column elimination
    def try_reduce_rank(U, V, W, target_rank, max_iters=50):
        """Attempt to reduce rank by removing columns and adjusting remaining ones."""
        current_U = [row[:] for row in U]
        current_V = [row[:] for row in V]
        current_W = [row[:] for row in W]
        current_rank = len(current_U[0])
        
        for _ in range(max_iters):
            if current_rank <= target_rank:
                break
            if time.time() > deadline:
                break
            
            # Try to remove a random column
            col_to_remove = random.randint(0, current_rank - 1)
            
            # Check if we can absorb this column into others
            # This is a simplified approach - in practice, we'd need to solve a linear system
            # For now, just try removing and see if verification still passes (it won't, but 
            # this establishes the framework)
            
            # Better approach: use SVD to find near-redundant columns
            # Build the tensor from current factors
            tensor = [[[0]*p for _ in range(m)] for _ in range(n)]
            for r in range(current_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor[i][j][k] += current_U[i*m+j][r] * current_V[j*p+k][r] * current_W[k*n+i][r]
            
            # Flatten tensor
            flat_tensor = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        flat_tensor.append(tensor[i][j][k])
            
            # Try to find a column that is nearly linearly dependent
            # This is complex, so we'll use a heuristic: try removing each column and 
            # see if the remaining columns can still span the tensor space
            # For efficiency, we'll just try a few random removals
            
            new_U = [row[:col_to_remove] + row[col_to_remove+1:] for row in current_U]
            new_V = [row[:col_to_remove] + row[col_to_remove+1:] for row in current_V]
            new_W = [row[:col_to_remove] + row[col_to_remove+1:] for row in current_W]
            
            # Verify if the reduced decomposition still works (it won't in general, but 
            # we're establishing the search framework)
            # In a real implementation, we'd need to solve for adjustments to the remaining
            # columns to maintain the tensor identity
            
            current_U = new_U
            current_V = new_V
            current_W = new_W
            current_rank = len(current_U[0])
        
        return current_U, current_V, current_W, current_rank
    
    # More effective approach: use structured random initialization and optimization
    def random_search(rank, num_trials=100):
        """Try random initializations and local optimization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small entries
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
            
            # Simple gradient-free local search
            for step in range(50):
                if time.time() > deadline:
                    break
                
                # Compute current error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for r in range(rank):
                                total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            error += (total - 1) ** 2
                
                if error < 1e-10:
                    # Found a valid decomposition!
                    return U, V, W, rank
                
                # Perturb a random entry
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, rank-1)
                    U[row][col] += random.uniform(-0.1, 0.1)
                elif mat == 'V':
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, rank-1)
                    V[row][col] += random.uniform(-0.1, 0.1)
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, rank-1)
                    W[row][col] += random.uniform(-0.1, 0.1)
        
        return None
    
    # Try ranks from 35 down to 20
    for target_rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        # Try random search for this rank
        result = random_search(target_rank, num_trials=20)
        if result is not None:
            U, V, W, rank = result
            if verify_decomposition(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break  # Found a valid decomposition at this rank
    
    return best_result

# EVOLVE-BLOCK-END
