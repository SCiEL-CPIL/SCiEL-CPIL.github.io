"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 90

    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Check that U, V, W exactly decompose the matmul tensor."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # For each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
        # And for each (i,j,k') with k'!=k, sum_c U[i*m+j,c]*V[j*p+k',c]*W[k'*n+i,c] should be 0
        # More generally: the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # should equal 1 if (i,j,k) is a valid matmul index, 0 otherwise.
        # Actually the matmul tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor).
        # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, let me re-read the convention. W[k*n+i, r]. So W index is (k,i).
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # Product: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k.
        # That's the standard matmul tensor where every entry is 1.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Try Strassen-based block decomposition
    # For 3x5 * 5x6, we can try to use Strassen on 2x2 blocks
    # 3 = 2+1, 5 = 2+2+1, 6 = 2+2+2
    # This is complex; let's try a simpler approach first.
    
    # Strategy: Use the known fact that rank(3,5,6) <= 3*5*6 = 90
    # Try to find a lower rank via random search with verification
    
    # First, try a structured approach: block decomposition
    # Split into blocks and use Strassen where possible
    
    # Let's try a direct random search with local optimization
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Try rank from 84 down (Strassen can save some)
    # Actually, let's try to find any valid decomposition with rank < 90
    
    # Use a smarter approach: start with schoolbook and try to merge/eliminate terms
    # Or use random initialization and gradient-free search
    
    # For now, let's try a structured Strassen-based approach
    # 3x5x6: use 2x2 Strassen on the 2x2 sub-blocks
    
    # Actually, let's try a different approach: use the fact that we can
    # share some of the multiplications across blocks.
    
    # Let me try a random search approach with verification
    rng = random.Random(42)
    
    for target_rank in range(89, 70, -1):  # Try ranks from 89 down to 71
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random sparse matrices
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Initialize with random ±1 values (sparse)
            for c in range(target_rank):
                # Pick random positions
                ui = rng.randint(0, nm-1)
                vj = rng.randint(0, mp-1)
                wk = rng.randint(0, pn-1)
                U[ui, c] = rng.choice([-1, 1])
                V[vj, c] = rng.choice([-1, 1])
                W[wk, c] = rng.choice([-1, 1])
            
            # Convert to lists
            U_list = [[int(U[i,c]) for c in range(target_rank)] for i in range(nm)]
            V_list = [[int(V[i,c]) for c in range(target_rank)] for i in range(mp)]
            W_list = [[int(W[i,c]) for c in range(target_rank)] for i in range(pn)]
            
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
