"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def find_rank9_block():
    """
    Find a rank-9 decomposition for M_{2,3,2}.
    Returns U, V, W lists or None.
    U: 2x3x9, V: 3x2x9, W: 2x2x9 (flattened to 2D for internal use)
    """
    # Dimensions for the block
    nb, mb, pb = 2, 3, 2
    rank = 9
    
    # We will search for U, V, W with entries in {-1, 0, 1}
    # To make it faster, we can fix W to be sparse or structured?
    # Or just random search.
    
    # Optimization: Use the linear system for W given U, V?
    # If we fix U and V, we can solve for W exactly if the system is well-posed.
    # Equations: sum_c W[k, i, c] * (U[i, j, c] * V[j, k, c]) = 1 for all i,j,k.
    # This is a system of 12 linear equations for the 36 entries of W (2*2*9).
    # We can pick random sparse U, V and solve for W.
    # If W has integer/half-integer entries, we accept it.
    
    for attempt in range(5000):
        # Generate random sparse U and V
        U = [[0]*rank for _ in range(nb*mb)]
        V = [[0]*rank for _ in range(mb*pb)]
        
        # Fill U and V with -1, 0, 1
        for i in range(nb*mb):
            for c in range(rank):
                if random.random() < 0.5:
                    U[i][c] = random.choice([-1, 1])
        
        for j in range(mb*pb):
            for c in range(rank):
                if random.random() < 0.5:
                    V[j][c] = random.choice([-1, 1])
        
        # Build linear system A * w_flat = b
        # w_flat is vector of W entries. Size 2*2*9 = 36.
        # Equations for each i, j, k:
        # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
        
        A = np.zeros((12, 36))
        b = np.ones(12)
        
        eq_idx = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    for c in range(rank):
                        u_val = U[i*mb+j][c]
                        v_val = V[j*pb+k][c]
                        if u_val != 0 and v_val != 0:
                            # W index is k*nb+i
                            w_idx = (k*nb+i)*rank + c
                            A[eq_idx, w_idx] = u_val * v_val
                    eq_idx += 1
        
        # Solve A w = b
        # Least squares solution
        try:
            w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        # Check if solution is close to integer or half-integer
        # And if it satisfies the equations exactly (within tolerance)
        w_flat = w_sol
        is_valid = True
        for val in w_flat:
            # Check if close to 0.5 * integer
            nearest_half = round(val * 2) / 2
            if abs(val - nearest_half) > 1e-6:
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Check residual
        if np.linalg.norm(A @ w_sol - b) > 1e-6:
            continue
            
        # Construct W
        W = [[0]*rank for _ in range(pb*nb)]
        for k in range(pb):
            for i in range(nb):
                for c in range(rank):
                    W[k*nb+i][c] = w_flat[(k*nb+i)*rank + c]
                    
        # Final verification
        if _check_decomposition(U, V, W, nb, mb, pb):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find rank-18 decomposition
    # Split B into two 3x2 blocks
    # Block 1: cols 0,1. Block 2: cols 2,3.
    # We need a rank-9 decomposition for M_{2,3,2}.
    
    if time.time() < deadline - 20:
        block_U, block_V, block_W = find_rank9_block()
        
        if block_U is not None:
            # Construct full rank-18 solution
            rank = 18
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Block 1 (cols 0,1 of B)
            # U is same for both blocks? No, U depends on A which is full 2x3.
            # Wait, A is 2x3. B is 3x4.
            # C = A * B.
            # C[:, 0:2] = A * B[:, 0:2].
            # C[:, 2:4] = A * B[:, 2:4].
            # The U matrix in the decomposition corresponds to the A part.
            # In the block decomposition, the "A" part is the same A.
            # So U_block is used for both?
            # Let's check the indices.
            # Term c in block 1:
            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For block 1, k is 0,1.
            # For block 2, k is 2,3.
            # We can map the block indices to the global indices.
            
            # Block 1 uses columns c=0..8
            # Block 2 uses columns c=9..17
            
            for i in range(n):
                for j in range(m):
                    # U is identical for both blocks in terms of A coefficients
                    # But we can use the same U_block values for both halves of the rank
                    for c in range(9):
                        U[i*m+j][c] = block_U[i*m+j][c]
                        U[i*m+j][c+9] = block_U[i*m+j][c]
                        
            for j in range(m):
                for k in range(2): # Block 1 k indices
                    V[j*p+k][j*2+k] = 0 # Placeholder, need to map
                # V mapping:
                # Block 1: k_local 0,1 -> k_global 0,1
                # Block 2: k_local 0,1 -> k_global 2,3
                
                # Fill V for Block 1 (cols 0..8)
                for k_local in range(2):
                    k_global = k_local
                    for c in range(9):
                        V[j*p+k_global][c] = block_V[j*2+k_local][c]
                
                # Fill V for Block 2 (cols 9..17)
                for k_local in range(2):
                    k_global = k_local + 2
                    for c in range(9):
                        V[j*p+k_global][c+9] = block_V[j*2+k_local][c]
                        
            for k in range(p):
                for i in range(n):
                    # W mapping
                    if k < 2:
                        k_local = k
                        for c in range(9):
                            W[k*n+i][c] = block_W[k_local*n+i][c]
                    else:
                        k_local = k - 2
                        for c in range(9):
                            W[k*n+i][c+9] = block_W[k_local*n+i][c]
                            
            # Verify
            if _check_decomposition(U, V, W, n, m, p):
                best_rank = 18
                best_U, best_V, best_W = U, V, W
                
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
