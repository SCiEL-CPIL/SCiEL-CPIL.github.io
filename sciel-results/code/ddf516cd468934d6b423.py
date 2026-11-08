"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll decompose the 4x9 matrix B into B1 (4x4) and B2 (4x5)
    # C = A*B1 + A*B2
    # For A*B1 (4x4x4): use Strassen-based decomposition
    # For A*B2 (4x4x5): use schoolbook or further decomposition
    
    # Actually, let's try a direct approach with a known decomposition
    # We'll use a combination of smaller known results
    
    # For now, let's try to find a decomposition using optimization
    # Start with a reasonable rank and try to minimize
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try different ranks, starting from a lower bound
    # Lower bound for 4x4x9 is at least 4*9 = 36 (trivial)
    # But we can do better with Strassen-like methods
    
    # Let's try rank 60 first (28+28+4 approach)
    # Actually, let's just return a valid decomposition with rank < 144
    
    # Use a simple but effective decomposition:
    # Split B into three 4x3 blocks: B = [B1 | B2 | B3]
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bi is 4x4x3
    
    # For 4x4x3, we can use schoolbook (rank 4*4*3=48) or better
    # Total: 3*48 = 144 (same as schoolbook) - not helpful
    
    # Better: use the fact that 4x4x4 has rank 28
    # Split B into B1 (4x4) and B2 (4x5)
    # C = A*B1 + A*B2
    # A*B1: rank 28
    # A*B2: 4x4x5, can split into 4x4x4 + 4x4x1, rank 28+4=32
    # Total: 28+32 = 60
    
    # But I need to actually construct these decompositions
    # For simplicity, let's use a random search approach
    
    # Initialize with schoolbook and try to reduce rank
    def schoolbook_decomp(n, m, p, rank):
        """Create a schoolbook-like decomposition with given rank (may not be valid if rank < n*m*p)"""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        # This is just a placeholder; actual construction is complex
        return U, V, W
    
    # Let's try a direct construction using known Strassen factors
    # For 2x2x2, Strassen's 7 factors:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A00)*(B00+B10)
    # M7 = (A10-A11)*(B01+B11)
    
    # This is getting complex. Let me just return a valid decomposition
    # with a known good rank.
    
    # For the time being, return the schoolbook decomposition
    # but with a note that we can improve it.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
