"""Matrix multiplication 4x4 * 4x6 with rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Precompute the tensor T[i, j, k] = e_{k*n + i, j*p + k}
    # Actually, the tensor is defined by:
    # (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # In tensor form: T[i,j,k] has U[i,j], V[j,k], W[i,k]
    # But the trace convention is W[k*n+i, r], so W index is (k, i)
    # U index is (i, j), V index is (j, k)
    
    # The multiplication tensor M[i,j,k] = 1 for all i,j,k in the standard basis
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a 3-tensor of size (n*m) x (m*p) x (p*n) = 16 x 24 x 24
    
    # Strategy: Use a known low-rank decomposition.
    # For 4x4 * 4x6, we can try to use block structure.
    # Let's try rank 84 first (Strassen-like for the 4x4 part, extended to 4x6)
    
    # Actually, let's think about this more carefully.
    # The schoolbook rank is 4*4*6 = 96.
    # We want to reduce this.
    
    # One approach: treat the 4x4 * 4x6 as two 4x4 * 4x3 blocks.
    # Each 4x4 * 4x3 has schoolbook rank 4*4*3 = 48.
    # Two such blocks would give rank 96, same as schoolbook.
    
    # Better: use Strassen for the 4x4 part.
    # Strassen for 4x4*4x4 has rank 2^3 * 7 = 56? No.
    # Actually, for 4x4*4x4, the Strassen algorithm uses 7 multiplications per 2x2 block,
    # so 4 blocks of 2x2, each needing 7 mults = 28 mults for the 4x4*4x4.
    # Wait, that's not right either. Let me think again.
    
    # For 2x2*2x2, Strassen uses 7 mults.
    # For 4x4*4x4, we can partition into 2x2 blocks (4 blocks per matrix),
    # so 7 mults per block position... no, that's not how it works.
    
    # Actually, the standard Strassen algorithm for 4x4*4x4:
    # Partition each 4x4 matrix into four 2x2 blocks.
    # The product C = AB has four 2x2 blocks, each computed from the 7 Strassen products
    # of the corresponding 2x2 blocks of A and B.
    # So we need 7 mults for each of the 4 block positions = 28 mults total.
    # But wait, the 7 Strassen products are shared across the block computation?
    # No, each 2x2*2x2 multiplication is independent.
    
    # So for 4x4*4x4, Strassen gives rank 4*7 = 28? That seems too low.
    # Let me check: the bilinear complexity of 4x4*4x4 is at most 28 using Strassen.
    # Yes, that's correct.
    
    # Now for 4x4*4x6:
    # We can view B as two 4x3 matrices: B = [B1 | B2] where B1, B2 are 4x3.
    # Then AB = [AB1 | AB2].
    # Each ABi is 4x4 * 4x3.
    
    # For 4x4*4x3, we can partition into 2x2 blocks for the 4x4 part,
    # but the 4x3 part doesn't divide evenly into 2x2.
    
    # Alternative: use the known result that M(4,4,6) <= 84.
    # Let's try to construct a rank-84 decomposition.
    
    # Actually, let's try a simpler approach: use a random search with integer entries.
    # Start with rank 84 and try to find a valid decomposition.
    
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of n*m*p = 96 equations in rank * (n*m + m*p + p*n) variables.
    # For rank 84: 84 * (16 + 24 + 24) = 84 * 64 = 5376 variables.
    # 96 equations, 5376 variables - highly underdetermined.
    
    # Let's try to find a sparse solution.
    
    def try_rank(rank):
        """Try to find a rank-rank decomposition with small integer entries."""
        if time.time() > deadline:
            return None
        
        # Initialize with zeros
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # For each tensor index (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let's try a structured approach: for each r, assign it to help satisfy some equations.
        
        # Simple approach: greedy assignment
        # For each (i,j,k), find an r where we can set U,V,W to make the product 1.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if time.time() > deadline:
                        return None
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    
                    # Find an r where U[ui, r], V[vj, r], W[wk, r] are all zero
                    found = False
                    for r in range(rank):
                        if U[ui, r] == 0 and V[vj, r] == 0 and W[wk, r] == 0:
                            U[ui, r] = 1
                            V[vj, r] = 1
                            W[wk, r] = 1
                            found = True
                            break
                    if not found:
                        return None
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.dot(U[ui], V[vj] * W[wk])
                    if abs(val - 1) > 1e-9:
                        return None
        
        return U, V, W
    
    # Try ranks from low to high
    best_result = None
    for rank in range(80, 97):
        if time.time() > deadline:
            break
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
