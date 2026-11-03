"""Program for rank-minimal tensor decomposition of 2x3x9 matrix multiplication."""
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    # Strategy: exploit block structure. B is 3x9, split into 3 blocks of 3x3.
    # C[:, 3g:3g+3] = A @ B[:, 3g:3g+3] for g=0,1,2.
    # So we need 3 copies of 2x3x3 multiplication.
    # If we have a rank-r0 decomposition for 2x3x3, total rank = 3*r0.
    # But we can potentially share U across blocks since A is the same.
    # Actually, in the tensor decomposition, U is indexed by (i,j) which is the same for all blocks.
    # So U is shared. V and W are block-specific.
    # Total rank = r0 * 3 if we don't share, but since U is shared, we still need 3*r0 terms
    # because each term has a specific V and W for a specific block.
    # Wait, no. The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    # For block g, we need terms where V and W correspond to block g.
    # If we use the same U for all blocks, we can have:
    #   For g=0: terms c=0..r0-1 with U_c, V_{g,c}, W_{g,c}
    #   For g=1: terms c=r0..2*r0-1 with U_{c-r0}, V_{1,c-r0}, W_{1,c-r0}
    #   For g=2: terms c=2*r0..3*r0-1 with U_{c-2*r0}, V_{2,c-2*r0}, W_{2,c-2*r0}
    # But U is the same matrix, so U_c for c in 0..r0-1 is the same as U_{c-r0} etc.
    # This means the total rank is still 3*r0, but U has only r0 distinct columns.
    # That's fine; the rank is still 3*r0.

    # So we need to find a low-rank decomposition for 2x3x3.
    # Let's search for it.

    def verify_decomp(U, V, W, n, m, p):
        """Verify that U, V, W decompose the tensor for dimensions n, m, p."""
        rank = len(U[0]) if U else 0
        # Check active entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    s = sum(U[a][c] * V[b][c] * W[d][c] for c in range(rank))
                    if s != 1:
                        return False
        # Check inactive entries (sample)
        random.seed(12345)
        for _ in range(500):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            d = random.randint(0, p * n - 1)
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k2 = b % p
            k1 = d // n
            i2 = d % n
            if i1 == i2 and j1 == j2 and k1 == k2:
                continue
            s = sum(U[a][c] * V[b][c] * W[d][c] for c in range(rank))
            if s != 0:
                return False
        return True

    def find_2x3x3_decomp(target_rank, time_limit=5.0):
        """Find a rank-r decomposition for 2x3x3 multiplication."""
        import time
        start = time.time()
        n2, m2, p2 = 2, 3, 3
        
        # Try random integer entries in {-1, 0, 1}
        random.seed(42)
        for attempt in range(2000):
            if time.time() - start > time_limit:
                break
            U = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n2 * m2)]
            V = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m2 * p2)]
            W = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p2 * n2)]
            if verify_decomp(U, V, W, n2, m2, p2):
                return U, V, W
        return None

    # Search for 2x3x3 decomposition with rank from 11 upward
    decomp_233 = None
    for r0 in range(11, 19):
        decomp_233 = find_2x3x3_decomp(r0, time_limit=3.0)
        if decomp_233 is not None:
            break

    if decomp_233 is not None:
        U2, V2, W2 = decomp_233
        r0 = len(U2[0])
        # Construct 2x3x9 decomposition
        # U is shared (same for all 3 blocks)
        # V and W are block-specific
        total_rank = 3 * r0
        U = [row[:] for row in U2]  # Copy U2, it will be repeated 3 times conceptually
        # Actually, U has shape (n*m, total_rank). We repeat U2 three times.
        U_full = []
        for row in U2:
            U_full.append(row * 3)  # Repeat each row 3 times
        
        V_full = [[0] * total_rank for _ in range(m * p)]
        W_full = [[0] * total_rank for _ in range(p * n)]
        
        for g in range(3):
            offset = g * r0
            for j in range(m):
                for k in range(p2):
                    b = j * p + g * p2 + k
                    V_full[b][offset + 0] = V2[j * p2 + k][0] if False else 0
            # Actually, V2 has shape (m*p2, r0) = (9, r0)
            # For block g, V row b = j*p + g*3 + k corresponds to V2 row j*p2 + k
            for j in range(m):
                for k in range(p2):
                    b = j * p + g * p2 + k
                    b2 = j * p2 + k
                    for c in range(r0):
                        V_full[b][offset + c] = V2[b2][c]
            
            for k in range(p2):
                for i in range(n):
                    d = (g * p2 + k) * n + i
                    d2 = k * n2 + i
                    for c in range(r0):
                        W_full[d][offset + c] = W2[d2][c]
        
        # Verify the full decomposition
        if verify_decomp(U_full, V_full, W_full, n, m, p):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
