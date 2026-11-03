"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Row/col indices for U: i*m+j with m=2 -> i*2+j
    Row/col indices for V: j*p+k with p=2 -> j*2+k
    Row/col indices for W: k*n+i with n=2 -> k*2+i
    
    Strassen formulas:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U rows: 0=(i0,j0)=a00, 1=(i0,j1)=a01, 2=(i1,j0)=a10, 3=(i1,j1)=a11
    # V rows: 0=(j0,k0)=b00, 1=(j0,k1)=b01, 2=(j1,k0)=b10, 3=(j1,k1)=b11
    # W rows: 0=(k0,i0)=C00, 1=(k0,i1)=C10, 2=(k1,i0)=C01, 3=(k1,i1)=C11
    
    # M1: (a00+a11)(b00+b11)
    U[0,0] += 1; U[3,0] += 1
    V[0,0] += 1; V[3,0] += 1
    W[0,0] += 1  # C00
    W[3,0] += 1  # C11
    
    # M2: (a10+a11)b00
    U[2,1] += 1; U[3,1] += 1
    V[0,1] += 1
    W[1,1] += 1  # C10
    W[3,1] += -1 # C11
    
    # M3: a00(b01-b11)
    U[0,2] += 1
    V[1,2] += 1; V[3,2] += -1
    W[2,2] += 1  # C01
    W[3,2] += 1  # C11
    
    # M4: a11(b10-b00)
    U[3,3] += 1
    V[2,3] += 1; V[0,3] += -1
    W[0,3] += 1  # C00
    W[1,3] += 1  # C10
    
    # M5: (a00+a01)b11
    U[0,4] += 1; U[1,4] += 1
    V[3,4] += 1
    W[0,4] += -1 # C00
    W[2,4] += 1  # C01
    
    # M6: (a10-a00)(b00+b01)
    U[2,5] += 1; U[0,5] += -1
    V[0,5] += 1; V[1,5] += 1
    W[3,5] += 1  # C11
    
    # M7: (a01-a11)(b10+b11)
    U[1,6] += 1; U[3,6] += -1
    V[2,6] += 1; V[3,6] += 1
    W[0,6] += 1  # C00
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 3, 3, 7
    
    # Get Strassen decomposition for 2x2 blocks
    SU, SV, SW = _strassen_2x2()
    
    # We'll build the full decomposition by combining:
    # - 3 Strassen 2x2x2 blocks (rank 7 each)
    # - Trivial 1x1x1 terms for the remaining entries (rank 1 each)
    
    # The 3x3x7 tensor has entries indexed by (i, j, k) where
    # i in [0,3), j in [0,3), k in [0,7)
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (21, rank)
    
    # We need to identify which (i,j,k) entries can be covered by Strassen blocks
    # and which must be done trivially.
    
    # A Strassen 2x2x2 block covers (i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1})
    # We need to pack non-overlapping 2x2x2 blocks in the 3x3x7 grid.
    
    # Let's find a good packing. We can have at most floor(3/2)*floor(3/2)*floor(7/2) = 1*1*3 = 3 blocks
    # if we're strict about non-overlap. But we can be smarter.
    
    # Actually, let's just use 3 blocks:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5}
    # Remaining entries: all where i=2, or j=2, or k in {6}, plus the entries not covered
    
    # Actually, let's be more careful. Let's list all (i,j,k) and see which are covered.
    
    # Let's use a different approach: build the full decomposition explicitly.
    # Start with rank 63 trivial, then replace 3 2x2x2 blocks with Strassen.
    
    # Trivial decomposition has one term per (i,j,k) entry.
    # For a 2x2x2 block, we have 8 trivial terms. Strassen has 7.
    # So each Strassen block saves 1 rank.
    
    # Let's identify 3 disjoint 2x2x2 blocks:
    # Block A: i in {0,1}, j in {0,1}, k in {0,1}
    # Block B: i in {0,1}, j in {2,2}... no, j needs to be 2 consecutive.
    
    # Let's use:
    # Block A: i in {0,1}, j in {0,1}, k in {0,1}
    # Block B: i in {0,1}, j in {0,1}, k in {3,4}  
    # Block C: i in {0,1}, j in {0,1}, k in {5,6}
    
    # These are disjoint in k. But what about j=2 and i=2 entries? They're not covered.
    
    # Let's just build it programmatically.
    
    # Start with all 63 trivial terms
    rank_total = n * m * p
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Assign trivial terms
    col = 0
    trivial_cols = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                trivial_cols[(i, j, k)] = col
                col += 1
    
    # Now replace 3 blocks with Strassen
    # We'll remove the 8 trivial terms for each block and add 7 Strassen terms
    
    # Let's pick 3 blocks:
    blocks = [
        (0, 0, 0),  # i0=0, j0=0, k0=0
        (0, 0, 3),  # i0=0, j0=0, k0=3
        (0, 0, 5),  # i0=0, j0=0, k0=5
    ]
    
    # For each block, we need to:
    # 1. Zero out the 8 trivial columns
    # 2. Add 7 new Strassen columns
    
    # But we can't just zero out and add - we need to rebuild.
    # Let's build a new decomposition from scratch.
    
    # New approach: build the decomposition by combining Strassen blocks and trivial terms.
    
    # Let's identify which (i,j,k) entries are covered by Strassen blocks.
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Build new decomposition
    new_rank = len(covered) - 3 * 8 + 3 * 7 + (rank_total - len(covered))
    # = len(covered) - 24 + 21 + (63 - len(covered))
    # = 63 - 3
    # = 60
    
    new_U = np.zeros((n * m, new_rank))
    new_V = np.zeros((m * p, new_rank))
    new_W = np.zeros((p * n, new_rank))
    
    col = 0
    
    # Add Strassen terms for each block
    for (i0, j0, k0) in blocks:
        for t in range(7):
            # For each of the 7 Strassen terms, add to U, V, W
            for (du, dv, dw) in [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]:
                pass  # This is getting complicated. Let me use the precomputed Strassen matrices.
        
        # Use the Strassen matrices directly
        # SU has shape (4,7), rows indexed by (i_local, j_local) -> i_local*2 + j_local
        # SV has shape (4,7), rows indexed by (j_local, k_local) -> j_local*2 + k_local
        # SW has shape (4,7), rows indexed by (k_local, i_local) -> k_local*2 + i_local
        
        for t in range(7):
            for il in range(2):
                for jl in range(2):
                    if SU[il*2+jl, t] != 0:
                        new_U[(i0+il)*m + (j0+jl), col] += SU[il*2+jl, t]
            for jl in range(2):
                for kl in range(2):
                    if SV[jl*2+kl, t] != 0:
                        new_V[(j0+jl)*p + (k0+kl), col] += SV[jl*2+kl, t]
            for kl in range(2):
                for il in range(2):
                    if SW[kl*2+il, t] != 0:
                        new_W[(k0+kl)*n + (i0+il), col] += SW[kl*2+il, t]
            col += 1
    
    # Add trivial terms for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    new_U[i * m + j, col] = 1
                    new_V[j * p + k, col] = 1
                    new_W[k * n + i, col] = 1
                    col += 1
    
    # Verify the decomposition
    # C = A @ B should equal sum_t U_t * V_t * W_t
    # For a test: let A = random(3,3), B = random(3,7)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor decomposition: C[i,k] = sum_t (sum_j U[i*m+j, t]) (sum_j V[j*p+k, t]) W[k*n+i, t]
    # Wait, that's not right. Let me re-examine.
    
    # The matrix multiplication tensor T[i,j,k] = delta (1 if we're computing C[i,k] from A[i,j]*B[j,k])
    # The decomposition is: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Let's verify with a simple test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C = A @ B
    
    C_recon = np.zeros((n, p))
    for t in range(new_rank):
        # U[:, t] has shape (n*m,), V[:, t] has shape (m*p,), W[:, t] has shape (p*n,)
        # The product for this term contributes to C[i,k]:
        # sum_j U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_recon[i, k] += new_U[i*m+j, t] * new_V[j*p+k, t] * new_W[k*n+i, t]
    
    if not np.allclose(C, C_recon, atol=1e-10):
        # Fallback to trivial decomposition
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
# EVOLVE-BLOCK-END
