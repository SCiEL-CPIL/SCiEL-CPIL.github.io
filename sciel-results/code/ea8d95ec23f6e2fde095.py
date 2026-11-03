"""
Program for 5x5 Matrix Multiplication Tensor Decomposition.

Strategy: Block-structured tensor decomposition.
The 5x5x5 matrix multiplication tensor is decomposed into a direct sum of
smaller tensor blocks by partitioning the index sets. For each block shape
(a x b x c), we use the best known algorithm:
- 2x2x2: Strassen (rank 7)
- 2x2x1 or 1x2x2 or 2x1x2: rank 4 (schoolbook)
- 1x1x1: rank 1
- 2x1x1 or 1x2x1 or 1x1x2: rank 1
- etc.

We search over all ways to partition {0..4} x {0..4} x {0..4} into rectangular
blocks and compute the total rank. The Kronecker product structure ensures
exactness.

Interface:
construct() -> {'U', 'V', 'W'}
U: shape (n*m, rank)
V: shape (m*p, rank)
W: shape (p*n, rank)
"""

import time
import itertools
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = 5, 5, 5

    # We partition the row index {0..4}, col index {0..4}, and output-col index {0..4}
    # into contiguous groups. Each group triple (I, J, K) defines a block.
    # For a block with |I|=a, |J|=b, |K|=c, the block tensor is the a×b×c matmul tensor.
    # Rank of block = best_known_rank(a, b, c).
    
    # Best known ranks for small matmul tensors:
    def best_rank(a, b, c):
        """Return the rank of the a×b×c matrix multiplication tensor."""
        if a == 1 and b == 1 and c == 1:
            return 1
        if a == 2 and b == 2 and c == 2:
            return 7  # Strassen
        if a == 2 and b == 2 and c == 1:
            return 4  # 2x2 * 2x1
        if a == 2 and b == 1 and c == 2:
            return 4  # 2x1 * 1x2
        if a == 1 and b == 2 and c == 2:
            return 4  # 1x2 * 2x2
        if a == 2 and b == 1 and c == 1:
            return 2  # 2x1 * 1x1
        if a == 1 and b == 2 and c == 1:
            return 2  # 1x2 * 1x1
        if a == 1 and b == 1 and c == 2:
            return 2  # 1x1 * 1x2
        if a == 1 and b == 1 and c == 1:
            return 1
        # General: schoolbook
        return a * b * c

    # Generate all ways to partition a set of size s into contiguous blocks.
    # A partition of {0..s-1} into contiguous blocks is defined by the cut points.
    def contiguous_partitions(s):
        """Yield all ways to partition {0..s-1} into contiguous blocks.
        Each partition is a list of (start, end) inclusive tuples."""
        if s == 1:
            yield [(0, 0)]
            return
        # Recursive: first block ends at position k (0-indexed), k from 0 to s-1
        for k in range(s):
            for rest in contiguous_partitions(s - 1 - k):
                # Shift rest indices
                shifted = [(a + k + 1, b + k + 1) for a, b in rest]
                yield [(0, k)] + shifted

    # Precompute all partitions of {0..4}
    partitions_5 = list(contiguous_partitions(5))
    
    # For each combination of partitions of I, J, K, compute total rank
    best_total_rank = n * m * p  # schoolbook
    best_partition = None
    
    # To keep search tractable, we limit to partitions with at most 3 blocks per dimension
    def max_blocks(part):
        return len(part)
    
    # Search over all partition triples
    # Number of partitions of 5: 2^4 = 16, so 16^3 = 4096 triples. Very manageable.
    for pi in partitions_5:
        for pj in partitions_5:
            for pk in partitions_5:
                # Each block is (I_range, J_range, K_range)
                # But we need to pair them: the block structure is a tensor product
                # Actually, the tensor is T[i,j,k] = delta[i,j,k] for matmul
                # The block decomposition: for each triple of blocks (I_a, J_b, K_c),
                # the sub-tensor is the |I_a| x |J_b| x |K_c| matmul tensor.
                # Total rank = sum over all triples of best_rank(|I_a|, |J_b|, |K_c|)
                
                total_rank = 0
                for (i0, i1) in pi:
                    a = i1 - i0 + 1
                    for (j0, j1) in pj:
                        b = j1 - j0 + 1
                        for (k0, k1) in pk:
                            c = k1 - k0 + 1
                            total_rank += best_rank(a, b, c)
                
                if total_rank < best_total_rank:
                    best_total_rank = total_rank
                    best_partition = (pi, pj, pk)
    
    # Now construct the actual U, V, W matrices for the best partition
    # The construction uses the Kronecker/block structure.
    # For each block (I, J, K) with sizes (a, b, c), we need a rank-r decomposition
    # of the a×b×c matmul tensor, then embed it into the full (25, 25, 25) tensor.
    
    # We need explicit low-rank decompositions for small tensors:
    def strassen_2x2x2():
        """Return (U, V, W) each of shape (4, 7) for 2x2x2 matmul via Strassen.
        U[i,j] corresponds to input A[i,j], V[j,k] to B[j,k], W[k,i] to C[k,i].
        Using trace convention W[k*n+i, r] for n=2."""
        # Standard Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # U: shape (4, 7), rows indexed by (i,j) -> i*2+j
        # V: shape (4, 7), rows indexed by (j,k) -> j*2+k
        # W: shape (4, 7), rows indexed by (k,i) -> k*2+i  (trace convention)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1  # A00, A11
        V[0,0] = 1; V[3,0] = 1  # B00, B11
        # C00 += M1, C11 += M1
        W[0,0] = 1  # C[0,0] -> k=0,i=0
        W[3,0] = 1  # C[1,1] -> k=1,i=1
        
        # M2: (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1  # A10, A11
        V[0,1] = 1  # B00
        W[2,1] = 1  # C[1,0] -> k=1,i=0
        
        # M3: A00(B01-B11)
        U[0,2] = 1  # A00
        V[1,2] = 1; V[3,2] = -1  # B01, -B11
        W[1,2] = 1  # C[0,1] -> k=0,i=1
        W[3,2] = 1  # C[1,1] -> k=1,i=1
        
        # M4: A11(B10-B00)
        U[3,3] = 1  # A11
        V[2,3] = 1; V[0,3] = -1  # B10, -B00
        W[0,3] = 1  # C[0,0] -> k=0,i=0
        W[2,3] = 1  # C[1,0] -> k=1,i=0
        
        # M5: (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1  # A00, A01
        V[3,4] = 1  # B11
        W[0,4] = -1  # C[0,0] -= M5
        W[1,4] = 1   # C[0,1] += M5
        
        # M6: (A10-A00)(B00+B01)
        U[2,5] = 1; U[0,5] = -1  # A10, -A00
        V[0,5] = 1; V[1,5] = 1  # B00, B01
        W[3,5] = 1  # C[1,1] += M6
        
        # M7: (A01-A11)(B10+B11)
        U[1,6] = 1; U[3,6] = -1  # A01, -A11
        V[2,6] = 1; V[3,6] = 1  # B10, B11
        W[0,6] = 1  # C[0,0] += M7
        
        return U, V, W

    def schoolbook_small(a, b, c):
        """Return (U, V, W) for a×b×c matmul, schoolbook rank a*b*c.
        U shape (a*b, a*b*c), V shape (b*c, a*b*c), W shape (c*a, a*b*c)."""
        rank = a * b * c
        U = np.zeros((a * b, rank))
        V = np.zeros((b * c, rank))
        W = np.zeros((c * a, rank))
        col = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i * b + j, col] = 1
                    V[j * c + k, col] = 1
                    W[k * a + i, col] = 1
                    col += 1
        return U, V, W

    def get_block_decomp(a, b, c):
        """Get (U, V, W) for a×b×c matmul tensor with minimal known rank."""
        if a == 2 and b == 2 and c == 2:
            return strassen_2x2x2()
        else:
            return schoolbook_small(a, b, c)

    # Build the full decomposition from the best partition
    pi, pj, pk = best_partition
    
    total_rank = 0
    # First pass: compute total rank
    for (i0, i1) in pi:
        a = i1 - i0 + 1
        for (j0, j1) in pj:
            b = j1 - j0 + 1
            for (k0, k1) in pk:
                c = k1 - k0 + 1
                total_rank += best_rank(a, b, c)
    
    # Allocate columns
    U_full = np.zeros((n * m, total_rank))  # (25, total_rank)
    V_full = np.zeros((m * p, total_rank))  # (25, total_rank)
    W_full = np.zeros((p * n, total_rank))  # (25, total_rank)
    
    col_offset = 0
    for (i0, i1) in pi:
        a = i1 - i0 + 1
        for (j0, j1) in pj:
            b = j1 - j0 + 1
            for (k0, k1) in pk:
                c = k1 - k0 + 1
                Ub, Vb, Wb = get_block_decomp(a, b, c)
                r = Ub.shape[1]
                
                # U rows: i in [i0..i1], j in [j0..j1] -> row index i*m+j
                for ri in range(a):
                    for rj in range(b):
                        U_full[(i0 + ri) * m + (j0 + rj), col_offset:col_offset + r] = Ub[ri, rj]
                
                # V rows: j in [j0..j1], k in [k0..k1] -> row index j*p+k
                for rj in range(b):
                    for rk in range(c):
                        V_full[(j0 + rj) * p + (k0 + rk), col_offset:col_offset + r] = Vb[rj, rk]
                
                # W rows: k in [k0..k1], i in [i0..i1] -> row index k*n+i (trace convention)
                for rk in range(c):
                    for ri in range(a):
                        W_full[(k0 + rk) * n + (i0 + ri), col_offset:col_offset + r] = Wb[rk, ri]
                
                col_offset += r
    
    # Verify the decomposition
    def verify_numpy(U, V, W):
        """Verify U @ V^T @ W^T structure: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify_numpy(U_full, V_full, W_full):
        result_rank = total_rank
    else:
        # Fallback: schoolbook
        result_rank = n * m * p
        U_full = np.zeros((n * m, result_rank))
        V_full = np.zeros((m * p, result_rank))
        W_full = np.zeros((p * n, result_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_full[i * m + j, col] = 1
                    V_full[j * p + k, col] = 1
                    W_full[k * n + i, col] = 1
                    col += 1
    
    # Also try: non-contiguous partitions might help, but contiguous is a good start.
    # Let's also try a "mixed" approach: use 2x2 Strassen blocks where possible
    # and handle the remainder with schoolbook.
    # This is already captured by the partition search above.
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
