"""Combined program: block-partition with Strassen for 5x5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    # Strassen decomposition for 2x2x2: 7 rank-1 tensors
    # Each tensor has U,V,W vectors of length 4 (flattened 2x2 matrices)
    def get_strassen_factors():
        # Standard Strassen 7 products
        # U: input A (2x2), V: input B (2x2), W: output C (2x2)
        # Flattened as [a00, a01, a10, a11]
        factors = []
        # M1 = (A00+A11)(B00+B11)
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (A10+A11)B00
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = A00(B10-B11)
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = A11(B01-B00)
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (A00+A01)B11
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (A10-A00)(B00+B01)
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (A01-A11)(B10+B11)
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors
    
    strassen_factors = get_strassen_factors()
    
    def build_block_decomposition(block_sizes):
        """Build decomposition using given block partition."""
        nb = len(block_sizes)
        offsets = [0]
        for s in block_sizes:
            offsets.append(offsets[-1] + s)
        
        # Compute total rank
        total_rank = 0
        for i in range(nb):
            for j in range(nb):
                for k in range(nb):
                    ni = block_sizes[i]
                    mj = block_sizes[j]
                    pk = block_sizes[k]
                    if ni == 2 and mj == 2 and pk == 2:
                        total_rank += 7
                    else:
                        total_rank += ni * mj * pk
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for i in range(nb):
            for j in range(nb):
                for k in range(nb):
                    if time.time() > deadline:
                        return None, None
                    ni = block_sizes[i]
                    mj = block_sizes[j]
                    pk = block_sizes[k]
                    
                    if ni == 2 and mj == 2 and pk == 2:
                        for (uf, vf, wf) in strassen_factors:
                            # Map local 2x2 indices to global
                            # U: row index = (i_global * m) + j_global, where i_global in [offsets[i], offsets[i]+ni)
                            # V: row index = (j_global * p) + k_global
                            # W: row index = (k_global * n) + i_global
                            for idx in range(4):
                                li = idx // 2
                                lj = idx % 2
                                gi = offsets[i] + li
                                gj = offsets[j] + lj
                                U[gi * m + gj][col] += uf[idx]
                                
                                lk = idx // 2  # For V, idx maps to (lj, lk)
                                # Actually, for V: local index idx corresponds to (row=j_local, col=k_local)
                                # In Strassen, V is B (m x p), so local row = j_local, col = k_local
                                # idx = j_local * 2 + k_local for 2x2
                                vj = idx // 2
                                vk = idx % 2
                                gvj = offsets[j] + vj
                                gvk = offsets[k] + vk
                                V[gvj * p + gvk][col] += vf[idx]
                                
                                # For W: local index idx corresponds to (row=k_local, col=i_local)
                                # W is (p x n), local row = k_local, col = i_local
                                wk = idx // 2
                                wi = idx % 2
                                gwk = offsets[k] + wk
                                gwi = offsets[i] + wi
                                W[gwk * n + gwi][col] += wf[idx]
                            col += 1
                    else:
                        for li in range(ni):
                            for lj in range(mj):
                                for lk in range(pk):
                                    gi = offsets[i] + li
                                    gj = offsets[j] + lj
                                    gk = offsets[k] + lk
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        if col != total_rank:
            return None, None
        
        return {"U": U, "V": V, "W": W}, total_rank
    
    def verify_exact(n, m, p, U, V, W):
        """Verify exact decomposition using Fraction arithmetic."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uc = Fraction(u_row[c])
                        vc = Fraction(v_row[c])
                        wc = Fraction(w_row[c])
                        s += uc * vc * wc
                    if s != 1:
                        return False
        return True
    
    # Try different block partitions
    best_result = None
    best_rank = float('inf')
    
    partitions_to_try = [
        [2, 2, 1],
        [1, 2, 2],
        [2, 1, 2],
        [1, 1, 3],
        [3, 1, 1],
        [1, 3, 1],
    ]
    
    for partition in partitions_to_try:
        if time.time() > deadline:
            break
        result, rank = build_block_decomposition(partition)
        if result is None:
            continue
        if rank < best_rank:
            # Verify
            if verify_exact(n, m, p, result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
            # Even if verification fails, keep track
            elif best_result is None:
                # Don't use unverified result as best, but remember it
                pass
    
    # Fallback: schoolbook (rank 125)
    if best_result is None:
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
        return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
