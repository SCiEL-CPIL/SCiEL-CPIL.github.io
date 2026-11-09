"""
Rank-minimal exact tensor decomposition for 2x4 times 4x7 matrix multiplication.

Strategy:
  - The multiplication tensor T has indices (i, j, k) with i in [0,2), j in [0,4), k in [0,7).
  - T[i,j,k] = 1 (this is the standard matmul tensor).
  - We seek a rank-r decomposition: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    where u has length n*m = 8, v has length m*p = 28, w has length p*n = 14.
  - Output convention: W[k*n + i, c] (i.e., w index = k*n + i).

  Known bounds:
  - Schoolbook rank: n*m*p = 2*4*7 = 56.
  - For 2x4x7, the rank is at most 56 (schoolbook).
  - Using block structure: partition into 2x2 blocks along the j and k axes.
    The 2x4x7 problem can be seen as:
      A (2x4) * B (4x7) -> C (2x7)
    
    We can use the fact that rank(2x2x2) <= 7 (Strassen) to reduce the rank.
    
    Partition j into two groups of 2: j1 in [0,2), j2 in [2,4)
    Partition k into groups: k1 in [0,2), k2 in [2,4), k3 in [4,7) (last has 3)
    
    Actually, let's think more carefully. The tensor is:
    T[i, j, k] = 1 for all valid i, j, k.
    
    A rank-1 term is u ⊗ v ⊗ w where:
      u is a vector in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    The contribution is sum over c of u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    For the standard matmul tensor, T[i,j,k] = 1, so we need:
    sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    
    Schoolbook: one term per (i,j,k) triple, giving rank 56.
    
    To reduce rank, we can group terms. For example, if we group over j for fixed i,k:
    For fixed i,k, we need sum over j of A[i,j] * B[j,k] = C[i,k].
    The tensor entry T[i,j,k] = 1 means the "multiplication" is just 1.
    
    Actually, the tensor decomposition for matrix multiplication is:
    T_{ijk} = sum_c U_{ij}^c * V_{jk}^c * W_{ki}^c
    
    where U is indexed by (i,j), V by (j,k), W by (k,i).
    
    For the standard matmul, T_{ijk} = 1.
    
    We can use the following approach: for each pair of "block rows" and "block columns",
    use a lower-rank decomposition of the corresponding sub-tensor.
    
    Let's partition:
      i: 2 rows
      j: 4 columns, split into 2 blocks of 2: J1=[0,1], J2=[2,3]
      k: 7 rows, split into blocks: K1=[0,1], K2=[2,3], K3=[4,5,6]
    
    For each combination of (j-block, k-block), we have a 2 x 2 x (|K|) subproblem.
    
    For j-block of size 2 and k-block of size 2: this is a 2x2x2 problem (i=2, j=2, k=2).
    Strassen gives rank 7 for 2x2x2.
    
    For j-block of size 2 and k-block of size 3: this is a 2x2x3 problem.
    We need the rank of the 2x2x3 matmul tensor.
    
    Let's compute the rank of 2x2x3:
    The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}.
    Schoolbook rank: 2*2*3 = 12.
    
    Can we do better? Using Strassen-like techniques:
    For 2x2x3, we can think of it as:
    C[:, 0:2] = A * B[:, 0:2]  (2x2x2, rank <= 7)
    C[:, 2:3] = A * B[:, 2:3]  (2x2x1, rank = 4 schoolbook)
    Total: 7 + 4 = 11.
    
    Or alternatively, use a unified decomposition.
    
    Let's just use the block approach:
    - J1 (j=0,1) with K1 (k=0,1): 2x2x2, rank 7
    - J1 (j=0,1) with K2 (k=2,3): 2x2x2, rank 7
    - J1 (j=0,1) with K3 (k=4,5,6): 2x2x3, rank <= 11
    - J2 (j=2,3) with K1 (k=0,1): 2x2x2, rank 7
    - J2 (j=2,3) with K2 (k=2,3): 2x2x2, rank 7
    - J2 (j=2,3) with K3 (k=4,5,6): 2x2x3, rank <= 11
    
    Total: 7+7+11+7+7+11 = 50.
    
    That's already better than 56.
    
    Can we do better for 2x2x3? Let's think...
    The 2x2x3 tensor: T[i,j,k] = 1.
    We need sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    Actually, let me just implement the block decomposition with Strassen for 2x2x2
    and schoolbook for 2x2x3, giving rank 50.
    
    Wait, I need to double-check the Strassen decomposition for 2x2x2 in the tensor format.
    
    Strassen for 2x2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[p, q], [r, s]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    p = M1 + M4 - M5 + M7
    q = M3 + M5
    r = M2 + M4
    s = M1 - M2 + M3 + M6
    
    Each M is a product of a linear combination of A entries and a linear combination of B entries.
    In the tensor decomposition, each M corresponds to a rank-1 term where:
      u_c has the coefficients of A entries in the linear combo
      v_c has the coefficients of B entries in the linear combo
      w_c has the coefficients of C entries (from the output formula)
    
    For M1 = (a+d)(e+h):
      u: a=1, d=1 (indices (0,0) and (1,1) in 2x2)
      v: e=1, h=1 (indices (0,0) and (1,1) in 2x2)
      w: p=1, s=1 (indices (0,0) and (1,1) in 2x2)
    
    Let me map this to the global indices.
    
    For the 2x2x2 subproblem at (j-block jb, k-block kb):
      j indices: jb*2 + 0, jb*2 + 1
      k indices: kb*2 + 0, kb*2 + 1
      i indices: 0, 1
    
    U index: i*m + j = i*4 + (jb*2 + local_j)
    V index: j*p + k = (jb*2 + local_j)*7 + (kb*2 + local_k)
    W index: k*n + i = (kb*2 + local_k)*2 + i
    
    For Strassen term t with local indices:
      u[local_i, local_j] -> U[i*4 + jb*2 + local_j]
      v[local_j, local_k] -> V[(jb*2 + local_j)*7 + kb*2 + local_k]
      w[local_k, local_i] -> W[(kb*2 + local_k)*2 + i]
    
    Let me code this up.
    
    For 2x2x3 (j-block of 2, k-block of 3):
    Use schoolbook: 2*2*3 = 12 terms. Or try to do better.
    
    Actually, let's just use schoolbook for 2x2x3 to be safe: rank 12.
    Total: 7+7+12+7+7+12 = 52.
    
    Hmm, that's 52, still better than 56.
    
    Let me try to get a better rank for 2x2x3.
    
    2x2x3: A is 2x2, B is 2x3, C is 2x3.
    C[i, k] = sum_j A[i,j] * B[j,k]
    
    We can write:
    C[:, 0:2] = A * B[:, 0:2]  -- 2x2x2, rank 7 (Strassen)
    C[:, 2] = A * B[:, 2]  -- 2x2x1, rank 4 (schoolbook)
    Total: 11.
    
    So for 2x2x3, rank 11.
    
    Total rank: 7+7+11+7+7+11 = 50.
    
    Let me implement this.
"""

import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition in blocks.
    # j is split into 2 blocks of size 2: J0 = [0,1], J1 = [2,3]
    # k is split into 3 blocks: K0 = [0,1], K1 = [2,3], K2 = [4,5,6]
    
    # For each (j_block, k_block) pair, we have a 2 x |J| x |K| subproblem.
    # |J| is always 2. |K| is 2 for K0, K1 and 3 for K2.
    
    # For 2x2x2: use Strassen (rank 7)
    # For 2x2x3: use Strassen for 2x2x2 part + schoolbook for 2x2x1 part (rank 7+4=11)
    
    def strassen_2x2x2(jb, kb):
        """
        Returns list of (u_dict, v_dict, w_dict) for the 2x2x2 subproblem.
        jb: j block index (0 or 1), so j = jb*2 + local_j
        kb: k block index (0 or 1), so k = kb*2 + local_k
        """
        terms = []
        
        # Strassen multipliers
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Local indices: (li, lj) for A, (lj, lk) for B, (lk, li) for C
        # a=(0,0), b=(0,1), c=(1,0), d=(1,1)
        # e=(0,0), f=(0,1), g=(1,0), h=(1,1)
        # p=(0,0), q=(0,1), r=(1,0), s=(1,1)
        
        # M1 = (a+d)(e+h)
        # p += M1, s += M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # M2 = (c+d)e
        # r += M2, s -= M2
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # M3 = a(f-h)
        # q += M3, s += M3
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # M4 = d(g-e)
        # p += M4, r += M4
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        terms.append((u4, v4, w4))
        
        # M5 = (a+b)h
        # p -= M5, q += M5
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1}
        terms.append((u5, v5, w5))
        
        # M6 = (c-a)(e+f)
        # s += M6
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # M7 = (b-d)(g+h)
        # p += M7
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        # Convert to global indices
        global_terms = []
        for u, v, w in terms:
            gu = {}
            gv = {}
            gw = {}
            for (li, lj), val in u.items():
                gi = li
                gj = jb * 2 + lj
                gu[gi * m + gj] = val
            for (lj, lk), val in v.items():
                gj = jb * 2 + lj
                gk = kb * 2 + lk
                gv[gj * p + gk] = val
            for (lk, li), val in w.items():
                gk = kb * 2 + lk
                gi = li
                gw[gk * n + gi] = val
            global_terms.append((gu, gv, gw))
        return global_terms
    
    def schoolbook_2x2x1(jb, kb):
        """
        Returns list of (u_dict, v_dict, w_dict) for the 2x2x1 subproblem.
        jb: j block index (0 or 1), so j = jb*2 + local_j
        kb: k block index for a single k element
        """
        terms = []
        for li in range(2):
            for lj in range(2):
                u = {(li, lj): 1}
                v = {(lj, 0): 1}
                w = {(0, li): 1}
                terms.append((u, v, w))
        
        global_terms = []
        for u, v, w in terms:
            gu = {}
            gv = {}
            gw = {}
            for (li, lj), val in u.items():
                gi = li
                gj = jb * 2 + lj
                gu[gi * m + gj] = val
            for (lj, lk), val in v.items():
                gj = jb * 2 + lj
                gk = kb * 2 + lk  # For 2x2x1, the single k is at position kb*2 + 0? No...
                # Wait, for the 2x2x3 case, the third k is at index kb*2 + 2 where kb is the block index for the first 2 k's.
                # Let me reconsider.
                gv[gj * p + gk] = val
            for (lk, li), val in w.items():
                gk = kb * 2 + lk
                gi = li
                gw[gk * n + gi] = val
            global_terms.append((gu, gv, gw))
        return global_terms
    
    def block_2x2x3(jb, kb):
        """
        For 2x2x3: use Strassen for the first 2 k's, schoolbook for the 3rd k.
        jb: j block index (0 or 1)
        kb: k block index, k starts at kb*2
        The 3 k's are at indices kb*2, kb*2+1, kb*2+2
        """
        terms = []
        # Strassen for k = kb*2, kb*2+1
        strassen_terms = strassen_2x2x2(jb, kb)
        terms.extend(strassen_terms)
        
        # Schoolbook for k = kb*2+2
        # This is a 2x2x1 problem with the single k at index kb*2+2
        for li in range(2):
            for lj in range(2):
                gu = {}
                gv = {}
                gw = {}
                gi = li
                gj = jb * 2 + lj
                gk = kb * 2 + 2  # The third k element
                gu[gi * m + gj] = 1
                gv[gj * p + gk] = 1
                gw[gk * n + gi] = 1
                terms.append((gu, gv, gw))
        
        return terms
    
    all_terms = []
    
    # j blocks: 0 (j=0,1), 1 (j=2,3)
    # k blocks: 0 (k=0,1), 1 (k=2,3), 2 (k=4,5,6)
    
    for jb in range(2):
        for kb in range(3):
            if kb < 2:
                # 2x2x2 subproblem
                all_terms.extend(strassen_2x2x2(jb, kb))
            else:
                # 2x2x3 subproblem
                all_terms.extend(block_2x2x3(jb, kb))
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (gu, gv, gw) in enumerate(all_terms):
        for idx, val in gu.items():
            U[idx][c] = val
        for idx, val in gv.items():
            V[idx][c] = val
        for idx, val in gw.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
