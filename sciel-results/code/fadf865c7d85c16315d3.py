"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def get_strassen_terms_2x2x2():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_dict, v_dict, w_dict).
        
        The 2x2x2 multiplication tensor T[i,j,k] = 1 if i==j==k else 0.
        We need 7 rank-1 tensors whose sum equals T.
        
        Standard Strassen decomposition:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11  
        M3 = A11(B21-B11)
        M4 = A22(B12-B02) -- corrected below
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        Where C = AB, A[i,j] is row i col j of A, B[j,k] is row j col k of B, C[i,k] is row i col k of C.
        The tensor entry T[i,j,k] = 1 iff the (i,k) entry of C uses A[i,j]*B[j,k].
        
        For 2x2x2: indices i,j,k in {0,1}
        T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1, others=0
        
        Let me verify each term contributes correctly.
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # u: A00->1, A11->1 => {(0,0):1, (1,1):1}
        # v: B00->1, B11->1 => {(0,0):1, (1,1):1}
        # w: C00 gets M1, C11 gets M1 => {(0,0):1, (1,1):1}
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        
        # M2 = (A10+A11)B00
        # u: A10->1, A11->1 => {(1,0):1, (1,1):1}
        # v: B00->1 => {(0,0):1}
        # w: C01 gets M2, C11 gets -M2 => {(0,1):1, (1,1):-1}
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        
        # M3 = A00(B10-B00)
        # u: A00->1 => {(0,0):1}
        # v: B10->1, B00->-1 => {(1,0):1, (0,0):-1}
        # w: C10 gets M3, C11 gets M3 => {(1,0):1, (1,1):1}
        terms.append((
            {(0,0): 1},
            {(1,0): 1, (0,0): -1},
            {(1,0): 1, (1,1): 1}
        ))
        
        # M4 = A11(B01-B11)
        # u: A11->1 => {(1,1):1}
        # v: B01->1, B11->-1 => {(0,1):1, (1,1):-1}
        # w: C00 gets M4, C01 gets M4 => {(0,0):1, (0,1):1}
        terms.append((
            {(1,1): 1},
            {(0,1): 1, (1,1): -1},
            {(0,0): 1, (0,1): 1}
        ))
        
        # M5 = (A00+A01)B11
        # u: A00->1, A01->1 => {(0,0):1, (0,1):1}
        # v: B11->1 => {(1,1):1}
        # w: C00 gets -M5, C10 gets M5 => {(0,0):-1, (1,0):1}
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ))
        
        # M6 = (A10-A00)(B00+B01)
        # u: A10->1, A00->-1 => {(1,0):1, (0,0):-1}
        # v: B00->1, B01->1 => {(0,0):1, (0,1):1}
        # w: C11 gets M6 => {(1,1):1}
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        
        # M7 = (A01-A11)(B10+B11)
        # u: A01->1, A11->-1 => {(0,1):1, (1,1):-1}
        # v: B10->1, B11->1 => {(1,0):1, (1,1):1}
        # w: C00 gets M7 => {(0,0):1}
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        
        return terms

    def verify_strassen_terms():
        """Verify that the Strassen terms correctly decompose the 2x2x2 tensor."""
        terms = get_strassen_terms_2x2x2()
        # Build the tensor
        T = np.zeros((2, 2, 2))
        for i in range(2):
            T[i, i, i] = 1  # T[i,j,k] = 1 iff i==j==k
        
        # Sum up the rank-1 terms
        S = np.zeros((2, 2, 2))
        for u_pat, v_pat, w_pat in terms:
            for (li, lj), val_u in u_pat.items():
                for (lj2, lk), val_v in v_pat.items():
                    if lj == lj2:
                        for (lk2, li2), val_w in w_pat.items():
                            if lk == lk2:
                                S[li, lj, lk] += val_u * val_v * val_w
        
        if not np.allclose(S, T, atol=1e-10):
            print(f"Strassen verification failed!\nS={S}\nT={T}")
            return False
        return True

    def generate_all_partitions(dim):
        """Generate all ways to partition indices 0..dim-1 into contiguous groups of size 1 or 2."""
        if dim == 0:
            return [[]]
        results = []
        def gen(idx, current):
            if idx >= dim:
                results.append(current[:])
                return
            # Try group of size 1
            current.append([idx])
            gen(idx+1, current)
            current.pop()
            # Try group of size 2
            if idx+1 < dim:
                current.append([idx, idx+1])
                gen(idx+2, current)
                current.pop()
        gen(0, [])
        return results

    def compute_rank_for_partition(i_groups, j_groups, k_groups):
        """Compute the total rank for a given partition using Strassen for 2x2x2 blocks."""
        total_rank = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total_rank += 7
                    else:
                        total_rank += si * sj * sk
        return total_rank

    def build_decomposition_for_partition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        total_rank = compute_rank_for_partition(i_groups, j_groups, k_groups)
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = get_strassen_terms_2x2x2()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        # Use Strassen for this 2x2x2 block
                        for (u_pat, v_pat, w_pat) in strassen_terms:
                            for (li, lj), val_u in u_pat.items():
                                gi, gj = ig[li], jg[lj]
                                U[gi * m + gj][col] = val_u
                            for (lj, lk), val_v in v_pat.items():
                                gj, gk = jg[lj], kg[lk]
                                V[gj * p + gk][col] = val_v
                            for (lk, li), val_w in w_pat.items():
                                gk, gi = kg[lk], ig[li]
                                W[gk * n + gi][col] = val_w
                            col += 1
                    else:
                        # Schoolbook for this sub-block
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify Strassen terms first
    if not verify_strassen_terms():
        # If Strassen is broken, just return schoolbook
        return best_result

    # Generate all possible partitions
    i_partitions = generate_all_partitions(n)  # n=3
    j_partitions = generate_all_partitions(m)  # m=5
    k_partitions = generate_all_partitions(p)  # p=6

    # Find the minimum rank partition
    min_rank = best_rank
    best_partition = None
    
    for i_g in i_partitions:
        for j_g in j_partitions:
            for k_g in k_partitions:
                r = compute_rank_for_partition(i_g, j_g, k_g)
                if r < min_rank:
                    min_rank = r
                    best_partition = (i_g, j_g, k_g)
    
    # Build the best partition
    if best_partition is not None:
        i_g, j_g, k_g = best_partition
        result = build_decomposition_for_partition(i_g, j_g, k_g)
        rank = result.pop("rank")
        
        # Verify
        if verify_numpy(result["U"], result["V"], result["W"], rank):
            best_result = result
            best_rank = rank
        else:
            # Fallback to schoolbook
            pass
    else:
        # Use schoolbook
        pass

    return best_result
# EVOLVE-BLOCK-END
