"""
Strategy: Block-Circulant Tensor Decomposition via Fourier Transform on Z_3.
We partition the 12 columns into 3 groups of 4.
The multiplication tensor T for 2x2 x 2x12 can be viewed as a tensor over the ring R = Q[x]/(x^3-1).
Since x^3-1 = (x-1)(x^2+x+1), R is isomorphic to Q x Q(omega).
We can decompose the problem into:
1. A "scalar" part corresponding to the x=1 root (sum of the 3 groups).
2. A "complex-like" part corresponding to the roots of x^2+x+1.

Effectively, we are looking for a decomposition of the form:
T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
where the indices are structured such that we can exploit the cyclic symmetry.

However, a more direct and robust approach for this specific "2x2x12" task is to
search for a decomposition that is a sum of 4 "base" decompositions, each of size 2x2x4,
where the 4 bases are related by cyclic shifts in the 12-column dimension.

Let the 12 columns be indexed by k in 0..11.
Group them into 3 blocks of 4: K0, K1, K2.
The Fourier transform on the group Z_3 (acting on the blocks) suggests we can diagonalize the block structure.

Actually, a simpler structural insight:
The tensor T[i, j, k] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
This is the standard matrix multiplication tensor.
We can try to find a rank R decomposition where R < 42.
The known lower bound for 2x2x2 is 7.
For 2x2x12, if we treat it as 3 parallel 2x2x4 problems, and if 2x2x4 has rank 12 (which is 3*4, trivial), 
then total rank is 36.
But we can do better.

Let's use a "Symmetry-Reduced" Search.
We assume the solution has a block-circulant structure in the k-dimension.
Specifically, we assume the decomposition consists of "orbits" under the cyclic group C_3 acting on the 3 blocks of 4 columns.
If a rank-1 term (u, v, w) is in the decomposition, then (u, v', w') where v', w' are shifted versions might also be in the decomposition or combine to form it.

To keep it simple and robust, we will search for a decomposition of the form:
T = T0 + T1 + T2
where T0, T1, T2 are 2x2x4 tensors corresponding to the three blocks of columns.
If we assume the 2x2x4 tensors are identical up to permutation, we can find a single 2x2x4 decomposition of rank r,
and then "tile" it 3 times? No, that gives rank 3r.
If r=14 (better than trivial 16 for 2x2x4?), rank 42.
If we can find a 2x2x4 decomposition of rank 12, total rank 36.
If we can find a 2x2x4 decomposition of rank 10, total rank 30.

Is there a better structure?
Consider the tensor as an element of (Q^4)^* \otimes (Q^{24})^* \otimes (Q^{24})^*.
Let's try a "Randomized Algebraic Search" constrained to have integer/half-integer entries.
We will initialize with the Strassen-42 solution and then perform a "Local Search" that allows
merging of columns if they are linearly dependent in a specific way, or swapping entries to reduce the rank.

Actually, the most promising "different strategy" is **Integer Programming / Constraint Satisfaction** on the support of the factors.
We define the set of non-zero entries in U, V, W.
We search for a minimal set of "triplets" (a, b, c) such that the tensor can be represented.
This is NP-hard, but for small dimensions, we can use a heuristic:
1. Start with rank 24 (theoretical minimum for 2x2x12 if we assume rank scales linearly with p? No, rank(2x2x1) = 4. So 3*4=12? No.
   Rank(2x2xn) is known to be 2n for n>=2? No, Strassen says rank(2x2x2)=7 < 8.
   For 2x2x4, is rank 12 possible?
   
Let's try to construct a rank-28 or rank-30 solution using a "Kronecker Product" of 2x2x2 (rank 7) and 1x1x2 (rank 2)?
If we take the tensor product of the 2x2x2 Strassen tensor (rank 7) and a 1x1x2 tensor (rank 2), we get a 2x2x4 tensor of rank 14.
Then 3 such blocks give rank 42.
This is the same as the current Strassen approach.

To improve, we need to "share" the rank-1 terms between blocks.
This is possible if the 2x2x4 tensor has a lower rank than 3 * rank(2x2x1) = 12?
Actually, rank(2x2x1) is 4. So 3*4=12 is the trivial rank for 2x2x4? No, 2x2x4 has 2*2*4=16 entries.
The trivial rank is 16.
The Kronecker product of 2x2x2 (rank 7) and 1x1x2 (rank 2) gives rank 14.
Can we do better than 14 for 2x2x4?
Yes, the rank of 2x2x4 is known to be 12.
If we use a rank-12 decomposition for each 2x2x4 block, the total rank is 36.
This is an improvement over 42.

Strategy:
1. Find a rank-12 decomposition for 2x2x4.
2. Tile it 3 times to get a rank-36 decomposition for 2x2x12.
3. Try to merge columns between the 3 blocks to reduce the rank further.

How to find rank-12 for 2x2x4?
We can use a known construction or search.
A known construction for 2x2xn uses the "Landsberg" or "Strassen-like" patterns.
For 2x2x4, we can view it as 2x2x2 + 2x2x2, but with sharing.
Actually, we can just hardcode a rank-12 solution for 2x2x4 if we can find one, or search for it.
Given the time limit, searching for a rank-12 2x2x4 tensor might be feasible.
Dimensions: 4x8x8 = 256 entries. Rank 12.
We can use a simple local search starting from a rank-14 solution (Kronecker).

Let's implement a search for a rank-12 2x2x4 decomposition.
If found, we tile it.
If not, we fall back to rank-14 (total 42).
Then we try to merge columns between the 3 tiles to see if we can go below 36.
"""

import time
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS

    # Target tensor T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def time_left():
        return deadline - time.time()

    def verify(U, V, W):
        if U is None:
            return False
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-9

    def get_strassen_42():
        """
        Generates the rank-42 decomposition using Strassen's algorithm
        applied to 6 disjoint 2x2x2 blocks.
        """
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        prods = [
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1
            ([(2,1), (3,1)], [(0,1)], [(1,1)]), # M2
            ([(0,1)], [(1,1), (3,-1)], [(2,1)]), # M3
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (1,1)]), # M4
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (2,1)]), # M5
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(1,1)]), # M6
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]) # M7
        ]
        
        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = prods[s]
                
                for idx, coeff in u_entries:
                    U[idx, col] = coeff
                    
                for idx, coeff in v_entries:
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff
                    
                for idx, coeff in w_entries:
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff
                    
        return U, V, W

    # Fallback: Rank 42
    U_42, V_42, W_42 = get_strassen_42()
    if not verify(U_42, V_42, W_42):
        # Absolute fallback: Schoolbook 48
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_rank = 42
    best_U, best_V, best_W = U_42, V_42, W_42

    # Strategy: Search for a rank-12 2x2x4 decomposition, then tile 3 times.
    # 2x2x4 tensor T4
    T4 = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                a = i * 2 + j
                b = j * 4 + k
                c = k * 2 + i
                T4[a, b, c] = 1.0

    def verify_2x2x4(U, V, W):
        if U.shape[0] != 4 or V.shape[0] != 8 or W.shape[0] != 8:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T4)
        return err < 1e-9

    def search_2x2x4_rank(target_rank, max_iter, seed):
        np.random.seed(seed)
        U = np.random.randn(4, target_rank)
        V = np.random.randn(8, target_rank)
        W = np.random.randn(8, target_rank)
        
        lr = 0.05
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time_left() < 10:
                break
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_rec - T4
            err = np.sum(diff**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-8:
                break
                
            Gu = np.einsum('abc,br,cr->ar', diff, V, W)
            Gv = np.einsum('abc,ar,cr->br', diff, U, W)
            Gw = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * Gu
            V -= lr * Gv
            W -= lr * Gw
            
            if it % 20 == 0:
                lr *= 0.95

        # Try to round to integers/half-integers
        U_r = np.round(best_U)
        V_r = np.round(best_V)
        W_r = np.round(best_W)
        if verify_2x2x4(U_r, V_r, W_r):
            return U_r, V_r, W_r
            
        U_h = np.round(best_U * 2) / 2.0
        V_h = np.round(best_V * 2) / 2.0
        W_h = np.round(best_W * 2) / 2.0
        if verify_2x2x4(U_h, V_h, W_h):
            return U_h, V_h, W_h
            
        return None, None, None

    # Try to find a rank-12 2x2x4 decomposition
    found_2x2x4 = False
    for rank_try in [12, 13, 14]:
        if time_left() < 15:
            break
        for seed in range(5):
            if time_left() < 5:
                break
            U4, V4, W4 = search_2x2x4_rank(rank_try, 1000, seed)
            if U4 is not None:
                # Construct the 2x2x12 solution by tiling
                # The 12 columns are 3 blocks of 4.
                # U is 4x(rank*3)
                # V is 24x(rank*3). Block t: rows 8t..8t+7 are V4.
                # W is 24x(rank*3). Block t: rows 8t..8t+7 are W4.
                
                total_rank = rank_try * 3
                U_total = np.zeros((4, total_rank))
                V_total = np.zeros((24, total_rank))
                W_total = np.zeros((24, total_rank))
                
                for t in range(3):
                    U_total[:, t*rank_try:(t+1)*rank_try] = U4
                    V_total[t*8:(t+1)*8, t*rank_try:(t+1)*rank_try] = V4
                    W_total[t*8:(t+1)*8, t*rank_try:(t+1)*rank_try] = W4
                    
                if verify(U_total, V_total, W_total):
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_U, best_V, best_W = U_total, V_total, W_total
                    found_2x2x4 = True
                    break
        if found_2x2x4:
            break

    # If we found a tiled solution, try to merge columns between blocks to reduce rank further.
    # This is a heuristic local search.
    if found_2x2x4 and time_left() > 20:
        # We have a solution of rank 3*r.
        # We can try to see if we can reduce the rank by 1, 2, etc.
        # by finding linear dependencies or replacing columns.
        # For simplicity, we just return the tiled solution if it's better than 42.
        pass

    # If we didn't find a better 2x2x4 solution, we stick with Strassen 42.
    # Or, we can try a direct search for rank 40, 39, 38 on the full tensor.
    if best_rank == 42 and time_left() > 30:
        # Direct search for lower rank on full tensor
        for target_rank in [40, 39, 38]:
            if time_left() < 10:
                break
            for seed in range(3):
                if time_left() < 5:
                    break
                np.random.seed(seed)
                U = np.random.randn(4, target_rank)
                V = np.random.randn(24, target_rank)
                W = np.random.randn(24, target_rank)
                
                lr = 0.02
                best_err = np.inf
                best_U_s, best_V_s, best_W_s = U.copy(), V.copy(), W.copy()
                
                for it in range(500):
                    if time_left() < 5:
                        break
                    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                    diff = T_rec - T
                    err = np.sum(diff**2)
                    
                    if err < best_err:
                        best_err = err
                        best_U_s, best_V_s, best_W_s = U.copy(), V.copy(), W.copy()
                        
                    if err < 1e-8:
                        break
                        
                    Gu = np.einsum('abc,br,cr->ar', diff, V, W)
                    Gv = np.einsum('abc,ar,cr->br', diff, U, W)
                    Gw = np.einsum('abc,ar,br->cr', diff, U, V)
                    
                    U -= lr * Gu
                    V -= lr * Gv
                    W -= lr * Gw
                    
                    if it % 20 == 0:
                        lr *= 0.95
                        
                U_r = np.round(best_U_s)
                V_r = np.round(best_V_s)
                W_r = np.round(best_W_s)
                if verify(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
                U_h = np.round(best_U_s * 2) / 2.0
                V_h = np.round(best_V_s * 2) / 2.0
                W_h = np.round(best_W_s * 2) / 2.0
                if verify(U_h, V_h, W_h):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_h, V_h, W_h
                    break

    # Final verification
    if verify(best_U, best_V, best_W):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_42.tolist(),
            "V": V_42.tolist(),
            "W": W_42.tolist()
        }
# EVOLVE-BLOCK-END
