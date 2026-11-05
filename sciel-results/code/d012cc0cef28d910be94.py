"""Rewritten program for Matrix Multiplication 3,5,7 using Strassen on 2x2x2 blocks."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _get_strassen_terms():
    """
    Return the 7 rank-1 tensor terms for the 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u, v, w are length-4 vectors.
    Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    For matrix mult: sum_r u_r[i*2+j] * v_r[j*2+k] * w_r[k*2+i] = 1 for all i,j,k in {0,1}.
    """
    terms = [
        (1, 0, 0, 1,  1, 0, 0, 1,  1, 0, 0, 1),
        (0, 0, 1, 1,  1, 0, 0, 0,  0, 0, 1, 0),
        (1, 0, 0, 0,  0, 1, 0, -1,  0, 1, 0, 1),
        (0, 0, 0, 1,  0, 0, 1, -1,  -1, 0, 1, 0),
        (1, 1, 0, 0,  0, 0, 0, 1,  1, 1, 0, 0),
        (0, 0, 1, -1,  1, 1, 0, 0,  0, 0, 1, -1),
        (0, 1, 0, -1,  0, 0, 1, 1,  0, 0, -1, 1),
    ]
    
    # Verify: sum over terms of u[i*2+j] * v[j*2+k] * w[k*2+i] == 1 for all i,j,k
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ui = i * 2 + j
                vi = j * 2 + k
                wi = k * 2 + i
                total = 0.0
                for t in terms:
                    total += t[ui] * t[vi] * t[wi]
                if abs(total - 1.0) > 1e-9:
                    return []
    
    return terms

def _apply_strassen_block(U, V, W, n, m, p, i0, j0, k0):
    """
    Replace the 8 schoolbook terms for the 2x2x2 block at (i0,j0,k0) with 7 Strassen terms.
    Returns (new_U, new_V, new_W, success)
    """
    terms = _get_strassen_terms()
    if not terms:
        return U, V, W, False
    
    # Find the 8 schoolbook column indices for this block
    block_cols = set()
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i, j, k = i0 + di, j0 + dj, k0 + dk
                col = i * (m * p) + j * p + k
                block_cols.add(col)
    
    old_rank = U.shape[1]
    new_rank = old_rank - 1
    
    new_U = np.zeros((n * m, new_rank))
    new_V = np.zeros((m * p, new_rank))
    new_W = np.zeros((p * n, new_rank))
    
    # Copy non-block columns
    col_new = 0
    for c in range(old_rank):
        if c not in block_cols:
            new_U[:, col_new] = U[:, c]
            new_V[:, col_new] = V[:, c]
            new_W[:, col_new] = W[:, c]
            col_new += 1
    
    # Add 7 Strassen columns
    for t in terms:
        u_loc = t[0:4]
        v_loc = t[4:8]
        w_loc = t[8:12]
        
        for li, val in enumerate(u_loc):
            if val != 0:
                di = li // 2
                dj = li % 2
                gi, gj = i0 + di, j0 + dj
                new_U[gi*m+gj, col_new] += val
        
        for li, val in enumerate(v_loc):
            if val != 0:
                dj = li // 2
                dk = li % 2
                gj, gk = j0 + dj, k0 + dk
                new_V[gj*p+gk, col_new] += val
        
        for li, val in enumerate(w_loc):
            if val != 0:
                dk = li // 2
                di = li % 2
                gk, gi = k0 + dk, i0 + di
                new_W[gk*n+gi, col_new] += val
        
        col_new += 1
    
    return new_U, new_V, new_W, True

def _find_best_packing(n, m, p, deadline):
    """
    Find the maximum number of non-overlapping 2x2x2 blocks.
    Uses backtracking with pruning for the small search space.
    """
    # All possible block positions
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                triples = frozenset((i0+di, j0+dj, k0+dk) for di in range(2) for dj in range(2) for dk in range(2))
                all_blocks.append((i0, j0, k0, triples))
    
    # Precompute conflict matrix
    num_blocks = len(all_blocks)
    conflicts = [[False] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i+1, num_blocks):
            if all_blocks[i][3] & all_blocks[j][3]:
                conflicts[i][j] = True
                conflicts[j][i] = True
    
    # Backtracking search for maximum independent set
    best_set = []
    
    def backtrack(start, current_set, used_mask):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        # Pruning: even if we add all remaining, can't beat best
        remaining = num_blocks - start
        if len(current_set) + remaining <= len(best_set):
            return
        
        for idx in range(start, num_blocks):
            if used_mask & (1 << idx):
                continue
            # Check conflict with current set
            conflict = False
            for cidx in current_set:
                if conflicts[idx][cidx]:
                    conflict = True
                    break
            if conflict:
                continue
            
            new_mask = used_mask | (1 << idx)
            current_set.append(idx)
            backtrack(idx + 1, current_set, new_mask)
            current_set.pop()
    
    backtrack(0, [], 0)
    return [all_blocks[i][:3] for i in best_set]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Find optimal packing of non-overlapping 2x2x2 blocks
    selected_blocks = _find_best_packing(n, m, p, deadline)
    
    # Apply Strassen to each selected block
    cur_U, cur_V, cur_W = best_U, best_V, best_W
    applied = 0
    for i0, j0, k0 in selected_blocks:
        if time.time() > deadline:
            break
        new_U, new_V, new_W, success = _apply_strassen_block(cur_U, cur_V, cur_W, n, m, p, i0, j0, k0)
        if success and _verify(new_U, new_V, new_W, n, m, p):
            cur_U, cur_V, cur_W = new_U, new_V, new_W
            applied += 1
    
    if applied > 0:
        best_U, best_V, best_W = cur_U, cur_V, cur_W
        best_rank = cur_U.shape[1]
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
