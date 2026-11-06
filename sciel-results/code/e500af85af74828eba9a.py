"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    rng = np.random.default_rng(42)
    
    all_4subs = list(combinations(range(n), 4))
    
    def q3_count(adj):
        count = 0
        for A in all_4subs:
            A_set = set(A)
            remaining = [v for v in range(n) if v not in A_set]
            A_arr = np.array(A, dtype=np.int64)
            for B in combinations(remaining, 4):
                B_arr = np.array(B, dtype=np.int64)
                bip = adj[A_arr[:, None], B_arr]
                if bip.sum(1).min() < 3 or bip.sum(0).min() < 3:
                    continue
                has_4 = False
                for i1 in range(4):
                    row1 = bip[i1]
                    for i2 in range(i1+1, 4):
                        if (row1 & bip[i2]).sum() >= 2:
                            has_4 = True
                            break
                    if has_4:
                        break
                if not has_4:
                    count += 1
        return count
    
    # Start from K_{3,13}
    adj = np.zeros((n, n), dtype=np.uint8)
    for a in range(3):
        for b in range(3, n):
            adj[a, b] = 1
            adj[b, a] = 1
    
    best_adj = adj.copy()
    best_edges = 39
    
    current_edges = 39
    current_q3 = 0
    current_obj = current_edges - 50.0 * current_q3
    
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        if rng.random() < 0.6:
            i, j = all_pairs[rng.integers(len(all_pairs))]
            if adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                ne = current_edges + 1
                nq = q3_count(adj)
                no = ne - 50.0 * nq
                d = no - current_obj
                if d >= 0 or rng.random() < np.exp(d / T):
                    current_edges = ne
                    current_q3 = nq
                    current_obj = no
                    if nq == 0 and ne > best_edges:
                        best_edges = ne
                        best_adj = adj.copy()
                else:
                    adj[i, j] = 0
                    adj[j, i] = 0
        else:
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
            if existing:
                i, j = existing[rng.integers(len(existing))]
                adj[i, j] = 0
                adj[j, i] = 0
                ne = current_edges - 1
                nq = q3_count(adj)
                no = ne - 50.0 * nq
                d = no - current_obj
                if d >= 0 or rng.random() < np.exp(d / T):
                    current_edges = ne
                    current_q3 = nq
                    current_obj = no
                    if nq == 0 and ne > best_edges:
                        best_edges = ne
                        best_adj = adj.copy()
                else:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        T = max(T_min, T * cooling)
    
    # Repair
    for _ in range(50):
        if time.time() > deadline:
            break
        found = False
        for A in all_4subs:
            A_set = set(A)
            remaining = [v for v in range(n) if v not in A_set]
            A_arr = np.array(A, dtype=np.int64)
            for B in combinations(remaining, 4):
                B_arr = np.array(B, dtype=np.int64)
                bip = best_adj[A_arr[:, None], B_arr]
                if bip.sum(1).min() < 3 or bip.sum(0).min() < 3:
                    continue
                has_4 = False
                for i1 in range(4):
                    row1 = bip[i1]
                    for i2 in range(i1+1, 4):
                        if (row1 & bip[i2]).sum() >= 2:
                            has_4 = True
                            break
                    if has_4:
                        break
                if not has_4:
                    u = A_arr[0]
                    v = B_arr[0]
                    best_adj[u, v] = 0
                    best_adj[v, u] = 0
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    return [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i, j]]
# EVOLVE-BLOCK-END
