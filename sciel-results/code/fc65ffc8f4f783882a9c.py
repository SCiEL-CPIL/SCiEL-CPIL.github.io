"""Matrix multiplication 4x4x5 rank minimization via term merging search."""
import random
import time

DIMENSIONS = [4, 4, 5]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def build_initial():
        cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    cols.append((u, v, w))
        return cols
    
    def verify(cols):
        """Verify tensor equality. Returns True if exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for (u, v, w) in cols:
                        a = u.get((i, j), 0)
                        if a == 0:
                            continue
                        b = v.get((j, k), 0)
                        if b == 0:
                            continue
                        c = w.get((k, i), 0)
                        if c == 0:
                            continue
                        total += a * b * c
                    if total != 1:
                        return False
        return True
    
    def try_merge(cols, idx1, idx2):
        u1, v1, w1 = cols[idx1]
        u2, v2, w2 = cols[idx2]
        
        u_new = {}
        for key in set(list(u1.keys()) + list(u2.keys())):
            val = u1.get(key, 0) + u2.get(key, 0)
            if val != 0:
                u_new[key] = val
        
        v_new = {}
        for key in set(list(v1.keys()) + list(v2.keys())):
            val = v1.get(key, 0) + v2.get(key, 0)
            if val != 0:
                v_new[key] = val
        
        w_new = {}
        for key in set(list(w1.keys()) + list(w2.keys())):
            val = w1.get(key, 0) + w2.get(key, 0)
            if val != 0:
                w_new[key] = val
        
        new_cols = [col for idx, col in enumerate(cols) if idx != idx1 and idx != idx2]
        new_cols.append((u_new, v_new, w_new))
        
        if verify(new_cols):
            return new_cols
        return None
    
    best_cols = build_initial()
    best_rank = len(best_cols)
    
    random.seed(98765)
    
    # Systematic: try all pairs from initial decomposition
    init_cols = build_initial()
    for it in range(300000):
        if time.time() - start_time > DEADLINE:
            break
        if best_rank <= 20:
            break
        
        # Mix: sometimes try from current best, sometimes from fresh
        if random.random() < 0.5 and len(best_cols) < 80:
            cur = best_cols
        else:
            cur = init_cols
        
        if len(cur) <= best_rank:
            continue
        
        idx1 = random.randint(0, len(cur) - 1)
        idx2 = random.randint(0, len(cur) - 1)
        if idx1 == idx2:
            continue
        
        new_cols = try_merge(cur, idx1, idx2)
        if new_cols is not None:
            if len(new_cols) < best_rank:
                best_rank = len(new_cols)
                best_cols = new_cols
    
    rank = len(best_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(best_cols):
        for (i, j), val in u.items():
            U[i * m + j][c] = val
        for (j, k), val in v.items():
            V[j * p + k][c] = val
        for (k, i), val in w.items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
