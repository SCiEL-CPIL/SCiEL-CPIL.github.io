"""Matrix multiplication 2x3 x 3x6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _build_target_tensor(n, m, p):
    T = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T.add((a, b, c))
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    for (a, b, c) in T:
        s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
        if abs(s - 1.0) > tol:
            return False
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T:
                    s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                    if abs(s) > tol:
                        return False
    return True

def _verify_exact(U, V, W, n, m, p):
    r = len(U[0])
    Uf = [[Fraction(U[i][k]) for k in range(r)] for i in range(len(U))]
    Vf = [[Fraction(V[i][k]) for k in range(r)] for i in range(len(V))]
    Wf = [[Fraction(W[i][k]) for k in range(r)] for i in range(len(W))]
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    for (a, b, c) in T:
        s = Fraction(0)
        for k in range(r):
            s += Uf[a][k] * Vf[b][k] * Wf[c][k]
        if s != 1:
            return False
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T:
                    s = Fraction(0)
                    for k in range(r):
                        s += Uf[a][k] * Vf[b][k] * Wf[c][k]
                    if s != 0:
                        return False
    return True

def _schoolbook(n, m, p):
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

def _try_binary_patterns(rank, n, m, p, deadline=None):
    """Try binary patterns with systematic index mapping."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    # Try various index mappings
    mappings = []
    
    # Mapping 1: (i,j,k) -> idx with different orderings
    for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        mappings.append(perm)
    
    for perm in mappings:
        if deadline and time.time() > deadline:
            return None
        
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for idx in range(rank):
            # Decode idx based on permutation
            i, j, k = 0, 0, 0
            if perm == (0,1,2):
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n*m)) % p
            elif perm == (0,2,1):
                i = idx % n
                k = (idx // n) % p
                j = (idx // (n*p)) % m
            elif perm == (1,0,2):
                j = idx % m
                i = (idx // m) % n
                k = (idx // (m*n)) % p
            elif perm == (1,2,0):
                j = idx % m
                k = (idx // m) % p
                i = (idx // (m*p)) % n
            elif perm == (2,0,1):
                k = idx % p
                i = (idx // p) % n
                j = (idx // (p*n)) % m
            else:
                k = idx % p
                j = (idx // p) % m
                i = (idx // (p*m)) % n
            
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a][idx] = 1
            V[b][idx] = 1
            W[c][idx] = 1
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_local_search(rank, n, m, p, deadline=None, n_restarts=100, iters_per_restart=100):
    """Local search with random restarts."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    T_list = list(T)
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            return None
        
        # Initialize
        U = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(nm)]
        V = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(mp)]
        W = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(rank)] for _ in range(pn)]
        
        best_obj = float('inf')
        best_state = None
        
        for it in range(iters_per_restart):
            if deadline and time.time() > deadline:
                return None
            
            # Compute objective
            obj = 0
            for (a, b, c) in T_list:
                s = 0
                for k in range(rank):
                    s += U[a][k] * V[b][k] * W[c][k]
                obj += (s - 1) ** 2
            
            # Sample non-target
            for _ in range(10):
                a = random.randint(0, nm-1)
                b = random.randint(0, mp-1)
                c = random.randint(0, pn-1)
                if (a, b, c) not in T:
                    s = 0
                    for k in range(rank):
                        s += U[a][k] * V[b][k] * W[c][k]
                    obj += s ** 2
            
            if obj < best_obj:
                best_obj = obj
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                if best_obj < 1e-10:
                    U_snap = [[round(v*2)/2 for v in row] for row in U]
                    V_snap = [[round(v*2)/2 for v in row] for row in V]
                    W_snap = [[round(v*2)/2 for v in row] for row in W]
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
            
            # Perturb
            mat = random.randint(0, 2)
            if mat == 0:
                row = random.randint(0, nm-1)
                arr = U
            elif mat == 1:
                row = random.randint(0, mp-1)
                arr = V
            else:
                row = random.randint(0, pn-1)
                arr = W
            col = random.randint(0, rank-1)
            arr[row][col] += random.choice([-1, -0.5, 0.5, 1])
        
        # Try snapping best state
        if best_state and best_obj < 0.1:
            U_snap = [[round(v*2)/2 for v in row] for row in best_state[0]]
            V_snap = [[round(v*2)/2 for v in row] for row in best_state[1]]
            W_snap = [[round(v*2)/2 for v in row] for row in best_state[2]]
            if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                return {"U": U_snap, "V": V_snap, "W": W_snap}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try binary patterns
    for r in range(18, min(30, best_rank)):
        if time.time() > deadline:
            break
        sol = _try_binary_patterns(r, n, m, p, deadline)
        if sol is not None:
            best = sol
            best_rank = r
            break
    
    # Try local search
    if best_rank == n * m * p:
        for r in range(18, min(25, best_rank)):
            if time.time() > deadline:
                break
            sol = _try_local_search(r, n, m, p, deadline, n_restarts=50, iters_per_restart=50)
            if sol is not None:
                best = sol
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
