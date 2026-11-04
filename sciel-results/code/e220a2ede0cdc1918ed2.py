import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor entries: T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.add((a, b, c))
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_solution = get_schoolbook()
    best_rank = n * m * p
    
    def compute_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def check_validity(U, V, W, tol=1e-9):
        T = compute_tensor(U, V, W)
        target = np.zeros((nm, mp, pn))
        for (a, b, c) in target_entries:
            target[a, b, c] = 1.0
        return np.allclose(T, target, atol=tol)
    
    def try_solution(U, V, W, rank):
        nonlocal best_solution, best_rank
        if check_validity(U, V, W):
            if rank < best_rank:
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = rank
                return True
        return False
    
    # Strategy: Try to find low-rank decompositions using structured initialization
    # Key insight: The tensor has a specific structure that might allow rank reduction
    
    # Try various rank values with different initialization strategies
    for R in range(35, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random initialization with small values
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            elif restart % 3 == 1:
                # Structured initialization based on Kronecker products
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                # Add random perturbations
                U += np.random.randn(nm, R) * 0.01
                V += np.random.randn(mp, R) * 0.01
                W += np.random.randn(pn, R) * 0.01
            else:
                # Sparse random initialization
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Randomly select a few non-zero entries
                    for _ in range(3):
                        a = np.random.randint(nm)
                        b = np.random.randint(mp)
                        c = np.random.randint(pn)
                        U[a, r] = np.random.choice([-1, 1])
                        V[b, r] = np.random.choice([-1, 1])
                        W[c, r] = np.random.choice([-1, 1])
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            for step in range(500):
                if time.time() > deadline:
                    break
                
                T = compute_tensor(U, V, W)
                target = np.zeros((nm, mp, pn))
                for (a, b, c) in target_entries:
                    target[a, b, c] = 1.0
                err = T - target
                loss = np.sum(err**2)
                
                if loss < 1e-10:
                    break
                
                grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
                grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
                grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                lr *= 0.995
            
            # Try to snap to half-integers
            for snap_factor in [1, 2, 4]:
                U_snap = np.round(U * snap_factor) / snap_factor
                V_snap = np.round(V * snap_factor) / snap_factor
                W_snap = np.round(W * snap_factor) / snap_factor
                
                if try_solution(U_snap, V_snap, W_snap, R):
                    found = True
                    break
        
        if found:
            continue
    
    return best_solution
