import numpy as np
import time
import sys
import math

N_SITES = 6
DIM = 2 ** N_SITES
DELTA = 0.8
FIELD = 0.3
TIMES = [0.25, 0.5, 1.0]

def build_hamiltonians():
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops_list):
        res = ops_list[0]
        for op in ops_list[1:]:
            res = np.kron(res, op)
        return res

    def get_term(i, op_i, j, op_j):
        ops = [I] * N_SITES
        ops[i] = op_i
        ops[j] = op_j
        return kron_all(ops)

    A = np.zeros((DIM, DIM), dtype=complex)
    B = np.zeros((DIM, DIM), dtype=complex)
    
    # Odd bonds: 0-1, 2-3, 4-5
    for bond in [(0, 1), (2, 3), (4, 5)]:
        i, j = bond
        A += get_term(i, sx, j, sx)
        A += get_term(i, sy, j, sy)
        A += DELTA * get_term(i, sz, j, sz)
        
    # Even bonds: 1-2, 3-4
    for bond in [(1, 2), (3, 4)]:
        i, j = bond
        B += get_term(i, sx, j, sx)
        B += get_term(i, sy, j, sy)
        B += DELTA * get_term(i, sz, j, sz)
        
    # Field
    A_field = np.zeros((DIM, DIM), dtype=complex)
    for i in range(N_SITES):
        ops = [I] * N_SITES
        ops[i] = sz
        A_field += kron_all(ops)
        
    A += 0.5 * FIELD * A_field
    B += 0.5 * FIELD * A_field
    
    return A, B

A, B = build_hamiltonians()
H = A + B

# Eigendecompositions
eigvals_A, eigvecs_A = np.linalg.eigh(A)
eigvals_B, eigvecs_B = np.linalg.eigh(B)
eigvals_H, eigvecs_H = np.linalg.eigh(H)

# Change of basis matrices
# V_B^dagger V_A
M_BA = eigvecs_B.conj().T @ eigvecs_A
# V_B^dagger V_H
M_BH = eigvecs_B.conj().T @ eigvecs_H

# Precompute exact propagators in B basis
M_BH = eigvecs_B.conj().T @ eigvecs_H
K_EXACTS = {}
for t in TIMES:
    dH = np.exp(-1j * t * eigvals_H)
    K_exact = M_BH @ np.diag(dH) @ M_BH.conj().T
    K_EXACTS[t] = K_exact

M_BA_dag = M_BA.conj().T

def compute_error(a, b):
    # a, b are lists of 12 floats
    total_error = 0.0
    for t in TIMES:
        K = np.eye(DIM, dtype=complex)
        
        for k in range(12):
            # Multiply by exp(-i b_k t B)
            dB = np.exp(-1j * b[k] * t * eigvals_B)
            K = K * dB
            
            # Multiply by exp(-i a_k t A)
            dA = np.exp(-1j * a[k] * t * eigvals_A)
            K = K @ M_BA
            K = K * dA
            K = K @ M_BA_dag
            
        K_exact = K_EXACTS[t]
        diff = K - K_exact
        # Spectral norm as required by task
        err = np.linalg.norm(diff, 2)
        total_error += err
        
    return total_error / len(TIMES)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Uniform
    init_a = [1.0/12] * 12
    init_b = [1.0/12] * 12
    
    best_a = init_a
    best_b = init_b
    best_error = compute_error(init_a, init_b)
    
    # Simple Gradient Descent with finite differences
    # Or Nelder-Mead on 24 params
    # Let's use Nelder-Mead for robustness
    
    params = init_a + init_b
    n = 24
    
    def eval_params(p):
        a = list(p[:12])
        b = list(p[12:24])
        return compute_error(a, b)
    
    simplex = [np.array(params)]
    for i in range(n):
        p = np.array(params)
        p[i] += 0.01
        simplex.append(p)
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 50
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        values = [eval_params(p) for p in simplex]
        indices = np.argsort(values)
        simplex = [simplex[i] for i in indices]
        values = [values[i] for i in indices]
        
        if values[0] < best_error:
            best_error = values[0]
            best_params = simplex[0]
            best_a = list(best_params[:12])
            best_b = list(best_params[12:24])
            
        if abs(values[0] - values[-1]) < 1e-8:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = eval_params(x_r)
        
        if f_r < values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = eval_params(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
            else:
                simplex[-1] = x_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = eval_params(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
    
    # Ensure sums are close to 1
    # The optimizer might drift sums slightly.
    # The problem requires sum(a)=1 and sum(b)=1.
    # We should project back to the constraint manifold if needed, 
    # or just return the best found.
    # Given the small search, projection is safer.
    
    sum_a = sum(best_a)
    sum_b = sum(best_b)
    
    if abs(sum_a - 1.0) > 1e-6:
        best_a = [x / sum_a for x in best_a]
    if abs(sum_b - 1.0) > 1e-6:
        best_b = [x / sum_b for x in best_b]
        
    # Clip to [-50, 50] just in case
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    return best_a, best_b
