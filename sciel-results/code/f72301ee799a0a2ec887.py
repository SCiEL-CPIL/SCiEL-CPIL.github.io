import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def verify_full(U, V, W):
        """Full verification: check all entries of the tensor."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    i = u // m
                    u_j = u % m
                    v_j = v // p
                    k_v = v % p
                    k_w = w // n
                    i_w = w % n
                    if i == i_w and u_j == v_j and k_v == k_w:
                        continue
                    val = 0.0
                    for r in range(R):
                        val += U[u, r] * V[v, r] * W[w, r]
                    if abs(val) > 1e-9:
                        return False
        return True

    def verify_partial(U, V, W):
        """Fast partial verification: check valid entries + random off-diagonal sample."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        for _ in range(500):
            u = random.randint(0, nm - 1)
            v = random.randint(0, mp - 1)
            w = random.randint(0, pn - 1)
            i = u // m
            u_j = u % m
            v_j = v // p
            k_v = v % p
            k_w = w // n
            i_w = w % n
            if i == i_w and u_j == v_j and k_v == k_w:
                continue
            val = 0.0
            for r in range(R):
                val += U[u, r] * V[v, r] * W[w, r]
            if abs(val) > 1e-9:
                return False
        return True

    def try_random_rank(R, time_budget=30.0):
        """Try to find a rank-R decomposition using random U,V and solving for W."""
        start_time = time.time()
        max_attempts = 20000
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Generate random U and V with small integer entries
            # Use a mix of sparse and denser patterns
            density = random.uniform(0.3, 0.7)
            
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(R):
                # Random pattern for U
                for idx in range(nm):
                    if random.random() < density:
                        U[idx, r] = random.choice([-2.0, -1.0, 1.0, 2.0])
                # Random pattern for V
                for idx in range(mp):
                    if random.random() < density:
                        V[idx, r] = random.choice([-2.0, -1.0, 1.0, 2.0])
            
            # Solve for W
            num_eq = n * m * p
            num_unk = pn * R
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(R):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                w_row = k * n + i
                                A[eq_idx, r * pn + w_row] = coeff
            
            try:
                w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W = w_flat.reshape(pn, R)
            
            # Try rounding to half-integers or integers
            for mult in [1.0, 2.0, 4.0]:
                W_rounded = np.round(W * mult) / mult
                
                # Quick check: verify valid entries
                valid_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = 0.0
                            for r in range(R):
                                val += U[idx_u, r] * V[idx_v, r] * W_rounded[idx_w, r]
                            if abs(val - 1.0) > 1e-9:
                                valid_ok = False
                                break
                        if not valid_ok:
                            break
                    if not valid_ok:
                        break
                
                if not valid_ok:
                    continue
                
                # Full verification
                if verify_full(U, V, W_rounded):
                    return U.tolist(), V.tolist(), W_rounded.tolist()
            
            # Also try without rounding
            if verify_full(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    def try_structured_rank16(time_budget=30.0):
        """Try structured patterns specifically for rank-16."""
        start_time = time.time()
        max_attempts = 5000
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            R = 16
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # Use a structured approach: each rank-1 term covers specific (i,j,k) patterns
            # Try to create terms that naturally satisfy the constraints
            for r in range(R):
                # Choose a random "type" of term
                term_type = random.randint(0, 3)
                
                if term_type == 0:
                    # Single (i,j,k) term with sign variations
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    s3 = random.choice([-1.0, 1.0])
                    U[i*m+j, r] = s1
                    V[j*p+k, r] = s2
                    # W will be solved
                elif term_type == 1:
                    # Two U entries, one V
                    i1 = random.randint(0, n-1)
                    i2 = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    s3 = random.choice([-1.0, 1.0])
                    U[i1*m+j, r] = s1
                    U[i2*m+j, r] = s2
                    V[j*p+k, r] = s3
                elif term_type == 2:
                    # One U, two V
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k1 = random.randint(0, p-1)
                    k2 = random.randint(0, p-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    s3 = random.choice([-1.0, 1.0])
                    U[i*m+j, r] = s1
                    V[j*p+k1, r] = s2
                    V[j*p+k2, r] = s3
                else:
                    # Two U, two V
                    i1 = random.randint(0, n-1)
                    i2 = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k1 = random.randint(0, p-1)
                    k2 = random.randint(0, p-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    s3 = random.choice([-1.0, 1.0])
                    s4 = random.choice([-1.0, 1.0])
                    U[i1*m+j, r] = s1
                    U[i2*m+j, r] = s2
                    V[j*p+k1, r] = s3
                    V[j*p+k2, r] = s4
            
            # Solve for W
            num_eq = n * m * p
            num_unk = pn * R
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(R):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                w_row = k * n + i
                                A[eq_idx, r * pn + w_row] = coeff
            
            try:
                w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W = w_flat.reshape(pn, R)
            
            for mult in [1.0, 2.0]:
                W_rounded = np.round(W * mult) / mult
                if verify_full(U, V, W_rounded):
                    return U.tolist(), V.tolist(), W_rounded.tolist()
            
            if verify_full(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try rank-16 first (most likely to succeed based on inspirations)
    remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
    if remaining > 10.0:
        time_budget = min(40.0, remaining - 5.0)
        result = try_structured_rank16(time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try random rank-16
    remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
    if remaining > 10.0:
        time_budget = min(35.0, remaining - 5.0)
        result = try_random_rank(16, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try rank-17
    remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
    if remaining > 10.0:
        time_budget = min(25.0, remaining - 5.0)
        result = try_random_rank(17, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try rank-18 (schoolbook, guaranteed)
    remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
    if remaining > 5.0:
        time_budget = min(15.0, remaining - 2.0)
        result = try_random_rank(18, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
