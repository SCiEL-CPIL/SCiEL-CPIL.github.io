import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 140
_decomp_cache = {}

def _make_schoolbook(m, n, p):
    rank = m * n * p
    U = np.zeros((m * n, rank))
    V = np.zeros((n * p, rank))
    W = np.zeros((p * m, rank))
    for i in range(m):
        for j in range(n):
            for k in range(p):
                idx = i * n * p + j * p + k
                U[i * n + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * m + i, idx] = 1
    return U, V, W

def _decompose(m, n, p):
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()
    
    key = (m, n, p)
    if key in _decomp_cache:
        return _decomp_cache[key]
        
    # Base cases
    if m == 0 or n == 0 or p == 0:
        return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0))
    if m == 1 and n == 1 and p == 1:
        return np.array([[1]]), np.array([[1]]), np.array([[1]])
        
    if m == 1:
        U, V, W = _make_schoolbook(1, n, p)
        _decomp_cache[key] = (U, V, W)
        return U, V, W
        
    if p == 1:
        U, V, W = _make_schoolbook(m, n, 1)
        _decomp_cache[key] = (U, V, W)
        return U, V, W
        
    if n == 1:
        U, V, W = _make_schoolbook(m, 1, p)
        _decomp_cache[key] = (U, V, W)
        return U, V, W
        
    if m == 2 and n == 2 and p == 2:
        # Strassen 2x2x2
        U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ])
        V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        W = np.array([
            [1, 0, 0, -1, 1, 0, -1],
            [0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, -1, 1, 0, 1, 1]
        ])
        _decomp_cache[key] = (U, V, W)
        return U, V, W

    # Try splits
    best_rank = m * n * p
    best_decomp = None
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                
                try:
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    rank = U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1] + \
                           U5.shape[1] + U6.shape[1] + U7.shape[1] + U8.shape[1]
                    
                    if rank < best_rank:
                        best_rank = rank
                        U_new = np.zeros((m * n, rank))
                        V_new = np.zeros((n * p, rank))
                        W_new = np.zeros((p * m, rank))
                        
                        offset = 0
                        def add_block(U_b, V_b, W_b, m_i, n_i, p_i, om, on, op):
                            nonlocal offset
                            r = U_b.shape[1]
                            if r == 0: return
                            
                            for i in range(m_i):
                                for j in range(n_i):
                                    U_new[(om+i)*n + (on+j), offset:offset+r] = U_b[i*n_i+j, :]
                            
                            for j in range(n_i):
                                for k in range(p_i):
                                    V_new[(on+j)*p + (op+k), offset:offset+r] = V_b[j*p_i+k, :]
                            
                            for k in range(p_i):
                                for i in range(m_i):
                                    W_new[(op+k)*m + (om+i), offset:offset+r] = W_b[k*m_i+i, :]
                            
                            offset += r
                            
                        add_block(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                        add_block(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                        add_block(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                        add_block(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                        add_block(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                        add_block(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                        add_block(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                        add_block(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        
                        best_decomp = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best_decomp is not None:
        _decomp_cache[key] = best_decomp
        return best_decomp
    else:
        U, V, W = _make_schoolbook(m, n, p)
        _decomp_cache[key] = (U, V, W)
        return U, V, W

def construct():
    global _start_time, _decomp_cache
    _start_time = time.time()
    _decomp_cache = {}
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
