"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True
    
    def verify_3x3(U, V, W):
        """Verify 3x3x3 block decomposition with local indices."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * 3 + j, c] * V[j * 3 + k, c] * W[k * 3 + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    def add_schoolbook_term(i, j, k):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        u[i * m + j] = 1.0
        v[j * p + k] = 1.0
        w[k * n + i] = 1.0
        add_term(u, v, w)
    
    def add_laderman_block(I, J, K):
        i0, i1, i2 = sorted(I)
        j0, j1, j2 = sorted(J)
        k0, k1, k2 = sorted(K)
        
        def make_u(coeffs):
            u = np.zeros(n * m)
            for (li, lj), val in coeffs.items():
                u[I[li] * m + J[lj]] = val
            return u
        
        def make_v(coeffs):
            v = np.zeros(m * p)
            for (lj, lk), val in coeffs.items():
                v[J[lj] * p + K[lk]] = val
            return v
        
        def make_w(coeffs):
            w = np.zeros(p * n)
            for (lk, li), val in coeffs.items():
                w[K[lk] * n + I[li]] = val
            return w
        
        # Laderman 3x3 decomposition - 23 terms
        # Using the standard formulation
        
        terms = []
        
        # T1: (A00+A11+A22)(B00+B11+B22)
        terms.append((
            make_u({(0,0):1, (1,1):1, (2,2):1}),
            make_v({(0,0):1, (1,1):1, (2,2):1}),
            make_w({(0,0):1, (1,1):1, (2,2):1})
        ))
        
        # T2: (A00-A11)(B00-B11)
        terms.append((
            make_u({(0,0):1, (1,1):-1}),
            make_v({(0,0):1, (1,1):-1}),
            make_w({(0,0):1, (1,1):1})
        ))
        
        # T3: (A00-A22)(B00-B22)
        terms.append((
            make_u({(0,0):1, (2,2):-1}),
            make_v({(0,0):1, (2,2):-1}),
            make_w({(0,0):1, (2,2):1})
        ))
        
        # T4: (A11-A22)(B11-B22)
        terms.append((
            make_u({(1,1):1, (2,2):-1}),
            make_v({(1,1):1, (2,2):-1}),
            make_w({(1,1):1, (2,2):1})
        ))
        
        # T5: (A01+A10+A20+A12+A21+A02)(B10+B01+B20+B12+B21+B02)
        terms.append((
            make_u({(0,1):1, (1,0):1, (2,0):1, (1,2):1, (2,1):1, (0,2):1}),
            make_v({(1,0):1, (0,1):1, (2,0):1, (1,2):1, (2,1):1, (0,2):1}),
            make_w({(0,1):1, (1,0):1, (0,2):1, (2,1):1, (1,2):1, (2,0):1})
        ))
        
        # T6: (A01-A10+A21-A12)(B10-B01+B21-B12)
        terms.append((
            make_u({(0,1):1, (1,0):-1, (2,1):1, (1,2):-1}),
            make_v({(1,0):1, (0,1):-1, (2,1):1, (1,2):-1}),
            make_w({(0,1):1, (1,0):1, (2,1):1, (1,2):1})
        ))
        
        # T7: (A01+A10-A21-A12)(B10+B01-B21-B12)
        terms.append((
            make_u({(0,1):1, (1,0):1, (2,1):-1, (1,2):-1}),
            make_v({(1,0):1, (0,1):1, (2,1):-1, (1,2):-1}),
            make_w({(0,1):1, (1,0):1, (2,1):1, (1,2):1})
        ))
        
        # T8: (A02+A20-A12-A21)(B20+B02-A12-A21)
        terms.append((
            make_u({(0,2):1, (2,0):1, (1,2):-1, (2,1):-1}),
            make_v({(2,0):1, (0,2):1, (1,2):-1, (2,1):-1}),
            make_w({(0,2):1, (2,0):1, (1,2):1, (2,1):1})
        ))
        
        # T9: (A02-A20+A12-A21)(B20-B02+A12-A21)
        terms.append((
            make_u({(0,2):1, (2,0):-1, (1,2):1, (2,1):-1}),
            make_v({(2,0):1, (0,2):-1, (1,2):1, (2,1):-1}),
            make_w({(0,2):1, (2,0):1, (1,2):1, (2,1):1})
        ))
        
        # T10: (A02+A20+A12+A21)(B20+B02+A12+A21)
        terms.append((
            make_u({(0,2):1, (2,0):1, (1,2):1, (2,1):1}),
            make_v({(2,0):1, (0,2):1, (1,2):1, (2,1):1}),
            make_w({(0,2):1, (2,0):1, (1,2):1, (2,1):1})
        ))
        
        # T11: (A10+A01+A22)(B10+B01+B22)
        terms.append((
            make_u({(1,0):1, (0,1):1, (2,2):1}),
            make_v({(1,0):1, (0,1):1, (2,2):1}),
            make_w({(0,1):1, (1,0):1, (2,2):1})
        ))
        
        # T12: (A10+A01-A22)(B10+B01-B22)
        terms.append((
            make_u({(1,0):1, (0,1):1, (2,2):-1}),
            make_v({(1,0):1, (0,1):1, (2,2):-1}),
            make_w({(0,1):1, (1,0):1, (2,2):1})
        ))
        
        # T13: (A20+A02+A11)(B20+B02+B11)
        terms.append((
            make_u({(2,0):1, (0,2):1, (1,1):1}),
            make_v({(2,0):1, (0,2):1, (1,1):1}),
            make_w({(0,2):1, (2,0):1, (1,1):1})
        ))
        
        # T14: (A20+A02-A11)(B20+B02-B11)
        terms.append((
            make_u({(2,0):1, (0,2):1, (1,1):-1}),
            make_v({(2,0):1, (0,2):1, (1,1):-1}),
            make_w({(0,2):1, (2,0):1, (1,1):1})
        ))
        
        # T15: (A12+A21+A00)(B12+B21+B00)
        terms.append((
            make_u({(1,2):1, (2,1):1, (0,0):1}),
            make_v({(1,2):1, (2,1):1, (0,0):1}),
            make_w({(2,1):1, (1,2):1, (0,0):1})
        ))
        
        # T16: (A12+A21-A00)(B12+B21-B00)
        terms.append((
            make_u({(1,2):1, (2,1):1, (0,0):-1}),
            make_v({(1,2):1, (2,1):1, (0,0):-1}),
            make_w({(2,1):1, (1,2):1, (0,0):1})
        ))
        
        # T17: (A10+A21)(B01+B12)
        terms.append((
            make_u({(1,0):1, (2,1):1}),
            make_v({(0,1):1, (1,2):1}),
            make_w({(1,1):1})
        ))
        
        # T18: (A20+A12)(B02+B21)
        terms.append((
            make_u({(2,0):1, (1,2):1}),
            make_v({(0,2):1, (2,1):1}),
            make_w({(2,2):1})
        ))
        
        # T19: (A01+A12)(B10+B20)
        terms.append((
            make_u({(0,1):1, (1,2):1}),
            make_v({(1,0):1, (2,0):1}),
            make_w({(0,0):1})
        ))
        
        # T20: (A02+A21)(B10+B01)
        terms.append((
            make_u({(0,2):1, (2,1):1}),
            make_v({(1,0):1, (0,1):1}),
            make_w({(0,1):1})
        ))
        
        # T21: (A10+A02)(B12+B20)
        terms.append((
            make_u({(1,0):1, (0,2):1}),
            make_v({(1,2):1, (2,0):1}),
            make_w({(2,1):1})
        ))
        
        # T22: (A20+A01)(B21+B12)
        terms.append((
            make_u({(2,0):1, (0,1):1}),
            make_v({(2,1):1, (1,2):1}),
            make_w({(1,2):1})
        ))
        
        # T23: (A11+A22+A00-A10-A20-A01-A02-A12-A21)(B11+B22+B00-B10-B20-B01-B02-B12-B21)
        terms.append((
            make_u({(1,1):1, (2,2):1, (0,0):1, (1,0):-1, (2,0):-1, (0,1):-1, (0,2):-1, (1,2):-1, (2,1):-1}),
            make_v({(1,1):1, (2,2):1, (0,0):1, (1,0):-1, (2,0):-1, (0,1):-1, (0,2):-1, (1,2):-1, (2,1):-1}),
            make_w({(1,1):1, (2,2):1, (0,0):1, (1,0):-1, (2,0):-1, (0,1):-1, (0,2):-1, (1,2):-1, (2,1):-1})
        ))
        
        for u, v, w in terms:
            add_term(u, v, w)
    
    # Define Laderman blocks
    laderman_blocks = [
        ({0, 1, 2}, {0, 1, 2}, {0, 1, 2}),
        ({0, 1, 2}, {0, 1, 2}, {3, 4, 5}),
    ]
    
    for I, J, K in laderman_blocks:
        add_laderman_block(I, J, K)
    
    covered = set()
    for I, J, K in laderman_blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook_term(i, j, k)
    
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END

