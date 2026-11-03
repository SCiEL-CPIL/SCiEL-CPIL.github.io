import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build the sparse tensor
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Try rank 24
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(trial)
        U = np.random.randn(nm, 24)
        V = np.random.randn(mp, 24)
        W = np.random.randn(pn, 24)
        
        lr = 0.01
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            # Vectorized gradient computation
            grad_U = np.zeros((nm, 24))
            grad_V = np.zeros((mp, 24))
            grad_W = np.zeros((pn, 24))
            
            for (a, b, c) in tensor_entries:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = val - 1
                grad_U[a, :] += err * V[b, :] * W[c, :]
                grad_V[b, :] += err * U[a, :] * W[c, :]
                grad_W[c, :] += err * U[a, :] * V[b, :]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            error = 0
            for (a, b, c) in tensor_entries:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                error += (val - 1) ** 2
            
            if error < 1e-10:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                valid = True
                for (a, b, c) in tensor_entries:
                    val = np.dot(U_rounded[a, :], V_rounded[b, :] * W_rounded[c, :])
                    if abs(val - 1) > 1e-10:
                        valid = False
                        break
                if valid:
                    np.random.seed(42)
                    for _ in range(100):
                        a = np.random.randint(nm)
                        b = np.random.randint(mp)
                        c = np.random.randint(pn)
                        if not any(a2 == a and b2 == b and c2 == c for (a2, b2, c2) in tensor_entries):
                            val = np.dot(U_rounded[a, :], V_rounded[b, :] * W_rounded[c, :])
                            if abs(val) > 1e-10:
                                valid = False
                                break
                
                if valid:
                    return {"U": [[float(x) for x in row] for row in U_rounded],
                            "V": [[float(x) for x in row] for row in V_rounded],
                            "W": [[float(x) for x in row] for row in W_rounded]}
        
        continue
    
    # Fallback: schoolbook
    rank = 72
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
