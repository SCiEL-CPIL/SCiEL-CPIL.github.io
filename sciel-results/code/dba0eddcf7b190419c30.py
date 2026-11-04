import numpy as np
import time

def construct():
    deadline = time.time() + 140
    n, m, p = 2, 4, 5
    terms = []

    def add_2x2x2(i_off, j_off, k_off):
        # Strassen decomposition for 2x2x2
        a00, a01 = i_off*m+j_off, i_off*m+j_off+1
        a10, a11 = (i_off+1)*m+j_off, (i_off+1)*m+j_off+1
        b00, b01 = j_off*p+k_off, j_off*p+k_off+1
        b10, b11 = (j_off+1)*p+k_off, (j_off+1)*p+k_off+1
        c00, c01 = k_off*n+i_off, (k_off+1)*n+i_off
        c10, c11 = k_off*n+i_off+1, (k_off+1)*n+i_off+1
        
        # M1 = (A00 + A11)(B00 + B11)
        u = np.zeros(8); u[a00] += 1; u[a11] += 1
        v = np.zeros(20); v[b00] += 1; v[b11] += 1
        w = np.zeros(10); w[c00] += 1; w[c11] += 1
        terms.append((u, v, w))

        # M2 = (A10 + A11)B10
        u = np.zeros(8); u[a10] += 1; u[a11] += 1
        v = np.zeros(20); v[b10] += 1
        w = np.zeros(10); w[c10] += 1; w[c11] += 1
        terms.append((u, v, w))

        # M3 = A00(B01 - B11)
        u = np.zeros(8); u[a00] += 1
        v = np.zeros(20); v[b01] += 1; v[b11] -= 1
        w = np.zeros(10); w[c01] += 1
        terms.append((u, v, w))

        # M4 = A11(B10 - B00)
        u = np.zeros(8); u[a11] += 1
        v = np.zeros(20); v[b10] += 1; v[b00] -= 1
        w = np.zeros(10); w[c10] += 1
        terms.append((u, v, w))

        # M5 = (A00 + A01)B00
        u = np.zeros(8); u[a00] += 1; u[a01] += 1
        v = np.zeros(20); v[b00] += 1
        w = np.zeros(10); w[c00] += 1; w[c01] += 1
        terms.append((u, v, w))

        # M6 = (A10 - A00)(B00 + B01)
        u = np.zeros(8); u[a10] += 1; u[a00] -= 1
        v = np.zeros(20); v[b00] += 1; v[b01] += 1
        w = np.zeros(10); w[c10] += 1
        terms.append((u, v, w))

        # M7 = (A01 - A11)(B10 + B11)
        u = np.zeros(8); u[a01] += 1; u[a11] -= 1
        v = np.zeros(20); v[b10] += 1; v[b11] += 1
        w = np.zeros(10); w[c01] += 1
        terms.append((u, v, w))

    def add_2x2x1(i_off, j_off, k_idx):
        # Rank 4 decomposition for 2x2x1
        a00, a01 = i_off*m+j_off, i_off*m+j_off+1
        a10, a11 = (i_off+1)*m+j_off, (i_off+1)*m+j_off+1
        b00, b01 = j_off*p+k_idx, (j_off+1)*p+k_idx
        c0, c1 = k_idx*n+i_off, k_idx*n+i_off+1

        # T1: A00 B00 -> C0
        u = np.zeros(8); u[a00] += 1
        v = np.zeros(20); v[b00] += 1
        w = np.zeros(10); w[c0] += 1
        terms.append((u, v, w))

        # T2: A01 B01 -> C0
        u = np.zeros(8); u[a01] += 1
        v = np.zeros(20); v[b01] += 1
        w = np.zeros(10); w[c0] += 1
        terms.append((u, v, w))

        # T3: A10 B00 -> C1
        u = np.zeros(8); u[a10] += 1
        v = np.zeros(20); v[b00] += 1
        w = np.zeros(10); w[c1] += 1
        terms.append((u, v, w))

        # T4: A11 B01 -> C1
        u = np.zeros(8); u[a11] += 1
        v = np.zeros(20); v[b01] += 1
        w = np.zeros(10); w[c1] += 1
        terms.append((u, v, w))

    # Split B into 2+2+1
    # C[:, 0:2] = A * B[:, 0:2]
    add_2x2x2(0, 0, 0)
    add_2x2x2(0, 2, 0)
    
    # C[:, 2:4] = A * B[:, 2:4]
    add_2x2x2(0, 0, 2)
    add_2x2x2(0, 2, 2)

    # C[:, 4] = A * B[:, 4]
    add_2x2x1(0, 0, 4)
    add_2x2x1(0, 2, 4)

    R = len(terms)
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
