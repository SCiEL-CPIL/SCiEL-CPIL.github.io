import random
import time
import numpy as np

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i

def verify(U, V, W, r):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[Aidx(i,j)], Va[Bidx(j,k)] * Wa[Cidx(i,k)])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return U, V, W

def is_zero_term(U, V, W, r, l):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l] != 0:
                    return False
    return True

def try_merge(U, V, W, r, l1, l2):
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = (U[Aidx(i,j)][l1] * V[Bidx(j,k)][l1] * W[Cidx(i,k)][l1] +
                           U[Aidx(i,j)][l2] * V[Bidx(j,k)][l2] * W[Cidx(i,k)][l2])
    
    M1 = T.reshape(n*m, p)
    for a in range(n*m):
        for b in range(a+1, n*m):
            for c in range(p):
                for d in range(c+1, p):
                    if abs(M1[a,c]*M1[b,d] - M1[a,d]*M1[b,c]) > 1e-9:
                        return None
    
    M2 = T.reshape(n, m*p)
    for a in range(n):
        for b in range(a+1, n):
            for c in range(m*p):
                for d in range(c+1, m*p):
                    if abs(M2[a,c]*M2[b,d] - M2[a,d]*M2[b,c]) > 1e-9:
                        return None
    
    found = False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k]) > 1e-12:
                    found = True
                    break
            if found: break
        if found: break
    
    if not found:
        return (np.zeros(n*m), np.zeros(m*p), np.zeros(p*n))
    
    u = np.ones(n*m)
    v = np.ones(m*p)
    w = np.ones(p*n)
    
    for _ in range(20):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    denom = v[Bidx(j,k)] * w[Cidx(i,k)]
                    if abs(denom) > 1e-12:
                        u[Aidx(i,j)] = T[i,j,k] / denom
                        break
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    denom = u[Aidx(i,j)] * w[Cidx(i,k)]
                    if abs(denom) > 1e-12:
                        v[Bidx(j,k)] = T[i,j,k] / denom
                        break
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    denom = u[Aidx(i,j)] * v[Bidx(j,k)]
                    if abs(denom) > 1e-12:
                        w[Cidx(i,k)] = T[i,j,k] / denom
                        break
    
    return (u, v, w)

def try_drop(U, V, W, r, l):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[Aidx(i,j)], Va[Bidx(j,k)] * Wa[Cidx(i,k)])
                s -= U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def local_search(U, V, W, r, deadline, rng):
    best_r = r
    best_U, best_V, best_W = U, V, W
    
    while time.time() < deadline - 5:
        improved = False
        
        for l in range(best_r):
            if is_zero_term(best_U, best_V, best_W, best_r, l):
                newU = [row[:l] + row[l+1:] for row in best_U]
                newV = [row[:l] + row[l+1:] for row in best_V]
                newW = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                best_U, best_V, best_W = newU, newV, newW
                improved = True
                break
        
        if improved:
            continue
        
        for l in range(best_r):
            if try_drop(best_U, best_V, best_W, best_r, l):
                newU = [row[:l] + row[l+1:] for row in best_U]
                newV = [row[:l] + row[l+1:] for row in best_V]
                newW = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                best_U, best_V, best_W = newU, newV, newW
                improved = True
                break
        
        if improved:
            continue
        
        merged = False
        for l1 in range(best_r):
            if time.time() > deadline - 5:
                break
            for l2 in range(l1+1, best_r):
                if time.time() > deadline - 5:
                    break
                result = try_merge(best_U, best_V, best_W, best_r, l1, l2)
                if result is not None:
                    u_new, v_new, w_new = result
                    keep_indices = [l for l in range(best_r) if l != l1 and l != l2]
                    newU = [row[:l1] + row[l1+1:l2] + row[l2+1:] for row in best_U]
                    newV = [row[:l1] + row[l1+1:l2] + row[l2+1:] for row in best_V]
                    newW = [row[:l1] + row[l1+1:l2] + row[l2+1:] for row in best_W]
                    for idx in range(len(newU)):
                        newU[idx].append(float(u_new[idx]))
                    for idx in range(len(newV)):
                        newV[idx].append(float(v_new[idx]))
                    for idx in range(len(newW)):
                        newW[idx].append(float(w_new[idx]))
                    best_r -= 1
                    best_U, best_V, best_W = newU, newV, newW
                    improved = True
                    merged = True
                    break
            if merged:
                break
        
        if not improved:
            break
    
    return best_r, best_U, best_V, best_W

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    U, V, W = baseline_66()
    r = 66
    r, U, V, W = local_search(U, V, W, r, deadline, rng)
    if not verify(U, V, W, r):
        U, V, W = baseline_66()
        r = 66
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
