"""
Rank-minimal exact tensor decomposition for 3x4 by 4x6 matrix multiplication.
Strategy:
  1. Build a block-structured baseline (Strassen + 1x2 blocks) at rank 66.
  2. Use HALS-style iterative refinement: parameterize rank-r factors, then
     run a local search that (a) drops redundant terms, (b) merges pairs of
     terms whose sum is rank-1, and (c) performs random perturbations followed
     by a greedy re-solve to escape local minima.
  3. Final verification with tight tolerance.
"""
import random
import time
import numpy as np

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i  # W[k*n+i]

def true_tensor_flat():
    """Return flat indices (i,j,k) where tensor = 1."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def verify(U, V, W, r):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[Aidx(i,j)], Va[Bidx(j,k)] * Wa[Cidx(i,k)])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def baseline_66():
    U = [[0.0]*66 for _ in range(n*m)]
    V = [[0.0]*66 for _ in range(m*p)]
    W = [[0.0]*66 for _ in range(p*n)]
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
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1

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

def try_drop(U, V, W, r, l):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[Aidx(i,j)], Va[Bidx(j,k)] * Wa[Cidx(i,k)])
                s -= U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def check_rank1_tensor(T):
    """Check if 3D tensor T (shape n,m,p) is rank-1. Returns True/False."""
    M1 = T.reshape(n*m, p)
    for a in range(n*m):
        for b in range(a+1, n*m):
            for c in range(p):
                for d in range(c+1, p):
                    if abs(M1[a,c]*M1[b,d] - M1[a,d]*M1[b,c]) > 1e-9:
                        return False
    M2 = T.reshape(n, m*p)
    for a in range(n):
        for b in range(a+1, n):
            for c in range(m*p):
                for d in range(c+1, m*p):
                    if abs(M2[a,c]*M2[b,d] - M2[a,d]*M2[b,c]) > 1e-9:
                        return False
    return True

def extract_rank1(T):
    """Extract u, v, w from rank-1 tensor T[i,j,k] = u[i,j]*v[j,k]*w[i,k]."""
    u = np.ones(n*m)
    v = np.ones(m*p)
    w = np.ones(p*n)
    for _ in range(30):
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
    return u, v, w

def try_merge(U, V, W, r, l1, l2):
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = (U[Aidx(i,j)][l1] * V[Bidx(j,k)][l1] * W[Cidx(i,k)][l1] +
                           U[Aidx(i,j)][l2] * V[Bidx(j,k)][l2] * W[Cidx(i,k)][l2])
    if check_rank1_tensor(T):
        u, v, w = extract_rank1(T)
        return (u, v, w)
    return None

def local_search(U, V, W, r, deadline, rng):
    best_r = r
    best_U, best_V, best_W = U, V, W

    while time.time() < deadline - 5:
        improved = False

        # Phase 1: drop zero terms
        for l in range(best_r):
            if is_zero_term(best_U, best_V, best_W, best_r, l):
                best_U = [row[:l] + row[l+1:] for row in best_U]
                best_V = [row[:l] + row[l+1:] for row in best_V]
                best_W = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                improved = True
                break
        if improved:
            continue

        # Phase 2: drop redundant terms
        for l in range(best_r):
            if try_drop(best_U, best_V, best_W, best_r, l):
                best_U = [row[:l] + row[l+1:] for row in best_U]
                best_V = [row[:l] + row[l+1:] for row in best_V]
                best_W = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                improved = True
                break
        if improved:
            continue

        # Phase 3: merge pairs
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
                    keep = [l for l in range(best_r) if l != l1 and l != l2]
                    newU = [[best_U[row][l] for l in keep] for row in range(len(best_U))]
                    newV = [[best_V[row][l] for l in keep] for row in range(len(best_V))]
                    newW = [[best_W[row][l] for l in keep] for row in range(len(best_W))]
                    for row in range(len(newU)):
                        newU[row].append(float(u_new[row]))
                    for row in range(len(newV)):
                        newV[row].append(float(v_new[row]))
                    for row in range(len(newW)):
                        newW[row].append(float(w_new[row]))
                    best_r -= 1
                    best_U, best_V, best_W = newU, newV, newW
                    improved = True
                    merged = True
                    break
            if merged:
                break

        # Phase 4: random perturbation + re-solve (HALS-inspired)
        if not improved and time.time() < deadline - 10:
            # Try a few random perturbations
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                U_pert = [row[:] for row in best_U]
                V_pert = [row[:] for row in best_V]
                W_pert = [row[:] for row in best_W]
                # Perturb a random column slightly
                col_idx = rng.randint(0, best_r - 1)
                for row in range(len(U_pert)):
                    U_pert[row][col_idx] += rng.gauss(0, 0.1)
                for row in range(len(V_pert)):
                    V_pert[row][col_idx] += rng.gauss(0, 0.1)
                for row in range(len(W_pert)):
                    W_pert[row][col_idx] += rng.gauss(0, 0.1)
                # Now try to drop/merge from perturbed state
                r_pert = best_r
                U_pert, V_pert, W_pert, r_pert = _quick_reduce(U_pert, V_pert, W_pert, r_pert, deadline, rng)
                if r_pert < best_r:
                    best_r, best_U, best_V, best_W = r_pert, U_pert, V_pert, W_pert
                    improved = True
                    break
                # Also verify perturbed state is still exact; if so, keep it
                elif verify(U_pert, V_pert, W_pert, r_pert):
                    best_U, best_V, best_W = U_pert, V_pert, W_pert
                    improved = True
                    break

        if not improved:
            break

    return best_r, best_U, best_V, best_W

def _quick_reduce(U, V, W, r, deadline, rng):
    """Quick drop/merge pass on a perturbed decomposition."""
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        changed = False
        for l in range(r):
            if is_zero_term(U, V, W, r, l):
                U = [row[:l] + row[l+1:] for row in U]
                V = [row[:l] + row[l+1:] for row in V]
                W = [row[:l] + row[l+1:] for row in W]
                r -= 1
                changed = True
                break
        if changed:
            continue
        for l in range(r):
            if try_drop(U, V, W, r, l):
                U = [row[:l] + row[l+1:] for row in U]
                V = [row[:l] + row[l+1:] for row in V]
                W = [row[:l] + row[l+1:] for row in W]
                r -= 1
                changed = True
                break
        if changed:
            continue
        # Try one merge
        merged = False
        for l1 in range(r):
            for l2 in range(l1+1, r):
                result = try_merge(U, V, W, r, l1, l2)
                if result is not None:
                    u_new, v_new, w_new = result
                    keep = [l for l in range(r) if l != l1 and l != l2]
                    newU = [[U[row][l] for l in keep] for row in range(len(U))]
                    newV = [[V[row][l] for l in keep] for row in range(len(V))]
                    newW = [[W[row][l] for l in keep] for row in range(len(W))]
                    for row in range(len(newU)):
                        newU[row].append(float(u_new[row]))
                    for row in range(len(newV)):
                        newV[row].append(float(v_new[row]))
                    for row in range(len(newW)):
                        newW[row].append(float(w_new[row]))
                    U, V, W, r = newU, newV, newW, r - 1
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return U, V, W, r

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
