import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """Strassen 2x2x2: U(4,7), V(4,7), W(4,7) with trace convention W[k*2+i]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A11-A10)(B00+B01)
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11)
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00)
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11)
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11)
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00)
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10)
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    return U, V, W

def _kronecker_4x4x4_from_2x2x2():
    """Build 4x4x4 decomposition (rank 49) from Strassen 2x2x2 via Kronecker product."""
    U2, V2, W2 = _strassen_2x2()
    r_base = 7
    r_full = r_base * r_base  # 49
    U = [[0]*r_full for _ in range(16)]
    V = [[0]*r_full for _ in range(16)]
    W = [[0]*r_full for _ in range(16)]
    
    for rb in range(r_base):
        for rs in range(r_base):
            col = rb * r_base + rs
            # U: a = ib*8 + i0*4 + jb*2 + j0
            for ib in range(2):
                for jb in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            a = ib*8 + i0*4 + jb*2 + j0
                            ab = ib*2 + jb
                            as_idx = i0*2 + j0
                            U[a][col] = U2[ab][rb] * U2[as_idx][rs]
            # V: b = jb2*8 + j02*4 + kb*2 + k0
            for jb2 in range(2):
                for j02 in range(2):
                    for kb in range(2):
                        for k0 in range(2):
                            b = jb2*8 + j02*4 + kb*2 + k0
                            bb = jb2*2 + kb
                            bs = j02*2 + k0
                            V[b][col] = V2[bb][rb] * V2[bs][rs]
            # W: c = kb2*8 + k02*4 + ib2*2 + i02
            for kb2 in range(2):
                for k02 in range(2):
                    for ib2 in range(2):
                        for i02 in range(2):
                            c = kb2*8 + k02*4 + ib2*2 + i02
                            cb = kb2*2 + ib2
                            cs = k02*2 + i02
                            W[c][col] = W2[cb][rb] * W2[cs][rs]
    return U, V, W

def _extend_4x4x4_to_4x4x8(U4, V4, W4):
    """Extend 4x4x4 (rank r) to 4x4x8 (rank 2r) by duplicating for the second half of p."""
    r_base = len(U4[0])
    r_full = 2 * r_base
    U = [[0]*r_full for _ in range(16)]
    V = [[0]*r_full for _ in range(32)]
    W = [[0]*r_full for _ in range(32)]
    
    for r in range(r_base):
        for i in range(16):
            U[i][r] = U4[i][r]
            U[i][r + r_base] = U4[i][r]
        for j in range(4):
            for k in range(4):
                V[j*8+k][r] = V4[j*4+k][r]
                V[j*8+(k+4)][r + r_base] = V4[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[k*4+i][r] = W4[k*4+i][r]
                W[(k+4)*4+i][r + r_base] = W4[k*4+i][r]
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify tensor decomposition using numpy for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Check all entries
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = float(np.dot(U_arr[a], np.multiply(V_arr[b], W_arr[c])))
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification: check random subset + structural invariants."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                actual = float(np.dot(U_arr[a], np.multiply(V_arr[b], W_arr[c])))
                if abs(actual - 1.0) > 1e-9:
                    return False
    
    # Check some off-diagonal entries (should be 0)
    np.random.seed(42)
    for _ in range(500):
        i = np.random.randint(n)
        j = np.random.randint(m)
        j2 = np.random.randint(m)
        k = np.random.randint(p)
        i2 = np.random.randint(n)
        k2 = np.random.randint(p)
        if j == j2 and i == i2 and k == k2:
            continue
        a = i * m + j
        b = j2 * p + k
        c = k2 * n + i2
        actual = float(np.dot(U_arr[a], np.multiply(V_arr[b], W_arr[c])))
        if abs(actual) > 1e-9:
            return False
    return True

def _try_strassen_based():
    """Build 4x4x8 via Strassen Kronecker: 2x2x2 -> 4x4x4 (rank 49) -> 4x4x8 (rank 98)."""
    U4, V4, W4 = _kronecker_4x4x4_from_2x2x2()
    U8, V8, W8 = _extend_4x4x4_to_4x4x8(U4, V4, W4)
    return U8, V8, W8

def _try_direct_kronecker_2x2x4():
    """Try building 4x4x8 as Kronecker of 2x2x2 (Strassen) and 2x2x4 (schoolbook, rank 16).
    
    4x4x8 = (2*2) x (2*2) x (2*4)
    A: 2x2 blocks of 2x2 -> block tensor is 2x2x2
    B: 2x2 blocks of 2x4 -> block tensor is 2x2x4  
    C: 2x2 blocks of 2x4 -> block tensor is 2x2x4 (trace convention)
    
    Actually this doesn't factor as a clean Kronecker product because the dimensions
    are asymmetric. Let me try a different approach.
    
    Alternative: view 4x4x8 as 4x4x4 with p doubled. The p-doubling gives rank 2*r(4x4x4).
    But maybe we can do better by exploiting the structure differently.
    
    For now, return None to indicate this approach isn't directly applicable.
    """
    return None

def _try_las_vegas_search(max_time=10):
    """Try to find a lower-rank decomposition using random initialization + optimization.
    
    Strategy: initialize with random factors of rank r < 98, then try to solve
    the nonlinear system T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    
    This is a hard problem, so we'll use a simple approach: start from the rank-98
    solution and try to find linear dependencies.
    """
    return None

def schoolbook_decomposition(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 128)
    best_result = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try Kronecker construction: 2x2x2 Strassen -> 4x4x4 (rank 49) -> 4x4x8 (rank 98)
    try:
        U8, V8, W8 = _try_strassen_based()
        if U8 is not None:
            # Fast verification first
            if _verify_fast(U8, V8, W8, n, m, p):
                # Full verification
                if _verify_decomposition(U8, V8, W8, n, m, p):
                    best_result = {"U": U8, "V": V8, "W": W8}
                    best_rank = len(U8[0])
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
