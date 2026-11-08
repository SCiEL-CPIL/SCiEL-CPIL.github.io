import time
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate, nn=2, mm=3, pp=8):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(nn):
            for a in range(mm):
                for k in range(pp):
                    acc = Fraction(0)
                    u_row = U[i * mm + a]
                    v_row = V[a * pp + k]
                    w_row = W[k * nn + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i*2+a] * S_V[r][a*2+k] * S_W[r][k*2+i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build 2x3x3 decomposition (rank 12)
    # Part A: 2x2x2 (Strassen, rank 7) for a in {0,1}, k in {0,1}
    # Part B: 2x2x1 (rank 2) for a in {0,1}, k=2
    # Part C: 2x1x3 (rank 3) for a=2, k in {0,1,2}
    
    # 2x2x1 rank-2 decomposition:
    # r=1: U[i,a]=1 for all i,a; V[a,0]=1 if a=0 else 0; W[0,i]=1 for all i
    #   Wait, V index is (a,k) with k fixed at 2. So V[a,2] for a in {0,1}
    #   V[0,2]=1, V[1,2]=0 for r=1; V[0,2]=0, V[1,2]=1 for r=2
    #   W[2,i]=1 for all i in both terms
    #   U[i,a]=1 for all i,a in both terms
    # Verify: C[i,2] = Σ_r U[i,a,r]*V[a,2,r]*W[2,i,r]
    #   r=1: U[i,a,1]*V[a,2,1]*W[2,i,1] = 1*(1 if a=0 else 0)*1 = 1 if a=0 else 0
    #   r=2: U[i,a,2]*V[a,2,2]*W[2,i,2] = 1*(0 if a=0 else 1)*1 = 1 if a=1 else 0
    #   Sum over a: for a=0: 1, for a=1: 1. Total = 2? No wait, we need Σ_a for each (i,k).
    #   
    #   Actually the tensor is T[i,a,k] and we need Σ_r U[i,a,r]V[a,k,r]W[k,i,r] = 1 for all i,a,k.
    #   For k=2, a in {0,1}:
    #   r=1: U[i,a,1]*V[a,2,1]*W[2,i,1] = 1*V[a,2,1]*1
    #   r=2: U[i,a,2]*V[a,2,2]*W[2,i,2] = 1*V[a,2,2]*1
    #   Sum = V[a,2,1] + V[a,2,2]
    #   We need this = 1 for a=0 and a=1.
    #   V[0,2,1]+V[0,2,2]=1, V[1,2,1]+V[1,2,2]=1.
    #   Set V[0,2,1]=1,V[0,2,2]=0; V[1,2,1]=0,V[1,2,2]=1. Works!
    #   And for other (a,k) pairs, these terms contribute 0 since V[a,k,r]=0 for k≠2.
    
    # 2x1x3 rank-3 decomposition:
    # For a=2, k in {0,1,2}:
    # r=1: U[i,2,1]=1 for all i; V[2,0,1]=1; W[0,i,1]=1 for all i
    # r=2: U[i,2,2]=1 for all i; V[2,1,2]=1; W[1,i,2]=1 for all i
    # r=3: U[i,2,3]=1 for all i; V[2,2,3]=1; W[2,i,3]=1 for all i
    # Verify: T[i,2,k] = Σ_r U[i,2,r]V[2,k,r]W[k,i,r]
    #   For k=0: only r=1 contributes: 1*1*1=1 ✓
    #   For k=1: only r=2: 1 ✓
    #   For k=2: only r=3: 1 ✓
    #   For a≠2: U[i,a,r]=0 for these terms, so no contribution ✓
    
    # Total 2x3x3 rank: 7+2+3 = 12

    # Now 2x3x8 = 2x3x3 (k=0,1,2) + 2x3x3 (k=3,4,5) + 2x3x2 (k=6,7)
    # 2x3x2: 2x2x2 (Strassen, rank 7) + 2x1x2 (rank 2)
    # 2x1x2: U[i,2,r]=1; V[2,k,r] and W[k,i,r]
    #   r=1: V[2,6,1]=1, W[6,i,1]=1 for all i
    #   r=2: V[2,7,2]=1, W[7,i,2]=1 for all i
    #   Total: 7+2=9
    
    # Grand total: 12+12+9 = 33

    rank_total = 33
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0

    def _add_2x3x3(k_offset):
        nonlocal col
        # Strassen for a in {0,1}, k in {k_offset, k_offset+1}
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2+la]
                    if u_val:
                        U[li*m+la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2+lk]
                    if v_val:
                        V[la*p+(k_offset+lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2+li]
                    if w_val:
                        W[(k_offset+lk)*n+li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return False
        
        # 2x2x1 for a in {0,1}, k=k_offset+2 (rank 2)
        # r=col: V[0,k_off+2]=1, W[k_off+2,i]=1 for all i
        for li in range(2):
            U[li*m+0][col] = 1
            U[li*m+1][col] = 1
        V[0*p+(k_offset+2)][col] = 1
        for li in range(2):
            W[(k_offset+2)*n+li][col] = 1
        col += 1
        if not _elapsed_ok():
            return False
        
        # r=col: V[1,k_off+2]=1, W[k_off+2,i]=1 for all i
        for li in range(2):
            U[li*m+0][col] = 1
            U[li*m+1][col] = 1
        V[1*p+(k_offset+2)][col] = 1
        for li in range(2):
            W[(k_offset+2)*n+li][col] = 1
        col += 1
        if not _elapsed_ok():
            return False
        
        # 2x1x3 for a=2, k in {k_offset, k_offset+1, k_offset+2} (rank 3)
        for lk in range(3):
            for li in range(2):
                U[li*m+2][col] = 1
            V[2*p+(k_offset+lk)][col] = 1
            for li in range(2):
                W[(k_offset+lk)*n+li][col] = 1
            col += 1
            if not _elapsed_ok():
                return False
        return True

    def _add_2x3x2(k_offset):
        nonlocal col
        # Strassen for a in {0,1}, k in {k_offset, k_offset+1}
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2+la]
                    if u_val:
                        U[li*m+la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2+lk]
                    if v_val:
                        V[la*p+(k_offset+lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2+li]
                    if w_val:
                        W[(k_offset+lk)*n+li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return False
        
        # 2x1x2 for a=2, k in {k_offset, k_offset+1} (rank 2)
        for lk in range(2):
            for li in range(2):
                U[li*m+2][col] = 1
            V[2*p+(k_offset+lk)][col] = 1
            for li in range(2):
                W[(k_offset+lk)*n+li][col] = 1
            col += 1
            if not _elapsed_ok():
                return False
        return True

    ok = _add_2x3x3(0)
    if ok and _elapsed_ok():
        ok = _add_2x3x3(3)
    if ok and _elapsed_ok():
        ok = _add_2x3x2(6)

    if not ok:
        return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
