import time

DIMENSIONS = [2, 2, 14]
_DEADLINE = None

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE[0]:
        raise TimeoutError("deadline exceeded")

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's algorithm on 7 independent 2x2 blocks.
    """
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = [t0 + 145.0]
    try:
        result = _build_rank_49()
        return result
    except Exception:
        return _build_rank_49()
    finally:
        _DEADLINE = None


def _build_rank_49():
    n, m, p = DIMENSIONS
    nb = p // m
    rk = nb * 7
    U = [[0.0] * rk for _ in range(n * m)]
    V = [[0.0] * rk for _ in range(m * p)]
    W = [[0.0] * rk for _ in range(p * n)]
    uf = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
          [1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
    for r in range(nb):
        ie = 2*r
        if_ = 2*r+1
        ig = m*(p//2)+2*r
        ih = m*(p//2)+2*r+1
        wc00 = 2*r*n
        wc10 = 2*r*n+1
        wc01 = (2*r+1)*n
        wc11 = (2*r+1)*n+1
        for s in range(7):
            c = 7*r+s
            for i in range(4):
                U[i][c] = float(uf[s][i])
            if s==0: V[ie][c]=1.0; V[ih][c]=1.0
            elif s==1: V[ie][c]=1.0
            elif s==2: V[if_][c]=1.0; V[ih][c]=-1.0
            elif s==3: V[ie][c]=-1.0; V[ig][c]=1.0
            elif s==4: V[ih][c]=1.0
            elif s==5: V[ie][c]=1.0; V[if_][c]=1.0
            elif s==6: V[ig][c]=1.0; V[ih][c]=1.0
            if s==0: W[wc00][c]=1.0; W[wc11][c]=1.0
            elif s==1: W[wc10][c]=1.0; W[wc11][c]=-1.0
            elif s==2: W[wc01][c]=1.0; W[wc11][c]=1.0
            elif s==3: W[wc00][c]=1.0; W[wc10][c]=1.0
            elif s==4: W[wc00][c]=-1.0; W[wc01][c]=1.0
            elif s==5: W[wc11][c]=1.0
            elif s==6: W[wc00][c]=1.0
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
