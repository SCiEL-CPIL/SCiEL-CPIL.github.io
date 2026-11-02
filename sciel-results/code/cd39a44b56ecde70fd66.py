import time
import random
_start = time.time()
DEADLINE = 145.0

DIMENSIONS = [4, 4, 4]

def _lf_add(a, b):
    d = dict(a)
    for k, v in b.items():
        d[k] = d.get(k, 0) + v
    # remove zeros
    return {k: v for k, v in d.items() if v != 0}

def _lf_scale(a, s):
    return {k: v * s for k, v in a.items() if v * s != 0}

def _lf_entry(source, r, c):
    return {(source, r, c): 1}

# Build A and B block entries (2x2 matrices of linear forms)
def _build_blocks(source):
    blocks = {}
    for a in range(2):
        for b in range(2):
            blk = [[None]*2 for _ in range(2)]
            for r in range(2):
                for c in range(2):
                    blk[r][c] = _lf_entry(source, 2*a+r, 2*b+c)
            blocks[(a,b)] = blk
    return blocks

def _blk_add(X, Y):
    return [[_lf_add(X[r][c], Y[r][c]) for c in range(2)] for r in range(2)]

def _blk_sub(X, Y):
    return [[_lf_add(X[r][c], _lf_scale(Y[r][c], -1)) for c in range(2)] for r in range(2)]

# inner 2x2 Strassen terms: list of (Lx_indices, Ly_indices, out_2x2)
# Lx_indices: list of (pos, coeff) into X's 2x2
INNER = [
    ( [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [[1,0],[0,1]] ),
    ( [((1,0),1),((1,1),1)], [((0,0),1)],           [[0,0],[1,-1]] ),
    ( [((0,0),1)],                [((1,0),1),((1,1),-1)], [[0,1],[0,1]] ),
    ( [((1,1),1)],                [((0,1),1),((1,0),-1)], [[1,0],[1,0]] ),
    ( [((0,0),1),((0,1),1)], [((1,1),1)],           [[-1,1],[0,0]] ),
    ( [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)], [[0,0],[0,1]] ),
    ( [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)], [[1,0],[0,0]] ),
]

# outer terms: (Lx_blocks, Ly_blocks, out_block_coeffs) where out_block_coeffs: dict block_pos->scalar
def _comb(blocks, spec):
    # spec: list of (block_pos, coeff)
    acc = [[None]*2 for _ in range(2)]
    for r in range(2):
        for c in range(2):
            acc[r][c] = {}
    for (bp, cf) in spec:
        B = blocks[bp]
        for r in range(2):
            for c in range(2):
                acc[r][c] = _lf_add(acc[r][c], _lf_scale(B[r][c], cf))
    return acc

OUTER = [
    ( [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], { (0,0):1, (1,1):1 } ),
    ( [((1,0),1),((1,1),1)], [((0,0),1)],           { (1,0):1, (1,1):-1 } ),
    ( [((0,0),1)],                [((1,0),1),((1,1),-1)], { (0,1):1, (1,1):1 } ),
    ( [((1,1),1)],                [((0,1),1),((1,0),-1)], { (0,0):1, (1,0):1 } ),
    ( [((0,0),1),((0,1),1)], [((1,1),1)],           { (0,0):-1, (0,1):1 } ),
    ( [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)], { (1,1):1 } ),
    ( [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)], { (0,0):1 } ),
]

def construct():
    global _start
    if time.time() - _start > DEADLINE:
        # fallback schoolbook (should not happen)
        return _schoolbook()
    A_blocks = _build_blocks('A')
    B_blocks = _build_blocks('B')
    # accumulate terms: list of (U_lf, V_lf, W_dict)
    terms = []
    for t in range(7):
        Lx_spec, Ly_spec, out_coeff = OUTER[t]
        Lx = _comb(A_blocks, Lx_spec)
        Ly = _comb(B_blocks, Ly_spec)
        for s in range(7):
            lx_idx, ly_idx, inner_out = INNER[s]
            # build Lx_inner = sum of Lx[pos]*coeff
            lx = {}
            for (pos, cf) in lx_idx:
                r, c = pos
                lx = _lf_add(lx, _lf_scale(Lx[r][c], cf))
            ly = {}
            for (pos, cf) in ly_idx:
                r, c = pos
                ly = _lf_add(ly, _lf_scale(Ly[r][c], cf))
            if not lx or not ly:
                continue
            wdict = {}
            for ba in out_coeff:
                cf_ba = out_coeff[ba]
                for r in range(2):
                    for c in range(2):
                        cf = cf_ba * inner_out[r][c]
                        if cf == 0:
                            continue
                        out_idx = (2*ba[0]+r) * 4 + (2*ba[1]+c)  # W index k*n+i? need convention
                        wdict[out_idx] = wdict.get(out_idx, 0) + cf
            terms.append((lx, ly, wdict))
    rank = len(terms)
    n = m = p = 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for col, (lx, ly, wdict) in enumerate(terms):
        for (src, r, c), cf in lx.items():
            assert src == 'A'
            U[r*m + c][col] = cf
        for (src, r, c), cf in ly.items():
            assert src == 'B'
            V[r*p + c][col] = cf
        for out_idx, cf in wdict.items():
            W[out_idx][col] = cf
    # verify
    if not _verify(U, V, W):
        return _schoolbook()
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, trials=8):
    n = m = p = 4
    rank = len(U[0])
    for _ in range(trials):
        A = [[random.randint(-3,3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3,3) for _ in range(p)] for _ in range(m)]
        # compute C via decomposition: C[i,k] = sum_r U[i,r]*W[k,r]*V[r_index? ]
        # Convention: U shape (n*m, rank) indexed [i*m+j]. V (m*p, rank) [j*p+k]. W (p*n, rank) [k*n+i].
        # C[i,k] = sum_c U[i*m + ?]... 
        # Standard: (A B)[i,k] = sum_j A[i,j] B[j,k].
        # Decomposition: C[i,k] = sum_c U[i,j?] ... The tensor: C_ik = sum_c U_{i,?} ...
        # Given shapes, the decomposition is: C[i,k] = sum_c (sum_j U[i*m+j, c]) * V[? ] ... 
        # Actually the convention: U[i*m+j, c] multiplies A[i,j]; V[j*p+k, c] multiplies B[j,k]; W[k*n+i, c] is output coeff.
        # C[i,k] = sum_c sum_j sum_? ... Let me use: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_k' V[j*p+k',c] B[j,k'])
        # Hmm that double counts j. Correct form: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) * W[k*n+i,c]
        # Wait that's product of two sums each over j -> not standard.
        # The tensor identity: T_{i,j,k,l} (A_ik? ) ... Let me just verify by building the full tensor.
        # C[i,k] = sum_{j} A[i,j] B[j,k].
        # Decomposition rank r: C[i,k] = sum_c U1[i, c] * U2[...] ... 
        # Given the interface, the correct reconstruction:
        # C[i,k] = sum_c W[k*n+i, c] * Xc * Yc where Xc = sum_j U[i*m+j, c] A[i,j], Yc = sum_j V[j*p+k, c] B[j,k]
        # This is the standard "U, V, W" with W combining. Let me test it.
        C = [[0]*p for _ in range(n)]
        for c in range(rank):
            Xc = 0
            for i in range(n):
                for j in range(m):
                    Xc += U[i*m+j][c] * A[i][j]
            Yc = 0
            for j in range(m):
                for k in range(p):
                    Yc += V[j*p+k][c] * B[j][k]
            for i in range(n):
                for k in range(p):
                    C[i][k] += W[k*n+i][c] * Xc * Yc
        # compare to A*B
        for i in range(n):
            for k in range(p):
                true = sum(A[i][j]*B[j][k] for j in range(m))
                if abs(C[i][k] - true) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 4
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
