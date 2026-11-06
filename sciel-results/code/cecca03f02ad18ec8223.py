import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

def _build():
    # Edge index
    eidx = [[-1]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            idx = i*(N-1) - i*(i+1)//2 + (j-i-1)
            eidx[i][j] = idx
            eidx[j][i] = idx
    
    # All 4-sets and cycles
    cycles = []
    for vs in combinations(range(N), 4):
        a,b,c,d = vs
        cycles.append((eidx[a][b], eidx[b][c], eidx[c][d], eidx[d][a]))
        cycles.append((eidx[a][b], eidx[b][d], eidx[d][c], eidx[c][a]))
        cycles.append((eidx[a][c], eidx[c][b], eidx[b][d], eidx[d][a]))
    
    # Edge to cycle indices
    e2c = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            e2c[e].append(ci)
    
    return eidx, cycles, e2c

_EIDX, _CYCLES, _E2C = _build()

def construct():
    t0 = time.time()
    deadline = t0 + 78
    
    colors = [0]*NUM_EDGES
    best = colors[:]
    best_n = 1
    
    # Greedy: assign new colors
    for e in range(NUM_EDGES):
        if time.time() > deadline - 15:
            break
        for nc in range(1, 120):
            ok = True
            for ci in _E2C[e]:
                cyc = _CYCLES[ci]
                c0 = colors[cyc[0]] if cyc[0]!=e else nc
                c1 = colors[cyc[1]] if cyc[1]!=e else nc
                c2 = colors[cyc[2]] if cyc[2]!=e else nc
                c3 = colors[cyc[3]] if cyc[3]!=e else nc
                if c0!=c1 and c0!=c2 and c0!=c3 and c1!=c2 and c1!=c3 and c2!=c3:
                    ok = False
                    break
            if ok:
                colors[e] = nc
                break
    
    n = len(set(colors))
    if n > best_n:
        best_n = n
        best = colors[:]
    
    # Local search
    rnd = random.Random(42)
    
    def check(e, nc, cols):
        for ci in _E2C[e]:
            cyc = _CYCLES[ci]
            c0 = cols[cyc[0]] if cyc[0]!=e else nc
            c1 = cols[cyc[1]] if cyc[1]!=e else nc
            c2 = cols[cyc[2]] if cyc[2]!=e else nc
            c3 = cols[cyc[3]] if cyc[3]!=e else nc
            if c0!=c1 and c0!=c2 and c0!=c3 and c1!=c2 and c1!=c3 and c2!=c3:
                return False
        return True
    
    def lsearch(cols, dl):
        while time.time() < dl - 2:
            imp = False
            for e in range(NUM_EDGES):
                if time.time() >= dl - 2:
                    break
                old = cols[e]
                for nc in range(120):
                    if nc == old:
                        continue
                    if check(e, nc, cols):
                        oc = len(set(cols))
                        cols[e] = nc
                        nc2 = len(set(cols))
                        if nc2 > oc:
                            imp = True
                            break
                        else:
                            cols[e] = old
            if not imp:
                break
        return cols
    
    att = 0
    while time.time() < deadline - 3:
        att += 1
        if att == 1:
            cur = best[:]
        else:
            cur = best[:]
            for _ in range(3):
                e = rnd.randint(0, NUM_EDGES-1)
                nc = rnd.randint(0, 119)
                if check(e, nc, cur):
                    cur[e] = nc
        cur = lsearch(cur, deadline)
        cn = len(set(cur))
        if cn > best_n:
            best_n = cn
            best = cur[:]
        if best_n >= 22:
            break
    
    return best
