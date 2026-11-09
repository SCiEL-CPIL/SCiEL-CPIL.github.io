# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using greedy removal.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    p = 7
    k = 3
    n = p * p
    total = n * n
    
    # F_49 helpers
    def add(a, b):
        return ((a[0]+b[0])%p, (a[1]+b[1])%p)
    def mul(a, b):
        return ((a[0]*b[0]+k*a[1]*b[1])%p, (a[0]*b[1]+a[1]*b[0])%p)
    def p2i(pt):
        return (pt[0][0]*p+pt[0][1])*p*p + pt[1][0]*p + pt[1][1]
    def i2p(i):
        yb=i%p; ya=(i//p)%p; xb=(i//p//p)%p; xa=(i//p//p//p)%p
        return ((xa,xb),(ya,yb))
    
    # Initial set
    sq = set((u*u)%p for u in range(p))
    N = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa+ya*ya-yb*yb*k)%p==0 and (ya,yb) not in sq:
                        continue
                    N.add((xa*p+xb)*p*p+ya*p+yb)
    
    # Directions
    dirs = [((1,0),(ma,mb)) for ma in range(p) for mb in range(p)]
    dirs.append(((0,0),(0,1)))
    nd = len(dirs)
    
    # Precompute line members
    lm = [None]*(total*nd)
    for xi in range(total):
        x = i2p(xi)
        for di, v in enumerate(dirs):
            mem = set()
            for ti in range(n):
                if ti==0: continue
                t = (ti%p, (ti//p)%p)
                tv = mul(t, v)
                np_ = add(x, tv)
                ni = p2i(np_)
                if ni != xi:
                    mem.add(ni)
            lm[xi*nd+di] = frozenset(mem)
    
    # Reverse: for each x, list of (y,d) where x in lm[y,d]
    rev = [[] for _ in range(total)]
    for yi in range(total):
        for di in range(nd):
            for xi in lm[yi*nd+di]:
                rev[xi].append((yi,di))
    
    # Valid count
    def compute_vc(Nset):
        vc = [0]*total
        for yi in range(total):
            for di in range(nd):
                if all(m in Nset for m in lm[yi*nd+di]):
                    vc[yi] += 1
        return vc
    
    best_sz = len(N)
    best_N = set(N)
    
    for _ in range(8):
        if time.time() > deadline:
            break
        Nset = set(best_N)
        vc = compute_vc(Nset)
        lst = list(Nset)
        random.shuffle(lst)
        for xi in lst:
            if time.time() > deadline:
                break
            if xi not in Nset:
                continue
            ok = True
            for (yi,di) in rev[xi]:
                if vc[yi] <= 1:
                    ok = False
                    break
                # Check if yi has a valid dir not containing xi
                found = False
                for d2 in range(nd):
                    if d2==di: continue
                    if all(m in Nset for m in lm[yi*nd+d2]) and xi not in lm[yi*nd+d2]:
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                Nset.remove(xi)
                for (yi,di) in rev[xi]:
                    vc[yi] = sum(1 for d2 in range(nd) if all(m in Nset for m in lm[yi*nd+d2]))
                if len(Nset) < best_sz:
                    best_sz = len(Nset)
                    best_N = set(Nset)
    
    res = []
    for i in best_N:
        pt = i2p(i)
        res.append([[pt[0][0],pt[0][1]],[pt[1][0],pt[1][1]]])
    return np.array(res, dtype=np.int64)

# EVOLVE-BLOCK-END
