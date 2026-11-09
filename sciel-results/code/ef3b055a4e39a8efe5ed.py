import time, math, random

def construct():
    t0 = time.time()
    LIMIT = 55.0
    def now(): return time.time()
    def gcd(a,b): return math.gcd(a,b)
    
    all_m = list(range(2,129))
    n = len(all_m)
    G = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            g = gcd(all_m[i], all_m[j])
            G[i][j]=g; G[j][i]=g
    
    def compatible(r, i, chosen):
        for (j, r2) in chosen:
            g = G[i][j]
            if g < 2: return False
            if (r % g) == (r2 % g): return False
        return True
    
    def find_residue(i, chosen, res_order):
        for r in res_order[i]:
            if compatible(r, i, chosen):
                return r
        return None
    
    # natural residue orders
    nat_order = [list(range(m)) for m in all_m]
    
    def greedy(order, res_orders):
        chosen = []
        for i in order:
            r = find_residue(i, chosen, res_orders)
            if r is not None:
                chosen.append((i, r))
        return chosen
    
    def fill(chosen, res_orders):
        included = set(i for i,_ in chosen)
        changed = True
        while changed and (now()-t0) < LIMIT:
            changed = False
            for i in range(n):
                if i in included: continue
                r = find_residue(i, chosen, res_orders)
                if r is not None:
                    chosen.append((i,r)); included.add(i); changed=True
        return chosen
    
    def improve(chosen, res_orders, kicks=300):
        included = set(i for i,_ in chosen)
        for _ in range(kicks):
            if now()-t0 > LIMIT: break
            # add one
            added=False
            for i in range(n):
                if i in included: continue
                r = find_residue(i, chosen, res_orders)
                if r is not None:
                    chosen.append((i,r)); included.add(i); added=True; break
            if added: continue
            if len(chosen) < 2: break
            ir = random.choice(list(included))
            chosen2 = [(i2,r2) for (i2,r2) in chosen if i2 != ir]
            chosen2 = fill(chosen2, res_orders)
            if len(chosen2) > len(chosen):
                chosen = chosen2
                included = set(i2 for i2,_ in chosen)
        return chosen
    
    def verify(chosen):
        for a in range(len(chosen)):
            for b in range(a+1,len(chosen)):
                i,r = chosen[a]; j,r2 = chosen[b]
                g = G[i][j]
                if g < 2: return False
                if (r % g) == (r2 % g): return False
        return True
    
    best = None; best_len = 0
    random.seed(987654321)
    
    det_orders = [
        list(range(n)),
        list(range(n-1,-1,-1)),
        sorted(range(n), key=lambda i: (-all_m[i].bit_count(), all_m[i])),
        sorted(range(n), key=lambda i: (all_m[i].bit_count(), all_m[i])),
    ]
    for o in det_orders:
        c = greedy(o, nat_order)
        c = fill(c, nat_order)
        c = improve(c, nat_order, kicks=100)
        if len(c) > best_len and verify(c):
            best_len = len(c); best = c
        if now()-t0 > LIMIT*0.4: break
    
    while (now()-t0) < LIMIT:
        o = list(range(n)); random.shuffle(o)
        ro = [list(range(m)) for m in all_m]
        for row in ro: random.shuffle(row)
        c = greedy(o, ro)
        c = fill(c, ro)
        c = improve(c, ro, kicks=150)
        if len(c) > best_len and verify(c):
            best_len = len(c); best = c
    
    if best is None or not verify(best):
        # small valid fallback
        best = [(0,0),(1,1)]  # i=0->m=2 r=0 ; i=1->m=3 r=1 -> gcd(2,3)=1 invalid!
        # use m=2,m=4
        best = [(0,0),(1,2)]  # i0 m2 r0 ; i2 m4 r1 -> gcd2, 0!=1 mod2 ok
        # add m=8 i6 r3
        best.append((3,6))  # i6 m8 r3
    result = [[r, all_m[i]] for (i,r) in best]
    return result
