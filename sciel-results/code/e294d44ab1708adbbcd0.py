import time, random

def construct():
    n = 15
    t0 = time.time()
    deadline = t0 + 50.0

    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng = random.Random(20240517)

    # Bitmask-based fast validity check
    def is_valid_mask(mask):
        deg = [0]*n
        for i in range(n):
            deg[i] = bin(mask >> (i*n) & ((1<<n)-1)).count('1')
        for u in range(n):
            au_mask = (mask >> (u*n)) & ((1<<n)-1)
            for v in range(u+1, n):
                if au_mask & (1<<v):
                    av_mask = (mask >> (v*n)) & ((1<<n)-1)
                    for s in range(n):
                        if s != v and (au_mask & (1<<s)):
                            ds = deg[s]
                            for t in range(n):
                                if t != u and t != s and (av_mask & (1<<t)) and deg[t] == ds:
                                    return False
        return True

    def ecount_mask(mask):
        c = 0
        for i in range(n):
            c += bin(mask >> (i*n) & ((1<<n)-1)).count('1')
        return c//2

    def mask_to_sets(mask):
        a = [set() for _ in range(n)]
        for i in range(n):
            mi = (mask >> (i*n)) & ((1<<n)-1)
            for j in range(i+1, n):
                if mi & (1<<j):
                    a[i].add(j); a[j].add(i)
        return a

    def sets_to_mask(a):
        mask = 0
        for i in range(n):
            for j in a[i]:
                if j > i:
                    mask |= (1 << (i*n + j))
        return mask

    def is_valid(adj):
        return is_valid_mask(sets_to_mask(adj))

    def ecount(adj):
        return ecount_mask(sets_to_mask(adj))

    def find_violation(adj):
        mask = sets_to_mask(adj)
        deg = [0]*n
        for i in range(n):
            deg[i] = bin(mask >> (i*n) & ((1<<n)-1)).count('1')
        for u in range(n):
            au_mask = (mask >> (u*n)) & ((1<<n)-1)
            for v in range(u+1, n):
                if au_mask & (1<<v):
                    av_mask = (mask >> (v*n)) & ((1<<n)-1)
                    for s in range(n):
                        if s != v and (au_mask & (1<<s)):
                            ds = deg[s]
                            for t in range(n):
                                if t != u and t != s and (av_mask & (1<<t)) and deg[t] == ds:
                                    return (s,u,v,t)
        return None

    best_mask = 0
    bestc = -1

    def consider_mask(m):
        nonlocal best_mask, bestc
        c = ecount_mask(m)
        if c > bestc and is_valid_mask(m):
            bestc = c
            best_mask = m

    # Baseline K7,8
    a0 = [set() for _ in range(n)]
    for i in range(7):
        for j in range(7,15):
            a0[i].add(j); a0[j].add(i)
    consider_mask(sets_to_mask(a0))

    def local_improve(start_mask):
        m = start_mask
        run_best = start_mask
        run_bc = ecount_mask(start_mask)
        while time.time() < deadline:
            missing = [(i,j) for i in range(n) for j in range(i+1,n) if not (m >> (i*n+j) & 1)]
            if not missing:
                break
            rng.shuffle(missing)
            progress = False
            for (u,v) in missing:
                if time.time() > deadline: break
                m2 = m | (1 << (u*n+v))
                if is_valid_mask(m2):
                    progress = True
                    c = ecount_mask(m2)
                    if c > run_bc:
                        run_bc = c
                        run_best = m2
                        m = m2
            if not progress:
                # kick: remove 1-2 random edges
                es = [(i,j) for i in range(n) for j in range(i+1,n) if (m >> (i*n+j) & 1)]
                if not es:
                    break
                k = rng.randint(1,2)
                rng.shuffle(es)
                for (x,y) in es[:k]:
                    m &= ~(1 << (x*n+y))
        return run_best, run_bc

    def local_search_swap(start_mask):
        m = start_mask
        run_best = start_mask
        run_bc = ecount_mask(start_mask)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            present = [(i,j) for i in range(n) for j in range(i+1,n) if (m >> (i*n+j) & 1)]
            missing = [(i,j) for i in range(n) for j in range(i+1,n) if not (m >> (i*n+j) & 1)]
            rng.shuffle(present)
            rng.shuffle(missing)
            for (ru,rv) in present:
                if time.time() > deadline: break
                for (au,av) in missing:
                    if time.time() > deadline: break
                    m2 = m & ~(1 << (ru*n+rv))
                    m2 |= (1 << (au*n+av))
                    if is_valid_mask(m2):
                        c = ecount_mask(m2)
                        if c > run_bc:
                            run_bc = c
                            run_best = m2
                            m = m2
                            improved = True
                            break
                if improved:
                    break
        return run_best, run_bc

    def greedy_add(order):
        m = 0
        for (u,v) in order:
            m2 = m | (1 << (u*n+v))
            if is_valid_mask(m2):
                m = m2
        return m

    def repair(start_mask):
        m = start_mask
        for _ in range(500):
            viol = find_violation(mask_to_sets(m))
            if viol is None:
                break
            s,u,v,t = viol
            cand = [(s,u),(u,v),(v,t)]
            done = False
            for (x,y) in cand:
                m2 = m & ~(1 << (x*n+y))
                if is_valid_mask(m2):
                    m = m2
                    done = True
                    break
            if not done:
                (x,y) = cand[rng.randrange(3)]
                m &= ~(1 << (x*n+y))
        return m

    iter_no = 0
    while time.time() < deadline:
        iter_no += 1
        # strategy 1: greedy from empty
        if iter_no % 3 == 1:
            order = list(all_edges); rng.shuffle(order)
            g = greedy_add(order)
            consider_mask(g)
            rb, rc = local_improve(g)
            consider_mask(rb)
            # swap local search
            rs, rcs = local_search_swap(rb)
            consider_mask(rs)
        # strategy 2: random dense + repair
        elif iter_no % 3 == 2:
            p = rng.uniform(0.4, 0.85)
            m = 0
            for (u,v) in all_edges:
                if rng.random() < p:
                    m |= (1 << (u*n+v))
            r = repair(m)
            consider_mask(r)
            rb, rc = local_improve(r)
            consider_mask(rb)
            rs, rcs = local_search_swap(rb)
            consider_mask(rs)
        # strategy 3: perturb best and improve
        else:
            m = best_mask
            # remove 2-3 random edges
            present = [(i,j) for i in range(n) for j in range(i+1,n) if (m >> (i*n+j) & 1)]
            if present:
                rng.shuffle(present)
                k = rng.randint(2,3)
                for (x,y) in present[:k]:
                    m &= ~(1 << (x*n+y))
            rb, rc = local_improve(m)
            consider_mask(rb)
            rs, rcs = local_search_swap(rb)
            consider_mask(rs)

    # final
    if bestc < 0:
        best_mask = sets_to_mask(a0)
    best = mask_to_sets(best_mask)
    edges = []
    for u in range(n):
        for v in best[u]:
            if v > u:
                edges.append([int(u), int(v)])
    return edges
