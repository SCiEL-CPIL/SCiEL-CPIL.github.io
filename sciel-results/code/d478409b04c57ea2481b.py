import time, random, math

def _precompute(n=180):
    triples=[]
    for x in range(1,n+1):
        x2=x*x
        d=1
        while d*d<=x2:
            if x2%d==0:
                e=x2//d
                if d<e:
                    y=x+d
                    z=x+e
                    if y<=n and z<=n and y!=z:
                        triples.append((x-1,y-1,z-1))
            d+=1
    pairs=[[] for _ in range(n)]
    for (i,j,k) in triples:
        pairs[i].append((j,k)); pairs[j].append((i,k)); pairs[k].append((i,j))
    return triples,pairs

def construct():
    t0=time.time()
    DEADLINE=85.0
    n=180
    triples,pairs=_precompute(n)
    random.seed(12345)
    
    def count_violations(S):
        c=0
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]: c+=1
        return c
    
    def greedy_add(S, size):
        improved=True
        while improved:
            improved=False
            for v in range(n):
                if not S[v]:
                    ok=True
                    for (u,w) in pairs[v]:
                        if S[u] and S[w]:
                            ok=False; break
                    if ok:
                        S[v]=1; size+=1; improved=True
        return S, size
    
    def repair(S):
        while True:
            found = None
            for (i,j,k) in triples:
                if S[i] and S[j] and S[k]:
                    found = (i,j,k)
                    break
            if not found:
                break
            i,j,k = found
            counts = {}
            for v in (i,j,k):
                c = 0
                for (u,w) in pairs[v]:
                    if S[u] and S[w]: c+=1
                counts[v] = c
            to_remove = min((i,j,k), key=lambda v: counts[v])
            S[to_remove] = 0
        return S
    
    best = [0]*n
    best_size = 0
    
    def run_sa(S_init, time_budget):
        nonlocal best, best_size
        S = S_init[:]
        S = repair(S)
        cur_v = count_violations(S)
        cur_size = sum(S)
        S, cur_size = greedy_add(S, cur_size)
        
        if cur_v == 0 and cur_size > best_size:
            best = S[:]
            best_size = cur_size
        
        T = 1.5
        step = 0
        t_start = time.time()
        while time.time() - t_start < time_budget and time.time() - t0 < DEADLINE:
            step += 1
            v = random.randrange(n)
            old = S[v]
            if old == 0:
                addc = 0
                for (u,w) in pairs[v]:
                    if S[u] and S[w]: addc += 1
                new_v = cur_v + addc
                new_size = cur_size + 1
            else:
                remc = 0
                for (u,w) in pairs[v]:
                    if S[u] and S[w]: remc += 1
                new_v = cur_v - remc
                new_size = cur_size - 1
            
            new_score = new_size - 1000.0 * new_v
            cur_score = cur_size - 1000.0 * cur_v
            dE = new_score - cur_score
            
            if dE >= 0 or random.random() < math.exp(dE / T):
                S[v] = 1 - old
                cur_v = new_v
                cur_size = new_size
                if new_v == 0 and cur_size > best_size:
                    best = S[:]
                    best_size = cur_size
            
            T *= 0.99995
            if T < 0.02: T = 0.02
        
        return S, cur_size
    
    # Diverse starts
    starts = []
    S1 = [0]*n
    for i in range(90, 180): S1[i] = 1
    starts.append(S1)
    
    S2 = [0]*n
    for i in range(n):
        if (i+1) % 2 == 0: S2[i] = 1
    starts.append(S2)
    
    S3 = [0]*n
    for i in range(100, 180): S3[i] = 1
    starts.append(S3)
    
    S4 = [0]*n
    for i in range(60, 180): S4[i] = 1
    starts.append(S4)
    
    S5 = [0]*n
    for i in range(n):
        if random.random() < 0.55: S5[i] = 1
    starts.append(S5)
    
    S6 = [0]*n
    for i in range(0, 90): S6[i] = 1
    for i in range(140, 180): S6[i] = 1
    starts.append(S6)
    
    # Run SA on each start with time budget
    for idx, S_init in enumerate(starts):
        if time.time() - t0 > DEADLINE:
            break
        time_budget = min(10.0, DEADLINE - (time.time() - t0))
        run_sa(S_init, time_budget)
    
    # Perturbation restarts from best
    while time.time() - t0 < DEADLINE - 5:
        S = best[:]
        # Perturb: flip ~15% of elements
        for i in range(n):
            if random.random() < 0.15:
                S[i] = 1 - S[i]
        time_budget = min(5.0, DEADLINE - (time.time() - t0))
        if time_budget < 1.0:
            break
        run_sa(S, time_budget)
    
    # Final greedy on best
    Sb = best[:]
    sz = sum(Sb)
    Sb, sz = greedy_add(Sb, sz)
    if sz > best_size:
        best = Sb
        best_size = sz
    
    out = [i+1 for i in range(n) if best[i]]
    if not out: out = [180]
    return out
