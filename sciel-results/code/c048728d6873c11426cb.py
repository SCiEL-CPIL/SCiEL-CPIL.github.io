import random, time

N=16
ALL_PAIRS=[(i,j) for i in range(N) for j in range(i+1,N)]
NOT_LE=[~((1<<(v+1))-1) for v in range(N)]

def _calc_score(adj,deg):
    best=0
    for u in range(N):
        au=adj[u]; du=deg[u]
        for v in range(u+1,N):
            if (au>>v)&1:
                c=au & adj[v] & NOT_LE[v]
                dv=deg[v]
                while c:
                    lsb=c & -c
                    w=lsb.bit_length()-1
                    s=du+dv+deg[w]
                    if s>best: best=s
                    c^=lsb
    return best

def _adj_deg(edges):
    adj=[0]*N
    for u,v in edges:
        adj[u]|=1<<v; adj[v]|=1<<u
    deg=[adj[u].bit_count() for u in range(N)]
    return adj,deg

def _build_24():
    edges=[]
    for a in range(10):
        for b in range(10,16):
            edges.append((a,b))
    for i in range(0,10,2):
        edges.append((i,i+1))
    return edges

def _local_search(edges, stop_time, best_score):
    edges=list(edges)
    edge_set=set(edges)
    adj,deg=_adj_deg(edges)
    cur=_calc_score(adj,deg)
    best=cur; best_edges=list(edges)
    if best<best_score: best_score=best
    no_improve=0
    it=0
    while time.time()<stop_time:
        it+=1
        # choose move size
        two = (random.random()<0.25)
        if two:
            # remove 2
            i1=random.randrange(len(edges)); e1=edges[i1]
            i2=random.randrange(len(edges)-1); 
            if i2>=i1: i2+=1
            e2=edges[i2]
            remove=[e1,e2]
            # add 2 non-edges
            add=[]
            while len(add)<2:
                a=random.randrange(N); b=random.randrange(N)
                if a==b: continue
                if a>b: a,b=b,a
                f=(a,b)
                if f in edge_set or f in remove or f in add: continue
                add.append(f)
        else:
            i=random.randrange(len(edges)); e1=edges[i]
            remove=[e1]
            add=[]
            while True:
                a=random.randrange(N); b=random.randrange(N)
                if a==b: continue
                if a>b: a,b=b,a
                f=(a,b)
                if f in edge_set or f in remove: continue
                add.append(f); break
        na,nd=_swap_edges(adj,deg,remove,add)
        ns=_calc_score(na,nd)
        if ns<=cur:
            # accept
            for u,v in remove:
                edge_set.discard((u,v))
            for u,v in add:
                edge_set.add((u,v))
            # update edges list
            for u,v in remove:
                edges.remove((u,v))
            for u,v in add:
                edges.append((u,v))
            adj,deg=na,nd
            cur=ns
            no_improve=0
            if cur<best:
                best=cur; best_edges=list(edges)
                if best<best_score: best_score=best
        else:
            no_improve+=1
            if no_improve>5000:
                break
    return best,best_edges

def construct():
    t0=time.time()
    deadline=t0+140.0
    random.seed(42)
    best_edges=_build_24()
    adj,deg=_adj_deg(best_edges)
    best_score=_calc_score(adj,deg)  # 24
    # also try a few structured/random starts to set best
    starts=[best_edges]
    for _ in range(8):
        starts.append(random.sample(ALL_PAIRS,65))
    for st in starts:
        if time.time()>deadline-5: break
        adj,deg=_adj_deg(st)
        s=_calc_score(adj,deg)
        if s<best_score:
            best_score=s; best_edges=list(st)
    # local search rounds
    while time.time()<deadline-3:
        r=random.random()
        if r<0.4:
            st=list(best_edges)
        elif r<0.7:
            st=random.sample(ALL_PAIRS,65)
        else:
            # perturb best
            st=list(best_edges)
            for _ in range(random.randint(2,4)):
                # random swap
                i=random.randrange(len(st)); e1=st[i]
                while True:
                    a=random.randrange(N); b=random.randrange(N)
                    if a==b: continue
                    if a>b: a,b=b,a
                    if (a,b) in set(st): continue
                    break
                st[i]=(a,b)
        stop=min(deadline-2, time.time()+25.0)
        s,e=_local_search(st, stop, best_score)
        if s<best_score:
            best_score=s; best_edges=e
        if best_score<=23:
            break
    return {'n':N,'edges':[[u,v] for u,v in best_edges]}
