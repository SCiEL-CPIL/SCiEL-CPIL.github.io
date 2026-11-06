import numpy as np
import time

def _topo_order(n, succs):
    # Kahn
    indeg = [0]*n
    for i in range(n):
        for s in succs[i]:
            indeg[s] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i]==0])
    order=[]
    while q:
        u=q.popleft()
        order.append(u)
        for s in succs[u]:
            indeg[s]-=1
            if indeg[s]==0:
                q.append(s)
    return order

def priority(instance):
    t0=time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = instance.get("predecessors", None)
    if preds is None:
        # fallback LPT
        return durations.copy()
    # normalize preds to list of lists
    succs=[[] for _ in range(n)]
    pred_lists=[]
    for i in range(n):
        p = preds[i] if i < len(preds) else []
        # p may be list/tuple/array
        pl = list(p) if p is not None else []
        pred_lists.append(pl)
        for j in pl:
            if 0 <= j < n and j != i:
                succs[j].append(i)
    order=_topo_order(n, succs)
    if len(order)!=n:
        # cycle or issue -> fallback
        return durations.copy()
    # forward pass
    ES=np.zeros(n); EF=np.zeros(n)
    for u in order:
        es=0.0
        for j in pred_lists[u]:
            if EF[j] > es: es=EF[j]
        ES[u]=es
        EF[u]=es+durations[u]
    T=float(max(EF)) if n else 0.0
    # backward pass
    LF=np.full(n, T); LS=np.full(n, T)
    for u in reversed(order):
        lf=T
        for s in succs[u]:
            if LS[s] < lf: lf=LS[s]
        LF[u]=lf
        LS[u]=lf-durations[u]
    TF=LS-ES
    # downstream transitive count
    down=np.zeros(n, dtype=int)
    # reverse topological: down[u] = number of distinct successors reachable
    # do BFS per node (n small)
    for u in range(n):
        seen=set()
        stack=list(succs[u])
        while stack:
            v=stack.pop()
            if v in seen: continue
            seen.add(v)
            stack.extend(succs[v])
        down[u]=len(seen)
    # deadline guard
    if time.time()-t0 > 140.0:
        return np.asarray(-TF, dtype=float)
    dmax=float(durations.max()) if n else 1.0
    downmax=float(down.max()) if n else 1.0
    if dmax<=0: dmax=1.0
    if downmax<=0: downmax=1.0
    priority = -TF + 0.1*(durations/dmax) + 0.1*(down/downmax)
    return np.asarray(priority, dtype=float)
