import time
import random
import numpy as np
from itertools import combinations

def _build():
    n=16
    edges=[]; eidx={}
    for i in range(n):
        for j in range(i+1,n):
            eidx[(i,j)]=len(edges); edges.append((i,j))
    k4s=[]
    for combo in combinations(range(n),4):
        a,b,c,d=combo
        k4s.append([eidx[(a,b)],eidx[(a,c)],eidx[(a,d)],eidx[(b,c)],eidx[(b,d)],eidx[(c,d)]])
    incident=[[] for _ in range(len(edges))]
    for ki,six in enumerate(k4s):
        for e in six: incident[e].append(ki)
    return edges,k4s,incident

_EDGES,_K4S,_INC=_build()
_K4=np.array(_K4S,dtype=np.int64)  # shape (1820,6)

def _init_state(C, seed=12345):
    rng=random.Random(seed)
    st=np.empty(len(_EDGES),dtype=np.int8)
    k=0
    for (i,j) in _EDGES:
        st[k]=(i+j)%C; k+=1
    return st

def _viol(state):
    # distinct per k4
    cols=state[_K4]  # (1820,6)
    # count distinct rows
    # use: for each row, number of distinct
    d=np.empty(_K4.shape[0],dtype=np.int8)
    for r in range(_K4.shape[0]):
        d[r]=len(set(cols[r].tolist()))
    return np.maximum(0,5-d).sum()

def _verify(state,C):
    cols=state[_K4]
    for r in range(_K4.shape[0]):
        if len(set(cols[r].tolist()))<5:
            return False
    return True

def _search(C, deadline, seed=0):
    rng=random.Random(seed)
    np_rng=np.random.RandomState(seed)
    st=_init_state(C,seed)
    total=int(_viol(st))
    if total==0:
        return st
    T=1.0
    cooling=0.9999
    while time.time()<deadline and total>0:
        e=np_rng.randint(0,len(_EDGES))
        old=st[e]
        new=int(np_rng.randint(0,C))
        if new==old:
            continue
        # incident k4s
        inc=_INC[e]
        before=0
        for ki in inc:
            before+=max(0,5-len(set(st[_K4[ki]].tolist())))
        st[e]=new
        after=0
        for ki in inc:
            after+=max(0,5-len(set(st[_K4[ki]].tolist())))
        delta=after-before
        if delta<=0:
            total+=delta
            T*=cooling
        else:
            if random.random()<np.exp(-delta/T):
                total+=delta
                T*=cooling
            else:
                st[e]=old
    if total==0 and _verify(st,C):
        return st
    return None

def construct():
    start=time.time()
    deadline=start+105.0
    best=None
    # try 16
    d1=start+40.0
    r=_search(16, d1, seed=1)
    if r is not None:
        best=r; bestC=16
    # try 15 with remaining
    if time.time()<deadline-2:
        r=_search(15, deadline, seed=2)
        if r is not None:
            best=r; bestC=15
    if best is not None:
        # return flat list
        return best.tolist()
    # fallback: additive mod 17 (valid, 17 colours)
    return [(i+j)%17 for i in range(16) for j in range(i+1,16)]
