import time
from collections import deque
import sys

def pebble(dag, fast_memory_budget):
    start = time.time()
    deadline = start + 85.0
    
    if isinstance(dag, dict):
        parents = dag['parents']
        outputs = list(dag['output_vertices'])
    else:
        parents = dag.parents
        outputs = list(dag.output_vertices)
    
    if not outputs:
        return []
    
    vertices = list(parents.keys())
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(v)
    
    inputs = set(v for v in vertices if len(parents[v]) == 0)
    output_set = set(outputs)
    
    needed = set()
    stack = list(outputs)
    while stack:
        u = stack.pop()
        if u in needed:
            continue
        needed.add(u)
        for p in parents[u]:
            if p not in needed:
                stack.append(p)
    
    indeg = {v: len([p for p in parents[v] if p in needed]) for v in needed}
    q = deque([v for v in needed if indeg[v] == 0])
    topo = []
    indeg2 = dict(indeg)
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children[u]:
            if w in needed:
                indeg2[w] -= 1
                if indeg2[w] == 0:
                    q.append(w)
    if len(topo) != len(needed):
        topo = list(needed)
    
    pos = {v: i for i, v in enumerate(topo)}
    farthest = {}
    for v in reversed(topo):
        m = pos[v]
        for w in children[v]:
            if w in needed:
                m = max(m, farthest.get(w, pos[w]))
        farthest[v] = m
    
    ops = []
    fast = set()
    computed = set()  # values that have been computed (and are in fast or slow memory)
    
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 200000))
    
    def evict():
        nonlocal fast
        while len(fast) > fast_memory_budget:
            # Choose victim: prefer to delete computed values, store inputs
            # Evict the value with the largest farthest (least urgent)
            candidates = list(fast)
            if not candidates:
                break
            victim = max(candidates, key=lambda x: farthest.get(x, 0))
            if victim in inputs:
                ops.append(['store', victim])
            else:
                ops.append(['delete', victim])
            fast.discard(victim)
    
    def need(v):
        nonlocal ops, fast, computed
        if time.time() > deadline:
            return
        if v in fast:
            return
        if v in inputs:
            ops.append(['load', v])
            fast.add(v)
        else:
            for p in parents[v]:
                if p in needed:
                    need(p)
            ops.append(['compute', v])
            fast.add(v)
            computed.add(v)
        evict()
    
    for o in outputs:
        need(o)
        # Store output to make it blue
        if o in fast:
            ops.append(['store', o])
            fast.discard(o)
    
    sys.setrecursionlimit(old_limit)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
