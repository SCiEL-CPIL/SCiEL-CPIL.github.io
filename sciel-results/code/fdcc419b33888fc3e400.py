# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    from collections import deque

    # ---- extract graph structure robustly ----
    parents = None
    outputs = None
    vertices_hint = None
    if isinstance(dag, dict):
        for k in ('parents','edges','preds','in','predecessors'):
            if k in dag and isinstance(dag[k], dict):
                parents = dag[k]
                break
        if parents is None:
            # maybe dag itself is the parents map
            if dag and all(isinstance(val, (list, tuple, set, frozenset)) for val in dag.values()):
                parents = dag
        for k in ('outputs','sinks','out','targets','results','final'):
            if k in dag:
                outputs = list(dag[k])
                break
        for k in ('vertices','nodes','all','inputs','keys','V'):
            if k in dag:
                vertices_hint = list(dag[k])
                break
    else:
        for k in ('parents','edges','preds','in','predecessors'):
            if hasattr(dag, k):
                parents = getattr(dag, k)
                break
        for k in ('outputs','sinks','out','targets','results','final'):
            if hasattr(dag, k):
                outputs = list(getattr(dag, k))
                break
        for k in ('vertices','nodes','all','inputs','keys','V'):
            if hasattr(dag, k):
                vertices_hint = list(getattr(dag, k))
                break

    if parents is None:
        return []
    if not isinstance(parents, dict):
        try:
            parents = dict(parents)
        except Exception:
            return []
    parents = {v: list(p) for v, p in parents.items()}

    allv = set(parents.keys())
    for p in parents.values():
        for x in p:
            allv.add(x)
    if vertices_hint:
        for x in vertices_hint:
            allv.add(x)

    children = {v: set() for v in allv}
    for v, p in parents.items():
        for x in p:
            children.setdefault(x, set()).add(v)

    if outputs is None:
        outputs = [v for v in allv if not children.get(v)]
    else:
        for v in outputs:
            allv.add(v)
            children.setdefault(v, set())
            parents.setdefault(v, [])

    inputs = [v for v in allv if not parents.get(v)]
    inputs_set = set(inputs)

    # ---- topological order ----
    indeg = {v: len(parents.get(v, [])) for v in allv}
    q = deque([v for v in allv if indeg[v] == 0])
    topo = []
    indeg2 = dict(indeg)
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children.get(u, ()):
            indeg2[c] = indeg2.get(c, 0) - 1
            if indeg2[c] == 0:
                q.append(c)
    for v in allv:
        if v not in topo:
            topo.append(v)

    budget = max(0, int(fast_memory_budget))

    # ---- which vertices to keep in slow memory (store) ----
    store_set = set(outputs)
    for v in allv:
        if v in inputs_set:
            continue
        if len(children.get(v, ())) >= 2:
            store_set.add(v)
    if budget < len(allv):
        for v in allv:
            if v not in inputs_set:
                store_set.add(v)

    fast = set()
    stored = set()
    ops = []

    def load(v):
        if len(ops) >= 20000:
            return
        ops.append(['load', v])
        fast.add(v)

    def storev(v):
        if len(ops) >= 20000:
            return
        ops.append(['store', v])
        fast.discard(v)
        stored.add(v)

    def deletev(v):
        if len(ops) >= 20000:
            return
        ops.append(['delete', v])
        fast.discard(v)

    def computev(v):
        if len(ops) >= 20000:
            return False
        ops.append(['compute', v])
        fast.add(v)
        return True

    def evict():
        # evict a fast value not in outputs, fewest remaining children
        cand = [x for x in fast if x not in outputs]
        if not cand:
            cand = [x for x in fast]
            if not cand:
                return
        best = min(cand, key=lambda x: len(children.get(x, ())))
        storev(best)

    import sys
    sys.setrecursionlimit(1000000)

    def ensure(v):
        if v in fast:
            return
        if v in stored:
            load(v)
            return
        if v in inputs_set:
            load(v)
            return
        for p in parents.get(v, ()):
            ensure(p)
        if computev(v):
            if v in store_set:
                storev(v)
        while len(fast) > budget and len(ops) < 19995:
            evict()

    for v in topo:
        if v in inputs_set:
            continue
        ensure(v)
        if len(ops) >= 19990:
            break

    # ensure all outputs finish blue (in slow memory)
    for v in outputs:
        if v in fast:
            storev(v)
        elif v not in stored:
            ensure(v)
            if v in fast:
                storev(v)

    for v in list(fast):
        if v not in outputs:
            deletev(v)

    if len(ops) > 20000:
        ops = ops[:20000]
    return ops
# EVOLVE-BLOCK-END
