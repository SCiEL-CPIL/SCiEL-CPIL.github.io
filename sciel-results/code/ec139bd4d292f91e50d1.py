def _greedy(parents, outputs, inputs, budget, max_ops, t0, deadline):
    verts = sorted(parents.keys())
    n = len(verts)
    outputs_set = set(outputs)
    state = {v:0 for v in verts}
    for v in inputs:
        state[v]=1
    active = set(inputs)  # wait, inputs are blue (state 1), not red. active = red vertices.
    active = set()  # no red initially
    ops = []
    transfers = 0
    ops_count = 0
    # children
    children = {v:[] for v in verts}
    for v in verts:
        for p in parents[v]:
            children[p].append(v)
    remaining_children = {v:len(children[v]) for v in verts}
    # topo
    topo = _topo_sort(parents, verts)
    topo_pos = {v:i for i,v in enumerate(topo)}
    uncomputed = set(verts)
    # next use pos
    def next_use(v):
        m = 10**9
        for c in children[v]:
            if c in uncomputed:
                m = min(m, topo_pos[c])
        return m
    def make_room(protected):
        nonlocal active, ops, transfers, ops_count
        while len(active) >= budget:
            cands = [v for v in active if v not in protected]
            if not cands:
                raise RuntimeError("no room to free")
            never = [v for v in cands if remaining_children[v]==0 and v not in outputs_set]
            if never:
                v = never[0]
                state[v]=0; active.remove(v); ops.append(['delete',v]); ops_count+=1
            else:
                v = min(cands, key=lambda x: next_use(x))
                state[v]=1; active.remove(v); ops.append(['store',v]); ops_count+=1; transfers+=1
    sys.setrecursionlimit(10000)
    def ensure_red(v, protected):
        nonlocal state, active, ops, transfers, ops_count
        if state[v]==2:
            return
        # protect red ancestors of v
        prot = set(protected)
        # add red ancestors
        stack=[v]
        seen=set()
        while stack:
            x=stack.pop()
            for p in parents[x]:
                if p not in seen:
                    seen.add(p)
                    if state[p]==2:
                        prot.add(p)
                    stack.append(p)
        for p in parents[v]:
            ensure_red(p, prot)
        # make room for v
        make_room(prot | set(parents[v]))
        # compute v
        for p in parents[v]:
            if state[p]!=2:
                raise RuntimeError("parent not red")
        state[v]=2; active.add(v); ops.append(['compute',v]); ops_count+=1
    # main loop
    for v in topo:
        if time.time()-t0 > deadline-5:
            return None
        if ops_count > max_ops-100:
            return None
        if state[v]==2:
            uncomputed.discard(v)
            # update remaining_children of parents
            for p in parents[v]:
                remaining_children[p]-=1
            continue
        if state[v]==1:
            # it's blue, but we need it red to... wait, v is being processed, we need to compute it? No, v is blue means it was stored. But v should be computed once. If v is blue, it was computed and stored. But in topo order, v is processed once. If v is blue here, that means it was computed earlier? No, in topo order each vertex computed once. v blue means... inputs are blue initially. Non-inputs start gone. So v blue here means v is an input (blue initially) OR was stored. But we process each v once in topo, and compute it. If v is an input, we don't compute it (it's given). Hmm.
            ...
