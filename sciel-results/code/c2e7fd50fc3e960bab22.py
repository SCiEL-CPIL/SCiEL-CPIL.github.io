def pebble(dag, fast_memory_budget):
    import time
    t0 = time.time()
    DEADLINE = 80.0

    # ---------- Parse ----------
    def get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    parents = None
    for pk in ('parents','preds','predecessors','ancestors','in_edges','deps','dependencies'):
        p = get(dag, pk)
        if p is not None:
            parents = p; break
    outputs = None
    for ok in ('outputs','sinks','targets','output_vertices','results','out'):
        o = get(dag, ok)
        if o is not None:
            outputs = o; break
    vertices = None
    for vk in ('vertices','nodes','V','vertex_list','node_list','all_vertices'):
        v = get(dag, vk)
        if v is not None:
            vertices = v; break

    if parents is None:
        if isinstance(dag, dict):
            vals = list(dag.values())[:10]
            if vals and all(isinstance(x,(list,tuple,set)) for x in vals):
                parents = dag
    if parents is None:
        raise ValueError("no parents")

    # normalize parents to dict v -> list
    if not isinstance(parents, dict):
        # maybe list of (v, parents)
        d = {}
        try:
            for item in parents:
                if isinstance(item,(list,tuple)) and len(item)==2:
                    d[item[0]] = list(item[1])
        except TypeError:
            # If parents itself is not iterable (e.g. int), this catches it
            pass
        parents = d

    # all vertices
    allv = set(parents.keys())
    for ps in parents.values():
        # Ensure ps is iterable
        if isinstance(ps, (list, tuple, set)):
            for p in ps:
                allv.add(p)
        elif isinstance(ps, int):
            # If it's an int, assume it's a single parent ID
            allv.add(ps)
        # Ignore other types for vertex collection if ambiguous
    
    if vertices is not None:
        allv |= set(vertices)

    # Build parent_map with robust type handling
    parent_map = {}
    for v in allv:
        ps = parents.get(v, [])
        if ps is None:
            parent_map[v] = []
        elif isinstance(ps, (list, tuple, set)):
            parent_map[v] = list(ps)
        elif isinstance(ps, int):
            parent_map[v] = [ps]
        else:
            # Try to convert if possible, else empty
            try:
                parent_map[v] = list(ps)
            except TypeError:
                parent_map[v] = []
    # Ensure all keys in parents are in allv/parent_map
    for v, ps in parents.items():
        if v not in parent_map:
            parent_map[v] = []
            allv.add(v)
            if isinstance(ps, (list, tuple, set)):
                for p in ps:
                    allv.add(p)
            elif isinstance(ps, int):
                allv.add(ps)

    # inputs
    inputs = {v for v in allv if not parent_map[v]}

    # children
    children = {v: [] for v in allv}
    for v, ps in parent_map.items():
        for p in ps:
            if p in children:
                children[p].append(v)

    if outputs is None:
        outputs = [v for v in allv if not children[v]]
    outputs = list(outputs)

    # Robust B
    try:
        B = max(1, int(fast_memory_budget))
    except (ValueError, TypeError):
        B = 1

    # ---------- Precompute ancestors & topological order ----------
    # topological order
    in_deg = {v: len(parent_map[v]) for v in allv}
    from collections import deque
    q = deque([v for v in allv if in_deg[v]==0])
    topo = []
    indeg = dict(in_deg)
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg[c]-=1
            if indeg[c]==0:
                q.append(c)
    # if cycle, append rest
    if len(topo) < len(allv):
        for v in allv:
            if v not in topo:
                topo.append(v)

    # ancestors (for eviction heuristic) - compute via topo reverse
    ancestors = {v: set() for v in allv}
    for v in reversed(topo):
        s = set()
        for p in parent_map[v]:
            if p in ancestors:
                s.add(p)
                s |= ancestors[p]
        ancestors[v] = s

    # remaining outputs set
    outputs_remaining = set(outputs)

    # ---------- Pebbling ----------
    fast = set()
    slow = set()   # computed & stored
    ops = []
    transfers = 0

    def is_input(v):
        return v in inputs

    def can_load(v):
        return v in slow or is_input(v)

    def recompute_size(v):
        return len(ancestors[v]) + 1

    def remaining_uses(v):
        # number of outputs in outputs_remaining that have v as ancestor (or v itself)
        cnt = 0
        for o in outputs_remaining:
            if v == o or v in ancestors[o]:
                cnt += 1
        return cnt

    def make_room():
        nonlocal transfers
        while len(fast) >= B:
            # choose victim
            best = None
            best_key = None
            for w in fast:
                # victim cost: 0 if input (delete), else 1 (store)
                now = 0 if is_input(w) else 1
                # prefer small now, then small remaining_uses, then small recompute
                ru = remaining_uses(w)
                rc = recompute_size(w)
                key = (now, ru, rc, str(w))
                if best_key is None or key < best_key:
                    best_key = key
                    best = w
            if best is None:
                # no victim (B==0?) -> shouldn't happen since B>=1 and we check len>=B
                break
            # evict best
            if is_input(best):
                ops.append(['delete', best])
                fast.discard(best)
                # inputs always loadable
            else:
                ops.append(['store', best])
                transfers += 1
                fast.discard(best)
                slow.add(best)

    def ensure(v):
        nonlocal transfers
        if v in fast:
            return
        if can_load(v):
            make_room()
            ops.append(['load', v])
            transfers += 1
            fast.add(v)
            slow.discard(v)
            return
        # compute v
        for p in parent_map[v]:
            ensure(p)
        make_room()
        ops.append(['compute', v])
        fast.add(v)

    # Process outputs one by one: ensure then store
    for o in outputs:
        if time.time() - t0 > DEADLINE:
            # emergency: finalize what we can
            break
        ensure(o)
        # store output
        if o in fast:
            ops.append(['store', o])
            transfers += 1
            fast.discard(o)
            slow.add(o)
        outputs_remaining.discard(o)

    # If deadline broke early, ensure remaining outputs quickly (best effort)
    for o in outputs:
        if o not in slow:
            ensure(o)
            if o in fast:
                ops.append(['store', o])
                transfers += 1
                fast.discard(o)
                slow.add(o)
        if time.time() - t0 > DEADLINE:
            break

    return ops
