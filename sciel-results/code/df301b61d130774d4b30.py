def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0

    parents = {}
    vertices = []
    outputs = []
    inputs = []

    if hasattr(dag, 'parents'):
        parents = dict(dag.parents)
        outputs = list(dag.outputs) if hasattr(dag, 'outputs') else []
        inputs = list(dag.inputs) if hasattr(dag, 'inputs') else []
        vertices = list(dag.vertices) if hasattr(dag, 'vertices') else list(parents.keys())
    elif isinstance(dag, dict):
        parents = dict(dag.get('parents', {}))
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))
        vertices = list(dag.get('vertices', list(parents.keys())))
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, (list, tuple)):
                v, ps = item[0], item[1]
                parents[v] = list(ps)
                vertices.append(v)
            else:
                vertices.append(item)
                parents[item] = []
        inputs = [v for v in vertices if len(parents[v]) == 0]

    for v in vertices:
        if v not in parents:
            parents[v] = []

    if not outputs:
        child_set = set()
        for v in parents:
            for p in parents[v]:
                child_set.add(p)
        outputs = [v for v in vertices if v not in child_set]

    if not inputs:
        inputs = [v for v in vertices if len(parents[v]) == 0]

    out_degree = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents[v]:
            out_degree[p] += 1

    state = {}
    for v in vertices:
        state[v] = 'blue' if v in inputs else 'gone'

    fast_mem = set()
    operations = []
    remaining_uses = dict(out_degree)

    def do_load(v):
        if state[v] == 'blue':
            operations.append(['load', v])
            state[v] = 'red'
            fast_mem.add(v)

    def do_store(v):
        if state[v] == 'red':
            operations.append(['store', v])
            state[v] = 'blue'
            fast_mem.discard(v)

    def do_compute(v):
        operations.append(['compute', v])
        state[v] = 'red'
        fast_mem.add(v)

    for v in vertices:
        if len(operations) >= 20000 or time.time() - start_time > DEADLINE:
            break

        for p in parents[v]:
            if state[p] == 'blue':
                if len(fast_mem) >= fast_memory_budget:
                    candidates = [u for u in fast_mem if u not in parents[v]]
                    if candidates:
                        evict_v = min(candidates, key=lambda u: remaining_uses.get(u, 0))
                        do_store(evict_v)
                    else:
                        for u in list(fast_mem):
                            do_store(u)
                            break
                do_load(p)

        do_compute(v)

        for p in parents[v]:
            remaining_uses[p] -= 1

        if v not in outputs and remaining_uses[v] == 0 and len(fast_mem) >= fast_memory_budget:
            do_store(v)

    for v in outputs:
        if state[v] == 'red':
            do_store(v)

    for v in list(fast_mem):
        if state[v] == 'red':
            do_store(v)

    return operations
