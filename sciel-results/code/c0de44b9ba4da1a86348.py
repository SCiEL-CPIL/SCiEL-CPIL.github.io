def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 80

    vertices = dag.get('vertices', list(range(dag.get('n', 0))))
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))

    top_order = dag.get('top_order', None)
    if top_order is None:
        in_degree = {v: 0 for v in vertices}
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children[p].append(v)
                in_degree[v] += 1
        from collections import deque
        q = deque([v for v in vertices if in_degree[v] == 0])
        top_order = []
        while q:
            v = q.popleft()
            top_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    q.append(c)
        if len(top_order) != len(vertices):
            top_order = list(vertices)

    fan_out = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            fan_out[p] += 1

    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(top_order):
        for p in parents.get(v, []):
            last_use[p] = i

    blue = set(inputs)
    red = set()
    operations = []

    def do_load(v):
        nonlocal blue, red
        operations.append(['load', v])
        blue.discard(v)
        red.add(v)

    def do_store(v):
        nonlocal blue, red
        operations.append(['store', v])
        red.discard(v)
        blue.add(v)

    def do_compute(v):
        nonlocal red
        operations.append(['compute', v])
        red.add(v)

    def do_delete(v):
        nonlocal red
        operations.append(['delete', v])
        red.discard(v)

    def ensure_red(v):
        nonlocal red, blue
        if v in red:
            return
        if v in blue:
            if len(red) >= fast_memory_budget:
                candidates = [x for x in red if x not in outputs]
                if not candidates:
                    candidates = list(red)
                if candidates:
                    to_evict = min(candidates, key=lambda x: fan_out.get(x, 0))
                    do_store(to_evict)
            do_load(v)
        else:
            do_load(v)

    for i, v in enumerate(top_order):
        if time.time() > deadline:
            break

        if len(parents.get(v, [])) > 0:
            for p in parents.get(v, []):
                ensure_red(p)

            if len(red) >= fast_memory_budget:
                candidates = [x for x in red if x not in outputs]
                if not candidates:
                    candidates = list(red)
                if candidates:
                    to_evict = min(candidates, key=lambda x: fan_out.get(x, 0))
                    do_store(to_evict)

            do_compute(v)

        for x in list(red):
            if x not in outputs and last_use.get(x, -1) < i:
                do_delete(x)

    for v in outputs:
        if v in red:
            do_store(v)
        elif v not in blue:
            blue.add(v)

    return operations
