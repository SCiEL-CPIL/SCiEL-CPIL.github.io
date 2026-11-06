import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85

    def parse_dag(dag):
        vertices = None
        parents = None
        inputs = None
        outputs = None

        if isinstance(dag, dict):
            if 'vertices' in dag:
                vertices = list(dag['vertices'])
            if 'parents' in dag:
                parents = {k: list(v) for k, v in dag['parents'].items()}
            elif 'edges' in dag:
                edges = dag['edges']
                parents = defaultdict(list)
                for e in edges:
                    if len(e) == 2:
                        a, b = e
                        parents[a].append(b)
                for k in list(parents.keys()):
                    for p in parents[k]:
                        if p not in parents:
                            parents[p] = []
            if 'inputs' in dag:
                inputs = list(dag['inputs'])
            if 'outputs' in dag:
                outputs = list(dag['outputs'])
        elif isinstance(dag, list):
            parents = {i: list(dag[i]) for i in range(len(dag))}
            vertices = list(range(len(dag)))

        if vertices is None:
            if parents:
                all_v = set()
                for v in parents:
                    all_v.add(v)
                    for p in parents[v]:
                        all_v.add(p)
                vertices = sorted(all_v)
            else:
                return None

        if parents is None:
            parents = {v: [] for v in vertices}

        if inputs is None:
            inputs = [v for v in vertices if not parents[v]]

        if outputs is None:
            children = defaultdict(list)
            for v in vertices:
                for p in parents[v]:
                    children[p].append(v)
            outputs = [v for v in vertices if not children[v]]

        return vertices, parents, inputs, outputs

    parsed = parse_dag(dag)
    if parsed is None:
        return []

    vertices, parents, inputs, outputs = parsed

    for v in vertices:
        if v not in parents:
            parents[v] = []

    in_degree = {v: len(parents[v]) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)

    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        return []

    last_use = {v: -1 for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents[v]:
            last_use[p] = i
    for v in outputs:
        last_use[v] = len(topo_order) - 1

    operations = []
    fast_mem = set(inputs)
    slow_mem = set()

    for i, v in enumerate(topo_order):
        if time.time() - start_time > DEADLINE:
            break

        if v in inputs:
            continue

        for p in parents[v]:
            if p not in fast_mem:
                if p in slow_mem:
                    operations.append(['load', p])
                    fast_mem.add(p)
                    slow_mem.discard(p)
                else:
                    operations.append(['load', p])
                    fast_mem.add(p)

        operations.append(['compute', v])
        fast_mem.add(v)

        while len(fast_mem) > fast_memory_budget:
            candidates = [x for x in fast_mem if last_use[x] < i and x not in outputs]
            if candidates:
                x = min(candidates, key=lambda x: last_use[x])
                operations.append(['delete', x])
                fast_mem.discard(x)
            else:
                candidates = [x for x in fast_mem if x not in outputs]
                if candidates:
                    x = min(candidates, key=lambda x: last_use[x])
                    operations.append(['store', x])
                    fast_mem.discard(x)
                    slow_mem.add(x)
                else:
                    break

    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem:
                operations.append(['load', v])
                fast_mem.add(v)
                slow_mem.discard(v)
            else:
                operations.append(['load', v])
                fast_mem.add(v)

    def validate(ops):
        fast = set(inputs)
        slow = set()

        for op, v in ops:
            if op == 'load':
                if v not in slow or v in fast or len(fast) >= fast_memory_budget:
                    return False
                fast.add(v)
                slow.discard(v)
            elif op == 'store':
                if v not in fast or v in slow:
                    return False
                slow.add(v)
                fast.discard(v)
            elif op == 'compute':
                for p in parents[v]:
                    if p not in fast:
                        return False
                if v not in fast and len(fast) >= fast_memory_budget:
                    return False
                fast.add(v)
            elif op == 'delete':
                if v not in fast:
                    return False
                fast.discard(v)
            else:
                return False

        for v in outputs:
            if v not in fast:
                return False

        return True

    if not validate(operations):
        operations = []
        fast = set(inputs)
        slow = set()

        for v in topo_order:
            if v in inputs:
                continue
            for p in parents[v]:
                if p not in fast:
                    if p in slow:
                        operations.append(['load', p])
                        fast.add(p)
                        slow.discard(p)
            operations.append(['compute', v])
            fast.add(v)
            while len(fast) > fast_memory_budget:
                operations.append(['store', v])
                fast.discard(v)
                slow.add(v)
                break

        for v in outputs:
            if v not in fast:
                if v in slow:
                    operations.append(['load', v])
                    fast.add(v)
                    slow.discard(v)

        if not validate(operations):
            return []

    return operations
