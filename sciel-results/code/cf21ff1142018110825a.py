"""Communication-minimizing DAG pebbling."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    MAX_OPS = 20000

    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    vertices = dag.get('vertices', [])
    budget = fast_memory_budget

    needed = set()
    def mark(v):
        if v in needed: return
        needed.add(v)
        for p in parents.get(v, []):
            mark(p)
    for o in outputs:
        mark(o)

    topo = [v for v in vertices if v in needed]
    children = {v: [] for v in topo}
    for v in topo:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)

    ops = []
    def emit(op, v):
        if len(ops) >= MAX_OPS:
            return
        ops.append([op, v])

    in_mem = set()

    for v in topo:
        if time.time() - start_time > DEADLINE:
            break

        if v in inputs:
            if v not in in_mem:
                if len(in_mem) >= budget:
                    deleted = False
                    for cand in list(in_mem):
                        needed_future = any(
                            c in needed and topo.index(c) > topo.index(v)
                            for c in children.get(cand, [])
                        ) if cand in set(children) else False
                        if not needed_future:
                            in_mem.discard(cand)
                            emit('delete', cand)
                            deleted = True
                            break
                    if not deleted and in_mem:
                        to_del = next(iter(in_mem))
                        in_mem.discard(to_del)
                        emit('delete', to_del)
                emit('load', v)
                in_mem.add(v)
        else:
            emit('compute', v)
            in_mem.add(v)

        if v in outputs:
            emit('store', v)
            in_mem.discard(v)

    return ops
# EVOLVE-BLOCK-END
