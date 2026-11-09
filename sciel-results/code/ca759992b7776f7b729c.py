"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = dag.get('vertices', [])
    edges = dag.get('edges', [])
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    topo_order = dag.get('topo_order', vertices)
    
    if not vertices:
        return []
    
    children = defaultdict(list)
    parents = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)
    
    use_count = {v: len(children[v]) for v in vertices}
    for v in outputs:
        use_count[v] += 10000
    
    in_fast = set()
    in_slow = set(inputs)
    ops = []
    
    def ensure_in_fast(v):
        if v in in_fast:
            return
        if v in in_slow:
            if len(in_fast) >= fast_memory_budget:
                evict()
            ops.append(['load', v])
            in_fast.add(v)
        else:
            for p in parents[v]:
                ensure_in_fast(p)
            if len(in_fast) >= fast_memory_budget:
                evict()
            ops.append(['compute', v])
            in_fast.add(v)
    
    def evict():
        if not in_fast:
            return
        candidates = [v for v in in_fast if v not in outputs]
        if not candidates:
            return
        v = min(candidates, key=lambda x: use_count.get(x, 0))
        if v not in in_slow:
            ops.append(['store', v])
            in_slow.add(v)
        in_fast.discard(v)
        ops.append(['delete', v])
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        ensure_in_fast(v)
    
    for v in outputs:
        if v not in in_fast:
            ensure_in_fast(v)
    
    return ops

# EVOLVE-BLOCK-END
