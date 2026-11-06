"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
    
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    topo_order = list(vertices)
    pos = {v: i for i, v in enumerate(topo_order)}
    
    # Last use position for each vertex
    last_use = {}
    for v in vertices:
        if children[v]:
            last_use[v] = max(pos[c] for c in children[v])
        else:
            last_use[v] = -1
    
    blue = set(inputs)
    red = set()
    computed = set(inputs)
    ops = []
    
    for i, v in enumerate(topo_order):
        if time.time() - start_time > DEADLINE:
            break
        if v in computed:
            continue
        
        for p in parents.get(v, []):
            if p not in blue:
                if p in red:
                    ops.append(['load', p])
                    blue.add(p)
                    red.discard(p)
        
        ops.append(['compute', v])
        computed.add(v)
        blue.add(v)
        
        while len(blue) > fast_memory_budget:
            candidates = [x for x in blue if x not in outputs]
            if not candidates:
                candidates = list(blue)
            
            def prio(x):
                lu = last_use.get(x, -1)
                if lu < i:
                    return (0, 0)
                return (1, -lu)
            
            to_evict = min(candidates, key=prio)
            ops.append(['store', to_evict])
            blue.discard(to_evict)
            red.add(to_evict)
    
    for v in outputs:
        if v not in blue:
            if v in red:
                ops.append(['load', v])
                blue.add(v)
                red.discard(v)
    
    if time.time() - start_time > DEADLINE:
        return ops[:20000]
    return ops[:20000]
# EVOLVE-BLOCK-END
