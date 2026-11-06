# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Returns list of [operation, vertex] with operations: load, store, compute, delete
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = list(dag.get('vertices', []))
    parents = dict(dag.get('parents', {}))
    outputs = list(dag.get('outputs', []))
    inputs = list(dag.get('inputs', []))
    
    if not vertices:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    # Determine input vertices (no parents or explicitly listed)
    input_set = set(inputs) if inputs else set()
    for v in vertices:
        if v not in parents or len(parents.get(v, [])) == 0:
            input_set.add(v)
    
    topo_order = list(vertices)
    
    # Find vertices that are ancestors of outputs
    output_set = set(outputs)
    needed = set(output_set)
    changed = True
    while changed:
        changed = False
        for v in list(needed):
            for p in parents.get(v, []):
                if p not in needed:
                    needed.add(p)
                    changed = True
    
    # Compute next-use index for each vertex
    vertex_to_idx = {v: i for i, v in enumerate(topo_order)}
    next_use = {}
    for v in vertices:
        if not children.get(v):
            next_use[v] = float('inf')
        else:
            idxs = [vertex_to_idx[c] for c in children[v] if c in vertex_to_idx]
            next_use[v] = min(idxs) if idxs else float('inf')
    
    # State
    fast_mem = set()
    on_disk = set()
    computed = set()
    operations = []
    
    def evict():
        if not fast_mem:
            return
        # Evict vertex with farthest next use; break ties by not-needed first
        def score(v):
            nu = next_use.get(v, float('inf'))
            not_needed = 1 if v not in needed else 0
            return (not_needed, nu)
        victim = max(fast_mem, key=score)
        fast_mem.discard(victim)
        on_disk.add(victim)
        operations.append(['store', victim])
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in input_set:
            if v not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evict()
                fast_mem.add(v)
                operations.append(['load', v])
            computed.add(v)
        else:
            ps = parents.get(v, [])
            for p in ps:
                if p not in fast_mem:
                    if p in on_disk:
                        if len(fast_mem) >= fast_memory_budget:
                            evict()
                        fast_mem.add(p)
                        operations.append(['load', p])
                    # else: shouldn't happen in topo order
            
            if len(fast_mem) >= fast_memory_budget:
                evict()
            fast_mem.add(v)
            operations.append(['compute', v])
            computed.add(v)
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if o not in fast_mem:
            if o in on_disk:
                if len(fast_mem) >= fast_memory_budget:
                    evict()
                fast_mem.add(o)
                operations.append(['load', o])
            else:
                operations.append(['compute', o])
                fast_mem.add(o)
    
    return operations
# EVOLVE-BLOCK-END
