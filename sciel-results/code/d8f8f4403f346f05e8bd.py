"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG with minimal transfers under fast memory budget.
    
    dag: dict with keys:
        - 'vertices': list of vertex IDs
        - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
        - 'outputs': list of output vertex IDs
        - 'inputs': list of input vertex IDs (start blue/in fast memory)
    fast_memory_budget: int, max number of red values in fast memory
    
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    # Parse DAG structure
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    if not vertices:
        return []
    
    # Build children map and in-degree for topological processing
    children = defaultdict(list)
    for v in vertices:
        par_list = parents.get(v, [])
        for p in par_list:
            children[p].append(v)
    
    # Determine topological order (parents already given in topo order per problem)
    # Compute for each vertex: how many times its value is needed (fan-out consideration)
    # and the "last use" position in a topological traversal
    
    topo_order = list(vertices)  # assume vertices are in topological order
    
    # Compute last-use index for each vertex in topological order
    # A vertex's value is needed when computing any of its children
    last_use = {}
    for idx, v in enumerate(topo_order):
        # The value of v is needed when computing its children
        max_child_idx = -1
        for c in children.get(v, []):
            # Find index of child in topo_order
            if c in topo_order:
                ci = topo_order.index(c)
                if ci > max_child_idx:
                    max_child_idx = ci
        last_use[v] = max_child_idx if max_child_idx >= 0 else idx
    
    # Greedy pebbling strategy:
    # - Process vertices in topological order
    # - For each vertex, ensure all parents are in fast memory (load if not)
    # - Compute the vertex (result goes to fast memory)
    # - If fast memory is full, evict the value with the farthest last_use
    # - At the end, ensure outputs are in fast memory
    
    # State: which vertices are currently in fast memory
    in_fast = set()
    operations = []
    
    # Initialize: inputs start in fast memory (blue)
    for v in inputs:
        in_fast.add(v)
    
    # Track which vertices have been computed
    computed = set(inputs)  # inputs are already "computed" (given)
    
    # Process non-input vertices in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
            
        if v in inputs:
            # Already in fast memory, skip
            continue
        
        # Ensure all parents are in fast memory
        par_list = parents.get(v, [])
        for p in par_list:
            if p not in in_fast:
                # Need to load p
                # First, if memory is full, evict something
                if len(in_fast) >= fast_memory_budget:
                    # Find eviction candidate: value with farthest last_use
                    best_evict = None
                    best_last_use = -1
                    for cand in in_fast:
                        lu = last_use.get(cand, 0)
                        if lu > best_last_use:
                            best_last_use = lu
                            best_evict = cand
                    if best_evict is not None:
                        in_fast.discard(best_evict)
                        operations.append(['store', best_evict])
                
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        
        # v's result is now in fast memory
        in_fast.add(v)
        
        # If memory exceeded, evict
        while len(in_fast) > fast_memory_budget:
            # Find eviction candidate: value with farthest last_use that isn't an output
            best_evict = None
            best_last_use = -1
            for cand in in_fast:
                if cand in outputs:
                    continue  # don't evict outputs
                lu = last_use.get(cand, 0)
                if lu > best_last_use:
                    best_last_use = lu
                    best_evict = cand
            if best_evict is not None:
                in_fast.discard(best_evict)
                operations.append(['store', best_evict])
            else:
                # All remaining are outputs; evict one with farthest last_use anyway
                best_evict = None
                best_last_use = -1
                for cand in in_fast:
                    lu = last_use.get(cand, 0)
                    if lu > best_last_use:
                        best_last_use = lu
                        best_evict = cand
                if best_evict is not None:
                    in_fast.discard(best_evict)
                    operations.append(['store', best_evict])
                break
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict something non-output
                best_evict = None
                best_last_use = -1
                for cand in in_fast:
                    if cand in outputs:
                        continue
                    lu = last_use.get(cand, 0)
                    if lu > best_last_use:
                        best_last_use = lu
                        best_evict = cand
                if best_evict is not None:
                    in_fast.discard(best_evict)
                    operations.append(['store', best_evict])
                else:
                    # Evict any non-output or just proceed
                    for cand in list(in_fast):
                        if cand not in outputs:
                            in_fast.discard(cand)
                            operations.append(['store', cand])
                            break
            
            operations.append(['load', v])
            in_fast.add(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
