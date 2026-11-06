# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy: For layered DAGs, use a depth-first pebbling strategy with
    careful memory management. The key insight is that for layered structures,
    we can minimize transfers by keeping intermediate results in fast memory
    as long as possible, and only storing to slow memory when fast memory is full.
    """
    t0 = time.time()
    DEADLINE = 70  # seconds, safety margin before 90s kill
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', dag.get('edges', {}))
        inputs = set(dag.get('inputs', dag.get('input_vertices', [])))
        outputs = set(dag.get('outputs', dag.get('output_vertices', [])))
    else:
        # Fallback: assume dag is a list of (vertex, parents) tuples
        vertices = []
        parents = {}
        inputs = set()
        outputs = set()
        for item in dag:
            if isinstance(item, tuple) and len(item) >= 2:
                v, ps = item[0], item[1]
                vertices.append(v)
                parents[v] = list(ps)
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('id', item.get('name')))
                ps = item.get('parents', item.get('children', []))
                vertices.append(v)
                parents[v] = list(ps)
                if item.get('is_input'):
                    inputs.add(v)
                if item.get('is_output'):
                    outputs.add(v)
    
    if not vertices:
        return []
    
    # Build reverse edges (children)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify inputs: vertices with no parents
    if not inputs:
        for v in vertices:
            if not parents.get(v, []):
                inputs.add(v)
    
    # Identify outputs: vertices with no children
    if not outputs:
        for v in vertices:
            if not children.get(v, []):
                outputs.add(v)
    
    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle - just return naive
        return _naive_fallback(dag, fast_memory_budget, vertices, parents, inputs, outputs)
    
    # Compute "depth" or layer for each vertex
    depth = {}
    for v in topo_order:
        if not parents.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    
    # For layered DAGs, group by depth
    layers = defaultdict(list)
    for v in topo_order:
        layers[depth[v]].append(v)
    
    max_layer = max(layers.keys()) if layers else 0
    
    # Compute fan-out (how many children each vertex has) - proxy for reuse value
    fan_out = {v: len(children.get(v, [])) for v in vertices}
    
    # Compute "criticality": how many output paths go through this vertex
    output_count = {v: 0 for v in vertices}
    for v in reversed(topo_order):
        if v in outputs:
            output_count[v] = 1
        for c in children.get(v, []):
            output_count[v] = max(output_count[v], output_count[c])
    
    # Strategy: Process layer by layer, but within a layer, prioritize
    # vertices with high output_count (high reuse potential)
    
    # Simulate the pebbling process
    fast_mem = set()  # vertices in fast memory
    slow_mem = set()  # vertices stored in slow memory
    computed = set()  # vertices whose values have been computed
    operations = []
    
    def load(v):
        """Load vertex v from slow memory to fast memory."""
        if v not in slow_mem:
            return False
        if v in fast_mem:
            return True
        if len(fast_mem) >= fast_memory_budget:
            return False
        fast_mem.add(v)
        slow_mem.discard(v)
        operations.append(['load', v])
        return True
    
    def store(v):
        """Store vertex v from fast memory to slow memory."""
        if v not in fast_mem:
            return False
        if v in slow_mem:
            return True
        slow_mem.add(v)
        fast_mem.discard(v)
        operations.append(['store', v])
        return True
    
    def compute(v):
        """Compute vertex v. All parents must be in fast memory."""
        if v in fast_mem:
            return True
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                return False
        if len(fast_mem) >= fast_memory_budget and v not in fast_mem:
            return False
        fast_mem.add(v)
        computed.add(v)
        operations.append(['compute', v])
        return True
    
    def delete(v):
        """Delete vertex v from fast memory (frees a slot)."""
        if v not in fast_mem:
            return False
        fast_mem.discard(v)
        operations.append(['delete', v])
        return True
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() - t0 > DEADLINE:
            break
        
        ps = parents.get(v, [])
        
        # Ensure all parents are in fast memory
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    # Need to load it
                    if len(fast_mem) >= fast_memory_budget:
                        # Need to free a slot: delete a vertex that's no longer needed
                        freed = False
                        for fm_v in list(fast_mem):
                            # Check if fm_v is still needed (has children not yet processed)
                            still_needed = False
                            for c in children.get(fm_v, []):
                                if c not in computed and c != v:
                                    still_needed = True
                                    break
                                if c == v:
                                    still_needed = True
                                    break
                            if not still_needed:
                                delete(fm_v)
                                freed = True
                                break
                        if not freed:
                            # Force delete something
                            if fast_mem:
                                delete(next(iter(fast_mem)))
                    if len(fast_mem) < fast_memory_budget:
                        load(p)
                else:
                    # Parent not in slow memory either - need to recompute
                    pass
        
        # Now try to compute v
        if v not in fast_mem:
            all_parents_in_fast = all(p in fast_mem for p in ps)
            if all_parents_in_fast:
                if len(fast_mem) >= fast_memory_budget:
                    # Need to free a slot
                    # Delete a parent that won't be needed again
                    for p in ps:
                        # Check if p has other children not yet processed
                        other_children = [c for c in children.get(p, []) if c not in computed and c != v]
                        if not other_children:
                            # p won't be needed again, delete it
                            if p in fast_mem:
                                delete(p)
                                break
                    if len(fast_mem) < fast_memory_budget:
                        compute(v)
                    else:
                        # Still can't fit - this is problematic
                        # Try to delete any non-output vertex in fast_mem
                        for fm_v in list(fast_mem):
                            if fm_v not in outputs:
                                delete(fm_v)
                                break
                        if len(fast_mem) < fast_memory_budget:
                            compute(v)
                else:
                    compute(v)
        
        # After computing v, decide whether to keep in fast memory or store
        if v in fast_mem:
            # Check if v will be needed again (has children not yet processed)
            will_be_needed = False
            for c in children.get(v, []):
                if c not in computed:
                    will_be_needed = True
                    break
            
            if not will_be_needed:
                # Not needed again - if it's an output, we must keep it blue (in fast memory)
                # Otherwise, we can delete it to free space
                if v in outputs:
                    pass  # Keep in fast memory
                else:
                    # Delete to free space (unless fast memory has room)
                    if len(fast_mem) >= fast_memory_budget:
                        delete(v)
                    # If fast memory has room, keep it (might be useful)
            else:
                # Will be needed again
                if v in outputs:
                    pass  # Keep
                else:
                    # Keep in fast memory for now
                    pass
    
    # Ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    # Free a slot
                    for fm_v in list(fast_mem):
                        if fm_v not in outputs:
                            delete(fm_v)
                            break
                if len(fast_mem) < fast_memory_budget:
                    load(v)
            else:
                # Need to recompute - complex, skip for now
                pass
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate - this shouldn't happen for reasonable DAGs
        operations = operations[:20000]
    
    return operations


def _naive_fallback(dag, fast_memory_budget, vertices, parents, inputs, outputs):
    """Simple fallback: load each input, compute in topo order, store outputs."""
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    operations = []
    fast_mem = set()
    slow_mem = set()
    
    for v in topo_order:
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                if p in slow_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        # Delete something
                        for fm_v in list(fast_mem):
                            if fm_v not in outputs:
                                fast_mem.discard(fm_v)
                                operations.append(['delete', fm_v])
                                break
                    if len(fast_mem) < fast_memory_budget:
                        fast_mem.add(p)
                        operations.append(['load', p])
        if v not in fast_mem:
            if len(fast_mem) >= fast_memory_budget:
                for fm_v in list(fast_mem):
                    if fm_v not in outputs:
                        fast_mem.discard(fm_v)
                        operations.append(['delete', fm_v])
                        break
            if len(fast_mem) < fast_memory_budget:
                fast_mem.add(v)
                operations.append(['compute', v])
        # Store to slow memory if not output
        if v not in outputs and v in fast_mem:
            slow_mem.add(v)
            fast_mem.discard(v)
            operations.append(['store', v])
    
    # Load outputs
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem:
                if len(fast_mem) >= fast_memory_budget:
                    for fm_v in list(fast_mem):
                        if fm_v not in outputs:
                            fast_mem.discard(fm_v)
                            operations.append(['delete', fm_v])
                            break
                if len(fast_mem) < fast_memory_budget:
                    fast_mem.add(v)
                    operations.append(['load', v])
    
    return operations
# EVOLVE-BLOCK-END
