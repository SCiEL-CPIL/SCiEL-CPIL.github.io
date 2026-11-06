"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Minimizes load/store operations by keeping values in fast memory
    as long as they'll be reused, and evicting when memory is full.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG structure
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        # Use baseline as fallback if parsing fails
    except ImportError:
        pebble_baseline = None
    
    # Try to understand dag format
    if isinstance(dag, dict):
        # Could be {vertex: [parents]} with special keys
        data_keys = set(dag.keys())
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents_map = dag.get('parents', {})
            outputs = set(dag.get('outputs', []))
            inputs = set(dag.get('inputs', []))
        elif 'nodes' in dag:
            vertices = dag['nodes']
            parents_map = dag.get('parents', {})
            outputs = set(dag.get('outputs', []))
            inputs = set(dag.get('inputs', []))
        else:
            # Assume {vertex: [parents]}
            vertices = list(dag.keys())
            parents_map = {}
            for v in vertices:
                if isinstance(dag[v], list):
                    parents_map[v] = dag[v]
                elif isinstance(dag[v], (int, str)):
                    parents_map[v] = [dag[v]]
                else:
                    parents_map[v] = list(dag[v]) if dag[v] else []
            # Check for special keys
            if 'outputs' in dag and not isinstance(dag['outputs'], list):
                pass
            outputs = set()
            inputs = set()
            all_parents = set()
            for v in vertices:
                for p in parents_map[v]:
                    all_parents.add(p)
            inputs = set(vertices) - all_parents
            # Outputs are vertices with no children
            all_children = set()
            for v in vertices:
                for p in parents_map[v]:
                    all_children.add(p)
            outputs = set(vertices) - all_children
    
    elif isinstance(dag, list):
        vertices = []
        parents_map = {}
        outputs = set()
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    vid = item[0]
                    par = item[1]
                    if isinstance(par, (list, tuple)):
                        par = list(par)
                    elif par is not None:
                        par = [par]
                    else:
                        par = []
                    vertices.append(vid)
                    parents_map[vid] = par
                    if len(item) >= 3:
                        if item[2] in (True, 1, 'output', 'out'):
                            outputs.add(vid)
                elif len(item) == 1:
                    vertices.append(item[0])
                    parents_map[item[0]] = []
            else:
                vertices.append(item)
                parents_map[item] = []
        
        # Infer inputs/outputs
        all_parents = set()
        for v in vertices:
            for p in parents_map[v]:
                all_parents.add(p)
        inputs = set(vertices) - all_parents
        if not outputs:
            all_children = set()
            for v in vertices:
                for p in parents_map[v]:
                    all_children.add(p)
            outputs = set(vertices) - all_children
    
    else:
        # Can't parse, use baseline
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Topological order (assume vertices list is in topo order)
    topo_order = vertices
    vertex_index = {v: i for i, v in enumerate(topo_order)}
    
    # Compute next use for each vertex
    next_use = {}
    for v in topo_order:
        uses = [vertex_index[c] for c in children_map.get(v, []) if c in vertex_index]
        next_use[v] = min(uses) if uses else float('inf')
    
    # Pebbling algorithm
    operations = []
    fast_memory = set()
    slow_memory = set()
    computed = set()
    
    # Inputs are initially available (blue) - assume they're in fast memory
    for inp in inputs:
        if inp in vertex_index:
            fast_memory.add(inp)
            computed.add(inp)
    
    for i, v in enumerate(topo_order):
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            continue  # Already handled
        
        # Check parents availability
        parents = parents_map.get(v, [])
        for p in parents:
            if p not in fast_memory and p not in slow_memory:
                # Parent not available - shouldn't happen in topo order
                break
        
        # Load parents from slow to fast if needed
        for p in parents:
            if p in slow_memory and p not in fast_memory:
                operations.append(['load', p])
                fast_memory.add(p)
                slow_memory.discard(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        computed.add(v)
        
        # Evict if over budget
        while len(fast_memory) > fast_memory_budget:
            # Find evictable vertex (not output, all children computed)
            evictable = []
            for u in fast_memory:
                if u in outputs:
                    continue
                all_children_done = all(c in computed for c in children_map.get(u, []))
                if all_children_done:
                    evictable.append(u)
            
            if not evictable:
                # Emergency: evict the one with farthest next use
                candidates = [u for u in fast_memory if u not in outputs]
                if candidates:
                    u = max(candidates, key=lambda x: next_use.get(x, float('inf')))
                    operations.append(['store', u])
                    fast_memory.discard(u)
                    slow_memory.add(u)
                else:
                    break
            else:
                # Evict the one with farthest next use (or no future use)
                u = max(evictable, key=lambda x: next_use.get(x, float('inf')))
                operations.append(['store', u])
                fast_memory.discard(u)
                slow_memory.add(u)
        
        # Proactively store vertices that are no longer needed
        # to keep memory light for future loads
        if len(fast_memory) >= fast_memory_budget - 1:
            for u in list(fast_memory):
                if u in outputs or u == v:
                    continue
                all_children_done = all(c in computed for c in children_map.get(u, []))
                if all_children_done and next_use.get(u, float('inf')) == float('inf'):
                    operations.append(['store', u])
                    fast_memory.discard(u)
                    slow_memory.add(u)
                if len(fast_memory) < fast_memory_budget:
                    break
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if o in slow_memory and o not in fast_memory:
            operations.append(['load', o])
            fast_memory.add(o)
            slow_memory.discard(o)
    
    # Enforce operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
