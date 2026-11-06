"""DAG pebbling: minimize communication under bounded fast memory."""
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 85.0
    
    # Robustly parse DAG structure
    vertices = []
    parents_map = {}
    inputs = []
    outputs = []
    
    def extract_from_dict(d):
        nonlocal vertices, parents_map, inputs, outputs
        if 'vertices' in d:
            vertices = list(d['vertices'])
        elif 'nodes' in d:
            vertices = list(d['nodes'])
        
        if 'parents' in d:
            for k, v in d['parents'].items():
                parents_map[k] = list(v)
        elif 'edges' in d:
            for e in d['edges']:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    p, c = e
                    if c not in parents_map:
                        parents_map[c] = []
                    parents_map[c].append(p)
        
        if 'inputs' in d:
            inputs = list(d['inputs'])
        elif 'sources' in d:
            inputs = list(d['sources'])
        
        if 'outputs' in d:
            outputs = list(d['outputs'])
        elif 'sinks' in d:
            outputs = list(d['outputs'])
    
    if isinstance(dag, dict):
        extract_from_dict(dag)
    elif isinstance(dag, (list, tuple)):
        for i, item in enumerate(dag):
            if isinstance(item, dict):
                v = item.get('id', item.get('vertex', i))
                ps = item.get('parents', item.get('deps', item.get('inputs', [])))
                vertices.append(v)
                parents_map[v] = list(ps)
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                v = item[0]
                ps = list(item[1:]) if len(item) > 1 else []
                vertices.append(v)
                parents_map[v] = ps
    elif hasattr(dag, 'vertices') or hasattr(dag, 'nodes'):
        vertices = list(getattr(dag, 'vertices', getattr(dag, 'nodes', [])))
        if hasattr(dag, 'parents'):
            pm = getattr(dag, 'parents')
            if isinstance(pm, dict):
                for k, v in pm.items():
                    parents_map[k] = list(v)
        elif hasattr(dag, 'edges'):
            for e in getattr(dag, 'edges', []):
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    p, c = e
                    if c not in parents_map:
                        parents_map[c] = []
                    parents_map[c].append(p)
        if hasattr(dag, 'inputs'):
            inputs = list(getattr(dag, 'inputs', []))
        if hasattr(dag, 'outputs'):
            outputs = list(getattr(dag, 'outputs', []))
    
    # Build children map
    children_map = {v: [] for v in vertices}
    for v, ps in parents_map.items():
        for p in ps:
            if p in children_map:
                children_map[p].append(v)
            else:
                children_map[p] = [v]
    
    # Identify inputs and outputs if not provided
    if not inputs:
        inputs = [v for v in vertices if not parents_map.get(v, [])]
    if not outputs:
        outputs = [v for v in vertices if not children_map.get(v, [])]
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Count usage
    use_count = {v: 0 for v in vertices}
    for v, ps in parents_map.items():
        for p in ps:
            use_count[p] = use_count.get(p, 0) + 1
    
    # Ensure all vertices are in maps
    for v in vertices:
        if v not in parents_map:
            parents_map[v] = []
        if v not in children_map:
            children_map[v] = []
    
    topo_order = list(vertices)
    
    # Greedy pebbling
    fast_set = set()
    ops = []
    
    # Initialize: load inputs
    for inp in inputs:
        if len(fast_set) < fast_memory_budget:
            fast_set.add(inp)
        else:
            candidates = [v for v in fast_set if v not in input_set]
            if candidates:
                victim = min(candidates, key=lambda x: use_count.get(x, 0))
                ops.append(('delete', victim))
                fast_set.discard(victim)
            fast_set.add(inp)
    
    # Process in topological order
    processed = set()
    for v in topo_order:
        if time.time() - start_time > deadline:
            break
        if v in input_set:
            processed.add(v)
            continue
        
        ps = parents_map.get(v, [])
        # Ensure parents are in fast memory
        for p in ps:
            if p not in fast_set:
                if p in input_set:
                    if len(fast_set) >= fast_memory_budget:
                        candidates = [x for x in fast_set if x not in input_set and x != v]
                        if candidates:
                            victim = min(candidates, key=lambda x: use_count.get(x, 0))
                            ops.append(('delete', victim))
                            fast_set.discard(victim)
                    ops.append(('load', p))
                    fast_set.add(p)
        
        ops.append(('compute', v))
        fast_set.add(v)
        processed.add(v)
        
        # Evict if over budget
        if len(fast_set) > fast_memory_budget:
            # Determine which vertices are still needed
            remaining = [x for x in topo_order if x not in processed]
            needed = set()
            for rv in remaining:
                for rp in parents_map.get(rv, []):
                    needed.add(rp)
            needed.update(output_set)
            
            evictable = [x for x in fast_set if x not in needed and x not in input_set]
            while len(fast_set) > fast_memory_budget and evictable:
                victim = min(evictable, key=lambda x: use_count.get(x, 0))
                ops.append(('delete', victim))
                fast_set.discard(victim)
                evictable = [x for x in evictable if x != victim]
            
            while len(fast_set) > fast_memory_budget:
                candidates = [x for x in fast_set if x not in output_set]
                if not candidates:
                    break
                victim = min(candidates, key=lambda x: use_count.get(x, 0))
                ops.append(('delete', victim))
                fast_set.discard(victim)
    
    # Ensure outputs are in fast memory
    for out in outputs:
        if out not in fast_set:
            if out in input_set:
                ops.append(('load', out))
                fast_set.add(out)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
