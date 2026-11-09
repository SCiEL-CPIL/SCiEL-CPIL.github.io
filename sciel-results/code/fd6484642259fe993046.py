# EVOLVE-BLOCK-START
import time
import heapq

def _parse_dag(dag):
    """Parse the dag structure."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        if not parents:
            edges = dag.get('edges', [])
            parents = {v: [] for v in vertices}
            for parent, child in edges:
                if child not in parents:
                    parents[child] = []
                parents[child].append(parent)
        return vertices, parents, inputs, outputs
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0 and isinstance(dag[0], dict):
            vertices = []
            parents = {}
            inputs = []
            outputs = []
            for i, v in enumerate(dag):
                vid = v.get('id', i)
                vertices.append(vid)
                par = v.get('parents', v.get('parent', []))
                if isinstance(par, (list, tuple)):
                    parents[vid] = list(par)
                else:
                    parents[vid] = [par] if par else []
                if v.get('is_input', False):
                    inputs.append(vid)
                if v.get('is_output', False):
                    outputs.append(vid)
            if not inputs:
                inputs = vertices[:1]
            if not outputs:
                outputs = vertices[-1:]
            return vertices, parents, inputs, outputs
        else:
            vertices = list(dag)
            parents = {v: [] for v in vertices}
            inputs = [vertices[0]] if vertices else []
            outputs = [vertices[-1]] if vertices else []
            return vertices, parents, inputs, outputs
    else:
        return [], {}, [], []

def _greedy_schedule(vertices, parents, inputs, outputs, fast_memory_budget):
    """Generate a greedy topological schedule with eviction."""
    ops = []
    in_mem = set()
    
    # Load all inputs
    for v in inputs:
        if v not in in_mem:
            ops.append(['load', v])
            in_mem.add(v)
    
    # Build child map for future needs
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    def is_needed_later(v, current_idx):
        """Check if v is a parent of any vertex at index >= current_idx."""
        for future_v in vertices[current_idx:]:
            if v in parents.get(future_v, []):
                return True
        return False
    
    # Process vertices in topological order
    for idx, v in enumerate(vertices):
        if v in inputs:
            continue
        
        # Ensure all parents are in memory
        for p in parents.get(v, []):
            if p not in in_mem:
                if len(in_mem) >= fast_memory_budget:
                    # Evict a safe vertex
                    evicted = None
                    for candidate in list(in_mem):
                        if candidate not in outputs:
                            if not is_needed_later(candidate, idx):
                                evicted = candidate
                                break
                    if evicted is None:
                        for candidate in list(in_mem):
                            if candidate not in outputs:
                                evicted = candidate
                                break
                    if evicted is not None:
                        ops.append(['delete', evicted])
                        in_mem.discard(evicted)
                
                if p not in in_mem:
                    ops.append(['load', p])
                    in_mem.add(p)
        
        # Make room for v if needed
        if len(in_mem) >= fast_memory_budget:
            evicted = None
            for candidate in list(in_mem):
                if candidate not in outputs and candidate != v:
                    if not is_needed_later(candidate, idx + 1):
                        evicted = candidate
                        break
            if evicted is None:
                for candidate in list(in_mem):
                    if candidate not in outputs and candidate != v:
                        evicted = candidate
                        break
            if evicted is not None:
                ops.append(['delete', evicted])
                in_mem.discard(evicted)
        
        ops.append(['compute', v])
        in_mem.add(v)
    
    # Store outputs
    for o in outputs:
        if o in in_mem:
            ops.append(['store', o])
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    vertices, parents, inputs, outputs = _parse_dag(dag)
    
    if not vertices:
        return []
    
    n = len(vertices)
    
    # If budget is large enough, simple topological order
    if fast_memory_budget >= n:
        ops = []
        in_mem = set()
        for v in vertices:
            if v not in in_mem:
                ops.append(['load', v])
                in_mem.add(v)
        for v in vertices:
            if v not in inputs:
                ops.append(['compute', v])
        for o in outputs:
            ops.append(['store', o])
        return ops
    
    # Try greedy schedule
    ops = _greedy_schedule(vertices, parents, inputs, outputs, fast_memory_budget)
    
    if time.time() > deadline:
        return ops
    
    return ops
# EVOLVE-BLOCK-END
