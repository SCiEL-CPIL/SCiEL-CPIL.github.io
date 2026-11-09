"""DAG pebbling with optimized transfer minimization."""
import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Minimize communication (transfers = loads + stores) for DAG pebbling.
    """
    deadline = time.time() + 50
    
    vertices, parents_map, inputs, outputs = _parse_dag(dag)
    
    if not vertices:
        return []
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # If budget can hold everything, optimal is: load inputs, compute all, store outputs
    # Transfers = |inputs| + |outputs|
    if fast_memory_budget >= len(vertices):
        ops = []
        for v in inputs:
            ops.append(['load', v])
        for v in vertices:
            if v not in input_set:
                ops.append(['compute', v])
        for v in outputs:
            ops.append(['store', v])
        return ops
    
    # For limited budget, use dynamic programming or greedy with recompute
    # Key: minimize loads + stores
    
    # Strategy: process in topological order, but manage fast memory carefully
    # When budget exceeded, recompute from scratch (reload inputs, recompute path)
    
    # Better: use the "pebbling" optimal strategy for the specific structure
    
    # For now, use a smart greedy that tracks dependencies
    ops = []
    fast_mem = set()
    
    # Topological order
    in_deg = {v: len(parents_map.get(v, [])) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    q = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    
    # Process in topological order
    for v in topo:
        if v in input_set:
            if v not in fast_mem:
                ops.append(['load', v])
            fast_mem.add(v)
        else:
            # Ensure all parents in fast_mem
            for p in parents_map.get(v, []):
                if p not in fast_mem:
                    # Recompute p: need its parents, etc.
                    # Simple: reload if input, else recompute recursively
                    _ensure_in_fast_mem(p, parents_map, input_set, fast_mem, ops)
            
            # Budget check: if over budget, evict non-essential
            while len(fast_mem) > fast_memory_budget:
                # Evict vertex not in outputs and not parent of any future vertex
                evict = None
                for u in list(fast_mem):
                    if u not in output_set:
                        # Check if u is needed for future computations
                        if not _is_needed_soon(u, v, topo, children, fast_mem, parents_map):
                            evict = u
                            break
                if evict:
                    ops.append(['delete', evict])
                    fast_mem.remove(evict)
                else:
                    # Can't evict, break
                    break
            
            ops.append(['compute', v])
            fast_mem.add(v)
        
        # Store output when computed
        if v in output_set:
            ops.append(['store', v])
    
    return ops


def _ensure_in_fast_mem(v, parents_map, input_set, fast_mem, ops):
    """Ensure vertex v is in fast memory, recomputing if necessary."""
    if v in fast_mem:
        return
    if v in input_set:
        ops.append(['load', v])
        fast_mem.add(v)
    else:
        for p in parents_map.get(v, []):
            _ensure_in_fast_mem(p, parents_map, input_set, fast_mem, ops)
        ops.append(['compute', v])
        fast_mem.add(v)


def _is_needed_soon(u, current, topo, children, fast_mem, parents_map):
    """Check if u is needed for any vertex after current in topo order."""
    # Simple check: does u have any children that come after current?
    idx = topo.index(current) if current in topo else len(topo)
    for c in children.get(u, []):
        if c in topo and topo.index(c) > idx:
            return True
    return False


def _parse_dag(dag):
    """Parse DAG into (vertices, parents_map, inputs, outputs)."""
    vertices = None
    parents_map = None
    inputs = None
    outputs = None
    
    if isinstance(dag, dict):
        for key in ['vertices', 'nodes', 'V']:
            if key in dag:
                vertices = list(dag[key])
                break
        for key in ['parents', 'preds', 'in_edges']:
            if key in dag:
                parents_map = {k: list(v) for k, v in dag[key].items()}
                break
        for key in ['inputs', 'sources']:
            if key in dag:
                inputs = list(dag[key])
                break
        for key in ['outputs', 'sinks']:
            if key in dag:
                outputs = list(dag[key])
                break
    elif hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
        if hasattr(dag, 'parents'):
            parents_map = {k: list(v) for k, v in dag.parents.items()}
        if hasattr(dag, 'inputs'):
            inputs = list(dag.inputs)
        if hasattr(dag, 'outputs'):
            outputs = list(dag.outputs)
    
    if vertices is None:
        if parents_map:
            vertices = sorted(parents_map.keys())
        else:
            return [], {}, [], []
    
    if parents_map is None:
        parents_map = {v: [] for v in vertices}
    for v in vertices:
        if v not in parents_map:
            parents_map[v] = []
    
    if inputs is None:
        inputs = [v for v in vertices if not parents_map.get(v, [])]
    
    if outputs is None:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map.get(v, [])]
    
    return (
        [int(v) for v in vertices],
        {int(k): [int(p) for p in v] for k, v in parents_map.items()},
        [int(v) for v in inputs],
        [int(v) for v in outputs]
    )
# EVOLVE-BLOCK-END
