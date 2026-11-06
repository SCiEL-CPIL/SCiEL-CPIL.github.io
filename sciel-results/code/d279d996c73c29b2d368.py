"""Communication-minimizing DAG pebbling with deadline guard."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

_START_TIME = time.time()
DEADLINE_MARGIN = 5.0  # seconds before the 90s wall-clock limit


def _elapsed():
    return time.time() - _START_TIME


def _is_deadline_exceeded():
    return _elapsed() > 90.0 - DEADLINE_MARGIN


def _parse_dag(dag):
    """
    Parse the dag structure. Expected formats:
      - dict with 'parents' and 'outputs' keys
      - list of (vertex, [parents]) tuples
      - other common formats
    
    Returns:
      vertices: list of vertex ids in topological order
      parents: dict vertex -> list of parent vertices
      outputs: set/list of output vertices
      n: number of vertices
    """
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        # Build full vertex list from parents keys and outputs
        all_verts = set(parents.keys())
        for v in parents:
            for p in parents[v]:
                all_verts.add(p)
        for v in outputs:
            all_verts.add(v)
        # Sort topologically: inputs (no parents listed) first, then by order
        # Assume the given order in parents dict is topological, or sort by depth
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        else:
            # Try to infer topological order
            vertices = list(parents.keys())
            # Add any parents not in keys
            for v in list(parents.keys()):
                for p in parents[v]:
                    if p not in parents:
                        parents[p] = []
            # Topological sort
            in_degree = {v: 0 for v in parents}
            for v, ps in parents.items():
                for p in ps:
                    if p in in_degree:
                        in_degree[p] = in_degree.get(p, 0)  # ensure exists
            # Actually let's do proper topo sort
            in_deg = {v: len(ps) for v, ps in parents.items()}
            children = {v: [] for v in parents}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            queue = deque([v for v, d in in_deg.items() if d == 0])
            topo = []
            while queue:
                u = queue.popleft()
                topo.append(u)
                for c in children.get(u, []):
                    in_deg[c] -= 1
                    if in_deg[c] == 0:
                        queue.append(c)
            if len(topo) != len(parents):
                # Fallback: just use the keys
                topo = list(parents.keys())
            vertices = topo
        n = len(vertices)
        return vertices, parents, outputs, n

    elif isinstance(dag, list):
        # List of (vertex, parents_list) or list of dicts
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)):
            # List of [vertex, [parents]]
            parents = {}
            outputs = []
            vertices = []
            for item in dag:
                if len(item) >= 2:
                    v = item[0]
                    ps = item[1]
                    parents[v] = list(ps) if ps else []
                    vertices.append(v)
                else:
                    v = item[0]
                    parents[v] = []
                    vertices.append(v)
            # outputs might be separate or inferred
            if isinstance(dag[0], dict):
                pass  # handled below
        # Check if items are dicts
        if len(dag) > 0 and isinstance(dag[0], dict):
            parents = {}
            vertices = []
            outputs = []
            for d in dag:
                v = d.get('vertex', d.get('id', d.get('v')))
                ps = d.get('parents', d.get('ps', []))
                is_out = d.get('is_output', d.get('output', False))
                parents[v] = list(ps) if ps else []
                vertices.append(v)
                if is_out:
                    outputs.append(v)
        else:
            # Assume last few are outputs or all are outputs
            parents = {}
            vertices = []
            for item in dag:
                if len(item) >= 2:
                    v = item[0]
                    ps = item[1]
                    parents[v] = list(ps) if ps else []
                    vertices.append(v)
                else:
                    v = item[0]
                    parents[v] = []
                    vertices.append(v)
            # Assume outputs are those with no children
            children = {v: [] for v in parents}
            for v, ps in parents.items():
                for p in ps:
                    if p in children:
                        children[p].append(v)
            outputs = [v for v in vertices if len(children[v]) == 0]
        n = len(vertices)
        return vertices, parents, outputs, n

    else:
        # Fallback: treat as list of vertices with no parents
        vertices = list(dag)
        parents = {v: [] for v in vertices}
        outputs = [vertices[-1]] if vertices else []
        n = len(vertices)
        return vertices, parents, outputs, n


def _naive_schedule_transfers(dag, fast_memory_budget):
    """
    Compute the number of transfers for the naive schedule.
    Naive: topological order, load parents if not in memory, compute, store result.
    """
    vertices, parents, outputs, n = _parse_dag(dag)
    if n == 0:
        return 0
    
    transfers = 0
    in_mem = set()
    
    for v in vertices:
        ps = parents.get(v, [])
        # Load parents that are not in memory
        for p in ps:
            if p not in in_mem:
                transfers += 1  # load
                in_mem.add(p)
        # Compute
        # Store result (if it's an output or will be needed later)
        # In naive schedule, we keep everything
        in_mem.add(v)
    
    # At the end, outputs must be blue (in memory), which they are
    # But we need to count stores? The task says "load/store/compute/delete"
    # Transfers are load and store operations
    
    # Actually, in the naive schedule, every vertex is computed once.
    # Each parent must be loaded at least once (first time).
    # Each computed result is stored (if it's an output) or kept in memory.
    
    # Let me reconsider: the naive schedule computes in topological order.
    # For each vertex v:
    #   - For each parent p not in memory: load p (1 transfer)
    #   - Compute v
    #   - If v is an output: store v (1 transfer) -- actually, outputs must finish blue, so they stay in memory
    #   - Actually, "store" means writing to slow memory, "load" means reading from slow memory
    #   - "blue" means in fast memory, "red" means in slow memory
    
    # So the naive schedule:
    #   - Start: all inputs are blue (in fast memory)
    #   - Process in topological order
    #   - For each vertex: ensure parents are blue (load if not), compute, result is blue
    #   - If memory full, we might need to delete (evict to slow memory = store)
    #   - At end, outputs must be blue
    
    # The naive baseline likely doesn't do any clever eviction - it just keeps everything
    # in fast memory if possible, or does minimal eviction.
    
    # Let me just return a reasonable estimate based on the structure.
    # The actual baseline is in pebble_baseline, so let me try to import it.
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        # Count transfers (load + store operations)
        transfers = sum(1 for op, v in result if op in ('load', 'store'))
        return transfers
    except:
        pass
    
    # Fallback naive count
    transfers = 0
    in_mem = set()
    for v in vertices:
        ps = parents.get(v, [])
        for p in ps:
            if p not in in_mem:
                transfers += 1  # load
                in_mem.add(p)
        in_mem.add(v)
    return transfers


def _count_transfers(ops):
    """Count load and store operations."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def pebble(dag, fast_memory_budget):
    """
    Pebble the DAG to minimize transfers.
    
    Returns list of [operation, vertex] where operation is in {load, store, compute, delete}.
    """
    start_time = time.time()
    
    vertices, parents, outputs, n = _parse_dag(dag)
    if n == 0:
        return []
    
    outputs_set = set(outputs)
    
    # Strategy: 
    # 1. Topological order processing
    # 2. Keep frequently-used vertices in fast memory
    # 3. Use fast_memory_budget wisely
    
    # Compute usage frequency: how many times each vertex is a parent
    usage = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            usage[p] = usage.get(p, 0) + 1
    
    # Compute depth (distance from inputs)
    depth = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in ps)
    
    # Strategy: process in topological order, keeping high-usage vertices in memory
    ops = []
    in_mem = set()
    
    # Pre-load inputs that have high usage
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Sort vertices: prioritize those with high-usage parents
    # Use a priority queue approach
    
    for v in vertices:
        if _is_deadline_exceeded():
            break
        
        ps = parents.get(v, [])
        
        # Ensure parents are in memory
        for p in ps:
            if p not in in_mem:
                # Need to load p
                # If memory is full, evict the least useful vertex
                while len(in_mem) >= fast_memory_budget and in_mem:
                    # Find the vertex in memory with lowest future usage
                    # that is not an output and not a parent of remaining vertices
                    best_evict = None
                    best_score = float('inf')
                    for u in in_mem:
                        if u in outputs_set:
                            continue
                        # Score: lower usage = better to evict
                        s = usage.get(u, 0)
                        if s < best_score:
                            best_score = s
                            best_evict = u
                    if best_evict is None:
                        # Evict any non-output
                        for u in in_mem:
                            if u not in outputs_set:
                                best_evict = u
                                break
                        if best_evict is None and in_mem:
                            best_evict = next(iter(in_mem))
                    if best_evict is not None:
                        ops.append(['store', best_evict])
                        in_mem.discard(best_evict)
                        break
                    else:
                        break
                
                if p not in in_mem:
                    ops.append(['load', p])
                    in_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        in_mem.add(v)
        
        # If v is an output, keep it in memory
        # Otherwise, if memory is tight, we might evict it later
    
    # Ensure all outputs are in memory at the end
    for o in outputs:
        if o not in in_mem:
            # Load it back
            if len(in_mem) >= fast_memory_budget:
                # Evict a non-output
                for u in list(in_mem):
                    if u not in outputs_set:
                        ops.append(['store', u])
                        in_mem.discard(u)
                        break
            ops.append(['load', o])
            in_mem.add(o)
    
    # Cap operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
