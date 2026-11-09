"""Rewritten program for communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Pebbles a DAG with a bounded fast memory budget.
    Returns a list of [operation, vertex] operations.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # --- 1. Parse DAG Structure ---
    # We need to handle potential variations in the dag object structure.
    # Common structures:
    # 1. dag is a dict: {'vertices': [...], 'parents': {v: [...], ...}, 'inputs': [...], 'outputs': [...]}
    # 2. dag is an object with attributes .vertices, .parents, etc.
    
    vertices = []
    parents = {}
    inputs = []
    outputs = []
    
    # Try dict access
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    # Try attribute access
    elif hasattr(dag, 'vertices'):
        vertices = dag.vertices
        parents = getattr(dag, 'parents', {})
        inputs = getattr(dag, 'inputs', [])
        outputs = getattr(dag, 'outputs', [])
    else:
        # Fallback: assume vertices are 0..N-1 and parents are provided in a specific way?
        # If we can't parse, return empty or a naive fallback.
        # For now, return an empty list to avoid crash, though it will score 0.
        # Better: try to infer from 'edges'
        if isinstance(dag, dict) and 'edges' in dag:
            edges = dag['edges']
            all_v = set()
            for u, v in edges:
                all_v.add(u)
                all_v.add(v)
            vertices = sorted(all_v)
            for u, v in edges:
                if v not in parents:
                    parents[v] = []
                parents[v].append(u)
            # Heuristic for inputs/outputs if not provided
            if not inputs:
                inputs = [v for v in vertices if not parents.get(v)]
            if not outputs:
                out_deg = {v: 0 for v in vertices}
                for u, v in edges:
                    out_deg[u] = out_deg.get(u, 0) + 1
                outputs = [v for v in vertices if out_deg.get(v, 0) == 0]
        else:
            return []

    if not vertices:
        return []

    # Ensure parents dict is complete for all vertices
    for v in vertices:
        if v not in parents:
            parents[v] = []

    # --- 2. Topological Sort ---
    in_degree = {v: len(parents[v]) for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
            
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        # Use a stack for DFS-like order or queue for BFS.
        # For pebbling, the order matters less for correctness if we use recursive loading,
        # but topological order is required for computation.
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        # Cycle detected or error, fallback
        return []

    # Map vertex to its topological index for eviction policy
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    # --- 3. Simulation ---
    fast_mem = set()
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    # We need a way to load a vertex into fast memory recursively if its parents are not there.
    # This is complex. A simpler approach for many DAGs is to process in topological order
    # and assume that if a parent is not in fast memory, we must have evicted it.
    # We can implement a 'ensure' function.
    
    def ensure_in_fast(v):
        """Ensure vertex v is in fast_mem, loading dependencies if necessary."""
        nonlocal fast_mem, operations, op_count
        if v in fast_mem:
            return
        
        # If v is an input, it's in slow memory.
        # If v is a computed node, it might have been evicted.
        # To load v, we need its parents in fast memory first (if v is to be computed)
        # OR we just load v directly if it's an input or was previously computed and stored?
        # The problem says "Inputs start blue". "Red value consumes one slot".
        # "load" moves from slow to fast. "store" moves from fast to slow.
        # "compute" requires parents in fast, result goes to fast.
        
        # If v is an input:
        if v in inputs:
            # It's in slow memory. Just load it.
            if len(fast_mem) >= fast_memory_budget:
                evict_one()
            operations.append(['load', v])
            op_count += 1
            fast_mem.add(v)
            return
        
        # If v is not an input, it must be computed.
        # To compute v, all parents must be in fast memory.
        for p in parents[v]:
            ensure_in_fast(p)
        
        # Now all parents are in fast memory. Compute v.
        if len(fast_mem) >= fast_memory_budget:
            evict_one()
        
        operations.append(['compute', v])
        op_count += 1
        fast_mem.add(v)
        
    def evict_one():
        """Evict a vertex from fast memory to make space."""
        nonlocal fast_mem, operations, op_count
        if not fast_mem:
            return
        
        # Eviction Policy: Evict the vertex with the highest topological index (furthest in future).
        # This is a heuristic.
        # Alternatively, evict the one with the fewest remaining children not yet processed.
        # For simplicity, highest topo index.
        max_v = max(fast_mem, key=lambda x: topo_index[x])
        
        # We can only store if the vertex was computed (i.e., not an input that was just loaded and not used?
        # Actually, any red value can be stored to blue.
        # But if it's an input, storing it just puts it back to slow.
        # If it's a computed node, storing it saves it for later.
        
        operations.append(['store', max_v])
        op_count += 1
        fast_mem.remove(max_v)
        
    # Main Loop
    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
            
        # If v is an output, it must end up in slow memory (blue).
        # But the problem says "all outputs must finish blue".
        # This implies we might need to store outputs at the end.
        # However, the loop processes all vertices.
        # Let's just ensure all vertices are computed.
        
        # If v is already in fast_mem, do nothing.
        # But wait, if v was evicted, it's not in fast_mem.
        # We need to ensure v is in fast_mem to "have" its value?
        # Actually, we only need to compute v once.
        # If v is already computed and in fast_mem, we don't need to do anything.
        # If v is already computed but evicted, do we need to reload it?
        # Only if it's a parent of a future node.
        # The 'ensure_in_fast' logic handles parents.
        # Do we need to compute v if it's not an input?
        # Yes, every non-input vertex must be computed at least once.
        
        # Check if v has been computed. We don't track this explicitly in fast_mem.
        # We need a 'computed' set.
        
        # Let's restart the simulation logic with a 'computed' set.
        pass

    # Restart simulation with proper tracking
    fast_mem = set()
    operations = []
    op_count = 0
    computed = set()
    
    def evict_one_v2():
        nonlocal fast_mem, operations, op_count
        if not fast_mem:
            return
        # Evict vertex with highest topo index
        max_v = max(fast_mem, key=lambda x: topo_index[x])
        operations.append(['store', max_v])
        op_count += 1
        fast_mem.remove(max_v)
        
    def ensure_in_fast_v2(v):
        nonlocal fast_mem, operations, op_count, computed
        if v in fast_mem:
            return
            
        if v in inputs:
            # Load input
            if len(fast_mem) >= fast_memory_budget:
                evict_one_v2()
            operations.append(['load', v])
            op_count += 1
            fast_mem.add(v)
            return
            
        # Non-input
        # Must be computed
        if v not in computed:
            # Ensure parents are in fast
            for p in parents[v]:
                ensure_in_fast_v2(p)
            
            # Compute v
            if len(fast_mem) >= fast_memory_budget:
                evict_one_v2()
            
            operations.append(['compute', v])
            op_count += 1
            fast_mem.add(v)
            computed.add(v)
        else:
            # v was computed but evicted.
            # We need to reload it?
            # If it's an intermediate result, it was stored to slow memory.
            # We can just 'load' it.
            if len(fast_mem) >= fast_memory_budget:
                evict_one_v2()
            operations.append(['load', v])
            op_count += 1
            fast_mem.add(v)

    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        # We need to ensure v is in fast memory?
        # Not necessarily, unless it's a parent of a future node.
        # But we must compute it.
        # If v is an input, we load it when needed.
        # If v is a computed node, we compute it when needed.
        # Do we need to actively process v in topo order?
        # Yes, to ensure dependencies are met.
        
        # Actually, we can just call ensure_in_fast_v2(v) for all v?
        # No, ensure_in_fast_v2 computes v if not computed.
        # But if v is not a dependency of any future node, we don't need it in fast memory at the end.
        # However, we must compute it.
        
        # Let's just iterate and ensure each vertex is computed.
        # If v is an input, ensure_in_fast_v2 will load it.
        # If v is a computed node, ensure_in_fast_v2 will compute it.
        
        ensure_in_fast_v2(v)
        
        # If v is an output, it must finish blue.
        # So we should store it?
        # "all outputs must finish blue"
        # If it's in fast_mem, we should store it.
        if v in outputs:
            if v in fast_mem:
                # Store it
                operations.append(['store', v])
                op_count += 1
                fast_mem.remove(v)
                
    return operations

# EVOLVE-BLOCK-END
