"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using recursive divide-and-conquer.
    
    dag: dict with:
        - 'vertices': list of vertex names
        - 'parents': dict mapping vertex -> list of parent vertices
        - 'outputs': list of output vertex names
    fast_memory_budget: int, number of red values in fast memory
    
    Returns: list of [operation, vertex]
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = dag['outputs']
    output_set = set(outputs)
    
    # Build structure
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    input_vertices = [v for v in vertices if len(parents.get(v, [])) == 0]
    
    # Count usage: how many times each vertex is needed as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            usage_count[p] += 1
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    if len(topo_order) != len(vertices):
        topo_order = vertices
    
    operations = []
    fast_mem = set()
    
    def load(v):
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
    
    def store(v):
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.discard(v)
    
    def compute(v):
        operations.append(['compute', v])
        fast_mem.add(v)
    
    def delete(v):
        if v in fast_mem:
            operations.append(['delete', v])
            fast_mem.discard(v)
    
    def evict_one():
        """Evict one vertex from fast memory to make space"""
        if not fast_mem:
            return
        # Prefer evicting vertices with low usage that aren't outputs
        candidates = [v for v in fast_mem if v not in output_set]
        if not candidates:
            candidates = list(fast_mem)
        # Evict the one with lowest future usage
        to_evict = min(candidates, key=lambda u: usage_count.get(u, 0))
        store(to_evict)
    
    def ensure_space():
        """Ensure fast memory has space for one more vertex"""
        while len(fast_mem) >= fast_memory_budget:
            evict_one()
    
    # Phase 1: Load all inputs
    for v in input_vertices:
        ensure_space()
        load(v)
    
    # Phase 2: Process vertices in topological order
    computed = set(input_vertices)
    for v in topo_order:
        if v in computed:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                ensure_space()
                load(p)
        
        # Compute v
        ensure_space()
        compute(v)
        computed.add(v)
        
        # If v won't be needed again and isn't an output, store it
        if usage_count.get(v, 0) == 0 and v not in output_set:
            store(v)
    
    # Phase 3: Ensure all outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            ensure_space()
            load(v)
    
    return operations

# EVOLVE-BLOCK-END
