"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Parse DAG structure
    def get_vertices(d):
        if hasattr(d, 'vertices'):
            return list(d.vertices)
        elif isinstance(d, dict) and 'vertices' in d:
            return list(d['vertices'])
        elif hasattr(d, 'nodes'):
            return list(d.nodes)
        elif isinstance(d, dict) and 'nodes' in d:
            return list(d['nodes'])
        else:
            if hasattr(d, 'parents'):
                parents = d.parents
            elif isinstance(d, dict) and 'parents' in d:
                parents = d['parents']
            else:
                raise ValueError("Cannot determine vertices from dag")
            verts = set()
            for v, ps in parents.items():
                verts.add(v)
                verts.update(ps)
            return list(verts)
    
    def get_parents(d):
        if hasattr(d, 'parents'):
            return dict(d.parents)
        elif isinstance(d, dict) and 'parents' in d:
            return dict(d['parents'])
        elif hasattr(d, 'edges'):
            parents = {}
            for e in d.edges:
                if len(e) == 2:
                    child, parent = e
                    parents.setdefault(child, []).append(parent)
            return parents
        else:
            raise ValueError("Cannot determine parents from dag")
    
    def get_inputs(d):
        if hasattr(d, 'inputs'):
            return list(d.inputs)
        elif isinstance(d, dict) and 'inputs' in d:
            return list(d['inputs'])
        else:
            parents = get_parents(d)
            return [v for v in get_vertices(d) if not parents.get(v, [])]
    
    def get_outputs(d):
        if hasattr(d, 'outputs'):
            return list(d.outputs)
        elif isinstance(d, dict) and 'outputs' in d:
            return list(d['outputs'])
        else:
            parents = get_parents(d)
            vertices = get_vertices(d)
            children = {}
            for v, ps in parents.items():
                for p in ps:
                    children.setdefault(p, []).append(v)
            return [v for v in vertices if not children.get(v, [])]
    
    try:
        vertices = get_vertices(dag)
        parents_map = get_parents(dag)
        inputs = get_inputs(dag)
        outputs = get_outputs(dag)
    except Exception:
        return []
    
    # Build children map
    children_map = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)
    
    # Topological sort
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Precompute next uses for each vertex
    next_uses = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents_map.get(v, []):
            next_uses[p].append(i)
    
    # Precompute recomputation cost: number of ancestors (including self) that would need recomputation
    # This is the size of the subgraph that depends on this vertex being available
    recompute_cost = {}
    for v in reversed(topo_order):
        ps = parents_map.get(v, [])
        if not ps:
            recompute_cost[v] = 1  # Input: just load it
        else:
            # Cost to recompute v = 1 (compute) + sum of costs to ensure parents are available
            # But parents might be shared, so this is an upper bound
            # Use a simpler metric: depth * fan_in as a proxy
            recompute_cost[v] = 1 + sum(1 for _ in ps)  # Simplified
    
    # Better: compute the actual number of vertices in the ancestor subgraph
    # Use dynamic programming from the top
    ancestor_count = {}
    for v in topo_order:
        ps = parents_map.get(v, [])
        if not ps:
            ancestor_count[v] = 1
        else:
            # Count unique ancestors - this is expensive, use a simpler metric
            ancestor_count[v] = 1 + max(ancestor_count.get(p, 0) for p in ps)
    
    # Use fan-out as a key metric
    fan_out = {v: len(children_map.get(v, [])) for v in vertices}
    
    # Fast memory state
    fast_mem = set()
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    # Track next use index for each vertex
    use_idx = {v: 0 for v in vertices}
    
    def get_next_use(v):
        idx = use_idx[v]
        uses = next_uses[v]
        if idx < len(uses):
            return uses[idx]
        return float('inf')
    
    def advance_use(v):
        use_idx[v] += 1
    
    def eviction_priority(v):
        """Higher priority = should be kept. Lower = should be evicted."""
        next_t = get_next_use(v)
        if next_t == float('inf'):
            # No more uses - safe to evict
            return -1000000
        
        # Priority based on:
        # 1. Distance to next use (closer = higher priority)
        # 2. Fan-out (more uses = higher priority)  
        # 3. Recomputation cost (expensive to recompute = higher priority)
        
        dist = next_t + 1  # Avoid division by zero
        fan = fan_out[v]
        cost = recompute_cost[v]
        
        # Weighted combination
        priority = (1.0 / dist) * 10 + fan * 5 + cost * 2
        return priority
    
    def evict_worst():
        """Evict the vertex with the lowest priority."""
        nonlocal op_count
        if not fast_mem:
            return None
        worst_v = None
        worst_priority = float('inf')
        for v in fast_mem:
            p = eviction_priority(v)
            if p < worst_priority:
                worst_priority = p
                worst_v = v
        if worst_v is not None:
            fast_mem.discard(worst_v)
            operations.append(['store', worst_v])
            op_count += 1
        return worst_v
    
    def ensure_space():
        """Ensure fast memory has at least one free slot."""
        while len(fast_mem) >= fast_memory_budget:
            evict_worst()
    
    def load_vertex(v):
        """Load vertex v into fast memory if not already there."""
        nonlocal op_count
        if v in fast_mem:
            return
        ensure_space()
        fast_mem.add(v)
        operations.append(['load', v])
        op_count += 1
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        if v in input_set:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents_map.get(v, []):
            load_vertex(p)
            advance_use(p)
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        
        # v is now in fast memory
        fast_mem.add(v)
        
        # If full, evict one
        if len(fast_mem) > fast_memory_budget:
            evict_worst()
    
    # At the end, ensure all outputs are stored
    for v in outputs:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        if v in fast_mem:
            operations.append(['store', v])
            op_count += 1
            fast_mem.discard(v)
    
    return operations[:MAX_OPS]

# EVOLVE-BLOCK-END

