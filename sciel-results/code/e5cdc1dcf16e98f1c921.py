"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _parse_dag(dag):
    """
    Parses the DAG into a list of vertices (integers), a parents map, and sets for inputs/outputs.
    Returns: (vertex_ids, parents_map, inputs, outputs)
    vertex_ids: list of integer IDs for all vertices in topological order.
    parents_map: dict {vertex_id: [parent_ids]}
    inputs: set of vertex_ids with no parents.
    outputs: set of vertex_ids with no children.
    """
    # We will assume the input dag is either:
    # 1. A dict: {v: [parents]} where v is hashable.
    # 2. A list: [[v, [parents]], ...]
    # We need to map all vertices to integers 0..N-1.
    
    # First, collect all unique vertices and their parents
    raw_vertices = set()
    raw_parents = {}
    
    if isinstance(dag, dict):
        for v, p_list in dag.items():
            raw_vertices.add(v)
            if p_list is None:
                p_list = []
            elif not isinstance(p_list, (list, tuple)):
                p_list = [p_list]
            raw_parents[v] = list(p_list)
            for p in p_list:
                raw_vertices.add(p)
    elif isinstance(dag, list):
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                p_list = item[1]
                if p_list is None:
                    p_list = []
                elif not isinstance(p_list, (list, tuple)):
                    p_list = [p_list]
                raw_vertices.add(v)
                raw_parents[v] = list(p_list)
                for p in p_list:
                    raw_vertices.add(p)
            else:
                # Single vertex
                v = item
                raw_vertices.add(v)
                if v not in raw_parents:
                    raw_parents[v] = []
    else:
        return [], {}, set(), set()
    
    # Map raw vertices to integer IDs
    # To ensure topological order, we should ideally sort them.
    # However, if the input is already topologically sorted (as per task description "topologically ordered"),
    # we can try to preserve that order or just compute it.
    # Let's compute topological order to be safe.
    
    # Build adjacency list for topo sort
    children_map = defaultdict(list)
    in_degree = defaultdict(int)
    
    # Initialize in_degree for all vertices
    for v in raw_vertices:
        if v not in in_degree:
            in_degree[v] = 0
            
    for v, p_list in raw_parents.items():
        for p in p_list:
            children_map[p].append(v)
            in_degree[v] += 1
            
    # Topological Sort (Kahn's algorithm)
    from collections import deque
    q = deque([v for v in raw_vertices if in_degree[v] == 0])
    topo_order = []
    while q:
        v = q.popleft()
        topo_order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                q.append(c)
                
    # If topo_order doesn't contain all vertices, there's a cycle or disconnected components not handled?
    # Assuming valid DAG. If some nodes are missing, append them arbitrarily (should not happen for valid DAG).
    if len(topo_order) < len(raw_vertices):
        for v in raw_vertices:
            if v not in topo_order:
                topo_order.append(v)
                
    # Create ID mapping
    id_map = {v: i for i, v in enumerate(topo_order)}
    
    # Convert parents_map to integer IDs
    int_parents_map = {}
    for v in topo_order:
        p_list = raw_parents.get(v, [])
        int_parents_map[id_map[v]] = [id_map[p] for p in p_list]
        
    # Identify inputs and outputs
    inputs = set(id_map[v] for v in raw_vertices if not raw_parents.get(v, []))
    outputs = set(id_map[v] for v in raw_vertices if not children_map.get(v, []))
    
    return topo_order, int_parents_map, inputs, outputs

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with bounded fast memory.
    Returns a list of [operation, vertex_id] where vertex_id is an integer.
    """
    start_time = time.time()
    DEADLINE = 85.0 # Safety margin for 90s limit
    
    topo_order, parents_map, inputs, outputs = _parse_dag(dag)
    
    if not topo_order:
        return []
        
    N = len(topo_order)
    
    # Build children map for integer IDs
    children_map = defaultdict(list)
    for v in range(N):
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # Algorithm:
    # 1. Inputs are initially in fast memory (blue).
    # 2. Process vertices in topological order.
    # 3. For each vertex v:
    #    a. Ensure all parents are in fast memory. If a parent is in slow memory, load it.
    #    b. Compute v (add to fast memory).
    #    c. If fast memory exceeds budget, evict a vertex.
    #       Eviction policy: Evict a vertex that is no longer needed (all children computed) 
    #       or has the farthest next use. Store it to slow memory.
    #    d. Proactively store vertices that are no longer needed to keep memory low.
    
    operations = []
    fast_memory = set(inputs)
    slow_memory = set()
    computed = set(inputs)
    
    # Precompute next use for each vertex to help eviction
    # next_use[v] = index of the next child of v in topological order, or inf if none
    next_use = [float('inf')] * N
    for v in range(N):
        if children_map[v]:
            # Find the earliest child in topological order
            # Since topo_order is 0..N-1, the children indices are just their values
            # We need the minimum index among children
            next_use[v] = min(children_map[v])
            
    # Process
    for v in range(N):
        if time.time() - start_time > DEADLINE:
            break
            
        if v in inputs:
            continue
            
        # 1. Load parents
        for p in parents_map.get(v, []):
            if p in slow_memory and p not in fast_memory:
                operations.append(['load', p])
                fast_memory.add(p)
                slow_memory.discard(p)
            # If p is not in fast and not in slow, it means it was computed and evicted to slow? 
            # Wait, "slow_memory" tracks what is stored there. 
            # Initially, nothing is in slow memory except what we store.
            # Inputs are in fast. 
            # When we compute v, v is in fast. 
            # When we evict u, u goes to slow. 
            # So if p is needed and not in fast, it MUST be in slow (because it was computed earlier).
            # Unless p is an input that was evicted? Inputs start in fast. If evicted, they go to slow.
            
        # 2. Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        computed.add(v)
        
        # 3. Evict if over budget
        while len(fast_memory) > fast_memory_budget:
            # Find best candidate to evict
            # Prioritize vertices with no remaining uses (next_use == inf)
            # Then vertices with farthest next_use
            evictable = []
            for u in fast_memory:
                if u in outputs:
                    continue # Don't evict outputs if possible, or do? 
                    # Task says "all outputs must finish blue". 
                    # If we evict an output, we must load it back at the end. 
                    # It's better to keep outputs in fast memory if budget allows.
                    # If budget is tight, we might have to evict and reload.
                
                evictable.append(u)
                
            if not evictable:
                break # Emergency: cannot evict, but this shouldn't happen if budget >= 1 and we don't have > budget non-evictable items?
                # Actually, if fast_memory_budget is 0, we can't hold anything. But inputs start blue?
                # "Inputs start blue". If budget is 0, we can't hold inputs? 
                # Assuming budget >= 1.
                
            # Sort by next_use descending (evict farthest first)
            # If next_use is inf, it's the best to evict.
            evictable.sort(key=lambda u: next_use[u], reverse=True)
            
            u_to_evict = evictable[0]
            
            operations.append(['store', u_to_evict])
            fast_memory.discard(u_to_evict)
            slow_memory.add(u_to_evict)
            
        # 4. Proactively store values no longer needed to free space
        # This helps prevent future loads
        if len(fast_memory) >= fast_memory_budget:
            for u in list(fast_memory):
                if len(fast_memory) < fast_memory_budget:
                    break
                if u in outputs or u == v:
                    continue
                # If all children are computed, u is no longer needed
                if all(c in computed for c in children_map.get(u, [])):
                    operations.append(['store', u])
                    fast_memory.discard(u)
                    slow_memory.add(u)
                    
    # 5. Ensure outputs are in fast memory at the end
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
