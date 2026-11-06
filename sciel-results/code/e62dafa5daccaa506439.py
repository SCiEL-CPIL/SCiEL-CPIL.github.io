"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Optimize DAG pebbling by minimizing transfers.
    
    Strategy: Greedy pebbling with lookahead eviction based on remaining uses.
    - Process nodes in topological order.
    - Keep nodes in fast memory that will be needed soonest and frequently.
    - Evict nodes needed latest or no longer needed.
    - Minimize redundant loads/stores.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        outputs = dag.get('outputs', [])
        if not vertices:
            vertices = dag.get('nodes', [])
            edges = dag.get('arcs', [])
            outputs = dag.get('sinks', [])
    elif isinstance(dag, (list, tuple)):
        vertices = []
        edges = []
        outputs = []
        for item in dag:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    v, parents, is_out = item[0], item[1], item[2]
                    vertices.append(v)
                    if is_out:
                        outputs.append(v)
                    for p in parents:
                        edges.append((p, v))
                elif len(item) == 2:
                    v, parents = item[0], item[1]
                    vertices.append(v)
                    for p in parents:
                        edges.append((p, v))
    else:
        # Fallback to baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    # If we can't parse the structure, use baseline
    if not vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    # Build adjacency structures
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    all_vertices = set()
    output_set = set()
    
    for v in vertices:
        all_vertices.add(v)
    
    for edge in edges:
        if len(edge) == 2:
            p, c = edge[0], edge[1]
            parents_map[c].append(p)
            children_map[p].append(c)
            all_vertices.add(p)
            all_vertices.add(c)
    
    for v in outputs:
        output_set.add(v)
        all_vertices.add(v)
    
    if not all_vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    # Topological sort
    in_degree = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for p in parents_map.get(v, []):
            in_degree[v] += 1
    
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    # Check time
    if time.time() > deadline:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
    
    # Identify input vertices and output vertices
    input_vertices = [v for v in topo_order if not parents_map.get(v)]
    output_vertices = [v for v in topo_order if v in output_set]
    
    if not output_vertices:
        output_vertices = [v for v in topo_order if not children_map.get(v)]
    
    # Compute topological position for each vertex
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Precompute: for each vertex, the list of children in topo order
    # And the "next use" index for each vertex
    # next_use[v] = the topological index of the child of v that is processed earliest
    # We need to track remaining uses dynamically.
    
    # For each vertex, store the list of children
    children_list = {v: children_map.get(v, []) for v in all_vertices}
    # Sort children by topological order
    for v in children_list:
        children_list[v].sort(key=lambda x: topo_pos[x])
    
    # Initialize state
    operations = []
    fast_mem = set()      # vertices currently in fast memory
    slow_mem = set()      # vertices currently in slow memory (computed and stored)
    computed = set()      # all vertices that have been computed
    
    # To efficiently find the "best" eviction candidate, we can maintain a heap
    # Key: (next_use_pos, -remaining_uses) -> We want to evict the one with MAX next_use_pos
    # Actually, we want to evict the one that is LEAST urgent.
    # Urgency: High if next_use is close AND remaining_uses is high.
    # So we evict the one with: LOWEST urgency.
    # Let's use a simple scan if the budget is small, or a heap if large.
    # Given constraints, a scan of `fast_mem` (size <= budget) is O(Budget).
    # If Budget is small (e.g., < 1000), scan is fine.
    
    # We track the "next child index" for each vertex to determine next use
    next_child_idx = {v: 0 for v in all_vertices}
    
    def get_next_use(v):
        """Returns the topological position of the next child of v, or inf if none."""
        idx = next_child_idx[v]
        children = children_list[v]
        if idx < len(children):
            return topo_pos[children[idx]]
        return float('inf')
    
    def get_remaining_uses(v):
        """Returns the number of children of v not yet processed."""
        idx = next_child_idx[v]
        children = children_list[v]
        return len(children) - idx
    
    def evict_one():
        """Evict the least urgent vertex from fast_mem."""
        if not fast_mem:
            return
        
        best_v = None
        best_score = -1 # We want to maximize "distance to next use"
        
        for v in fast_mem:
            next_use = get_next_use(v)
            rem_uses = get_remaining_uses(v)
            
            # If no remaining uses, evict immediately (score = inf)
            if rem_uses == 0:
                best_v = v
                best_score = float('inf')
                break
            
            # Score: We prefer to evict nodes with later next use.
            # If next use is the same, prefer evicting nodes with fewer remaining uses.
            # Score = next_use * 1000 - rem_uses
            score = next_use * 1000 - rem_uses
            
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v is not None:
            operations.append(['store', best_v])
            fast_mem.discard(best_v)
            slow_mem.add(best_v)
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
        
        # Load parents that are not in fast memory
        for p in parents_map.get(v, []):
            if p not in fast_mem:
                # Load from slow memory
                operations.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        fast_mem.add(v)
        
        # If over budget, evict something
        while len(fast_mem) > fast_memory_budget:
            evict_one()
            
        # Optional: Proactive eviction?
        # If we have space, we might evict nodes that are no longer needed to free space for future loads?
        # No, if we have space, keep them. It's better to have them in fast memory.
        
        # Update next_child_idx for children of v
        # Actually, we update next_child_idx for v's children? No, we update the "consumption" of v.
        # When we process a child c, we consume one use of its parent p.
        # So, after processing v, we should mark that v has "used" one of its parents?
        # No, the "use" is by the child.
        # When we process node v, we are consuming the values of its parents.
        # So for each parent p of v, we should increment next_child_idx[p]?
        # Yes! The parent p is "used" by v.
        
        for p in parents_map.get(v, []):
            next_child_idx[p] += 1
    
    # Ensure all outputs are in fast memory at the end
    for v in output_vertices:
        if v not in fast_mem:
            operations.append(['load', v])
            fast_mem.add(v)
            slow_mem.discard(v)
    
    # Final cleanup: store everything that's not an output
    for v in list(fast_mem):
        if v not in output_set:
            operations.append(['store', v])
            fast_mem.discard(v)
            slow_mem.add(v)
    
    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
