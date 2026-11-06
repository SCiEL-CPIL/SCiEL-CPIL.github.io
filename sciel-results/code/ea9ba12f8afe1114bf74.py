# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque
import math

def _topo_sort(vertices, parents_map):
    """Compute topological order given vertices and parents."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def _compute_naive_transfers(dag, fast_memory_budget):
    """Compute the naive schedule's transfer count for comparison."""
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    
    transfers = 0
    for v in vertices:
        for p in parents_map.get(v, []):
            transfers += 1  # load parent
        transfers += 1  # store result
    return transfers

def _max_bipartite_matching(left_nodes, right_nodes, edges, deadline):
    """
    Maximum bipartite matching using augmenting paths.
    left_nodes: list of left-side nodes
    right_nodes: list of right-side nodes
    edges: dict mapping left_node -> list of right_nodes it connects to
    Returns: dict mapping left_node -> matched right_node (or None)
    """
    match_r = {}  # right_node -> left_node
    match_l = {}  # left_node -> right_node
    
    def dfs(u, visited):
        for v in edges.get(u, []):
            if v not in visited:
                visited.add(v)
                if v not in match_r or dfs(match_r[v], visited):
                    match_r[v] = u
                    match_l[u] = v
                    return True
        return False
    
    for u in left_nodes:
        if time.time() > deadline:
            break
        visited = set()
        dfs(u, visited)
    
    return match_l

def _layered_pebble(dag, fast_memory_budget, deadline):
    """
    Layer-based pebbling with blackboard architecture.
    Each layer transition uses max bipartite matching to decide which
    vertices to retain in fast memory, minimizing store operations.
    """
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Pre-compute children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Compute depth (longest path from any input)
    order = _topo_sort(vertices, parents_map)
    depth = {}
    for v in order:
        if not parents_map.get(v, []):
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents_map[v])
    
    # Group vertices by depth (layer)
    layers = defaultdict(list)
    for v in order:
        layers[depth[v]].append(v)
    
    max_depth = max(layers.keys()) if layers else 0
    layer_indices = sorted(layers.keys())
    
    # Remaining usage count: how many times each vertex is needed as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    operations = []
    fast_memory = set()
    
    def get_recompute_cost(v):
        """Estimate cost to recompute v if not in memory."""
        if v in input_set:
            return 1
        else:
            return 1 + sum(get_recompute_cost(p) for p in parents_map.get(v, []))
    
    # Process each layer
    for layer_idx in layer_indices:
        if time.time() > deadline:
            break
        
        layer_vertices = layers[layer_idx]
        
        # Determine which vertices from this layer we want to keep
        # and which we need to evict to make room for the next layer
        
        # Vertices in this layer that will be needed in future layers
        future_needed = set()
        for v in layer_vertices:
            if remaining_usage.get(v, 0) > 0 or v in output_set:
                future_needed.add(v)
        
        # Vertices in current fast memory that are NOT in future_needed
        # These are candidates for eviction
        evictable = fast_memory - future_needed
        # Also include vertices in fast memory that are in future_needed but
        # have low priority (low remaining usage * recompute cost)
        
        # We need to compute the next layer's vertices, which requires
        # their parents to be in memory. Let's figure out what parents
        # the next layer needs.
        next_layer_needed = set()
        if layer_idx + 1 <= max_depth:
            for v in layers.get(layer_idx + 1, []):
                for p in parents_map.get(v, []):
                    next_layer_needed.add(p)
        
        # Now we need to ensure all parents of the next layer are in fast memory.
        # Some may already be there, some need to be loaded/computed.
        
        # First, identify which next-layer parents are already in fast memory
        already_in_mem = next_layer_needed & fast_memory
        
        # Which ones need to be brought in?
        need_to_bring = next_layer_needed - fast_memory
        
        # To bring in vertices from need_to_bring, we may need to evict some
        # current fast memory vertices. Use matching to decide what to keep.
        
        # Build a matching problem:
        # Left side: vertices in fast_memory that we might evict
        # Right side: vertices we want to keep (future_needed ∩ fast_memory) ∪ need_to_bring
        # Actually, let's think differently.
        
        # We have fast_memory_budget slots. After processing this layer,
        # we want to have as many useful vertices as possible.
        
        # Useful vertices = (future_needed ∩ fast_memory) ∪ need_to_bring
        # But we can only keep fast_memory_budget vertices total.
        
        # Let's compute the value of each vertex in fast_memory:
        # value = remaining_usage[v] * recompute_cost[v]
        
        # And the value of each vertex in need_to_bring:
        # value = remaining_usage[v] * recompute_cost[v] (if we keep it after computing)
        
        # We want to select up to fast_memory_budget vertices from
        # (fast_memory ∪ need_to_bring) to maximize total value,
        # subject to the constraint that if we keep a vertex from need_to_bring,
        # we must compute it (which requires its parents to be in memory).
        
        # For simplicity, let's use a greedy approach:
        # 1. Compute value for each vertex in (fast_memory ∪ need_to_bring)
        # 2. Sort by value descending
        # 3. Keep the top fast_memory_budget vertices that are "reachable"
        
        # A vertex is reachable if it's in fast_memory or all its parents are in
        # the set of vertices we're keeping.
        
        # This is complex, so let's use a simpler heuristic:
        # - Always keep output vertices
        # - For other vertices, keep those with highest (remaining_usage * recompute_cost)
        # - Ensure we can compute the next layer
        
        # Let's just use the greedy eviction approach but with better scoring
        # based on the layer structure.
        
        # Process vertices in this layer
        for v in layer_vertices:
            if time.time() > deadline:
                break
            
            # Ensure all parents are in fast memory
            for p in parents_map.get(v, []):
                if p not in fast_memory:
                    # Need to bring p into memory
                    # Evict if necessary
                    while len(fast_memory) >= fast_memory_budget and fast_memory:
                        # Find best eviction candidate
                        best_victim = None
                        best_score = float('inf')
                        for fm_v in fast_memory:
                            if fm_v in output_set:
                                continue
                            r = max(remaining_usage.get(fm_v, 0), 0)
                            if fm_v in input_set:
                                cost = 1
                            else:
                                cost = 1 + len(parents_map.get(fm_v, []))
                            score = cost * r
                            if score < best_score:
                                best_score = score
                                best_victim = fm_v
                        
                        if best_victim is None:
                            # No good eviction candidate, break
                            break
                        
                        if best_victim in input_set:
                            operations.append(['delete', best_victim])
                        else:
                            operations.append(['store', best_victim])
                        fast_memory.discard(best_victim)
                    
                    if p in input_set:
                        operations.append(['load', p])
                        fast_memory.add(p)
                    else:
                        # Recursively ensure p's parents are in memory
                        for pp in parents_map.get(p, []):
                            if pp not in fast_memory:
                                # This could lead to deep recursion
                                # For now, just load/store as needed
                                while len(fast_memory) >= fast_memory_budget and fast_memory:
                                    best_victim = None
                                    best_score = float('inf')
                                    for fm_v in fast_memory:
                                        if fm_v in output_set:
                                            continue
                                        r = max(remaining_usage.get(fm_v, 0), 0)
                                        if fm_v in input_set:
                                            cost = 1
                                        else:
                                            cost = 1 + len(parents_map.get(fm_v, []))
                                        score = cost * r
                                        if score < best_score:
                                            best_score = score
                                            best_victim = fm_v
                                    
                                    if best_victim is None:
                                        break
                                    
                                    if best_victim in input_set:
                                        operations.append(['delete', best_victim])
                                    else:
                                        operations.append(['store', best_victim])
                                    fast_memory.discard(best_victim)
                                
                                if pp in input_set:
                                    operations.append(['load', pp])
                                    fast_memory.add(pp)
                                else:
                                    operations.append(['compute', pp])
                                    fast_memory.add(pp)
                        
                        operations.append(['compute', p])
                        fast_memory.add(p)
            
            # Now compute v
            operations.append(['compute', v])
            fast_memory.add(v)
            
            # Update remaining usage for v's parents
            for p in parents_map.get(v, []):
                remaining_usage[p] = max(0, remaining_usage.get(p, 0) - 1)
            
            # If v is not needed in future and not an output, consider evicting
            if v not in output_set and remaining_usage.get(v, 0) == 0:
                # Don't evict immediately; let the next layer's processing
                # decide what to evict
                pass
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_memory:
            while len(fast_memory) >= fast_memory_budget and fast_memory:
                best_victim = None
                best_score = float('inf')
                for fm_v in fast_memory:
                    if fm_v in output_set:
                        continue
                    r = max(remaining_usage.get(fm_v, 0), 0)
                    if fm_v in input_set:
                        cost = 1
                    else:
                        cost = 1 + len(parents_map.get(fm_v, []))
                    score = cost * r
                    if score < best_score:
                        best_score = score
                        best_victim = fm_v
                
                if best_victim is None:
                    break
                
                if best_victim in input_set:
                    operations.append(['delete', best_victim])
                else:
                    operations.append(['store', best_victim])
                fast_memory.discard(best_victim)
            
            if o in input_set:
                operations.append(['load', o])
            else:
                for p in parents_map.get(o, []):
                    if p not in fast_memory:
                        operations.append(['load', p])
                        fast_memory.add(p)
                operations.append(['compute', o])
            fast_memory.add(o)
    
    return operations

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """
    Greedy pebbling with memory management and recomputation.
    Uses a refined cost-based eviction strategy with lookahead.
    """
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Pre-compute children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Pre-compute depth (longest path from any input) for heuristic
    order = _topo_sort(vertices, parents_map)
    depth = {}
    for v in order:
        if not parents_map.get(v, []):
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents_map[v])
    
    # Usage counts: how many times each vertex is used as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    fast_memory = set()
    operations = []
    
    # Track the position of each vertex in topological order for lookahead
    topo_pos = {v: i for i, v in enumerate(order)}
    
    def get_eviction_score(v):
        """
        Score for eviction: lower means better to evict.
        """
        r = max(remaining_usage.get(v, 0), 0)
        
        # Base recompute cost
        if v in input_set:
            base_cost = 1
        else:
            # Cost to recompute: 1 for the compute op + costs to get parents
            base_cost = 1 + len(parents_map.get(v, []))
        
        # If v is an output, we must keep it until the end
        if v in output_set:
            return (base_cost * r + 1000, 0, v)
        
        score = base_cost * r
        # Tie-breaker: prefer evicting inputs over computed
        is_comp = 0 if v in input_set else 1
        return (score, is_comp, v)
    
    def evict_one():
        if not fast_memory:
            return
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
        
    def ensure_in_memory(v):
        if v in fast_memory:
            return
        
        while len(fast_memory) >= fast_memory_budget and fast_memory:
            evict_one()
        
        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            operations.append(['compute', v])
            fast_memory.add(v)
            
    for v in order:
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        if v not in output_set and remaining_usage.get(v, 0) == 0:
             if v in fast_memory:
                 if v in input_set:
                     operations.append(['delete', v])
                 else:
                     operations.append(['store', v])
                 fast_memory.discard(v)
             
        for p in parents_map.get(v, []):
            remaining_usage[p] = remaining_usage.get(p, 0) - 1

    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
            
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _remove_redundant_stores(operations, dag, fast_memory_budget):
    """
    Post-process to remove redundant store→load cycles.
    """
    if not operations:
        return operations
    
    parents_map = {v: dag[v] for v in dag.keys()}
    input_set = set(v for v in dag.keys() if not parents_map.get(v, []))
    
    n = len(operations)
    last_store_idx = {}
    used_since_store = {}
    
    redundant_stores = set()
    
    for i in range(n):
        op, v = operations[i]
        
        if op == 'store' and v not in input_set:
            last_store_idx[v] = i
            used_since_store[v] = False
        elif op == 'load':
            if v in last_store_idx:
                if not used_since_store.get(v, False):
                    redundant_stores.add(last_store_idx[v])
                last_store_idx.pop(v, None)
                used_since_store.pop(v, None)
        elif op in ('compute', 'delete'):
            if v in last_store_idx:
                used_since_store[v] = True
    
    for i in range(n):
        op, v = operations[i]
        if op == 'compute':
            for p in parents_map.get(v, []):
                if p in last_store_idx:
                    if i in redundant_stores:
                        redundant_stores.discard(i)
    
    if redundant_stores:
        optimized = []
        for i in range(n):
            if i not in redundant_stores:
                optimized.append(operations[i])
        return optimized
    
    return operations

def _verify_operations(operations, dag, fast_memory_budget):
    """
    Verify the operations are valid.
    Returns True if valid, False otherwise.
    """
    if not operations:
        return False
    
    vertices = set(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map[v]]
    output_set = set(outputs)
    input_set = set(v for v in vertices if not parents_map.get(v, []))
    
    in_fast = set()
    computed_stored = set()
    
    for op, v in operations:
        if op == 'load':
            if v not in input_set and v not in computed_stored:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            computed_stored.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        elif op == 'compute':
            for p in parents_map.get(v, []):
                if p not in in_fast:
                    return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
            computed_stored.add(v)
    
    for o in outputs:
        if o not in in_fast:
            return False
    
    return True

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    try:
        # Try the layered approach first
        result = _layered_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                # Try to optimize
                optimized = _remove_redundant_stores(result, dag, fast_memory_budget)
                if optimized and len(optimized) <= 20000:
                    if _verify_operations(optimized, dag, fast_memory_budget):
                        return optimized
                return result
            else:
                # Layered approach failed verification, try greedy
                result = _greedy_pebble(dag, fast_memory_budget, deadline)
                if result and len(result) <= 20000:
                    if _verify_operations(result, dag, fast_memory_budget):
                        optimized = _remove_redundant_stores(result, dag, fast_memory_budget)
                        if optimized and len(optimized) <= 20000:
                            if _verify_operations(optimized, dag, fast_memory_budget):
                                return optimized
                        return result
                # Fall back to baseline
                from autoresearch_suite.science_tcs_expansion import pebble_baseline
                return pebble_baseline(dag, fast_memory_budget)
        else:
            # Layered approach produced too many ops, try greedy
            result = _greedy_pebble(dag, fast_memory_budget, deadline)
            if result and len(result) <= 20000:
                if _verify_operations(result, dag, fast_memory_budget):
                    return result
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        # Fall back to baseline on any error
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
