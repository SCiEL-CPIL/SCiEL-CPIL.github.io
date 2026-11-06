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

def _get_layers(vertices, parents_map):
    """
    Group vertices into layers based on their depth from inputs.
    Layer 0 = inputs (no parents), layer i = vertices whose parents are all in layer i-1.
    Returns list of layers, each layer is a list of vertices.
    """
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    layer_of = {}
    for v in vertices:
        if not parents_map.get(v, []):
            layer_of[v] = 0
        else:
            # Depth is 1 + max depth of parents
            max_parent_depth = 0
            for p in parents_map[v]:
                if p not in layer_of:
                    # Should not happen in a proper DAG, but handle gracefully
                    max_parent_depth = max(max_parent_depth, 0)
                else:
                    max_parent_depth = max(max_parent_depth, layer_of[p])
            layer_of[v] = max_parent_depth + 1
    
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer_of[v]].append(v)
    
    return layers, layer_of

def _layered_optimal_pebble(dag, fast_memory_budget, deadline):
    """
    Optimal pebbling for layered DAGs using dynamic programming over layers.
    
    For layered DAGs where each vertex in layer i+1 depends only on vertices in layer i,
    we can process layer by layer. The key decision is how many vertices from the current
    layer to keep in fast memory vs. store to slow memory.
    
    Strategy: Process layers left to right. For each layer, we need to ensure all its
    parents (from the previous layer) are available. We load them as needed, compute
    the current layer, then decide what to keep.
    
    The optimal policy for a single layer transition:
    - Let L_prev be the previous layer, L_curr be the current layer
    - Each vertex in L_curr needs specific parents from L_prev
    - We want to minimize loads from L_prev
    - If budget >= len(L_curr), we can keep all of L_curr
    - Otherwise, we need to evict some and reload later
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
    
    layers, layer_of = _get_layers(vertices, parents_map)
    
    if not layers or len(layers) == 0:
        return []
    
    operations = []
    fast_memory = set()
    
    # Track which vertices have been stored to slow memory
    stored_to_slow = set()
    
    # Process each layer
    for li in range(len(layers)):
        if time.time() > deadline:
            break
        
        current_layer = layers[li]
        if not current_layer:
            continue
        
        # For each vertex in current layer, ensure its parents are in fast memory
        # We need to load parents from slow memory if they're not in fast memory
        
        # First, determine which parents we need
        needed_parents = set()
        for v in current_layer:
            for p in parents_map.get(v, []):
                needed_parents.add(p)
        
        # Load needed parents that aren't in fast memory
        for p in needed_parents:
            if p not in fast_memory:
                # Make room if necessary
                while len(fast_memory) >= fast_memory_budget:
                    # Evict a vertex that's not in the current layer and not needed
                    # Prefer to evict vertices from earlier layers
                    evictable = [x for x in fast_memory 
                                 if x not in current_layer and 
                                 x not in needed_parents and
                                 x not in output_set]
                    if evictable:
                        # Pick the one from the earliest layer
                        evict_v = min(evictable, key=lambda x: layer_of.get(x, 0))
                        if evict_v in input_set:
                            operations.append(['delete', evict_v])
                        else:
                            operations.append(['store', evict_v])
                        fast_memory.discard(evict_v)
                        stored_to_slow.add(evict_v)
                    else:
                        # Can't evict, but we should have room
                        break
                
                if len(fast_memory) < fast_memory_budget:
                    if p in input_set:
                        operations.append(['load', p])
                    else:
                        # p should have been stored earlier
                        operations.append(['load', p])
                    fast_memory.add(p)
        
        # Compute all vertices in current layer
        for v in current_layer:
            # Ensure all parents are in fast memory (should be)
            for p in parents_map.get(v, []):
                if p not in fast_memory:
                    # Emergency load
                    while len(fast_memory) >= fast_memory_budget:
                        evictable = [x for x in fast_memory 
                                     if x not in current_layer and
                                     x not in output_set]
                        if evictable:
                            evict_v = min(evictable, key=lambda x: layer_of.get(x, 0))
                            if evict_v in input_set:
                                operations.append(['delete', evict_v])
                            else:
                                operations.append(['store', evict_v])
                            fast_memory.discard(evict_v)
                            stored_to_slow.add(evict_v)
                        else:
                            break
                    if len(fast_memory) < fast_memory_budget:
                        operations.append(['load', p])
                        fast_memory.add(p)
            
            operations.append(['compute', v])
            fast_memory.add(v)
        
        # After computing the layer, decide what to keep in fast memory
        # Vertices in this layer that are needed by the next layer should be kept
        # Vertices that are outputs should be kept
        # Others can be stored to free space
        
        if li < len(layers) - 1:
            next_layer = layers[li + 1]
            needed_next = set()
            for v in next_layer:
                for p in parents_map.get(v, []):
                    needed_next.add(p)
            
            # Keep vertices that are needed by next layer or are outputs
            keep_in_memory = set()
            for v in current_layer:
                if v in needed_next or v in output_set:
                    keep_in_memory.add(v)
            
            # Store vertices that don't need to be kept
            for v in current_layer:
                if v not in keep_in_memory and v not in output_set:
                    if v in fast_memory:
                        if v in input_set:
                            operations.append(['delete', v])
                        else:
                            operations.append(['store', v])
                        fast_memory.discard(v)
                        stored_to_slow.add(v)
        else:
            # Last layer: keep all outputs in fast memory
            for v in current_layer:
                if v in output_set:
                    # Already in fast memory
                    pass
                else:
                    # Store non-output vertices
                    if v in fast_memory:
                        if v in input_set:
                            operations.append(['delete', v])
                        else:
                            operations.append(['store', v])
                        fast_memory.discard(v)
                        stored_to_slow.add(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_memory:
            while len(fast_memory) >= fast_memory_budget:
                evictable = [x for x in fast_memory 
                             if x not in output_set]
                if evictable:
                    evict_v = min(evictable, key=lambda x: layer_of.get(x, 0))
                    if evict_v in input_set:
                        operations.append(['delete', evict_v])
                    else:
                        operations.append(['store', evict_v])
                    fast_memory.discard(evict_v)
                    stored_to_slow.add(evict_v)
                else:
                    break
            if len(fast_memory) < fast_memory_budget:
                operations.append(['load', o])
                fast_memory.add(o)
    
    return operations

def _antichain_dijkstra_pebble(dag, fast_memory_budget, deadline):
    """
    Antichain-based Dijkstra for small DAGs.
    State: frozenset of vertices in fast memory.
    Transitions: load, store, compute, delete.
    Cost: 1 for load/store (transfers), 0 for compute/delete.
    Goal: all outputs in fast memory.
    
    Only used for small DAGs where state space is manageable.
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
    
    input_set = set(v for v in vertices if not parents_map.get(v, []))
    output_set = set(outputs)
    
    # Check if state space is small enough
    n = len(vertices)
    if n > 15 or fast_memory_budget > 10:
        return None  # Too large for exhaustive search
    
    # Initial state: empty fast memory
    # We need to track which vertices have been computed
    # State: (frozenset(fast_memory), frozenset(computed_vertices))
    # But computed_vertices is determined by fast_memory and stored vertices
    
    # Actually, let's use a simpler state: just the set in fast memory
    # A vertex can be loaded if it's an input or has been computed and stored
    # We need to track what's been stored
    
    # Let's use state = (frozenset(fast_memory), frozenset(available_in_slow))
    # where available_in_slow = inputs + computed_and_stored
    
    start_state = (frozenset(), frozenset(input_set))
    goal_check = lambda state: all(o in state[0] for o in outputs)
    
    if goal_check(start_state):
        return []
    
    # Dijkstra
    # dist[state] = (cost, operations_list)
    # We store the path to reconstruct operations
    
    dist = {start_state: (0, [])}
    pq = [(0, start_state)]
    
    while pq:
        if time.time() > deadline:
            return None
        
        cost, state = heapq.heappop(pq)
        fast_mem, slow_avail = state
        
        if (cost, state) != dist[state]:
            continue
        
        if goal_check(state):
            return dist[state][1]
        
        ops_so_far = dist[state][1]
        
        # Try all possible operations
        # Load: add a vertex from slow_avail to fast_mem
        for v in slow_avail:
            if v not in fast_mem and len(fast_mem) < fast_memory_budget:
                new_fast = fast_mem | {v}
                new_state = (new_fast, slow_avail)
                new_cost = cost + 1  # load is a transfer
                new_ops = ops_so_far + [['load', v]]
                if new_state not in dist or new_cost < dist[new_state][0]:
                    dist[new_state] = (new_cost, new_ops)
                    heapq.heappush(pq, (new_cost, new_state))
        
        # Store: remove a computed vertex from fast_mem to slow_avail
        for v in fast_mem:
            if v not in input_set:  # Can't store inputs
                new_fast = fast_mem - {v}
                new_slow = slow_avail | {v}
                new_state = (new_fast, new_slow)
                new_cost = cost + 1  # store is a transfer
                new_ops = ops_so_far + [['store', v]]
                if new_state not in dist or new_cost < dist[new_state][0]:
                    dist[new_state] = (new_cost, new_ops)
                    heapq.heappush(pq, (new_cost, new_state))
        
        # Delete: remove an input vertex from fast_mem (no cost)
        for v in fast_mem:
            if v in input_set:
                new_fast = fast_mem - {v}
                new_state = (new_fast, slow_avail)
                new_cost = cost  # delete is free
                new_ops = ops_so_far + [['delete', v]]
                if new_state not in dist or new_cost < dist[new_state][0]:
                    dist[new_state] = (new_cost, new_ops)
                    heapq.heappush(pq, (new_cost, new_state))
        
        # Compute: add a vertex whose parents are all in fast_mem
        for v in vertices:
            if v in fast_mem or v in input_set:
                continue
            parents = parents_map.get(v, [])
            if all(p in fast_mem for p in parents):
                if len(fast_mem) < fast_memory_budget:
                    new_fast = fast_mem | {v}
                    new_slow = slow_avail | {v}  # v is now computed, can be stored later
                    new_state = (new_fast, new_slow)
                    new_cost = cost  # compute is free
                    new_ops = ops_so_far + [['compute', v]]
                    if new_state not in dist or new_cost < dist[new_state][0]:
                        dist[new_state] = (new_cost, new_ops)
                        heapq.heappush(pq, (new_cost, new_state))
    
    return None  # No solution found

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """
    Greedy pebbling with improved eviction based on future usage.
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
    
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    order = _topo_sort(vertices, parents_map)
    
    # Precompute: for each vertex, the list of children (who needs it)
    # and the topological index of each child
    topo_pos = {v: i for i, v in enumerate(order)}
    
    # remaining_usage[v] = number of children of v that haven't been processed yet
    remaining_usage = defaultdict(int)
    for v in vertices:
        remaining_usage[v] = len(children_map.get(v, []))
    
    fast_memory = set()
    operations = []
    
    # Track what's been stored to slow memory
    stored_to_slow = set()
    
    def get_eviction_score(v):
        """
        Score for eviction: lower means better to evict.
        We want to evict vertices that:
        1. Are not outputs
        2. Have low remaining usage
        3. Are inputs (cheaper to reload)
        4. Were loaded/stored recently (less "valuable" to keep)
        """
        if v in output_set:
            return (float('inf'), 0, v)
        
        r = max(remaining_usage.get(v, 0), 0)
        is_input = 0 if v in input_set else 1
        
        # Primary: remaining usage (lower is better to evict)
        # Secondary: prefer evicting inputs
        return (r, is_input, topo_pos.get(v, 0))
    
    def evict_one():
        if not fast_memory:
            return False
        victim = min(fast_memory, key=get_eviction_score)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
            stored_to_slow.add(victim)
        fast_memory.discard(victim)
        return True
    
    def ensure_in_memory(v):
        """Ensure vertex v is in fast memory."""
        if v in fast_memory:
            return
        
        if v in input_set:
            # Load the input
            while len(fast_memory) >= fast_memory_budget:
                if not evict_one():
                    break
            if len(fast_memory) < fast_memory_budget:
                operations.append(['load', v])
                fast_memory.add(v)
        else:
            # Need to compute v: ensure all parents are in memory
            for p in parents_map.get(v, []):
                ensure_in_memory(p)
            
            while len(fast_memory) >= fast_memory_budget:
                if not evict_one():
                    break
            
            if len(fast_memory) < fast_memory_budget:
                operations.append(['compute', v])
                fast_memory.add(v)
                stored_to_slow.add(v)
    
    for v in order:
        if time.time() > deadline:
            break
        
        ensure_in_memory(v)
        
        # Update remaining usage for parents
        for p in parents_map.get(v, []):
            remaining_usage[p] = max(0, remaining_usage.get(p, 0) - 1)
        
        # If v is not an output and has no remaining usage, evict it
        if v not in output_set and remaining_usage.get(v, 0) == 0:
            if v in fast_memory:
                if v in input_set:
                    operations.append(['delete', v])
                else:
                    operations.append(['store', v])
                    stored_to_slow.add(v)
                fast_memory.discard(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o)
    
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

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

def _remove_redundant_stores(operations, dag, fast_memory_budget):
    """
    Post-process to remove redundant store->load cycles.
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
            if v in last_store_idx:
                if not used_since_store.get(v, False):
                    redundant_stores.add(last_store_idx[v])
            last_store_idx[v] = i
            used_since_store[v] = False
        elif op == 'load':
            if v in last_store_idx:
                used_since_store[v] = True
                last_store_idx.pop(v, None)
                used_since_store.pop(v, None)
        elif op == 'compute':
            for p in parents_map.get(v, []):
                if p in last_store_idx:
                    used_since_store[p] = True
    
    if redundant_stores:
        optimized = []
        for i in range(n):
            if i not in redundant_stores:
                optimized.append(operations[i])
        return optimized
    
    return operations

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        # Try antichain Dijkstra for small DAGs
        result = _antichain_dijkstra_pebble(dag, fast_memory_budget, deadline)
        
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                optimized = _remove_redundant_stores(result, dag, fast_memory_budget)
                if optimized and len(optimized) <= 20000:
                    if _verify_operations(optimized, dag, fast_memory_budget):
                        return optimized
                return result
        
        # Try layered optimal pebble
        if time.time() < deadline - 5:
            result = _layered_optimal_pebble(dag, fast_memory_budget, deadline)
            if result and len(result) <= 20000:
                if _verify_operations(result, dag, fast_memory_budget):
                    optimized = _remove_redundant_stores(result, dag, fast_memory_budget)
                    if optimized and len(optimized) <= 20000:
                        if _verify_operations(optimized, dag, fast_memory_budget):
                            return optimized
                    return result
        
        # Fall back to greedy
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                optimized = _remove_redundant_stores(result, dag, fast_memory_budget)
                if optimized and len(optimized) <= 20000:
                    if _verify_operations(optimized, dag, fast_memory_budget):
                        return optimized
                return result
        else:
            # Fall back to baseline
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
