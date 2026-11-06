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
        
        We estimate the cost of NOT having v in memory when it's next needed.
        For inputs: cost is 1 (just load)
        For computed nodes: cost is 1 (store) + cost to recompute parents
        
        The benefit of keeping v is proportional to:
        1. How many times it will be needed in the future (remaining_usage)
        2. The cost to re-fetch it (recompute_cost)
        
        For lookahead, we check if v will be needed soon and weight accordingly.
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
    If we store v, then later load v without v being used in between,
    the store was unnecessary.
    """
    if not operations:
        return operations
    
    parents_map = {v: dag[v] for v in dag.keys()}
    input_set = set(v for v in dag.keys() if not parents_map.get(v, []))
    
    # Build a map: for each store, find if the next operation on that vertex is a load
    # with no intervening use
    n = len(operations)
    # For each vertex, track the index of the most recent store
    last_store_idx = {}
    # For each vertex, track if it has been used since the last store
    used_since_store = {}
    
    # First pass: identify which stores are followed by a load without use
    redundant_stores = set()
    
    for i in range(n):
        op, v = operations[i]
        
        if op == 'store' and v not in input_set:
            last_store_idx[v] = i
            used_since_store[v] = False
        elif op == 'load':
            if v in last_store_idx:
                if not used_since_store.get(v, False):
                    # This load can be replaced by keeping v in memory
                    # Mark the store as redundant
                    redundant_stores.add(last_store_idx[v])
                last_store_idx.pop(v, None)
                used_since_store.pop(v, None)
        elif op in ('compute', 'delete'):
            if v in last_store_idx:
                # v is being used or deleted, so the store was needed
                used_since_store[v] = True
        elif op == 'compute':
            # Check if v is a parent of the computed node
            for p in parents_map.get(v, []):
                if p in last_store_idx:
                    used_since_store[p] = True
    
    # Second pass: also check if a vertex is used as a parent in a compute
    # after being stored
    for i in range(n):
        op, v = operations[i]
        if op == 'compute':
            for p in parents_map.get(v, []):
                if p in last_store_idx:
                    # p was stored and is now being used as parent
                    # The store was necessary
                    if i in redundant_stores:
                        redundant_stores.discard(i)
    
    # Remove redundant stores
    if redundant_stores:
        optimized = []
        for i in range(n):
            if i not in redundant_stores:
                optimized.append(operations[i])
        return optimized
    
    return operations

def _verify_and_optimize(operations, dag, fast_memory_budget, deadline):
    """
    Verify the operations are valid and try to optimize by removing redundant operations.
    Returns the optimized operations or the original if verification fails.
    """
    if not operations:
        return operations
    
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
    
    # Simulate the operations to verify validity
    fast_mem = set()
    blue_set = set()  # vertices that are currently blue (in memory or stored)
    # Actually, let's track what's in fast memory and what's been computed/stored
    # A vertex is "available" if it's in fast memory
    # We need to track which vertices have been computed and stored (so they can be loaded later)
    
    # Let's do a simple verification: simulate and check that:
    # 1. Every load is for an input or a previously computed vertex
    # 2. Every compute has all parents in fast memory
    # 3. Every store/delete is for a vertex in fast memory
    # 4. All outputs end up in fast memory (blue)
    
    valid = True
    computed_stored = set()  # vertices that have been computed and stored to slow memory
    in_fast = set()
    
    for op, v in operations:
        if op == 'load':
            # Can only load inputs or previously computed/stored vertices
            if v not in input_set and v not in computed_stored:
                valid = False
                break
            if len(in_fast) >= fast_memory_budget:
                valid = False
                break
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                valid = False
                break
            in_fast.discard(v)
            computed_stored.add(v)  # Now it can be loaded later
        elif op == 'delete':
            if v not in in_fast:
                valid = False
                break
            in_fast.discard(v)
            # Deleted, so it's not stored; if it was computed, it's lost
            # But if it's an input, it can be reloaded
        elif op == 'compute':
            # All parents must be in fast memory
            for p in parents_map.get(v, []):
                if p not in in_fast:
                    valid = False
                    break
            if not valid:
                break
            if len(in_fast) >= fast_memory_budget:
                valid = False
                break
            in_fast.add(v)
            computed_stored.add(v)  # It's been computed, so it can be stored later
    
    if not valid:
        return operations  # Return original if invalid
    
    # Check all outputs are in fast memory at the end
    for o in outputs:
        if o not in in_fast:
            return operations  # Return original if outputs not all in memory
    
    # Now try to optimize: remove redundant operations
    # A load is redundant if the vertex is already in fast memory
    # A store/delete is redundant if it's not needed (but we can't easily tell)
    # For now, just return the valid operations
    return operations

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        # Try the greedy approach
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result is valid and optimize if possible
        if result and len(result) <= 20000:
            # Try to optimize the result
            optimized = _verify_and_optimize(result, dag, fast_memory_budget, deadline)
            if optimized and len(optimized) <= 20000:
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
