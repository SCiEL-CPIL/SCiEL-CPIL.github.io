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

def _compute_subtree_sizes(vertices, parents_map, input_set):
    """
    Precompute the size of the sub-DAG rooted at each vertex.
    subtree_size[v] = 1 if v is an input, else 1 + sum(subtree_size[p] for p in parents[v])
    This represents the number of operations needed to recompute v from scratch.
    """
    order = _topo_sort(vertices, parents_map)
    subtree_size = {}
    for v in order:
        if v in input_set:
            subtree_size[v] = 1
        else:
            subtree_size[v] = 1 + sum(subtree_size.get(p, 1) for p in parents_map.get(v, []))
    return subtree_size

def _greedy_pebble(dag, fast_memory_budget, deadline):
    """
    Greedy pebbling with memory management and recomputation.
    Uses subtree-size-based eviction scoring for better decisions.
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
    
    # Pre-compute topological order
    order = _topo_sort(vertices, parents_map)
    
    # Pre-compute subtree sizes for eviction scoring
    subtree_size = _compute_subtree_sizes(vertices, parents_map, input_set)
    
    # Usage counts: how many times each vertex is used as a parent
    usage_count = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            usage_count[p] += 1
    remaining_usage = dict(usage_count)
    
    fast_memory = set()
    operations = []
    
    # Track which vertices have been computed (for verification)
    computed_set = set()
    
    def get_eviction_score(v):
        """
        Score for eviction: lower means better to evict.
        Uses subtree_size * remaining_usage as the primary metric.
        """
        r = max(remaining_usage.get(v, 0), 0)
        
        # If v is an output, we must keep it until the end
        if v in output_set:
            return (float('inf'), 0, v)
        
        # Primary score: cost to recompute * how many more times we need it
        score = subtree_size.get(v, 1) * r
        
        # Tie-breaker: prefer evicting inputs (cheaper to reload)
        is_input = 0 if v in input_set else 1
        
        return (score, is_input, v)
    
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
        """
        Ensure vertex v is in fast memory.
        Uses iterative approach to avoid deep recursion.
        """
        if v in fast_memory:
            return
        
        # We need to bring v and all its ancestors into memory
        # Build the list of vertices to bring in (v and its parents, recursively)
        to_bring = []
        stack = [v]
        visited = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node not in fast_memory:
                to_bring.append(node)
                if node not in input_set:
                    for p in parents_map.get(node, []):
                        stack.append(p)
        
        # Sort to_bring in topological order (parents before children)
        topo_pos = {v: i for i, v in enumerate(order)}
        to_bring.sort(key=lambda x: topo_pos.get(x, 0))
        
        for node in to_bring:
            if node in fast_memory:
                continue
            
            # Make room if needed
            while len(fast_memory) >= fast_memory_budget and fast_memory:
                evict_one()
            
            if node in input_set:
                operations.append(['load', node])
            else:
                # Ensure parents are in memory (they should be, since we're in topo order)
                for p in parents_map.get(node, []):
                    if p not in fast_memory:
                        # This shouldn't happen if to_bring is in topo order
                        ensure_in_memory(p)
                operations.append(['compute', node])
                computed_set.add(node)
            
            fast_memory.add(node)
    
    for v in order:
        if time.time() > deadline:
            break
            
        ensure_in_memory(v)
        
        # If v is not needed in future and not an output, evict it immediately
        if v not in output_set and remaining_usage.get(v, 0) == 0:
            if v in fast_memory:
                if v in input_set:
                    operations.append(['delete', v])
                else:
                    operations.append(['store', v])
                fast_memory.discard(v)
            
        for p in parents_map.get(v, []):
            remaining_usage[p] = max(0, remaining_usage.get(p, 0) - 1)

    # Ensure all outputs are in fast memory
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
    A store is redundant if the same vertex is loaded again before being used.
    """
    if not operations:
        return operations
    
    parents_map = {v: dag[v] for v in dag.keys()}
    input_set = set(v for v in dag.keys() if not parents_map.get(v, []))
    
    n = len(operations)
    # Track for each vertex: index of last store, and whether it was used since
    last_store_idx = {}
    used_since_store = {}
    
    redundant_stores = set()
    
    for i in range(n):
        op, v = operations[i]
        
        if op == 'store' and v not in input_set:
            # If there was a previous store of v that wasn't used, mark it redundant
            if v in last_store_idx:
                if not used_since_store.get(v, False):
                    redundant_stores.add(last_store_idx[v])
            last_store_idx[v] = i
            used_since_store[v] = False
        elif op == 'load':
            # Loading v means we're using a stored value (or input)
            if v in last_store_idx:
                # The store was used, so it's not redundant
                used_since_store[v] = True
                last_store_idx.pop(v, None)
                used_since_store.pop(v, None)
        elif op == 'compute':
            # Computing v uses its parents, so any stored parents are "used"
            for p in parents_map.get(v, []):
                if p in last_store_idx:
                    used_since_store[p] = True
        elif op == 'delete':
            # Deleting doesn't count as "using" the value for recompute purposes
            pass
    
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

def _local_optimize(operations, dag, fast_memory_budget, deadline):
    """
    Try local swaps to reduce transfer count.
    For each pair of adjacent operations, check if swapping reduces transfers.
    This is a simple hill-climbing approach.
    """
    if not operations or time.time() > deadline:
        return operations
    
    parents_map = {v: dag[v] for v in dag.keys()}
    input_set = set(v for v in dag.keys() if not parents_map.get(v, []))
    
    current = list(operations)
    current_transfers = _count_transfers(current)
    improved = True
    max_iterations = 50
    
    while improved and time.time() < deadline:
        improved = False
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            best_swap = None
            best_transfers = current_transfers
            
            for i in range(len(current) - 1):
                # Try swapping operations i and i+1
                trial = current.copy()
                trial[i], trial[i+1] = trial[i+1], trial[i]
                
                # Quick check: does this still verify?
                if not _verify_operations(trial, dag, fast_memory_budget):
                    continue
                
                t = _count_transfers(trial)
                if t < best_transfers:
                    best_transfers = t
                    best_swap = i
            
            if best_swap is not None:
                current[best_swap], current[best_swap+1] = current[best_swap+1], current[best_swap]
                current_transfers = best_transfers
                improved = True
    
    return current

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        # Try the greedy approach with improved eviction
        result = _greedy_pebble(dag, fast_memory_budget, deadline)
        
        # Verify the result
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                # Try to optimize redundant stores
                optimized = _remove_redundant_stores(result, dag, fast_memory_budget)
                if optimized and len(optimized) <= 20000:
                    if _verify_operations(optimized, dag, fast_memory_budget):
                        # Try local optimization if we have time
                        if time.time() < deadline - 10:
                            local_opt = _local_optimize(optimized, dag, fast_memory_budget, deadline)
                            if local_opt and len(local_opt) <= 20000:
                                if _verify_operations(local_opt, dag, fast_memory_budget):
                                    return local_opt
                        return optimized
                return result
            else:
                # Greedy failed verification, fall back to baseline
                from autoresearch_suite.science_tcs_expansion import pebble_baseline
                return pebble_baseline(dag, fast_memory_budget)
        else:
            # Greedy produced too many ops, fall back to baseline
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
