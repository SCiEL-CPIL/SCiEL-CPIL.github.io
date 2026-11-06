# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

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

def _compute_subtree_size(v, parents_map, children_map, memo=None):
    """Compute the number of descendants of v (subtree size)."""
    if memo is None:
        memo = {}
    if v in memo:
        return memo[v]
    if not children_map[v]:
        memo[v] = 1
        return 1
    size = 1
    for c in children_map[v]:
        size += _compute_subtree_size(c, parents_map, children_map, memo)
    memo[v] = size
    return size

def _greedy_pebble_heap(dag, fast_memory_budget, deadline):
    """
    Improved greedy pebbling with better eviction strategy.
    Uses a cost-aware eviction that balances remaining usage against recomputation cost.
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
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
            
    # Pre-compute metrics
    remaining_usage = {v: len(children_map[v]) for v in vertices}
    
    order = _topo_sort(vertices, parents_map)
    order_index = {v: i for i, v in enumerate(order)}
    
    # Compute subtree sizes for recomputation cost estimation
    subtree_memo = {}
    subtree_size = {v: _compute_subtree_size(v, parents_map, children_map, subtree_memo) for v in vertices}
    
    # Fast memory state
    fast_mem = set()
    # State: 'in_fast', 'in_slow', 'stored', 'gone'
    state = {v: 'in_slow' if v in input_set else 'gone' for v in vertices}
    
    ops = []
    ops_count = 0
    max_ops = 20000
    
    def get_eviction_cost(v):
        """
        Returns the cost of evicting v. Lower cost = better eviction candidate.
        We want to evict vertices that are cheap to restore.
        
        Cost factors:
        1. If remaining_usage == 0: cost = 0 (must evict, no penalty)
        2. If input: cost = 1 (just reload, cheap)
        3. If computed: cost = subtree_size[v] * remaining_usage[v] (expensive to recompute)
        4. Add small tie-breaker based on vertex id
        """
        r = remaining_usage.get(v, 0)
        if r == 0:
            return (0, 0, v)
        
        if v in input_set:
            # Inputs are cheap to reload
            return (1, r, v)
        else:
            # Computed vertices: cost is proportional to recompute effort
            # subtree_size gives us the number of operations to recompute
            # remaining_usage gives us how many times we'll need it
            cost = subtree_size[v] * max(r, 1)
            return (cost, r, v)
    
    def evict_one():
        nonlocal ops_count
        if not fast_mem:
            return False
        
        # Find the vertex with lowest eviction cost
        victim = min(fast_mem, key=get_eviction_cost)
        
        r = remaining_usage.get(victim, 0)
        
        if r == 0:
            # No more uses, just delete
            ops.append(['delete', victim])
            ops_count += 1
        else:
            # Has more uses
            if victim in input_set:
                ops.append(['delete', victim])
                ops_count += 1
                state[victim] = 'in_slow'
            else:
                ops.append(['store', victim])
                ops_count += 1
                state[victim] = 'stored'
        
        fast_mem.discard(victim)
        return True
    
    def make_space(n=1):
        """Ensure at least n slots are available in fast memory."""
        nonlocal ops_count
        while len(fast_mem) >= fast_memory_budget and ops_count < max_ops:
            if not evict_one():
                break
    
    def ensure_in_memory(v):
        """Ensure vertex v is in fast memory. Returns True if successful."""
        nonlocal ops_count
        if v in fast_mem:
            return True
        
        if ops_count >= max_ops:
            return False
        
        # Make space if needed
        make_space(1)
        
        if len(fast_mem) >= fast_memory_budget:
            # Still no space, can't proceed
            return False
        
        if v in input_set:
            ops.append(['load', v])
            ops_count += 1
            fast_mem.add(v)
            state[v] = 'in_fast'
            return True
        else:
            # Check if it was stored
            if state.get(v) == 'stored':
                ops.append(['load', v])
                ops_count += 1
                fast_mem.add(v)
                state[v] = 'in_fast'
                return True
            else:
                # Must compute
                # Ensure all parents are in memory first
                for p in parents_map.get(v, []):
                    if not ensure_in_memory(p):
                        return False
                
                if ops_count >= max_ops:
                    return False
                
                ops.append(['compute', v])
                ops_count += 1
                fast_mem.add(v)
                state[v] = 'in_fast'
                return True
    
    # Process in Topological Order
    for v in order:
        if time.time() > deadline:
            break
        if ops_count >= max_ops:
            break
            
        ensure_in_memory(v)
        
        # Update remaining_usage of parents
        for p in parents_map.get(v, []):
            if p in remaining_usage:
                remaining_usage[p] -= 1
        
        # If v is not an output and has no remaining usage, we could evict it now
        # to free up space, but we'll let the eviction strategy handle it
        # when we need space.
        
    # Ensure outputs are in memory
    for o in outputs:
        if o not in fast_mem:
            if time.time() > deadline or ops_count >= max_ops:
                break
            ensure_in_memory(o)
        
    return ops

def _count_transfers(operations):
    return sum(1 for op, v in operations if op in ('load', 'store'))

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80
    
    try:
        result = _greedy_pebble_heap(dag, fast_memory_budget, deadline)
        
        if result and len(result) <= 20000:
            # Verify validity loosely
            return result
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
