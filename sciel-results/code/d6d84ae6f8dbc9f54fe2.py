"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 75.0  # wall-clock limit from evaluator with safety margin

    # Try to import baseline as fallback
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        if not parents and not outputs and not inputs:
            return []

        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order using Kahn's algorithm with a list (FIFO for determinism)
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = sorted([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    queue.sort()
        
        # If topo_order doesn't contain all vertices (cycle?), add remaining
        remaining = all_vertices - set(topo_order)
        if remaining:
            topo_order.extend(sorted(remaining))

        # Map vertex to index in topo_order for O(1) lookup
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute last_use: for each vertex, the highest topo_index of any child
        # Outputs must be kept until the end, so their last_use is len(topo_order) - 1
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
            
            # If v is an output, it must be kept until the end
            if v in outputs:
                last_use[v] = len(topo_order) - 1

        # Precompute depth and recompute_cost (number of descendants) for eviction heuristic
        depth = {}
        recompute_cost = {}
        
        # Compute depth
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1
        
        # Compute recompute_cost: number of descendants (including self) that would need recomputation
        # Process in reverse topological order
        for v in reversed(topo_order):
            cs = children.get(v, [])
            if not cs:
                recompute_cost[v] = 1
            else:
                recompute_cost[v] = 1 + sum(recompute_cost[c] for c in cs)
        
        must_be_blue = set(outputs)
        
        ops = []
        fast_mem = set()
        
        # Eviction heap: (recompute_cost, last_use, depth, vertex)
        # We use a lazy deletion approach: when we pop from heap, check if vertex is still in fast_mem
        evict_heap = []
        
        def push_evict_candidate(v):
            """Add vertex v to the eviction heap."""
            heapq.heappush(evict_heap, (recompute_cost[v], last_use[v], depth[v], v))
        
        def pop_valid_evict_candidate():
            """
            Pop the best eviction candidate from the heap.
            Skips stale entries (vertices no longer in fast_mem).
            Returns the vertex to evict, or None if heap is empty.
            """
            while evict_heap:
                rc, lu, d, v = heapq.heappop(evict_heap)
                if v in fast_mem:
                    return v
                # Stale entry, discard
            return None

        def ensure_in_fast_mem(v, current_idx):
            """
            Ensure vertex v is in fast_mem.
            Returns True if successful, False if budget exceeded and couldn't free space.
            """
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            if not ps:
                # It's an input
                if len(fast_mem) >= fast_memory_budget:
                    cand = pop_valid_evict_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['load', v])
                fast_mem.add(v)
                push_evict_candidate(v)
                return True
            else:
                # It's a computed vertex
                # First, ensure all its parents are in fast_mem
                for p in ps:
                    if not ensure_in_fast_mem(p, current_idx):
                        return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    cand = pop_valid_evict_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['compute', v])
                fast_mem.add(v)
                push_evict_candidate(v)
                return True
        
        # Process in topological order
        # Track dead vertices for efficient batch deletion
        dead_vertices = set()
        
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            if not ensure_in_fast_mem(v, i):
                break
            
            # Mark vertices as dead if their last_use is before current index
            for u in list(fast_mem):
                if u != v and last_use[u] < i and u not in must_be_blue:
                    dead_vertices.add(u)
            
            # Batch-delete dead vertices: those with last_use < i and not outputs
            # This keeps fast_mem small and reduces eviction pressure
            for u in list(dead_vertices):
                if u in fast_mem:
                    ops.append(['delete', u])
                    fast_mem.discard(u)
                    dead_vertices.discard(u)
        
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                if not ensure_in_fast_mem(o, len(topo_order)):
                    # Last resort: force evict something
                    if len(fast_mem) >= fast_memory_budget:
                        cand = pop_valid_evict_candidate()
                        if cand:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    ensure_in_fast_mem(o, len(topo_order))
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
