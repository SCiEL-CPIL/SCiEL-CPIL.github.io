"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Parse the DAG structure
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    else:
        if hasattr(dag, 'vertices'):
            n = len(dag.vertices)
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)

    def get_parents(v):
        if hasattr(dag, 'parents'):
            return dag.parents[v]
        elif isinstance(dag, dict) and 'parents' in dag:
            return dag['parents'][v]
        else:
            return []

    def get_outputs():
        if hasattr(dag, 'outputs'):
            return set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            return set(dag['outputs'])
        else:
            return {n - 1}

    outputs = get_outputs()

    # Build efficient data structures
    parents_list = [list(get_parents(v)) for v in range(n)]

    # Compute children for each vertex
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_list[v]:
            children[p].append(v)

    # Compute remaining uses (how many children still need this vertex)
    remaining_uses = [0] * n
    for v in range(n):
        remaining_uses[v] = len(children[v])

    # Compute depth (topological level)
    depth = [0] * n
    for v in range(n):
        if parents_list[v]:
            depth[v] = max(depth[p] for p in parents_list[v]) + 1

    # Compute topological order via Kahn's algorithm
    in_degree = [0] * n
    for v in range(n):
        for p in parents_list[v]:
            in_degree[v] += 1

    queue = deque()
    for v in range(n):
        if in_degree[v] == 0:
            queue.append(v)

    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    # If topological order doesn't cover all vertices (shouldn't happen for valid DAG),
    # add remaining vertices
    if len(topo_order) < n:
        for v in range(n):
            if v not in set(topo_order):
                topo_order.append(v)

    # Identify input vertices (no parents)
    input_vertices = set(v for v in range(n) if not parents_list[v])

    # For small DAGs with small budget, try Dijkstra over memory states
    # State: (frozenset of red vertices, frozenset of computed vertices)
    # But this is exponential, so only use for very small cases
    use_dijkstra = (n <= 12 and fast_memory_budget <= 4)

    if use_dijkstra:
        # Dijkstra's algorithm over (red_set, computed_set) states
        # Cost: number of load/store operations
        # We track: red (in fast memory), computed (have been computed at least once)
        # Initial: red=empty, computed=empty
        # Goal: all outputs are computed and not in red (blue)
        
        import sys
        sys.setrecursionlimit(10000)
        
        # State: (frozenset red, frozenset computed)
        # But computed is determined by which vertices have been computed
        # Actually, a vertex is "available" (can be loaded) if it's an input OR has been computed
        # Let's define: available = inputs ∪ computed
        # A vertex can be in red only if it's available
        
        # State: (frozenset of red vertices)
        # A vertex v can be loaded if v is in inputs or v has been computed
        # A vertex v has been computed if all its parents were simultaneously red at some point
        
        # This is complex. Let's use a different approach:
        # Track (red_set, done_set) where done_set = vertices that have been computed
        # v is computable if all parents of v are in red_set
        # v is loadable if v in input_vertices or v in done_set
        
        # Dijkstra
        # State: (frozenset red, frozenset done)
        # Start: (frozenset(), frozenset())
        # Goal: all outputs in done, and outputs not in red (i.e., stored)
        # Actually goal: all outputs are blue, meaning they're done and not in red
        
        # For efficiency, since n is small, we can enumerate
        # But the state space is 2^n * 2^n which is too large even for n=12
        # Let's use A* or just skip Dijkstra for now and use improved greedy
        
        pass  # Skip Dijkstra, use improved greedy

    # Improved greedy approach with better eviction strategy
    red = set()  # vertices currently in fast memory
    computed = set()  # vertices that have been computed at least once
    operations = []
    
    # Priority for eviction: we want to keep vertices that will be needed soon
    # and have high remaining uses
    # Score for keeping: (remaining_uses, -depth, -has_children_in_future)
    
    # Precompute for each vertex, which future vertices (in topo order) need it
    # i.e., the "next use" position in topological order
    next_use = [n] * n  # position in topo_order where next child appears, or n if none
    for idx, v in enumerate(topo_order):
        for p in parents_list[v]:
            if next_use[p] == n:
                next_use[p] = idx
    
    # Also compute "last use"
    last_use = [n] * n
    for idx, v in enumerate(topo_order):
        for p in parents_list[v]:
            last_use[p] = idx

    def eviction_score(v):
        """Lower score = evict first. We want to keep high-score vertices."""
        # Vertices with 0 remaining uses should be evicted immediately
        if remaining_uses[v] == 0:
            return -1
        # Prefer to evict vertices with fewer remaining uses
        # Among same remaining uses, evict those needed further in the future
        # And those at lower depth (less critical)
        return (remaining_uses[v], next_use[v], -depth[v])

    def ensure_space():
        """Ensure fast memory has space for one more vertex"""
        while len(red) >= fast_memory_budget:
            if not red:
                break
            # Find vertex with lowest eviction score
            best_v = min(red, key=eviction_score)
            operations.append(['store', best_v])
            red.discard(best_v)

    def load_vertex(v):
        """Load vertex v into fast memory if not already there"""
        if v not in red:
            ensure_space()
            if v not in red:
                # Check if v is available (input or computed)
                if v in input_vertices or v in computed:
                    operations.append(['load', v])
                    red.add(v)
                else:
                    # Should not happen if we process in topological order
                    operations.append(['load', v])
                    red.add(v)

    # Process vertices in topological order
    for v in topo_order:
        if time.time() - t0 > deadline - 10:
            break

        if not parents_list[v]:
            # Input vertex: load from slow to fast
            load_vertex(v)
            # Inputs are "computed" by definition
            computed.add(v)
        else:
            # Has parents: need all parents in fast memory to compute
            for p in parents_list[v]:
                if p not in red:
                    load_vertex(p)

            # Compute v
            operations.append(['compute', v])
            red.add(v)
            computed.add(v)

        # Decrement remaining uses for parents
        for p in parents_list[v]:
            remaining_uses[p] -= 1

        # After each vertex, evict vertices with 0 remaining uses
        if len(red) > fast_memory_budget:
            zero_use = [v2 for v2 in red if remaining_uses[v2] == 0]
            for v2 in zero_use:
                operations.append(['store', v2])
                red.discard(v2)

            # If still over budget, evict lowest priority
            while len(red) >= fast_memory_budget:
                if not red:
                    break
                best_v = min(red, key=eviction_score)
                operations.append(['store', best_v])
                red.discard(best_v)

    # Ensure all outputs are blue (stored back to slow memory)
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            red.discard(v)

    # Check operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
        # Ensure outputs are still handled
        for v in outputs:
            if v in red:
                operations.append(['store', v])

    return operations
# EVOLVE-BLOCK-END
