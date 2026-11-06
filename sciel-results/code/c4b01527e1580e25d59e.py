"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Precompute next-use positions for each vertex
    # next_use[v] = list of indices in needed_vertices where v is used as a parent
    next_use = defaultdict(list)
    for i, v in enumerate(needed_vertices):
        for p in parent_dict.get(v, []):
            if p in needed_set:
                next_use[p].append(i)

    # State
    in_fast = set()
    in_slow = set(inputs)  # Inputs start in slow memory
    computed = set()
    operations = []
    
    # Track current position for FNU
    current_pos = 0

    def get_next_use(u, pos):
        """Get the next use position of u after position pos, or infinity if never used again."""
        uses = next_use.get(u, [])
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uses):
            return uses[lo]
        return float('inf')

    def get_remaining_uses(u, pos):
        """Count how many times u is used after position pos."""
        uses = next_use.get(u, [])
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        return len(uses) - lo

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    def eviction_cost(u):
        """Compute cost of evicting u using a smooth surrogate over remaining uses.
        Lower cost = better eviction candidate (more expensive to evict).
        Uses log-sum-exp style aggregation to smooth the discrete usage pattern."""
        nu = get_next_use(u, current_pos)
        rem = get_remaining_uses(u, current_pos)
        if nu == float('inf'):
            # Never used again: very cheap to evict (just delete)
            return -100.0
        # Distance to next use
        dist = nu - current_pos
        # Smooth aggregation of remaining uses: sum of exp(-alpha * distance)
        # This gives higher weight to imminent uses
        uses = next_use.get(u, [])
        # Get remaining use positions
        remaining_positions = []
        for pos in uses:
            if pos > current_pos:
                remaining_positions.append(pos - current_pos)
        
        if not remaining_positions:
            smooth_pressure = 0.0
        else:
            # Soft-max style: sum of exp(-beta * d) for each remaining use
            beta = 0.3  # Decay rate
            smooth_pressure = sum(max(0.0, 1.0 - beta * d) for d in remaining_positions)
        
        # Cost: we want to KEEP vertices with high pressure (many/imminent uses)
        # and EVICT vertices with low pressure
        # Lower cost = better to evict
        keep_value = dist * 0.5 + smooth_pressure * 2.0
        return -keep_value  # Negative: lower is better to evict

    def recomputation_cost(u):
        """Estimate cost of recomputing u if evicted: number of parent loads needed."""
        if u in inputs:
            return 1  # Just reload from slow
        parents = [p for p in parent_dict.get(u, []) if p in needed_set]
        # Cost is roughly the number of parents that would need to be loaded
        return len(parents)

    def evict_one(protect_set):
        """Evict one vertex from fast memory using smooth eviction strategy.
        protect_set: set of vertices that should not be evicted."""
        if not in_fast:
            return None
        
        best_evict = None
        best_cost = float('inf')
        
        for u in list(in_fast):
            if u in protect_set:
                continue
            c = eviction_cost(u)
            if c < best_cost:
                best_cost = c
                best_evict = u
        
        if best_evict is None:
            # All protected, pick any non-protected if possible, else any
            for u in in_fast:
                if u not in protect_set:
                    best_evict = u
                    break
            if best_evict is None:
                best_evict = next(iter(in_fast))
        
        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
        
        return best_evict

    def proactive_delete():
        """Delete vertices in fast memory that have no more future uses."""
        for u in list(in_fast):
            if get_next_use(u, current_pos) == float('inf'):
                delete(u)

    def lookahead_loads(needed_parents, budget):
        """Pre-load parents that will be needed in the next few vertices.
        Returns list of (vertex, priority) sorted by priority."""
        if len(in_fast) >= budget:
            return []
        
        # Look ahead up to 5 vertices
        ahead = []
        for i in range(current_pos + 1, min(current_pos + 6, len(needed_vertices))):
            v = needed_vertices[i]
            for p in parent_dict.get(v, []):
                if p in needed_set and p not in in_fast:
                    ahead.append((p, i - current_pos))
        
        # Deduplicate, keeping highest priority (closest)
        seen = {}
        for p, dist in ahead:
            if p not in seen or dist < seen[p]:
                seen[p] = dist
        
        return sorted(seen.items(), key=lambda x: x[1])

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break
        
        current_pos = vertex_to_idx.get(v, current_pos)
        
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        
        # Proactive delete: free up slots for vertices no longer needed
        proactive_delete()

        # Lookahead: pre-load some parents that will be needed soon
        # Only if we have spare capacity
        if len(in_fast) < fast_memory_budget:
            preloads = lookahead_loads(parents, fast_memory_budget)
            # Load up to 2 preloaded vertices if capacity allows
            for pl_vertex, _ in preloads[:2]:
                if len(in_fast) < fast_memory_budget and pl_vertex not in in_fast:
                    if pl_vertex in in_slow:
                        load(pl_vertex)
                    elif pl_vertex in computed:
                        load(pl_vertex)
                    # If not in slow and not computed, skip (would need compute)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load parent p
            if len(in_fast) >= fast_memory_budget:
                # Evict: Find best candidate to remove
                remaining_parents = set(parents) - {p}
                evict_one(remaining_parents | {v})
            
            if p in in_slow or p in computed:
                load(p)
            else:
                # p not in slow or computed - shouldn't happen in valid DAG
                load(p)

        # Compute v
        if v not in in_fast:
            compute(v)
        else:
            # v was already in fast? Maybe from a previous computation?
            if v in inputs:
                pass
            else:
                pass

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evict_one(outputs)
            load(v)

    # Final check: ensure all outputs are in fast
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) < fast_memory_budget:
                load(v)
            else:
                if in_fast:
                    u = next(iter(in_fast))
                    if u in in_slow:
                        delete(u)
                    else:
                        store(u)
                    load(v)

    # Post-pass: try to reduce operations by merging consecutive loads/stores
    # and removing redundant deletes
    if operations:
        refined = []
        for op in operations:
            if refined and op[0] == 'delete' and refined[-1][0] == 'store' and op[1] == refined[-1][1]:
                # store followed by delete is redundant
                continue
            if refined and op[0] == 'load' and refined[-1][0] == 'load' and op[1] == refined[-1][1]:
                # duplicate load
                continue
            refined.append(op)
        operations = refined

    return operations
# EVOLVE-BLOCK-END
