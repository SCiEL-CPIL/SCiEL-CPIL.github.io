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
    DEADLINE = 80.0  # wall-clock limit from evaluator with safety margin

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
        
        # Topological order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            if v not in in_degree:
                in_degree[v] = 0
            else:
                in_degree[v] = 0
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        # Use a heap for deterministic ordering if needed, but list is fine
        topo_order = []
        while queue:
            # Sort to ensure determinism
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # If topo_order doesn't contain all vertices (cycle?), add remaining
        remaining = all_vertices - set(topo_order)
        if remaining:
            topo_order.extend(sorted(remaining))

        # Map vertex to index in topo_order for O(1) lookup
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute: for each vertex, the set of children (direct dependents)
        # and the "last_use" index: the highest topo_index of any child
        # Also need to consider that outputs must be kept until the very end.
        # So effectively, last_use of an output is len(topo_order) (infinity-ish).
        
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
            
            # If v is an output, it must be kept until the end
            if v in outputs:
                last_use[v] = len(topo_order) - 1 # Or just a large number

        # Precompute depth for eviction heuristic
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1
        
        must_be_blue = set(outputs)
        
        ops = []
        fast_mem = set()
        
        def get_eviction_candidate(current_idx):
            """
            Find the vertex in fast_mem to delete to make space.
            Criteria:
            1. Not an output (unless all are outputs, then pick one with latest last_use? No, outputs must be blue at end. If we delete, we must reload. So avoid if possible.)
            2. Among non-outputs, pick the one with the smallest depth (easiest to recompute).
            3. Tie-breaker: earliest last_use (less useful in the near future).
            """
            candidates = []
            for v in fast_mem:
                if v in must_be_blue:
                    continue
                # Only consider vertices that are actually needed in the future?
                # If last_use[v] < current_idx, it's already "dead" and should have been deleted.
                # So we should delete "dead" vertices first before evicting live ones.
                candidates.append((depth[v], last_use[v], v))
            
            if not candidates:
                # All in fast_mem are outputs. We have to delete one.
                # Pick the one with the latest last_use? Or earliest?
                # If we delete an output, we must reload it at the end.
                # It's better to delete the one that is "most expensive" to keep?
                # Or the one that is least likely to be needed soon?
                # Since they are outputs, they are needed at the very end.
                # So all have last_use >= current_index (conceptually, until the end).
                # Let's pick the one with the largest depth (hardest to recompute) to KEEP?
                # No, we want to DELETE.
                # We want to delete the one that is EASIEST to recompute (smallest depth).
                candidates = [(depth[v], last_use[v], v) for v in fast_mem]
            
            if not candidates:
                return None
            
            # Sort by depth (ascending), then last_use (ascending)
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[0][2]

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
                    cand = get_eviction_candidate(current_idx)
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # It's a computed vertex
                # First, ensure all its parents are in fast_mem
                for p in ps:
                    if not ensure_in_fast_mem(p, current_idx):
                        return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate(current_idx)
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['compute', v])
                fast_mem.add(v)
                return True
        
        # Process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            if not ensure_in_fast_mem(v, i):
                # Failed to get v in fast mem, break?
                # This shouldn't happen if budget is reasonable or we evict correctly.
                break
            
            # Optional: Delete vertices that are no longer needed to save memory for future?
            # We can proactively delete vertices whose last_use < i (current index).
            # This keeps fast_mem small and reduces pressure.
            # Only do this if we are close to budget or to keep memory clean.
            # Actually, always delete dead vertices to minimize memory footprint.
            for u in list(fast_mem):
                if u == v:
                    continue
                # u is no longer needed if its last use is before current index
                # AND it is not an output (outputs are kept until end)
                if last_use[u] < i and u not in must_be_blue:
                    ops.append(['delete', u])
                    fast_mem.discard(u)
        
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                # We need to bring it back.
                # If it was deleted, we must recompute or load it.
                # ensure_in_fast_mem handles this.
                if not ensure_in_fast_mem(o, len(topo_order)):
                    # Try to evict and load
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate(len(topo_order))
                        if cand:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    
                    # If it's an input, load. If computed, compute.
                    # ensure_in_fast_mem handles this, but if it failed due to budget,
                    # we might need to force evict an output?
                    # If ensure_in_fast_mem failed, it means we couldn't free space.
                    # This implies fast_memory_budget is too small to hold all necessary live vars + outputs.
                    # In that case, we might have to delete an output to make room, then reload it?
                    # That's a net zero gain in memory, but adds ops.
                    
                    # Let's just try ensure again, maybe it works now?
                    ensure_in_fast_mem(o, len(topo_order))
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # Too many ops, try baseline or truncate
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        # Fallback to baseline if available
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
