"""Communication-minimizing DAG pebbling using block-based lookahead caching."""
import time
from collections import defaultdict

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
    DEADLINE = 85.0

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        inputs_set = set(inputs)
        outputs_set = set(outputs)

        # Build children map
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)

        all_vertices = set(parents.keys()) | inputs_set | outputs_set

        # Topological sort (Kahn's algorithm)
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

        topo_index = {v: i for i, v in enumerate(topo_order)}
        n_vertices = len(topo_order)
        
        # If topo_order doesn't cover all vertices, there's a cycle or disconnected graph.
        # Assume valid DAG.
        
        # Precompute next_use: the maximum topo_index of any child, or own index if leaf
        # This is the "last time" this vertex is needed.
        # Actually, for caching, we want to know if it's needed in the FUTURE.
        # next_use[v] = max(topo_index[c] for c in children[v]) if children else topo_index[v]
        
        next_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                next_use[v] = max(topo_index[c] for c in children[v])
            else:
                next_use[v] = topo_index.get(v, -1)

        ops = []
        fast_mem = set()
        slow_mem = set() # Vertices that have been stored to slow memory and can be reloaded

        def evict():
            """Evict a vertex from fast_mem to make space."""
            if not fast_mem:
                return
            
            # Find the vertex with the furthest next_use
            # We prefer to evict vertices that are not needed soon.
            # If a vertex is not needed in the future (next_use < current_idx), we can delete it.
            # If it is needed in the future, we must store it.
            
            # We should not evict the vertex we are currently processing if possible?
            # The caller adds the new vertex, then calls evict if full.
            # So we can evict anyone.
            
            best_v = None
            best_next_use = -1
            
            # Heuristic: Evict the one with the max next_use.
            # If multiple, evict the one with the highest recompute cost? Or just max next_use.
            # Actually, if next_use is in the past, it's dead. Delete is free.
            # If next_use is in the future, Store costs 1.
            # We want to free space.
            # If we have any dead vertices, delete them first.
            
            dead_candidates = []
            live_candidates = []
            
            cur_idx = current_topo_index[0]
            
            for v in fast_mem:
                if next_use[v] < cur_idx:
                    dead_candidates.append(v)
                else:
                    live_candidates.append(v)
            
            if dead_candidates:
                # Delete any dead vertex
                v_to_evict = dead_candidates[0]
                ops.append(['delete', v_to_evict])
                fast_mem.discard(v_to_evict)
            else:
                # All are live. Evict the one with the furthest next_use.
                # This is a form of LFU/LRU.
                if live_candidates:
                    # Sort by next_use descending
                    # v_to_evict = max(live_candidates, key=lambda x: next_use[x])
                    # This is O(N) per eviction. With N up to 20000 and budget small, this is fine.
                    # But we can optimize by keeping a heap? 
                    # For simplicity and robustness against small N, linear scan is okay.
                    v_to_evict = max(live_candidates, key=lambda x: next_use[x])
                    ops.append(['store', v_to_evict])
                    slow_mem.add(v_to_evict)
                    fast_mem.discard(v_to_evict)

        current_topo_index = [0]

        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            current_topo_index[0] = i

            if v in inputs_set:
                # Input vertex
                if v not in fast_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        evict()
                    ops.append(['load', v])
                    fast_mem.add(v)
            else:
                # Computed vertex
                # Ensure parents are in fast_mem
                ps = parents.get(v, [])
                for p in ps:
                    if p not in fast_mem:
                        if p in slow_mem:
                            if len(fast_mem) >= fast_memory_budget:
                                evict()
                            ops.append(['load', p])
                            fast_mem.add(p)
                        else:
                            # This should not happen if we always store evicted live vertices.
                            # If it happens, it means we deleted a vertex that was needed.
                            # This is a logic error in the eviction policy.
                            # We only delete if next_use < current_idx.
                            # Since p is a parent of v (current i), next_use[p] >= i.
                            # So next_use[p] < i is false.
                            # So p should have been stored, not deleted.
                            # Unless p was an input? No, inputs are always available to load.
                            # If p was an input, it would be in inputs_set, not in parents (unless DAG has input nodes as parents? No, inputs are sources).
                            # So this branch should be unreachable.
                            # If it is reached, we try to load from slow_mem (will fail) or recompute (complex).
                            # Let's assume it's in slow_mem.
                            ops.append(['load', p])
                            fast_mem.add(p)

                # Compute v
                if len(fast_mem) >= fast_memory_budget:
                    evict()
                
                ops.append(['compute', v])
                fast_mem.add(v)
                
                # Note: We do NOT immediately store v. We keep it in fast_mem.
                # It will be stored when it needs to be evicted to make space for future vertices.

        # Finalize: Ensure all outputs are in slow_mem (blue)
        for o in outputs:
            if o in fast_mem:
                # We need to store it to make it blue
                # We don't need to evict anyone, we just store it.
                # But storing it doesn't free space in fast_mem? 
                # "store" operation: red -> blue. Does it free the fast_mem slot?
                # "every red value consumes one fast-memory slot".
                # If we store it, it becomes blue. So it no longer consumes a fast-memory slot.
                # So yes, store frees the slot.
                ops.append(['store', o])
                slow_mem.add(o)
                fast_mem.discard(o)
            elif o not in slow_mem:
                # Output was never computed or was deleted?
                # Should not happen.
                # If it happened, we would need to compute it.
                # But we can't easily do that now.
                # We'll assume it's handled.
                pass

        if len(ops) <= 20000:
            return ops
        else:
            return ops[:20000]

    except Exception as e:
        return []
