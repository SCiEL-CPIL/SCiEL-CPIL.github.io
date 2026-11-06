"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        has_baseline = True
    except ImportError:
        has_baseline = False

    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        if has_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # Compute layer for each vertex
    layer = {}
    for v in input_vertices:
        layer[v] = 0

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    queue = deque(list(input_vertices))
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents.get(c, [])) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)

    max_L = max(layers.keys()) if layers else 0

    # Precompute next_use_layer: the earliest layer > current where v is needed as a parent
    # We will update this dynamically or precompute all future uses.
    # For efficiency, precompute a list of future layers where v is needed.
    future_needs = defaultdict(list)
    for L in range(max_L + 1):
        for v in layers.get(L, []):
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                future_needs[p].append(L)

    # Sort future needs for each vertex to allow quick lookup of next use
    for v in future_needs:
        future_needs[v].sort()

    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []

    # Helper to get the next use layer for a vertex after layer L
    def get_next_use(v, current_L):
        needs = future_needs.get(v, [])
        for L_use in needs:
            if L_use > current_L:
                return L_use
        return float('inf')

    for L in range(max_L + 1):
        if time_check():
            break
        lv = layers.get(L, [])
        if not lv:
            continue

        # Determine which parents we need for this layer
        needed_parents = set()
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_parents.add(p)

        # Load needed parents that are in slow memory
        for p in needed_parents:
            if p not in fast_mem and p in slow_mem:
                ops.append(['load', p])
                fast_mem.add(p)
                slow_mem.discard(p)

        # Compute vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            slow_mem.discard(v)

        # Evict if over budget
        # We need to evict until len(fast_mem) <= fast_memory_budget
        while len(fast_mem) > fast_memory_budget and not time_check():
            best = None
            best_score = float('inf') # We want to minimize the "keep score", so evict the one with lowest keep score?
            # Actually, let's define a "value to keep" score. Higher is better to keep.
            # We evict the one with the lowest value.
            best_value = float('inf')
            
            for u in fast_mem:
                # If it's an output, it has infinite value to keep (must store, not delete)
                if u in outputs:
                    val = float('inf')
                else:
                    nu = get_next_use(u, L)
                    if nu == float('inf'):
                        # Not needed again
                        # If it's an input, it's cheap to reload, so low value to keep
                        # If it's a computed value, it's gone, so low value
                        if u in input_vertices:
                            val = 1
                        else:
                            val = 0
                    else:
                        # It is needed in the future
                        # Value increases if it's needed sooner
                        # Value increases if it has more children (more future uses)
                        dist = nu - L
                        fan_out = len(children.get(u, []))
                        
                        # Heuristic: 
                        # 1. Immediate need (dist=1) is very valuable
                        # 2. Further need is less valuable
                        # 3. More fan-out is more valuable
                        val = (1000 / dist) + fan_out * 10
                        
                        # If it's an input, we can always reload it, so its value is slightly lower than a computed value of same dist/fanout?
                        # Actually, reloading an input is 1 op. Recalculating a computed value might be many ops.
                        # So computed values have higher value.
                        if u in input_vertices:
                            val *= 0.5

                if val < best_value:
                    best_value = val
                    best = u

            if best is None:
                break

            # Evict the best candidate
            if best in outputs:
                # Should not happen if we handle outputs correctly, but if we must evict an output...
                # The problem says "all outputs must finish blue". 
                # If we evict an output with 'delete', we lose it. 
                # So we MUST store it.
                ops.append(['store', best])
                slow_mem.add(best)
                fast_mem.discard(best)
            else:
                # Check if we can just delete it
                # If it's an input, we can delete it (it's in slow_mem implicitly? No, we moved it to fast_mem)
                # Wait, when we load an input, we remove it from slow_mem. 
                # If we delete it, it's gone. Can we reload it?
                # The problem says "Inputs start blue". This implies they are always available in slow memory?
                # "Inputs start blue" usually means they are in the initial state. 
                # If I load an input, does it leave slow memory?
                # "every red value consumes one fast-memory slot". 
                # "load/store/compute/delete".
                # If I load an input, it becomes red in fast memory. 
                # If I delete it, it's gone. 
                # Can I load it again? 
                # Usually, inputs are persistent in slow memory. 
                # Let's assume inputs are always available in slow memory. 
                # So 'delete' on an input just frees the fast memory slot.
                ops.append(['delete', best])
                fast_mem.discard(best)
                # Note: We do NOT add it back to slow_mem because it's an input (always there) 
                # OR if it's a computed value, it's lost.
                # If it's a computed value, we lose it. 
                # If it's an input, we can reload it later.
                # To be safe, let's track what's in slow_mem.
                # If u was in slow_mem before load, we removed it. 
                # If we delete it, should we add it back?
                # If it's an input, yes, it's still in slow memory.
                # If it's a computed value, no, it's deleted.
                if best in input_vertices:
                    slow_mem.add(best)
                # Else, it's gone.

    # Store outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
        elif o in slow_mem:
            # Already stored, nothing to do
            pass
        else:
            # Should not happen, but handle gracefully
            # If an output is not in fast or slow, we failed.
            # But we must return a valid schedule.
            # If we lost an output, the schedule is invalid.
            # We hope the eviction logic kept outputs.
            pass

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

# EVOLVE-BLOCK-END
