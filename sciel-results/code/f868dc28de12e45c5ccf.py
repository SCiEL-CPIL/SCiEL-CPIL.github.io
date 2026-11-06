def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # Parse DAG
    n = dag.get('n', None)
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    if n is None:
        n = len(parents)
        
    topo_order = dag.get('topo_order', list(range(n)))

    if fast_memory_budget <= 0:
        return []

    # Build children map and usage counts
    children = [[] for _ in range(n)]
    future_usage = [0] * n
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
            future_usage[p] += 1

    # State
    fast = set(inputs)
    slow = set()
    ops = []

    # Helper to evict a vertex from fast to slow
    def evict_from_fast(prefer_keep=None):
        """
        Evicts a vertex from fast to slow.
        prefer_keep: A vertex ID that should not be evicted if possible.
        Returns True if eviction was successful, False if fast was empty or only contained prefer_keep.
        """
        if not fast:
            return False
        
        # Candidates: all in fast, excluding prefer_keep if it's in fast
        candidates = [x for x in fast if x != prefer_keep]
        
        if not candidates:
            # If we must evict and only prefer_keep is there, we have to evict it?
            # Or fail?
            # If we fail, we can't load new items.
            # Let's evict prefer_keep as a last resort if we have no choice.
            candidates = list(fast)
            if not candidates:
                return False

        # Choose victim: min future_usage
        victim = min(candidates, key=lambda x: future_usage[x])
        
        fast.discard(victim)
        slow.add(victim)
        ops.append(['store', victim])
        return True

    # Process vertices in topological order
    for v in topo_order:
        if time.time() > deadline:
            break

        if v in inputs:
            continue

        # Ensure all parents are in fast
        for p in parents[v]:
            if p not in fast:
                if p in slow:
                    # Need to load p
                    if len(fast) < fast_memory_budget:
                        fast.add(p)
                        ops.append(['load', p])
                    else:
                        # Evict something to make space
                        if evict_from_fast(prefer_keep=p):
                            fast.add(p)
                            ops.append(['load', p])
                        else:
                            # Could not evict
                            pass
                else:
                    # p is not in fast and not in slow.
                    # Critical error in state tracking.
                    pass

        # Compute v
        fast.add(v)
        ops.append(['compute', v])

        # Enforce fast memory budget
        while len(fast) > fast_memory_budget:
            if not evict_from_fast(prefer_keep=None):
                break

    # Ensure outputs are in fast
    for o in outputs:
        if o not in fast:
            if o in slow:
                if len(fast) < fast_memory_budget:
                    fast.add(o)
                    ops.append(['load', o])
                else:
                    if evict_from_fast(prefer_keep=o):
                        fast.add(o)
                        ops.append(['load', o])
                    else:
                        pass
            else:
                pass

    return ops
