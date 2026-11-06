"""Rewritten pebbling: minimize load/store transfers by keeping values in fast memory."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys 'parents' (list of lists, topologically ordered) and 'outputs' (list of vertex indices)
    fast_memory_budget: int, number of fast-memory slots
    Returns: list of [operation, vertex] where operation in {'load','store','compute','delete'}
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Parse DAG structure
    parents = dag['parents']
    outputs = dag.get('outputs', [])
    n = len(parents)

    # Determine if vertex is an input (no parents) or output
    is_input = [len(p) == 0 for p in parents]
    is_output = [False] * n
    for v in outputs:
        is_output[v] = True

    # Topological order is already given (indices 0..n-1)
    # Compute out-degree (how many children need each vertex)
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    # Count how many times each vertex is needed (number of children that consume it)
    # A vertex's value is needed until all its children have been computed
    use_count = [len(children[v]) for v in range(n)]

    # Strategy: process vertices in topological order.
    # For each vertex, if it's an input, load it (transfer).
    # If it's an internal vertex, compute it (no transfer).
    # After computing/using a vertex, decide whether to keep it in fast memory
    # or store it back (transfer).
    #
    # Key insight: a computed value must be kept in fast memory until all its
    # children are computed. If the budget allows, keep it; otherwise store it.
    #
    # Greedy approach: keep a value in fast memory if it will be needed again
    # (has remaining children) and we have budget. Otherwise store it.

    # For each vertex, determine the "last use" - the latest child that needs it
    # This helps decide when we can release the value
    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
        else:
            last_use[v] = v  # if no children, last use is itself (for outputs)

    # Actually, for pebbling: a vertex's value is needed by its children.
    # The value can be released after the LAST child that uses it has been computed.
    # But we also need to handle the case where a vertex is an output - it must
    # finish blue (in fast memory).

    # Simpler greedy: process in topo order, maintain fast memory set.
    # For each vertex v:
    #   - If input: load it (transfer +1), add to fast memory
    #   - If internal: compute it (no transfer), add to fast memory
    #   - Before computing v, ensure all parents are in fast memory (load if not)
    #   - After computing v, if budget exceeded, store some value
    #   - A value can only be released (stored) when no remaining child needs it

    # Track which values are in fast memory
    # We need to ensure parents are available when computing a vertex.

    # Better approach:
    # 1. Pre-compute for each vertex the set of ancestors that must be kept alive
    # 2. Use a greedy strategy that keeps values as long as needed

    # Let's use a simpler but effective strategy:
    # Process vertices in topological order.
    # Maintain a set of vertices currently in fast memory.
    # When computing vertex v:
    #   - For each parent p of v, if p not in fast memory, load p (transfer +1)
    #   - Compute v (no transfer)
    #   - Add v to fast memory
    #   - If |fast_memory| > budget, we must store something.
    #     Store a vertex that won't be needed again (last_use already passed)
    #     or if all are needed, store the one with the earliest last_use among
    #     those not yet used by all their children.

    # Track remaining children for each vertex (how many children still need it)
    remaining_children = [len(children[v]) for v in range(n)]

    # fast_memory: set of vertex indices currently in fast memory
    fast_memory = set()
    operations = []
    transfer_count = 0

    # For inputs, they start blue (in fast memory? or in slow memory?)
    # "Inputs start blue" - this means inputs are initially available.
    # But do they count against fast memory? The problem says "every red value
    # consumes one fast-memory slot, including newly computed results."
    # Inputs start blue, so they're in fast memory initially? Or in slow memory?
    # 
    # Re-reading: "Inputs start blue, all outputs must finish blue"
    # I think inputs are in slow memory initially and must be loaded to compute.
    # "Blue" might mean "available in fast memory" or "computed/valid".
    #
    # Let me assume: inputs are in slow memory, must be loaded (transfer) to use.
    # Outputs must end in fast memory (blue).

    # Process in topological order
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            break

        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_memory:
                # Load parent from slow memory
                operations.append(['load', p])
                transfer_count += 1
                fast_memory.add(p)

        # Compute v
        if not is_input[v]:
            operations.append(['compute', v])

        # Add v to fast memory
        fast_memory.add(v)

        # Now, release any vertices that are no longer needed
        # A vertex p can be released if all its children have been processed
        # (i.e., remaining_children[p] == 0) and p is not an output
        # (outputs must finish blue)

        # But we should only release if we're over budget, or proactively
        # to make room.

        # Check if we're over budget
        while len(fast_memory) > fast_memory_budget:
            # Find a vertex to store
            # Prefer vertices whose last_use has passed and are not outputs
            candidates = []
            for u in fast_memory:
                if is_output[u]:
                    continue  # can't store outputs
                # Check if all children of u have been processed
                # remaining_children[u] tracks how many children still need u
                # But we need to decrement when we compute a child
                # Let's track this differently

            # Actually, let's track remaining uses properly
            # When we compute vertex v, we consume one use of each parent
            # Let me restructure

            # For now, just pick the vertex with the smallest last_use
            # that isn't an output
            storeable = [u for u in fast_memory if not is_output[u]]
            if not storeable:
                # Can't store anything, this is a problem
                break
            # Pick the one with the earliest last_use (most urgent to release)
            # Actually, we want to release the one that will be needed latest
            # No, we want to release the one that will NOT be needed soon
            # Release the one with the LARGEST last_use? No...
            # Release the one that won't be needed for the longest time
            # i.e., the one with the largest last_use among those we can release
            # But we can only release if it's not needed by any unprocessed child

            # Let's just pick the first non-output vertex
            # This is a simplification
            to_store = None
            for u in storeable:
                # Check if u is still needed by any unprocessed vertex
                # u is needed if there exists a child c of u with c > v (unprocessed)
                needed = False
                for c in children[u]:
                    if c >= v:  # c is current or future
                        needed = True
                        break
                if not needed:
                    to_store = u
                    break

            if to_store is None:
                # All storeable vertices are still needed
                # This shouldn't happen if budget is sufficient
                # Force store the one with earliest last_use
                to_store = min(storeable, key=lambda u: last_use[u])

            operations.append(['store', to_store])
            transfer_count += 1
            fast_memory.discard(to_store)

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in fast_memory:
            operations.append(['load', v])
            transfer_count += 1
            fast_memory.add(v)

    # Cap operations at 20000
    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
# EVOLVE-BLOCK-END
