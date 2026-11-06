"""Communication-minimizing DAG pebbling with recompute strategy."""
import time

def pebble(dag, fast_memory_budget):
    """
    Minimize memory transfers (load/store) for a layered DAG under a bounded
    fast-memory budget, using a recompute-based divide-and-conquer strategy.

    dag: dict with keys 'layers' (list of lists of vertex ids), 'parents'
         (dict vertex -> list of parents), 'outputs' (list of output vertices).
         Inputs (layer 0) start blue; all outputs must finish blue.
    fast_memory_budget: max number of red (fast-memory) values allowed.
    """
    start_time = time.time()
    deadline = start_time + 50.0

    layers = dag['layers']
    parents = dag['parents']
    outputs = dag['outputs']

    if not layers:
        return []
    if len(layers) == 1:
        return []

    B = max(1, int(fast_memory_budget))
    MAX_OPS = 20000

    # Build depth map
    depth = {}
    for li, layer in enumerate(layers):
        for v in layer:
            depth[v] = li

    # Build child map
    children = {}
    for v, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(v)

    # Find all vertices needed (ancestors of outputs)
    all_needed = set()
    stack = list(outputs)
    while stack:
        v = stack.pop()
        if v in all_needed:
            continue
        all_needed.add(v)
        for p in parents.get(v, []):
            if depth.get(p, -1) > 0:  # skip layer 0 inputs from "needed to compute"
                stack.append(p)

    # Add layer 0 inputs that are actually used
    input_needs = set()
    for v in all_needed:
        for p in parents.get(v, []):
            if depth.get(p, -1) == 0:
                input_needs.add(p)

    n_layers = len(layers)

    # Group needed vertices by layer (excluding layer 0)
    layer_vertices = {}
    for v in all_needed:
        li = depth.get(v, -1)
        if li > 0:
            layer_vertices.setdefault(li, []).append(v)
    for li in layer_vertices:
        layer_vertices[li].sort()

    # Determine which layer-0 inputs are needed
    # For each input, check if any child (transitively) leads to an output
    # We'll track which inputs are needed per "chunk"

    ops = []
    ops_count = 0

    def emit(op, v):
        nonlocal ops_count
        if ops_count >= MAX_OPS:
            raise RuntimeError("too many ops")
        ops.append([op, v])
        ops_count += 1

    # Core idea: divide-and-conquer over layers.
    # For a range [a, b] (layers a to b), with targets in layer b:
    # - If b - a == 1: compute directly from layer a to layer b.
    # - Otherwise: split at midpoint m.
    #   1. Compute frontier (layer m vertices needed) from [a, m].
    #   2. Compute targets from [m, b] using frontier.
    #   The frontier must be kept in fast memory or stored/reloaded.

    def get_frontier(a, m, targets, b):
        """Get vertices at layer m that are needed to compute targets in [m, b]."""
        needed = set()
        stack = list(targets)
        while stack:
            v = stack.pop()
            if v in needed:
                continue
            needed.add(v)
            dv = depth.get(v, -1)
            if dv <= m:
                continue
            for p in parents.get(v, []):
                if depth.get(p, -1) >= m:
                    stack.append(p)
        return [v for v in needed if depth.get(v, -1) == m]

    def get_inputs_needed(a, m, frontier):
        """Get layer-a inputs needed to compute frontier."""
        if a == 0:
            needed_inputs = set()
            stack = list(frontier)
            while stack:
                v = stack.pop()
                dv = depth.get(v, -1)
                if dv <= a:
                    if dv == a:
                        needed_inputs.add(v)
                    continue
                for p in parents.get(v, []):
                    if depth.get(p, -1) >= a:
                        stack.append(p)
            return sorted(needed_inputs)
        else:
            # For non-zero a, "inputs" are vertices at layer a
            # We just return the frontier itself if a == m, else recurse
            if a == m:
                return sorted(frontier)
            return sorted(frontier)

    def compute_range(a, b, targets, keep_in_fast=False):
        """
        Compute targets (in layers [a+1, b]) given that layer-a vertices 
        are available. Returns list of ops.
        If keep_in_fast, the computed vertices at layer b are left in fast memory.
        """
        if a >= b:
            return []

        targets = sorted(set(targets), key=lambda x: (depth.get(x, -1), x))
        # Filter targets to [a+1, b]
        targets = [v for v in targets if a < depth.get(v, -1) <= b]
        if not targets:
            return []

        span = b - a
        if span == 1:
            # Direct computation: load inputs from layer a, compute layer b
            result = []
            # Find inputs at layer a needed for targets
            inputs_needed = set()
            stack = list(targets)
            while stack:
                v = stack.pop()
                dv = depth.get(v, -1)
                if dv <= a:
                    if dv == a:
                        inputs_needed.add(v)
                    continue
                for p in parents.get(v, []):
                    if depth.get(p, -1) >= a:
                        stack.append(p)

            # Load inputs
            for inp in sorted(inputs_needed):
                result.append(['load', inp])

            # Compute targets
            for v in targets:
                result.append(['compute', v])

            # If not keeping in fast, store to slow
            if not keep_in_fast:
                for v in targets:
                    result.append(['store', v])

            return result

        # Divide and conquer
        m = (a + b) // 2
        frontier = get_frontier(a, m, targets, b)

        if not frontier:
            # No frontier needed, just compute targets directly
            return compute_range(a, b, targets, keep_in_fast)

        # Compute frontier from [a, m]
        # The frontier values need to be available for the second phase
        # If len(frontier) <= B, keep in fast memory
        keep_frontier = len(frontier) <= B

        ops1 = compute_range(a, m, frontier, keep_in_fast=keep_frontier)

        # Compute targets from [m, b]
        ops2 = compute_range(m, b, targets, keep_in_fast=keep_in_fast)

        return ops1 + ops2

    # For the top-level call, layer 0 inputs are in slow memory (blue).
    # We need to load them when needed.
    # The compute_range function handles loading.

    # But we need to be careful: the "load" operation loads from slow memory.
    # After computation, values in fast memory can be stored back.

    # Let's trace through the algorithm to verify correctness.
    # The key issue: when we "load" an input, it goes into fast memory.
    # When we "compute", the result goes into fast memory.
    # When we "store", it goes to slow memory and leaves fast memory.
    # When we "delete", it leaves fast memory (lost).

    # The fast memory budget B means at most B values can be in fast memory at once.
    # We need to track this and ensure we don't exceed it.

    # Let me rewrite with explicit fast-memory tracking.

    # Reset
    ops = []
    ops_count = 0
    fast_mem = set()

    def emit(op, v):
        nonlocal ops_count
        if ops_count >= MAX_OPS:
            raise RuntimeError("too many ops")
        ops.append([op, v])
        ops_count += 1

    # Simulate the state machine to verify
    def simulate(ops_list):
        """Simulate to verify correctness."""
        state = {}  # vertex -> 'blue' or 'red'
        # Inputs start blue
        for v in layers[0]:
            state[v] = 'blue'
        fast_count = 0

        for op, v in ops_list:
            if op == 'load':
                if state.get(v) != 'blue':
                    return False
                fast_count += 1
                if fast_count > B:
                    return False
                state[v] = 'red'
            elif op == 'store':
                if state.get(v) != 'red':
                    return False
                fast_count -= 1
                state[v] = 'blue'
            elif op == 'compute':
                # All parents must be red (in fast memory)
                for p in parents.get(v, []):
                    if state.get(p) != 'red':
                        return False
                fast_count += 1
                if fast_count > B:
                    return False
                state[v] = 'red'
            elif op == 'delete':
                if state.get(v) != 'red':
                    return False
                fast_count -= 1
                state[v] = None  # deleted
            else:
                return False
        # All outputs must be blue
        for v in outputs:
            if state.get(v) != 'blue':
                return False
        return True

    # The divide-and-conquer approach needs careful handling of fast memory.
    # Let me use a simpler but effective approach:
    # Process in chunks, keeping track of fast memory explicitly.

    # For each layer, determine what needs to be computed.
    # Use a greedy strategy: keep as much as possible in fast memory.

    # Actually, let me implement a proper recursive algorithm with
    # explicit fast-memory management.

    # The key insight: in a layered DAG, we can use the following strategy:
    # 1. Split the computation into two halves.
    # 2. Compute the first half, keeping the boundary in fast memory.
    # 3. Compute the second half using the boundary.
    # 4. The boundary vertices must fit in fast memory (or be stored/reloaded).

    # Let me implement this properly.

    def compute_recursive(a, b, target_vertices):
        """
        Compute target_vertices (in layers [a+1, b]) given layer a is in slow memory.
        Returns list of ops. Manages fast memory implicitly.
        """
        if time.time() > deadline:
            return []

        if a >= b:
            return []

        target_vertices = sorted(set(target_vertices), key=lambda x: (depth.get(x, -1), x))
        target_vertices = [v for v in target_vertices if a < depth.get(v, -1) <= b]
        if not target_vertices:
            return []

        if b - a == 1:
            # Direct: load parents from layer a, compute layer b
            ops_local = []
            # Find all layer-a vertices needed
            inputs_needed = set()
            stk = list(target_vertices)
            while stk:
                v = stk.pop()
                dv = depth.get(v, -1)
                if dv <= a:
                    if dv == a:
                        inputs_needed.add(v)
                    continue
                for p in parents.get(v, []):
                    if depth.get(p, -1) >= a:
                        stk.append(p)

            for inp in sorted(inputs_needed):
                ops_local.append(['load', inp])
            for v in target_vertices:
                ops_local.append(['compute', v])
            # Store results to slow memory (they need to be blue at end if output,
            # or available for later loads)
            for v in target_vertices:
                ops_local.append(['store', v])
            return ops_local

        # Divide
        m = (a + b) // 2
        frontier = get_frontier(a, m, target_vertices, b)

        if not frontier:
            # Degenerate case, just compute directly
            return compute_recursive(a, b, target_vertices)

        # Compute frontier from [a, m]
        ops1 = compute_recursive(a, m, frontier)

        # Now frontier values are in slow memory (blue).
        # Compute target_vertices from [m, b]
        ops2 = compute_recursive(m, b, target_vertices)

        return ops1 + ops2

    # Call the recursive function
    all_outputs = sorted(outputs, key=lambda x: (depth.get(x, -1), x))
    result_ops = compute_recursive(0, n_layers - 1, all_outputs)

    # Verify the result
    if not simulate(result_ops):
        # Fallback: simple layer-by-layer with recompute
        ops = []
        ops_count = 0
        fast_mem = set()

        for li in range(1, n_layers):
            verts = layer_vertices.get(li, [])
            if not verts:
                continue

            # Find needed parents
            needed_parents = set()
            for v in verts:
                for p in parents.get(v, []):
                    needed_parents.add(p)

            # Load needed parents that are not in fast memory
            for p in sorted(needed_parents, key=lambda x: (depth.get(x, -1), x)):
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)

            # Compute
            for v in verts:
                ops.append(['compute', v])
                fast_mem.add(v)

            # Store computed values to slow memory
            for v in verts:
                ops.append(['store', v])
                fast_mem.discard(v)

            # Clean up fast memory
            fast_mem.clear()

        # Ensure outputs are blue (they should be after store)
        # Verify
        if not simulate(ops):
            # Last resort: compute everything and store
            ops = []
            for li in range(1, n_layers):
                verts = layer_vertices.get(li, [])
                for v in verts:
                    # Load all parents
                    for p in parents.get(v, []):
                        ops.append(['load', p])
                    ops.append(['compute', v])
                    ops.append(['store', v])

    return result_ops
