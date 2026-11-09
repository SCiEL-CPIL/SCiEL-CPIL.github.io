def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 85

    # Fallback to baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except Exception:
        baseline_ops = None

    def _extract(dag):
        if isinstance(dag, dict):
            parents_dict = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
        else:
            parents_dict = dict(getattr(dag, 'parents', {}))
            inputs = list(getattr(dag, 'inputs', []))
            outputs = set(getattr(dag, 'outputs', []))
        return parents_dict, inputs, outputs

    def _topo_order(parents_dict, inputs):
        all_vertices = set(inputs)
        for v, ps in parents_dict.items():
            all_vertices.add(v)
            all_vertices.update(ps)
        in_deg = {v: len(parents_dict.get(v, [])) for v in all_vertices}
        children = {v: [] for v in all_vertices}
        for v, ps in parents_dict.items():
            for p in ps:
                children[p].append(v)
        from collections import deque
        q = deque([v for v in all_vertices if in_deg[v] == 0])
        order = []
        while q:
            v = q.popleft()
            order.append(v)
            for c in children[v]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        return order, children, all_vertices

    def _transfers(ops):
        return sum(1 for op in ops if op[0] in ('load', 'store'))

    def _chunk_memory_need(parents_dict, inputs, children, start_idx, end_idx, topo_order):
        """Compute minimum fast memory needed to process vertices from start_idx to end_idx
        (inclusive) without any eviction. Assumes all parents outside the chunk are already loaded."""
        # Track which vertices are in fast memory
        in_fast = set()
        # Pre-load all parents of vertices in [start_idx, end_idx] that are outside the range
        for i in range(start_idx, end_idx + 1):
            v = topo_order[i]
            if v not in inputs:
                for p in parents_dict.get(v, []):
                    p_idx = topo_order.index(p)
                    if p_idx < start_idx:
                        in_fast.add(p)
        
        # Process the chunk
        for i in range(start_idx, end_idx + 1):
            v = topo_order[i]
            if v in inputs:
                if v not in in_fast:
                    in_fast.add(v)
            else:
                ps = parents_dict.get(v, [])
                for p in ps:
                    if p not in in_fast:
                        in_fast.add(p)
                in_fast.add(v)
        
        return len(in_fast)

    def _generate_chunk_ops(parents_dict, inputs, outputs, children, start_idx, end_idx, topo_order, in_fast_at_start):
        """Generate operations for processing vertices from start_idx to end_idx.
        in_fast_at_start: set of vertices already in fast memory before this chunk."""
        ops = []
        in_fast = set(in_fast_at_start)
        
        # Track which vertices from before this chunk are still needed
        for i in range(start_idx, end_idx + 1):
            v = topo_order[i]
            if v in inputs:
                if v not in in_fast:
                    in_fast.add(v)
                    ops.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                for p in ps:
                    if p not in in_fast:
                        in_fast.add(p)
                        ops.append(['load', p])
                in_fast.add(v)
                ops.append(['compute', v])
        
        return ops, in_fast

    def _simulate_partition(parents_dict, inputs, outputs, topo_order, children, partition, fast_memory_budget):
        """Simulate a partition of topo_order into chunks.
        partition: list of (start_idx, end_idx) pairs."""
        if time.time() > deadline - 2:
            return None
        
        ops = []
        in_fast = set()
        
        for start_idx, end_idx in partition:
            # Check if we can process this chunk without exceeding fast_memory_budget
            # We need to account for vertices already in fast memory
            # Simulate to check memory usage
            test_in_fast = set(in_fast)
            for i in range(start_idx, end_idx + 1):
                v = topo_order[i]
                if v in inputs:
                    if v not in test_in_fast:
                        test_in_fast.add(v)
                else:
                    ps = parents_dict.get(v, [])
                    for p in ps:
                        if p not in test_in_fast:
                            test_in_fast.add(p)
                    test_in_fast.add(v)
            
            if len(test_in_fast) > fast_memory_budget:
                # This partition doesn't work with the given budget
                return None
            
            # Process the chunk
            chunk_ops, in_fast = _generate_chunk_ops(parents_dict, inputs, outputs, children, 
                                                       start_idx, end_idx, topo_order, in_fast)
            ops.extend(chunk_ops)
        
        # Store all outputs
        for v in outputs:
            if v in in_fast:
                in_fast.discard(v)
                ops.append(['store', v])
        
        return ops

    try:
        parents_dict, inputs, outputs = _extract(dag)
        topo_order, children, all_vertices = _topo_order(parents_dict, inputs)
        n = len(topo_order)
        
        if n == 0:
            return []
        
        # Find output indices
        output_indices = [topo_order.index(v) for v in outputs if v in topo_order]
        max_output_idx = max(output_indices) if output_indices else n - 1
        
        # Precompute: for each position i, what's the maximum chunk end we can have
        # such that memory stays within budget
        # Use a greedy approach: find the longest valid chunk starting from each position
        
        # First, let's try a simple strategy: process in chunks where each chunk
        # has memory need <= fast_memory_budget
        
        # Compute memory need for chunks of various lengths
        # For efficiency, we'll use a sliding window approach
        
        # Let's try different partition strategies
        best_ops = None
        best_transfers = float('inf')
        
        # Strategy 1: Greedy - take the longest possible chunk at each step
        if time.time() < deadline - 5:
            partition = []
            i = 0
            while i <= max_output_idx:
                # Find the longest chunk starting at i that fits in budget
                # Binary search on end_idx
                lo, hi = i, min(i + 200, max_output_idx)  # Limit search for speed
                best_end = i
                
                # First check if single vertex works
                if _chunk_memory_need(parents_dict, inputs, children, i, i, topo_order) <= fast_memory_budget:
                    # Binary search for longest valid chunk
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if _chunk_memory_need(parents_dict, inputs, children, i, mid, topo_order) <= fast_memory_budget:
                            best_end = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                
                if best_end == i:
                    # Single vertex
                    partition.append((i, i))
                    i += 1
                else:
                    partition.append((i, best_end))
                    i = best_end + 1
            
            ops = _simulate_partition(parents_dict, inputs, outputs, topo_order, children, partition, fast_memory_budget)
            if ops is not None:
                tr = _transfers(ops)
                if tr < best_transfers:
                    best_transfers = tr
                    best_ops = ops
        
        # Strategy 2: DP over partition points
        # dp[i] = minimum transfers to process vertices 0..i-1
        # This is expensive, so only try for small n
        if n <= 200 and time.time() < deadline - 10:
            INF = float('inf')
            dp = [INF] * (n + 1)
            dp[0] = 0
            parent = [-1] * (n + 1)
            
            for i in range(1, n + 1):
                # Try all possible chunk starts j where j <= i
                for j in range(max(0, i - 300), i):  # Limit chunk length for speed
                    chunk_ops, _ = _generate_chunk_ops(parents_dict, inputs, outputs, children, 
                                                        j, i - 1, topo_order, set())
                    chunk_tr = _transfers(chunk_ops)
                    
                    # Check memory constraint
                    mem_need = _chunk_memory_need(parents_dict, inputs, children, j, i - 1, topo_order)
                    if mem_need > fast_memory_budget:
                        continue
                    
                    if dp[j] + chunk_tr < dp[i]:
                        dp[i] = dp[j] + chunk_tr
                        parent[i] = j
            
            if dp[n] < INF:
                # Reconstruct partition
                partition = []
                i = n
                while i > 0:
                    j = parent[i]
                    partition.append((j, i - 1))
                    i = j
                partition.reverse()
                
                ops = _simulate_partition(parents_dict, inputs, outputs, topo_order, children, partition, fast_memory_budget)
                if ops is not None:
                    tr = _transfers(ops)
                    if tr < best_transfers:
                        best_transfers = tr
                        best_ops = ops
        
        # Strategy 3: Fixed-size chunks
        if time.time() < deadline - 5:
            for chunk_size in [10, 20, 50, 100]:
                partition = []
                for i in range(0, max_output_idx + 1, chunk_size):
                    end = min(i + chunk_size - 1, max_output_idx)
                    partition.append((i, end))
                
                ops = _simulate_partition(parents_dict, inputs, outputs, topo_order, children, partition, fast_memory_budget)
                if ops is not None:
                    tr = _transfers(ops)
                    if tr < best_transfers:
                        best_transfers = tr
                        best_ops = ops
        
        # Fallback: simple sequential processing
        if best_ops is None:
            ops = []
            in_fast = set()
            for v in topo_order:
                if v in inputs:
                    if v not in in_fast:
                        in_fast.add(v)
                        ops.append(['load', v])
                else:
                    ps = parents_dict.get(v, [])
                    for p in ps:
                        if p not in in_fast:
                            in_fast.add(p)
                            ops.append(['load', p])
                    in_fast.add(v)
                    ops.append(['compute', v])
            
            for v in outputs:
                if v in in_fast:
                    in_fast.discard(v)
                    ops.append(['store', v])
            
            best_ops = ops
        
        # Compare with baseline
        if baseline_ops is not None:
            bt = _transfers(baseline_ops)
            if best_ops is not None and _transfers(best_ops) < bt:
                return best_ops
            return baseline_ops
        else:
            return best_ops if best_ops is not None else []

    except Exception:
        if baseline_ops is not None:
            return baseline_ops
        return []
