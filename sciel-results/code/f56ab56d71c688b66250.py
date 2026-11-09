def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 85
    
    # Fallback to baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except:
        baseline_ops = None
    
    try:
        # Extract DAG info
        if isinstance(dag, dict):
            parents_dict = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
        else:
            parents_dict = dict(getattr(dag, 'parents', {}))
            inputs = list(getattr(dag, 'inputs', []))
            outputs = set(getattr(dag, 'outputs', []))
        
        # Compute topological order
        all_vertices = set(inputs)
        for v, ps in parents_dict.items():
            all_vertices.add(v)
            all_vertices.update(ps)
        
        in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
        children = {v: [] for v in all_vertices}
        for v, ps in parents_dict.items():
            for p in ps:
                children[p].append(v)
        
        from collections import deque
        queue = deque([v for v in all_vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        if len(topo_order) < len(all_vertices):
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        pos_in_topo = {v: i for i, v in enumerate(topo_order)}
        n = len(topo_order)
        
        # Precompute for each vertex: last use position (global)
        last_use = {v: pos_in_topo[v] for v in topo_order}
        for v in topo_order:
            for c in children.get(v, []):
                last_use[v] = max(last_use[v], pos_in_topo[c])
        
        # For the entire DAG, outputs must be kept until the end
        for v in outputs:
            last_use[v] = n - 1
        
        # Precompute child positions for each vertex
        child_positions = {v: [pos_in_topo[c] for c in children.get(v, [])] for v in topo_order}
        
        # Memoization for solve_range
        memo = {}
        
        def compute_cut_size(k, start_idx, end_idx):
            """Compute the size of the cut at position k"""
            cut_size = 0
            for v in topo_order[start_idx:k]:
                for cp in child_positions[v]:
                    if cp >= k and cp < end_idx:
                        cut_size += 1
                        break
            return cut_size
        
        def estimate_transfers(start_idx, end_idx, pre_loaded_count):
            """
            Estimate the number of transfers needed for a sub-range.
            Uses a smooth surrogate based on cut sizes and memory pressure.
            """
            range_size = end_idx - start_idx
            if range_size <= 0:
                return 0
            
            # For a linear pass, estimate transfers based on:
            # 1. Number of external inputs that need loading
            # 2. Number of vertices that need to be kept (memory pressure)
            
            sub_set = set(topo_order[start_idx:end_idx])
            
            ext_input_count = 0
            for v in topo_order[start_idx:end_idx]:
                for p in parents_dict.get(v, []):
                    if p not in sub_set:
                        ext_input_count += 1
                        break
            
            # Memory pressure: max concurrent vertices in memory
            # Approximate by average fan-in + 1
            avg_fan_in = sum(len(parents_dict.get(v, [])) for v in topo_order[start_idx:end_idx]) / max(1, range_size)
            
            # If memory is tight, more transfers needed
            memory_pressure = max(0, (avg_fan_in + 1) - fast_memory_budget)
            
            # Base transfers: at least ext_input_count
            base_transfers = ext_input_count + pre_loaded_count
            
            # Additional transfers due to memory pressure (smooth approximation)
            # Use soft-max style: if pressure > 0, add proportional transfers
            extra_transfers = memory_pressure * range_size * 0.5
            
            return base_transfers + extra_transfers
        
        def solve_range(start_idx, end_idx, pre_loaded_set):
            """
            Pebble the sub-DAG defined by topo_order[start_idx:end_idx].
            pre_loaded_set: Set of vertices already in fast memory at start.
            Returns: (ops, final_mem_set, transfer_count)
            """
            if start_idx >= end_idx:
                return [], pre_loaded_set, 0
            
            # Check memo - use frozenset of pre_loaded_set as key
            memo_key = (start_idx, end_idx, frozenset(pre_loaded_set))
            if memo_key in memo:
                return memo[memo_key]
            
            sub_topo = topo_order[start_idx:end_idx]
            sub_set = set(sub_topo)
            
            # Determine external inputs and outputs for this sub-range
            ext_inputs = set()
            ext_outputs = set()
            
            for v in sub_topo:
                for p in parents_dict.get(v, []):
                    if p not in sub_set:
                        ext_inputs.add(p)
                for c in children.get(v, []):
                    if c not in sub_set:
                        ext_outputs.add(v)
            
            # Try linear pass first
            in_mem = set(pre_loaded_set)
            simulated_ops = []
            fits = True
            transfer_count = 0
            
            for i, v in enumerate(sub_topo):
                # Load inputs not in memory
                for p in parents_dict.get(v, []):
                    if p not in in_mem:
                        if len(in_mem) >= fast_memory_budget:
                            fits = False
                            break
                        in_mem.add(p)
                        simulated_ops.append(['load', p])
                        transfer_count += 1
                    if not fits:
                        break
                
                if not fits:
                    break
                
                # Compute v
                in_mem.add(v)
                simulated_ops.append(['compute', v])
                
                # Delete if not needed later in this sub-range and not an ext_output
                needed_later = False
                for c in children.get(v, []):
                    if c in sub_set and pos_in_topo[c] > i:
                        needed_later = True
                        break
                if v in ext_outputs:
                    needed_later = True
                
                if not needed_later:
                    in_mem.discard(v)
                    simulated_ops.append(['delete', v])
                
                if len(in_mem) > fast_memory_budget:
                    fits = False
                    break
            
            if fits:
                result = (simulated_ops, in_mem, transfer_count)
                if len(memo) < 10000:  # Limit memo size
                    memo[memo_key] = result
                return result
            
            # Linear doesn't fit, need to split
            # Use LP-inspired approach: find split that minimizes estimated total transfers
            
            range_size = end_idx - start_idx
            
            # Generate candidates: use denser sampling for better quality
            if range_size <= 200:
                candidates = list(range(start_idx + 1, end_idx))
            elif range_size <= 1000:
                step = max(1, range_size // 200)
                candidates = list(range(start_idx + 1, end_idx, step))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            else:
                step = max(1, range_size // 300)
                candidates = list(range(start_idx + 1, end_idx, step))
                if candidates[-1] != end_idx - 1:
                    candidates.append(end_idx - 1)
            
            best_k = -1
            best_cost = float('inf')
            
            for k in candidates:
                # Compute cut size
                cut_size = compute_cut_size(k, start_idx, end_idx)
                
                if cut_size > fast_memory_budget:
                    continue
                
                # Smooth cost model: use log-sum-exp style soft-min
                # Estimate transfers for left and right subproblems
                left_size = k - start_idx
                right_size = end_idx - k
                
                # For left: pre_loaded_set is the input
                left_transfers = estimate_transfers(start_idx, k, len(pre_loaded_set))
                
                # For right: pre_loaded is the cut vertices (approximately cut_size)
                right_transfers = estimate_transfers(k, end_idx, cut_size)
                
                # Cut penalty: vertices that must be kept in memory
                # Use smooth penalty: cut_size^1.5 to penalize large cuts more
                cut_penalty = (cut_size ** 1.5) if cut_size > 0 else 0
                
                # Balance penalty: prefer balanced splits but allow imbalance if it reduces transfers
                imbalance = abs(left_size - right_size)
                balance_penalty = imbalance * 0.1  # Small penalty for imbalance
                
                # Total cost
                cost = left_transfers + right_transfers + cut_penalty + balance_penalty
                
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            
            # Local search polish: try neighbors of best_k
            if best_k != -1:
                local_candidates = set()
                # Add neighbors within a window
                window = max(5, range_size // 50)
                for delta in range(-window, window + 1):
                    nk = best_k + delta
                    if start_idx < nk < end_idx:
                        local_candidates.add(nk)
                
                for k in local_candidates:
                    if k == best_k:
                        continue
                    cut_size = compute_cut_size(k, start_idx, end_idx)
                    if cut_size > fast_memory_budget:
                        continue
                    
                    left_size = k - start_idx
                    right_size = end_idx - k
                    
                    left_transfers = estimate_transfers(start_idx, k, len(pre_loaded_set))
                    right_transfers = estimate_transfers(k, end_idx, cut_size)
                    cut_penalty = (cut_size ** 1.5) if cut_size > 0 else 0
                    imbalance = abs(left_size - right_size)
                    balance_penalty = imbalance * 0.1
                    
                    cost = left_transfers + right_transfers + cut_penalty + balance_penalty
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
            
            if best_k == -1:
                # No valid split found. Try to find any k with cut_size <= budget
                for k in range(start_idx + 1, end_idx):
                    cut_size = compute_cut_size(k, start_idx, end_idx)
                    if cut_size <= fast_memory_budget:
                        best_k = k
                        break
            
            if best_k == -1:
                # No valid split at all. Return linear ops (may be invalid) or baseline
                if baseline_ops is not None:
                    return baseline_ops, pre_loaded_set, 0
                result = (simulated_ops, in_mem, transfer_count)
                if len(memo) < 10000:
                    memo[memo_key] = result
                return result
            
            # Recurse
            ops1, mem1, tc1 = solve_range(start_idx, best_k, pre_loaded_set)
            ops2, mem2, tc2 = solve_range(best_k, end_idx, mem1)
            
            result = (ops1 + ops2, mem2, tc1 + tc2)
            if len(memo) < 10000:
                memo[memo_key] = result
            return result
        
        # Initial call
        ops, final_mem, total_transfers = solve_range(0, n, set())
        
        # Ensure outputs are stored
        for v in outputs:
            if v in final_mem:
                ops.append(['store', v])
                final_mem.discard(v)
                total_transfers += 1
        
        # Check operation limit
        if len(ops) > 20000:
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        # Check time
        if time.time() > deadline:
            if baseline_ops is not None:
                return baseline_ops
            return []
        
        return ops
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        if baseline_ops is not None:
            return baseline_ops
        return []
