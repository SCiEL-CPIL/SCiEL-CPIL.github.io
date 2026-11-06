"""XOR-circuit synthesis with multi-granularity shared subexpression extraction."""
import time
import numpy as np
import itertools

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like, m x n binary matrix
        depth_limit: maximum circuit depth (<=16)
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Normalize input
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)
    
    # Input wires: 0..n-1
    # Constant zero wire: n
    # Gate k produces wire: n + 1 + k
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets for efficient manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)
    
    input_mask = (1 << n) - 1
    
    def _make_baseline():
        """Simple baseline: no sharing, just XOR each row's inputs independently."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = []
                for j in range(n):
                    if bits & (1 << j):
                        indices.append(j)
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}
    
    def _verify(circuit):
        """Verify circuit correctness using bitset simulation."""
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        
        for i in range(m):
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True
    
    def _compute_depth(gates, outputs):
        """Compute max depth of circuit."""
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for i in range(n + 1):
            wire_depth[i] = 0
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth
    
    def _build_final_outputs(expressions, gates, next_wire, n, m, wire_depth):
        """Build final outputs from expressions, adding necessary gates."""
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                expr_list = []
                for w in range(n + 1 + len(gates)):
                    if expr & (1 << w):
                        expr_list.append(w)
                
                if len(expr_list) == 1:
                    outputs.append(expr_list[0])
                else:
                    prev = expr_list[0]
                    for w in expr_list[1:]:
                        gates.append([prev, w])
                        new_wire = next_wire
                        next_wire += 1
                        if new_wire < len(wire_depth):
                            wire_depth[new_wire] = max(wire_depth[prev], wire_depth[w]) + 1
                        prev = new_wire
                    outputs.append(prev)
        return outputs
    
    def _add_shared_gates(S, gates, next_wire, wire_depth, n):
        """Add gates to compute XOR of set bits in S. Returns (shared_wire, next_wire)."""
        S_list = []
        for k in range(n):
            if S & (1 << k):
                S_list.append(k)
        
        if len(S_list) == 1:
            return S_list[0], next_wire
        
        if len(S_list) == 2:
            gates.append([S_list[0], S_list[1]])
            shared_wire = next_wire
            next_wire += 1
            if shared_wire < len(wire_depth):
                wire_depth[shared_wire] = max(wire_depth[S_list[0]], wire_depth[S_list[1]]) + 1
            return shared_wire, next_wire
        else:
            prev = S_list[0]
            for k in range(1, len(S_list)):
                gates.append([prev, S_list[k]])
                new_wire = next_wire
                next_wire += 1
                if new_wire < len(wire_depth):
                    wire_depth[new_wire] = max(wire_depth[prev], wire_depth[S_list[k]]) + 1
                prev = new_wire
            return prev, next_wire
    
    def _global_soft_min_search(row_bits, n, m, time_budget, depth_limit):
        """
        Soft-min (log-sum-exp) inspired global search:
        Score each candidate subset by how many rows it helps reduce.
        This smooth objective discovers intermediates that are 'close' to multiple targets.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        max_possible_wires = n + 1 + m * m + 500
        wire_depth = [0] * max_possible_wires
        
        # Pre-compute: for each row, the set of inputs
        row_input_sets = [row_bits[i] & input_mask for i in range(m)]
        
        # We will iteratively find the best subset to extract
        # A subset S is "good" if it appears in many rows and reduces total gate count
        
        max_iterations = min(200, m * 10)
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
            
            # Find all non-trivial subsets that appear in at least 2 rows
            # For efficiency, we look at pairs and triples of inputs
            # Actually, let's look at all common subsets of size 2+
            
            best_subset = 0
            best_score = 0
            best_rows = []
            
            # Strategy: Look at all pairs of inputs that appear together in multiple rows
            # Then check larger subsets
            
            # First pass: find all input pairs that appear in >= 2 rows
            pair_counts = {}
            for i in range(m):
                bits = expressions[i] & input_mask
                # Get list of set bits
                indices = []
                for j in range(n):
                    if bits & (1 << j):
                        indices.append(j)
                for idx in indices:
                    for jdx in indices:
                        if jdx > idx:
                            key = (idx, jdx)
                            pair_counts[key] = pair_counts.get(key, 0) + 1
            
            # For each pair with count >= 2, try extending to larger subsets
            # But for efficiency, just use the pair itself or extend greedily
            
            for (a, b), count in pair_counts.items():
                if count < 2:
                    continue
                
                # Start with this pair
                S = (1 << a) | (1 << b)
                
                # Try to extend: add inputs that are in all rows containing S
                # Find rows that contain both a and b
                rows_with_S = []
                for i in range(m):
                    if (expressions[i] & (1 << a)) and (expressions[i] & (1 << b)):
                        rows_with_S.append(i)
                
                if len(rows_with_S) < 2:
                    continue
                
                # Try adding more bits that are common to all these rows
                common_to_all = input_mask
                for i in rows_with_S:
                    common_to_all &= expressions[i]
                
                # S can be extended by any subset of common_to_all
                # Let's try the maximal extension
                extended_S = S | common_to_all
                # But we need to be careful: extended_S must be a subset of expressions[i] for all i in rows_with_S
                # By construction, common_to_all is a subset of expressions[i] for all i in rows_with_S
                # And S is a subset of expressions[i] for all i in rows_with_S
                # So extended_S is a subset of expressions[i] for all i in rows_with_S
                
                # Score: number of rows helped * (size of S - 1)
                # The gate savings is (len(S_list) - 1) per row
                S_list_count = bin(extended_S & input_mask).count('1')
                if S_list_count < 2:
                    continue
                
                score = len(rows_with_S) * (S_list_count - 1)
                
                if score > best_score:
                    best_score = score
                    best_subset = extended_S
                    best_rows = rows_with_S
            
            if best_subset == 0 or best_score <= 1:
                break
            
            S = best_subset
            rows_to_update = best_rows
            
            # Add gates for S
            shared_wire, next_wire = _add_shared_gates(S, gates, next_wire, wire_depth, n)
            
            # Update expressions
            shared_bit = 1 << shared_wire
            for i in rows_to_update:
                expressions[i] = (expressions[i] & ~S) | shared_bit
        
        # Build final outputs
        outputs = _build_final_outputs(expressions, gates, next_wire, n, m, wire_depth)
        
        return {'gates': gates, 'outputs': outputs}
    
    def _pairwise_greedy_with_extension(row_bits, n, m, time_budget, depth_limit):
        """
        Pairwise greedy sharing with subset extension:
        When finding the best pair, extend the common subset to the maximal
        common subset across all rows that contain it.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        max_possible_wires = n + 1 + m * m + 500
        wire_depth = [0] * max_possible_wires
        
        max_iterations = min(300, m * m * 2)
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
            
            best_pair = None
            best_subset = 0
            best_savings = 0
            best_rows = []
            
            for i in range(m):
                if expressions[i] == 0:
                    continue
                for j in range(i + 1, m):
                    if expressions[j] == 0:
                        continue
                    # Find common input wires
                    common = expressions[i] & expressions[j] & input_mask
                    if common == 0:
                        continue
                    
                    # Find all rows that contain 'common'
                    rows_with_common = []
                    for k in range(m):
                        if (expressions[k] & common) == common:
                            rows_with_common.append(k)
                    
                    if len(rows_with_common) < 2:
                        continue
                    
                    # Extend common to maximal common subset
                    extended = common
                    for k in rows_with_common:
                        extended &= expressions[k]
                    
                    # extended is the maximal common subset
                    # But we need extended to be a superset of common
                    # Actually, extended = common & (intersection of all expressions[k])
                    # So extended is a subset of common
                    # We want to extend common, so we should look at the intersection
                    # of all rows that contain common, but only add bits that are
                    # in all of them.
                    
                    # Let's recompute: the maximal subset T such that T is a subset
                    # of expressions[k] for all k in rows_with_common
                    # and T is a superset of common
                    # This is: common | (intersection of expressions[k] for k in rows_with_common)
                    # No, that's not right either.
                    
                    # Actually, the maximal common subset is just the intersection
                    # of expressions[k] for all k in rows_with_common.
                    # This intersection contains common (since common is a subset
                    # of expressions[i] and expressions[j], and i,j are in rows_with_common).
                    
                    max_common = input_mask
                    for k in rows_with_common:
                        max_common &= expressions[k]
                    
                    # max_common is the maximal subset common to all rows in rows_with_common
                    # It contains common.
                    
                    savings = bin(max_common).count('1') - 1
                    if savings > best_savings:
                        best_savings = savings
                        best_pair = (i, j)
                        best_subset = max_common
                        best_rows = rows_with_common
            
            if best_pair is None or best_savings <= 0:
                break
            
            S = best_subset
            rows_to_update = best_rows
            
            shared_wire, next_wire = _add_shared_gates(S, gates, next_wire, wire_depth, n)
            
            shared_bit = 1 << shared_wire
            for i in rows_to_update:
                expressions[i] = (expressions[i] & ~S) | shared_bit
        
        outputs = _build_final_outputs(expressions, gates, next_wire, n, m, wire_depth)
        
        return {'gates': gates, 'outputs': outputs}
    
    def _row_based_sharing(row_bits, n, m, time_budget, depth_limit):
        """
        Row-based sharing: group rows by similarity and share subcircuits.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        max_possible_wires = n + 1 + m * m + 500
        wire_depth = [0] * max_possible_wires
        
        # Sort rows by bit count (descending) to process dense rows first
        row_indices = list(range(m))
        row_indices.sort(key=lambda i: bin(row_bits[i]).count('1'), reverse=True)
        
        max_iterations = min(200, m * 5)
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
            
            best_subset = 0
            best_savings = 0
            best_rows = []
            
            # For each subset of size 2-4, find how many rows contain it
            # Use a sampling approach for large n
            if n <= 20:
                # Try all pairs
                for a in range(n):
                    for b in range(a + 1, n):
                        S = (1 << a) | (1 << b)
                        count = 0
                        rows = []
                        for i in range(m):
                            if (expressions[i] & S) == S:
                                count += 1
                                rows.append(i)
                        if count >= 2:
                            # Extend
                            extended = S
                            for i in rows:
                                extended &= expressions[i]
                            savings = bin(extended).count('1') - 1
                            if savings > best_savings:
                                best_savings = savings
                                best_subset = extended
                                best_rows = rows
            else:
                # Sample-based: try random pairs
                import random
                for _ in range(100):
                    a = random.randint(0, n - 1)
                    b = random.randint(0, n - 1)
                    if a == b:
                        continue
                    S = (1 << a) | (1 << b)
                    count = 0
                    rows = []
                    for i in range(m):
                        if (expressions[i] & S) == S:
                            count += 1
                            rows.append(i)
                    if count >= 2:
                        extended = S
                        for i in rows:
                            extended &= expressions[i]
                        savings = bin(extended).count('1') - 1
                        if savings > best_savings:
                            best_savings = savings
                            best_subset = extended
                            best_rows = rows
            
            if best_subset == 0 or best_savings <= 0:
                break
            
            S = best_subset
            rows_to_update = best_rows
            
            shared_wire, next_wire = _add_shared_gates(S, gates, next_wire, wire_depth, n)
            
            shared_bit = 1 << shared_wire
            for i in rows_to_update:
                expressions[i] = (expressions[i] & ~S) | shared_bit
        
        outputs = _build_final_outputs(expressions, gates, next_wire, n, m, wire_depth)
        
        return {'gates': gates, 'outputs': outputs}
    
    # Try multiple strategies and keep the best
    best_result = None
    best_gate_count = float('inf')
    
    strategies = [
        ("pairwise_greedy", _pairwise_greedy_with_extension),
        ("global_soft_min", _global_soft_min_search),
        ("row_based", _row_based_sharing),
    ]
    
    for name, strategy_func in strategies:
        if time.time() > deadline - 5.0:
            break
        try:
            result = strategy_func(row_bits, n, m, deadline, depth_limit)
            if _verify(result):
                depth = _compute_depth(result['gates'], result['outputs'])
                if depth <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_gate_count = len(result['gates'])
                        best_result = result
        except Exception:
            pass
    
    # Fallback to baseline
    if best_result is None:
        baseline = _make_baseline()
        if _verify(baseline):
            return baseline
    
    return best_result if best_result is not None else {'gates': [], 'outputs': [n] * m}
