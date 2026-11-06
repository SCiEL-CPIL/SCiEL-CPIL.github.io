import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)

    if m == 0:
        return {'gates': [], 'outputs': []}

    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}

    # Convert rows to bitsets (Python ints) for efficient manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)

    def _make_baseline():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
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
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _greedy_global_cse(row_bits, n, m, time_budget):
        """
        Global Common Subexpression Elimination using bitsets.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = set(range(n))
        
        max_iterations = 200
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
                
            best_pair = None
            best_count = 1
            
            current_pool = list(pool)
            pool_size = len(current_pool)
            
            for i in range(pool_size):
                a = current_pool[i]
                a_bit = 1 << a
                for j in range(i + 1, pool_size):
                    b = current_pool[j]
                    b_bit = 1 << b
                    count = 0
                    for expr in expressions:
                        if (expr & a_bit) and (expr & b_bit):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_pair = (a, b)
                        
            if best_pair is None:
                break
                
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.add(g)
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for idx, expr in enumerate(expressions):
                if (expr & a_bit) and (expr & b_bit):
                    new_expr = (expr & ~a_bit & ~b_bit) | g_bit
                    expressions[idx] = new_expr
                    
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                wires = []
                for w in range(next_wire):
                    if (expr >> w) & 1:
                        wires.append(w)
                        
                if len(wires) == 1:
                    outputs.append(wires[0])
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
                    
        return {'gates': gates, 'outputs': outputs}

    def _basis_greedy(row_bits, n, m, time_budget):
        """
        Basis construction greedy:
        1. Start with the set of target vectors.
        2. Maintain a set of 'available' wires (inputs + gates).
        3. At each step, pick the pair of available wires whose XOR has the highest 'value'.
           Value is defined as the reduction in total Hamming weight of the remaining target vectors
           if we were to replace any occurrence of that XOR in the targets.
           Actually, a better heuristic for XOR circuits is:
           Pick pair (a, b) that appears together in the most target expressions.
           This is similar to CSE but we treat the targets as a multiset of vectors.
           
           Let's try a different angle:
           We want to synthesize a basis for the span of the rows.
           If we can find a set of basis vectors B such that each row is a sparse combination of B,
           and B itself is cheap to synthesize, we win.
           
           Algorithm:
           1. Compute a basis for the row space using Gaussian elimination (standard).
           2. This gives us a set of 'pivot' vectors.
           3. Now we need to synthesize these pivot vectors and the combinations.
           
           Actually, the 'global CSE' above is already doing something similar.
           Let's try a 'minimum weight basis' approach.
           
           Alternative:
           Use a greedy algorithm to build a set of gates.
           Score of a gate g = a^b is:
           - How many target rows contain both a and b? (Saves gates in final reduction)
           - Does it create a new vector that is a 'basis element' for the row space?
           
           Let's stick to the CSE but improve the scoring.
           Current CSE picks the pair with max co-occurrence.
           This is good.
           
           Let's add a 'polish' step: local search on the gate list.
           Try swapping the order of gates or replacing gates.
        """
        # This function is a placeholder for now, we will rely on the improved CSE below.
        return None

    def _advanced_cse(row_bits, n, m, time_budget):
        """
        Improved CSE with lookahead and tie-breaking.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = set(range(n))
        
        # Pre-compute weights of expressions?
        # We want to minimize the total number of wires in the final expressions.
        
        max_iterations = 300
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
                
            best_pair = None
            best_score = 0
            best_savings = 0
            
            current_pool = list(pool)
            pool_size = len(current_pool)
            
            # Optimization: Only consider pairs from inputs or recent gates?
            # No, global is better for small n.
            
            for i in range(pool_size):
                a = current_pool[i]
                a_bit = 1 << a
                for j in range(i + 1, pool_size):
                    b = current_pool[j]
                    b_bit = 1 << b
                    
                    # Count co-occurrence
                    count = 0
                    # Also calculate potential savings in Hamming weight
                    # If we replace a,b with g, the weight of expr decreases by 1 if both were present.
                    # But g is a new wire.
                    # The main saving is in the number of XOR gates needed to reduce the expression.
                    # An expression with k wires needs k-1 gates.
                    # If we replace 2 wires with 1, we save 1 gate for that expression.
                    # So savings = count.
                    
                    for expr in expressions:
                        if (expr & a_bit) and (expr & b_bit):
                            count += 1
                            
                    # Tie-breaker: Prefer pairs that result in a vector with lower Hamming weight?
                    # Or pairs that are 'simpler'?
                    # Let's just use count.
                    
                    if count > best_score:
                        best_score = count
                        best_pair = (a, b)
                        best_savings = count
                        
            if best_pair is None or best_savings <= 1:
                break
                
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.add(g)
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for idx, expr in enumerate(expressions):
                if (expr & a_bit) and (expr & b_bit):
                    new_expr = (expr & ~a_bit & ~b_bit) | g_bit
                    expressions[idx] = new_expr
                    
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                wires = []
                for w in range(next_wire):
                    if (expr >> w) & 1:
                        wires.append(w)
                        
                if len(wires) == 1:
                    outputs.append(wires[0])
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
                    
        return {'gates': gates, 'outputs': outputs}

    # Try strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _make_baseline()
    if _verify(baseline):
        d = _compute_depth(baseline['gates'], baseline['outputs'])
        if d <= depth_limit:
            best_result = baseline
            best_gate_count = len(baseline['gates'])
            
    # 2. Advanced CSE
    try:
        time_left = deadline - time.time()
        if time_left > 0.1:
            result = _advanced_cse(row_bits, n, m, deadline - 0.5)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_result = result
                        best_gate_count = len(result['gates'])
    except Exception:
        pass
        
    # 3. Original Greedy CSE (sometimes finds different things due to tie-breaking or order)
    try:
        time_left = deadline - time.time()
        if time_left > 0.5:
            result = _greedy_global_cse(row_bits, n, m, deadline - 0.5)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_result = result
                        best_gate_count = len(result['gates'])
    except Exception:
        pass

    # 4. Randomized Restarts of CSE
    # The knowledge suggests randomized restarts can help escape local minima.
    # We will run the advanced CSE with random tie-breaking or random initial ordering.
    # Since the CSE is deterministic based on the pool order, we can shuffle the pool or the pair selection.
    
    def _randomized_cse(row_bits, n, m, time_budget, seed):
        import random
        random.seed(seed)
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = set(range(n))
        
        max_iterations = 200
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
                
            best_pair = None
            best_score = 0
            
            current_pool = list(pool)
            random.shuffle(current_pool) # Randomize order to break ties
            pool_size = len(current_pool)
            
            for i in range(pool_size):
                a = current_pool[i]
                a_bit = 1 << a
                for j in range(i + 1, pool_size):
                    b = current_pool[j]
                    b_bit = 1 << b
                    
                    count = 0
                    for expr in expressions:
                        if (expr & a_bit) and (expr & b_bit):
                            count += 1
                            
                    if count > best_score:
                        best_score = count
                        best_pair = (a, b)
                        
            if best_pair is None or best_score <= 1:
                break
                
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.add(g)
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for idx, expr in enumerate(expressions):
                if (expr & a_bit) and (expr & b_bit):
                    new_expr = (expr & ~a_bit & ~b_bit) | g_bit
                    expressions[idx] = new_expr
                    
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                wires = []
                for w in range(next_wire):
                    if (expr >> w) & 1:
                        wires.append(w)
                        
                if len(wires) == 1:
                    outputs.append(wires[0])
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
                    
        return {'gates': gates, 'outputs': outputs}

    # Run a few randomized restarts if time permits
    restarts = 0
    max_restarts = 10
    while time.time() < deadline - 1.0 and restarts < max_restarts:
        try:
            seed = int((time.time() - start_time) * 1000000) % 1000000
            result = _randomized_cse(row_bits, n, m, deadline - 0.5, seed)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_result = result
                        best_gate_count = len(result['gates'])
        except Exception:
            pass
        restarts += 1

    if best_result is None:
        return baseline
        
    return best_result
