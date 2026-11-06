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
    
    # Convert rows to sets of wire indices
    # Wire i corresponds to input i
    row_sets = []
    for i in range(m):
        bits = set()
        for j in range(n):
            if A[i, j]:
                bits.add(j)
        row_sets.append(bits)
        
    # Baseline: just XOR inputs
    def _make_baseline():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = sorted(row_sets[i])
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                prev = bits[0]
                for b in bits[1:]:
                    gates.append([prev, b])
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
            expected = 0
            for j in range(n):
                if A[i, j]:
                    expected |= (1 << j)
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

    def _greedy_global_cse(row_sets, n, m, time_budget):
        # expressions: list of sets of wire ids
        expressions = [set(s) for s in row_sets]
        gates = []
        next_wire = n + 1
        
        # pool: set of all available wire ids (inputs + created gates)
        pool = set(range(n))
        
        # pair_counts: dict (a, b) -> count of expressions containing both a and b
        # We only care about pairs where a < b to avoid duplicates
        # But expressions can contain any wires from the pool
        pair_counts = {}
        
        def update_pair_counts(expr, old_wires, new_wires):
            # Remove counts for pairs involving old_wires that are no longer together
            # Add counts for pairs involving new_wires
            pass # Simplified: we will recompute or update carefully
            
        # Actually, maintaining pair counts dynamically is tricky.
        # For m<=16, n<=16, the number of wires is small.
        # We can just scan all pairs in the pool each iteration.
        
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
                
            best_pair = None
            best_count = 1 # Must be > 1 to be beneficial (savings >= 1 gate)
            
            # Get current pool
            current_pool = list(pool)
            
            # Find best pair
            for i in range(len(current_pool)):
                a = current_pool[i]
                for j in range(i+1, len(current_pool)):
                    b = current_pool[j]
                    # Count how many expressions contain both a and b
                    count = 0
                    for expr in expressions:
                        if a in expr and b in expr:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_pair = (a, b)
                        
            if best_pair is None:
                break
                
            a, b = best_pair
            # Create new gate G = a ^ b
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.add(g)
            
            # Update expressions
            for expr in expressions:
                if a in expr and b in expr:
                    expr.remove(a)
                    expr.remove(b)
                    expr.add(g)
                    
            # Optional: If an expression becomes empty, it's 0.
            # If it has size 1, it's just that wire.
            
        # Finalize outputs
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if not expr:
                outputs.append(n)
            elif len(expr) == 1:
                outputs.append(list(expr)[0])
            else:
                # Need to reduce expr to a single wire
                # Chain XORs
                wires = list(expr)
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
            
    # 2. Greedy Global CSE
    try:
        result = _greedy_global_cse(row_sets, n, m, deadline)
        if _verify(result):
            d = _compute_depth(result['gates'], result['outputs'])
            if d <= depth_limit:
                if len(result['gates']) < best_gate_count:
                    best_result = result
                    best_gate_count = len(result['gates'])
    except Exception:
        pass
        
    if best_result is None:
        return baseline
        
    return best_result
