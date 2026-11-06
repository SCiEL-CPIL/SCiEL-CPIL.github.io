"""XOR-circuit synthesis with gate sharing optimization."""
import time
import numpy as np

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
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
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
    
    # For each row, find the set of input indices that are 1
    row_sets = []
    for i in range(m):
        row = A[i]
        indices = np.where(row != 0)[0]
        row_sets.append(frozenset(indices.tolist()))
    
    def _make_baseline():
        """Simple baseline: no sharing, just XOR each row's inputs independently."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            row = A[i]
            indices = np.where(row != 0)[0].tolist()
            if not indices:
                outputs.append(n)
            elif len(indices) == 1:
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
        wire_val = [0] * (n + 1 + len(gates))
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            if a >= len(wire_val) or b >= len(wire_val):
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        
        for i in range(m):
            expected = 0
            for j in range(n):
                if A[i, j]:
                    expected |= (1 << j)
            if outputs[i] >= len(wire_val):
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True
    
    def _compute_depth(gates, outputs):
        """Compute max depth of circuit."""
        if not gates:
            return 0
        wire_depth = [0] * (n + 1 + len(gates))
        for i in range(n + 1):
            wire_depth[i] = 0
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < len(wire_depth):
                max_depth = max(max_depth, wire_depth[w])
        return max_depth
    
    def _greedy_sharing(row_sets, n, m, time_budget):
        """
        Greedy sharing algorithm:
        1. Start with each row as its own set of input wires
        2. Repeatedly find the pair of outputs with maximum overlap in input wires
        3. Extract the overlap as a shared sub-circuit
        4. Update expressions
        """
        expressions = [set(row_sets[i]) for i in range(m)]
        gates = []
        next_wire = n + 1
        wire_depth = [0] * (n + 1 + len(gates) + 100)  # Pre-allocate some space
        
        # Track depth for each wire
        for i in range(n + 1):
            wire_depth[i] = 0
        
        max_iterations = min(200, m * m * 2)
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
            
            best_pair = None
            best_subset = None
            best_savings = 0
            
            # Try all pairs of outputs
            for i in range(m):
                if not expressions[i]:
                    continue
                for j in range(i + 1, m):
                    if not expressions[j]:
                        continue
                    # Find common input wires (only inputs 0..n-1)
                    common = expressions[i] & expressions[j]
                    common_inputs = {w for w in common if 0 <= w < n}
                    if len(common_inputs) >= 2:
                        savings = len(common_inputs) - 1
                        if savings > best_savings:
                            best_savings = savings
                            best_pair = (i, j)
                            best_subset = common_inputs
            
            if best_pair is None:
                break
            
            i, j = best_pair
            S = best_subset
            S_list = sorted(S)
            
            # Create gates to compute XOR(S)
            if len(S_list) == 2:
                gates.append([S_list[0], S_list[1]])
                shared_wire = next_wire
                next_wire += 1
                if shared_wire < len(wire_depth):
                    wire_depth[shared_wire] = max(wire_depth[S_list[0]], wire_depth[S_list[1]]) + 1
            else:
                prev = S_list[0]
                for k in range(1, len(S_list)):
                    gates.append([prev, S_list[k]])
                    new_wire = next_wire
                    next_wire += 1
                    if new_wire < len(wire_depth):
                        wire_depth[new_wire] = max(wire_depth[prev], wire_depth[S_list[k]]) + 1
                    prev = new_wire
                shared_wire = prev
            
            # Update expressions: replace S in both with shared_wire
            expressions[i] = (expressions[i] - S) | {shared_wire}
            expressions[j] = (expressions[j] - S) | {shared_wire}
        
        # Build final outputs
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if not expr:
                outputs.append(n)
            elif len(expr) == 1:
                w = list(expr)[0]
                outputs.append(w)
            else:
                expr_list = sorted(expr)
                prev = expr_list[0]
                for w in expr_list[1:]:
                    gates.append([prev, w])
                    new_wire = next_wire
                    next_wire += 1
                    if new_wire < len(wire_depth):
                        wire_depth[new_wire] = max(wire_depth[prev], wire_depth[w]) + 1
                    prev = new_wire
                outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}
    
    # Try greedy sharing
    try:
        result = _greedy_sharing(row_sets, n, m, deadline)
        if _verify(result):
            depth = _compute_depth(result['gates'], result['outputs'])
            if depth <= depth_limit:
                return result
    except Exception:
        pass
    
    # Fallback to baseline
    baseline = _make_baseline()
    if _verify(baseline):
        return baseline
    
    # Last resort: return empty (shouldn't happen)
    return {'gates': [], 'outputs': [n] * m}
