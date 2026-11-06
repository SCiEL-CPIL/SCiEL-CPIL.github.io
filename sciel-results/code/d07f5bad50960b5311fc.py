"""XOR-circuit synthesis via global vector-space greedy basis construction."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix using a global vector-space
    greedy basis construction.
    
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
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets for efficient manipulation
    # We represent the current state of the matrix as a list of bitsets (integers)
    # Row i is a bitset of n bits.
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)
    
    # Also keep the original row bits for verification
    orig_row_bits = list(row_bits)
    
    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        if len(outputs) != m:
            return False
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        for i in range(m):
            expected = orig_row_bits[i]
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

    def _baseline():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = orig_row_bits[i]
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

    def _global_greedy():
        # State:
        # 1. `expressions`: list of m bitsets representing the current form of each output row in terms of available wires.
        # 2. `available_wires`: list of wire indices that exist.
        # 3. `wire_val`: dict mapping wire index to its value (bitset of inputs).
        # 4. `wire_depth`: dict mapping wire index to its depth.
        # 5. `gates`: list of gates created.
        
        expressions = list(orig_row_bits)
        gates = []
        next_wire = n + 1
        
        wire_val = {}
        wire_depth = {}
        for i in range(n):
            wire_val[i] = 1 << i
            wire_depth[i] = 0
        wire_val[n] = 0
        wire_depth[n] = 0
        
        available_wires = list(range(n + 1))
        
        # Pre-compute current weights to track improvement
        # We want to minimize the sum of weights of the expressions.
        # Adding a gate w = a ^ b costs 1 gate.
        # If w is added to an expression, its weight decreases by 1 (if a and b were present) 
        # OR increases? 
        # If we replace {a, b} with {w}, the weight of the expression changes from 2 to 1.
        # So the total weight decreases by 1 for each expression that contains both a and b.
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
                
            best_pair = None
            best_score = -1
            best_w = None
            best_depth = 0
            best_saving = 0
            
            L = len(available_wires)
            
            # To speed up, we can limit the search space if L is large.
            # But L is small.
            
            for i in range(L):
                a = available_wires[i]
                da = wire_depth[a]
                for j in range(i + 1, L):
                    b = available_wires[j]
                    db = wire_depth[b]
                    
                    new_depth = max(da, db) + 1
                    if new_depth > depth_limit:
                        continue
                    
                    saving = 0
                    for k in range(m):
                        e = expressions[k]
                        if (e & (1 << a)) and (e & (1 << b)):
                            saving += 1
                    
                    # We want to maximize saving.
                    # If saving == 0, creating the gate is useless for current expressions.
                    # If saving >= 1, it reduces total weight.
                    # However, we must ensure the gate itself is "worth it".
                    # The gate costs 1.
                    # The saving in final synthesis is `saving` gates (since each occurrence of {a,b} becomes {w}, reducing the XOR chain length by 1).
                    # So net gain is `saving - 1`.
                    # We should only proceed if saving >= 2 (net gain >= 1).
                    # If saving == 1, net gain is 0, but it might enable further savings?
                    # Usually, greedy algorithms only take positive gains.
                    # Let's stick to saving >= 2 for strict improvement, or saving >= 1 if we want to flatten depth?
                    # Actually, if saving == 1, we replace 2 wires with 1. The final synthesis of that expression becomes 1 gate shorter.
                    # But we added 1 gate. Net 0.
                    # However, `w` might be used by other expressions later?
                    # No, `w` is only useful if it's in the expression.
                    # If we don't add `w` to the expression, it's not in the expression.
                    # We MUST add `w` to the expression if we create it from `a` and `b` that are in the expression.
                    # So if saving == 1, we add `w` to 1 expression.
                    # That expression's weight drops by 1.
                    # Total gates: +1 (new gate) - 1 (shorter chain) = 0.
                    # So it's a wash.
                    # However, if we don't take it, we might miss a structural simplification?
                    # Let's try taking saving >= 2 first. If none, try saving == 1?
                    # Actually, taking saving == 1 is safe and might help depth or future steps?
                    # Let's prioritize high saving.
                    
                    score = saving
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_w = next_wire
                        best_depth = new_depth
                        best_saving = saving
            
            if best_pair is None or best_score < 2:
                # If best saving is 1, it's a wash in gates.
                # But it reduces the "complexity" (weight) of the expressions.
                # Let's allow saving == 1 if it reduces the maximum weight of any expression?
                # Or just break to avoid infinite loops of zero-gain moves?
                # The previous code broke on score <= 0.
                # Let's break if best_score < 2 to ensure we only make strictly improving moves in terms of gate count.
                # Wait, if we break at < 2, we might leave expressions with weight 2 that could be reduced to 1 with a wash?
                # No, if weight is 2, we need 1 gate. If we make it weight 1, we need 0 gates.
                # Creating the gate costs 1. Net 0.
                # So it doesn't help the final gate count.
                # It only helps depth.
                # Since we minimize gate count, we should probably stop.
                break
                
            a, b = best_pair
            w = best_w
            
            # Commit
            gates.append([a, b])
            wire_val[w] = wire_val[a] ^ wire_val[b]
            wire_depth[w] = best_depth
            
            for k in range(m):
                e = expressions[k]
                if (e & (1 << a)) and (e & (1 << b)):
                    e ^= (1 << a)
                    e ^= (1 << b)
                    e |= (1 << w)
                    expressions[k] = e
            
            available_wires.append(w)
            next_wire += 1
            
        # Final Synthesis
        outputs = []
        # We need to synthesize the remaining expressions.
        # To minimize depth and gates, we can build a shared tree if possible.
        # But for simplicity and safety, we chain.
        # Optimization: If an expression has weight 1, use that wire.
        # If weight 0, use 0.
        # If weight > 1, chain.
        
        # To improve sharing in the final phase, we can check if two expressions are identical?
        # If expr_i == expr_j, we can compute it once and output the same wire for both.
        
        # Let's group identical expressions.
        expr_to_wire = {}
        final_gates = []
        final_next_wire = next_wire
        
        for k in range(m):
            e = expressions[k]
            if e in expr_to_wire:
                outputs.append(expr_to_wire[e])
            else:
                # Synthesize this expression
                if e == 0:
                    wire_out = n
                else:
                    bits = []
                    for w_id in available_wires:
                        if e & (1 << w_id):
                            bits.append(w_id)
                    
                    if len(bits) == 1:
                        wire_out = bits[0]
                    else:
                        # Chain
                        prev = bits[0]
                        for b_id in bits[1:]:
                            final_gates.append([prev, b_id])
                            new_w = final_next_wire
                            final_next_wire += 1
                            prev = new_w
                        wire_out = prev
                
                expr_to_wire[e] = wire_out
                outputs.append(wire_out)
        
        # Append final gates
        gates.extend(final_gates)
        
        return {'gates': gates, 'outputs': outputs}

    def _polish(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        if not gates:
            return circuit
            
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        # Identify used wires
        used = set(outputs)
        for a, b in gates:
            used.add(a)
            used.add(b)
            
        # A gate is necessary if its output is in 'used'
        # and its inputs are valid.
        # We can remove gates whose output is not used.
        
        necessary = [False] * len(gates)
        
        # Map wire -> gate index
        wire_to_gate = {}
        for k in range(len(gates)):
            w = n + 1 + k
            wire_to_gate[w] = k
            
        # Mark outputs
        for w in outputs:
            if w in wire_to_gate:
                necessary[wire_to_gate[w]] = True
                
        # Propagate
        changed = True
        while changed:
            changed = False
            for k in range(len(gates) - 1, -1, -1):
                if necessary[k]:
                    a, b = gates[k]
                    for w in [a, b]:
                        if w in wire_to_gate:
                            idx = wire_to_gate[w]
                            if not necessary[idx]:
                                necessary[idx] = True
                                changed = True
                                
        # Build new circuit
        new_gates = []
        wire_map = {}
        new_next_wire = n + 1
        
        for k in range(len(gates)):
            if necessary[k]:
                a, b = gates[k]
                a_new = wire_map.get(a, a)
                b_new = wire_map.get(b, b)
                new_gates.append([a_new, b_new])
                old_w = n + 1 + k
                wire_map[old_w] = new_next_wire
                new_next_wire += 1
                
        new_outputs = []
        for w in outputs:
            new_outputs.append(wire_map.get(w, w))
            
        return {'gates': new_gates, 'outputs': new_outputs}

    # Run strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
        
    # 2. Global Greedy
    if time.time() < deadline - 10.0:
        try:
            res = _global_greedy()
            res = _polish(res)
            if _verify(res):
                depth = _compute_depth(res['gates'], res['outputs'])
                if depth <= depth_limit:
                    if len(res['gates']) < best_gate_count:
                        best_result = res
                        best_gate_count = len(res['gates'])
        except Exception:
            pass
            
    # 3. Fallback
    if best_result is None:
        best_result = {'gates': [], 'outputs': [n] * m}
        
    return best_result

if __name__ == "__main__":
    matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    res = synthesize(matrix, 16)
    print(res)
