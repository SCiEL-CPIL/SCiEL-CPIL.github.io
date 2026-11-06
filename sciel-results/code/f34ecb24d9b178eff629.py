import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin from 90s

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)

    if m == 0:
        return {'gates': [], 'outputs': []}

    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}

    # Convert rows to bitsets (Python ints)
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

    def _finalize_circuit(expressions, gates, next_wire, n, m):
        """
        Takes the final state of expressions and gates, and constructs the full circuit
        by adding any remaining XORs needed to reduce each expression to a single wire.
        """
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

    def _beam_search_cse(row_bits, n, m, beam_width=3, max_steps=50):
        """
        Beam search for CSE: maintain multiple candidate states, expand each, keep best beam_width.
        """
        if hasattr(int, 'bit_count'):
            def popcount(x):
                return x.bit_count()
        else:
            def popcount(x):
                return bin(x).count('1')

        # State: (expressions, gates, next_wire, pool)
        # Score: total gates + penalty for remaining complexity
        def state_score(state):
            exprs, gates, next_wire, pool = state
            base = len(gates)
            # Penalty for remaining work: sum of popcounts - m (for single-wire expressions)
            remaining = 0
            for e in exprs:
                pc = popcount(e)
                if pc > 1:
                    remaining += pc - 1
                elif pc == 1:
                    remaining += 0
                else:
                    remaining += 0
            return base + 0.5 * remaining

        # Initial state
        init_exprs = list(row_bits)
        init_gates = []
        init_next_wire = n + 1
        init_pool = frozenset(range(n))
        beam = [(init_exprs, init_gates, init_next_wire, init_pool)]

        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break

            candidates = []
            for exprs, gates, next_wire, pool in beam:
                pool_list = list(pool)
                pool_size = len(pool_list)
                if pool_size < 2:
                    continue

                # Generate all candidate pairs
                for i in range(pool_size):
                    a = pool_list[i]
                    a_bit = 1 << a
                    for j in range(i + 1, pool_size):
                        b = pool_list[j]
                        b_bit = 1 << b
                        count = 0
                        for expr in exprs:
                            if (expr & a_bit) and (expr & b_bit):
                                count += 1
                        if count > 1:
                            g = next_wire
                            new_gates = gates + [[a, b]]
                            new_next_wire = next_wire + 1
                            new_pool = pool | {g}
                            g_bit = 1 << g

                            new_exprs = []
                            for expr in exprs:
                                if (expr & a_bit) and (expr & b_bit):
                                    new_expr = (expr & ~a_bit & ~b_bit) | g_bit
                                    new_exprs.append(new_expr)
                                else:
                                    new_exprs.append(expr)

                            candidates.append((new_exprs, new_gates, new_next_wire, new_pool))

            if not candidates:
                break

            # Score and keep top beam_width
            scored = []
            for c in candidates:
                s = state_score(c)
                scored.append((s, c))

            scored.sort(key=lambda x: x[0])
            beam = [c for _, c in scored[:beam_width]]

            # Check if any state is fully reduced
            all_done = False
            for exprs, gates, next_wire, pool in beam:
                all_reduced = True
                for e in exprs:
                    pc = popcount(e)
                    if pc > 1:
                        all_reduced = False
                        break
                if all_reduced:
                    all_done = True
                    break
            if all_done:
                break

        if not beam:
            return None

        # Pick best state from beam
        best_state = min(beam, key=state_score)
        exprs, gates, next_wire, pool = best_state
        return _finalize_circuit(exprs, gates, next_wire, n, m)

    def _local_polish(circuit, max_iterations=20):
        """
        Try to improve circuit by swapping adjacent gate orderings or reordering.
        Returns improved circuit if found, else original.
        """
        gates = circuit['gates']
        outputs = circuit['outputs']
        if len(gates) < 2:
            return circuit

        best = circuit
        best_depth = _compute_depth(gates, outputs)
        best_gate_count = len(gates)

        for _ in range(max_iterations):
            if time.time() > deadline - 2.0:
                break

            improved = False
            # Try swapping adjacent gates
            for i in range(len(gates) - 1):
                new_gates = gates[:]
                new_gates[i], new_gates[i+1] = new_gates[i+1], new_gates[i]
                
                test_circuit = {'gates': new_gates, 'outputs': outputs}
                if _verify(test_circuit):
                    d = _compute_depth(new_gates, outputs)
                    gc = len(new_gates)
                    if d <= depth_limit and (gc < best_gate_count or (gc == best_gate_count and d < best_depth)):
                        best = test_circuit
                        best_depth = d
                        best_gate_count = gc
                        gates = new_gates
                        improved = True
                        break

            if not improved:
                break

        return best

    # Try strategies
    best_result = None
    best_gate_count = float('inf')
    best_depth = float('inf')
    
    # 1. Baseline
    baseline = _make_baseline()
    if _verify(baseline):
        d = _compute_depth(baseline['gates'], baseline['outputs'])
        if d <= depth_limit:
            best_result = baseline
            best_gate_count = len(baseline['gates'])
            best_depth = d

    # 2. Beam search CSE
    try:
        beam_result = _beam_search_cse(row_bits, n, m, beam_width=4, max_steps=40)
        if beam_result and _verify(beam_result):
            d = _compute_depth(beam_result['gates'], beam_result['outputs'])
            if d <= depth_limit:
                gc = len(beam_result['gates'])
                if gc < best_gate_count or (gc == best_gate_count and d < best_depth):
                    best_gate_count = gc
                    best_depth = d
                    best_result = beam_result
    except Exception:
        pass

    # 3. Greedy CSE with multiple random restarts
    try:
        num_restarts = 6
        for seed in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            result = _greedy_cse_with_randomness(row_bits, n, m, deadline - 2.0, seed=seed)
            
            if result and _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    gc = len(result['gates'])
                    if gc < best_gate_count or (gc == best_gate_count and d < best_depth):
                        best_gate_count = gc
                        best_depth = d
                        best_result = result
                        
    except Exception:
        pass

    # 4. Local polish on best result
    if best_result is not None and time.time() < deadline - 1.0:
        try:
            polished = _local_polish(best_result, max_iterations=10)
            if polished and _verify(polished):
                d = _compute_depth(polished['gates'], polished['outputs'])
                if d <= depth_limit:
                    gc = len(polished['gates'])
                    if gc < best_gate_count or (gc == best_gate_count and d < best_depth):
                        best_result = polished
        except Exception:
            pass

    if best_result is None:
        return baseline
        
    return best_result
