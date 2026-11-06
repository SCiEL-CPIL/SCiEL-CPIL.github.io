import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitsets for efficient GF(2) manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)

    # If all zeros
    if not any(row_bits):
        return {'gates': [], 'outputs': [n] * m}

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

    def _baseline_circuit():
        """Independent XOR of inputs for each output."""
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

    def _greedy_cse_v3(time_budget, seed):
        """
        Greedy Common Subexpression Elimination with basis-aware scoring.
        Uses a soft-min surrogate to balance gate count and intermediate weight.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1))  # inputs 0..n-1 and zero wire n

        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            # Count frequency of each wire in exprs
            freq = {}
            for w in pool:
                bit = 1 << w
                cnt = 0
                for e in exprs:
                    if e & bit:
                        cnt += 1
                if cnt >= 2:
                    freq[w] = cnt

            if not freq:
                break

            # Only consider wires with freq >= 2
            candidates = list(freq.keys())
            if len(candidates) < 2:
                break

            # Sort by frequency descending
            candidates.sort(key=lambda w: freq[w], reverse=True)
            top_k = min(len(candidates), 40)
            top_candidates = candidates[:top_k]

            best_pair = None
            best_score = 1.0

            # Soft-min surrogate: score = benefit - alpha * weight_penalty
            # weight_penalty approximates the cost of the new intermediate
            alpha = 0.1

            for i in range(len(top_candidates)):
                a = top_candidates[i]
                a_bit = 1 << a
                for j in range(i + 1, len(top_candidates)):
                    b = top_candidates[j]
                    b_bit = 1 << b

                    # Count how many exprs contain both a and b
                    benefit = 0
                    for e in exprs:
                        if (e & a_bit) and (e & b_bit):
                            benefit += 1

                    if benefit <= 1:
                        continue

                    # Weight penalty: popcount of a XOR b (the new intermediate)
                    weight = bin(a_bit ^ b_bit).count('1')
                    # Normalize by n to keep scale reasonable
                    weight_penalty = alpha * (weight / max(n, 1))

                    score = benefit - weight_penalty

                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        # Build outputs
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                w = 0
                temp = e
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1

                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    # XOR all wires together
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)

        return {'gates': gates, 'outputs': outputs}

    def _basis_greedy_v2(time_budget, seed):
        """
        Basis-aware greedy: prioritize pairs that reduce total Hamming weight
        of the expression set, not just co-occurrence count.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1))

        max_iter = 400
        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            # Compute total weight of current expression set
            total_weight = sum(bin(e).count('1') for e in exprs)

            # Find all pairs of wires that co-occur
            co_occ = {}
            for e in exprs:
                if e == 0:
                    continue
                wires_in_e = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires_in_e.append(w)
                    temp >>= 1
                    w += 1

                for idx_a in range(len(wires_in_e)):
                    for idx_b in range(idx_a + 1, len(wires_in_e)):
                        a = wires_in_e[idx_a]
                        b = wires_in_e[idx_b]
                        key = (a, b)
                        co_occ[key] = co_occ.get(key, 0) + 1

            if not co_occ:
                break

            # Find the pair with highest benefit: co-occurrence count
            # but penalize if the new intermediate has high weight
            best_pair = None
            best_score = 1.0
            alpha = 0.05

            items = list(co_occ.items())
            rng.shuffle(items)
            for (a, b), cnt in items:
                if cnt <= 1:
                    continue
                a_bit = 1 << a
                b_bit = 1 << b
                weight = bin(a_bit ^ b_bit).count('1')
                weight_penalty = alpha * (weight / max(n, 1))
                score = cnt - weight_penalty
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

            # Check if we made progress
            new_total_weight = sum(bin(e).count('1') for e in exprs)
            if new_total_weight >= total_weight and len(gates) > 5:
                # No weight reduction, might be stuck; continue but with caution
                pass

        # Build outputs
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1

                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)

        return {'gates': gates, 'outputs': outputs}

    def _steiner_mst(time_budget, seed):
        """
        Steiner tree / MST-inspired approach.
        Greedily compute target vectors in order of cheapest cost from
        already-computed vectors, using existing intermediates when possible.
        """
        rng = random.Random(seed)

        wire_to_vec = {}
        vec_to_wire = {}

        for i in range(n):
            vec = 1 << i
            wire_to_vec[i] = vec
            vec_to_wire[vec] = i
        wire_to_vec[n] = 0
        vec_to_wire[0] = n

        next_wire = n + 1
        gates = []

        targets = list(set(row_bits))
        targets = [t for t in targets if t != 0]

        if not targets:
            return {'gates': [], 'outputs': [n] * m}

        pending = list(targets)
        pending_set = set(pending)

        max_iter = 600
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            if not pending:
                break

            best_target = None
            best_cost = float('inf')
            best_indices = None

            # Shuffle pending for randomization
            rng.shuffle(pending)

            for t in pending:
                if t in vec_to_wire:
                    best_target = t
                    best_cost = 0
                    best_indices = None
                    break

                # Try to compute t from two existing vectors
                # t = u XOR w => u = t XOR w
                found = False
                for w_vec, w_wire in wire_to_vec.items():
                    u_vec = t ^ w_vec
                    if u_vec in vec_to_wire:
                        u_wire = vec_to_wire[u_vec]
                        # Cost: 1 gate
                        cost = 1
                        if cost < best_cost:
                            best_cost = cost
                            best_target = t
                            best_indices = (u_wire, w_wire)
                            found = True
                            break

                if found:
                    continue

                # Fallback: direct computation from inputs
                pc = bin(t).count('1')
                cost_direct = max(0, pc - 1)
                if cost_direct < best_cost:
                    best_cost = cost_direct
                    best_target = t
                    best_indices = 'direct'

            if best_target is None:
                break

            t = best_target

            if t in vec_to_wire:
                pending.remove(t)
                pending_set.discard(t)
                continue

            if best_indices == 'direct':
                indices = [j for j in range(n) if (t >> j) & 1]
                if len(indices) == 0:
                    wire_id = n
                elif len(indices) == 1:
                    wire_id = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    wire_id = prev
            else:
                u_wire, w_wire = best_indices
                gates.append([u_wire, w_wire])
                wire_id = next_wire
                next_wire += 1

            wire_to_vec[wire_id] = t
            vec_to_wire[t] = wire_id
            pending.remove(t)
            pending_set.discard(t)

        # Build outputs
        outputs = []
        for i in range(m):
            e = row_bits[i]
            if e == 0:
                outputs.append(n)
            elif e in vec_to_wire:
                outputs.append(vec_to_wire[e])
            else:
                # Fallback: direct
                indices = [j for j in range(n) if (e >> j) & 1]
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

    # First, get a baseline
    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    # Try multiple strategies with different seeds
    strategies = [
        ('greedy_cse_v3', _greedy_cse_v3),
        ('basis_greedy_v2', _basis_greedy_v2),
        ('steiner_mst', _steiner_mst),
    ]

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for strategy_name, strategy_func in strategies:
        for seed in seeds:
            if time.time() > deadline - 2.0:
                break

            try:
                result = strategy_func(deadline - 2.0, seed)
                if _verify(result):
                    d = _compute_depth(result['gates'], result['outputs'])
                    if d <= depth_limit:
                        gc = len(result['gates'])
                        if gc < best_gate_count:
                            best_gate_count = gc
                            best_result = result
            except Exception:
                pass

    return best_result
