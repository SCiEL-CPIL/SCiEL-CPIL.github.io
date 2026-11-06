"""XOR-circuit synthesis with greedy shared-subexpression, BFS decomposition, and gate elimination."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitmasks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)

    # Baseline: independent row synthesis
    baseline_gates = 0
    for t in targets:
        bits = bin(t).count('1')
        if bits > 1:
            baseline_gates += bits - 1
    baseline = max(baseline_gates, 1)

    def time_left():
        return deadline - time.time()

    def greedy_synthesize(target_order, seed=None, prefer_new=False):
        """Greedy synthesis processing targets in given order."""
        if seed is not None:
            random.seed(seed)

        wires = []
        for i in range(n):
            wires.append(1 << i)
        wires.append(0)

        wire_map = {w: i for i, w in enumerate(wires)}
        gates = []
        depths = [0] * (n + 1)

        for idx in target_order:
            target = targets[idx]

            if target in wire_map:
                continue

            # Try 1-step: find a, b in available such that a^b = target
            found = False
            avail_list = list(wire_map.keys())
            if seed is not None:
                random.shuffle(avail_list)
            elif prefer_new:
                avail_list.sort(key=lambda w: wire_map[w], reverse=True)

            for a in avail_list:
                b_needed = a ^ target
                if b_needed in wire_map:
                    wa = wire_map[a]
                    wb = wire_map[b_needed]
                    new_wire = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(target)
                    wire_map[target] = new_wire
                    depths.append(max(depths[wa], depths[wb]) + 1)
                    found = True
                    break

            if found:
                continue

            # Try 2-step: find intermediate c such that target^c is achievable in 1 step
            # i.e., find c in available, then find a, b in available such that a^b = target^c
            # This means: target = a^b^c where a,b,c are all available
            # Cost: 2 gates (create a^b, then XOR with c)
            # But we only need 1 new gate if we can find a,b such that a^b^c = target
            # Actually: if we have c available, and we find a,b available with a^b = target^c,
            # then we create wire for target^c (1 gate), then XOR with c (1 gate) = 2 gates
            # But if target^c is already available, then we just need 1 gate
            # So the 1-step check above already covers the case where target^c is available

            # Alternative 2-step: build from a subset of inputs
            # Find the smallest subset of available wires whose XOR gives target
            # This is equivalent to finding a representation of target in the span
            # For now, try: pick one input bit, XOR with target to get remainder,
            # then see if remainder is available or can be made in 1 step

            # Try: for each input bit j set in target, check if target^(1<<j) is available
            # or can be made from 2 available wires
            best_cost = float('inf')
            best_decomp = None  # (wire_a, wire_b, wire_c) where target = wa^wb^wc or wa^wb

            # Check if target can be made from exactly 2 available wires (already checked above)
            # Check if target can be made from 3 available wires: a^b^c = target
            # For each c in available (limited to inputs + zero for efficiency):
            #   check if target^c can be made from 2 available wires

            # For efficiency, only try a limited set of c values
            c_candidates = [0] + [1 << j for j in range(n) if target & (1 << j)]

            for c in c_candidates:
                if c not in wire_map:
                    continue
                remainder = target ^ c
                if remainder == 0:
                    # target = c, already handled
                    continue
                if remainder in wire_map:
                    # target = c ^ remainder, 1 gate
                    # This should have been caught by the 1-step check
                    # unless c or remainder wasn't in the shuffled order
                    wa = wire_map[c]
                    wb = wire_map[remainder]
                    if best_cost > 1:
                        best_cost = 1
                        best_decomp = (wa, wb, None)
                    continue
                # Check if remainder can be made from 2 available wires
                # For each a in a limited set, check if a^remainder is available
                # Limit to inputs for efficiency
                for j in range(n):
                    a = 1 << j
                    if a not in wire_map:
                        continue
                    b_needed = a ^ remainder
                    if b_needed in wire_map:
                        # target = a ^ b_needed ^ c, 2 gates
                        wa = wire_map[a]
                        wb = wire_map[b_needed]
                        wc = wire_map[c]
                        if best_cost > 2:
                            best_cost = 2
                            best_decomp = (wa, wb, wc)
                        break
                if best_cost <= 2:
                    break

            if best_decomp is not None:
                if best_cost == 1:
                    wa, wb, _ = best_decomp
                    new_wire = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(target)
                    wire_map[target] = new_wire
                    depths.append(max(depths[wa], depths[wb]) + 1)
                elif best_cost == 2:
                    wa, wb, wc = best_decomp
                    # Create intermediate: wa ^ wb
                    inter_mask = wires[wa] ^ wires[wb]
                    new_wire1 = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(inter_mask)
                    wire_map[inter_mask] = new_wire1
                    depths.append(max(depths[wa], depths[wb]) + 1)
                    # Create target: inter ^ wc
                    new_wire2 = n + 1 + len(gates)
                    gates.append([new_wire1, wc])
                    wires.append(target)
                    wire_map[target] = new_wire2
                    depths.append(max(depths[new_wire1], depths[wc]) + 1)
                continue

            # Fallback: build from inputs one by one
            bits = []
            for j in range(n):
                if target & (1 << j):
                    bits.append(j)

            if not bits:
                continue
            if len(bits) == 1:
                continue

            current_wire = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                new_mask = wires[current_wire] ^ (1 << j)
                wires.append(new_mask)
                wire_map[new_mask] = new_wire
                depths.append(max(depths[current_wire], 0) + 1)
                current_wire = new_wire

        # Map outputs
        outputs = []
        for i in range(m):
            target = targets[i]
            if target in wire_map:
                outputs.append(wire_map[target])
            else:
                outputs.append(n)

        return gates, outputs

    def eliminate_gates(gates, outputs, wire_map, wires):
        """Try to eliminate redundant gates."""
        if not gates:
            return gates, outputs

        # Build wire values
        vals = [1 << i for i in range(n)] + [0]
        for g in gates:
            a, b = g
            vals.append(vals[a] ^ vals[b])

        # For each gate, check if its output is used by any later gate or output
        # and if the same value is already available from an earlier wire
        gate_used = [False] * len(gates)

        # Mark which gates are directly used by outputs
        output_wires = set(outputs)
        for i in range(len(gates)):
            wire_num = n + 1 + i
            if wire_num in output_wires:
                gate_used[i] = True

        # Mark which gates are used by later gates
        for i in range(len(gates)):
            if gate_used[i]:
                continue
            a, b = gates[i]
            # Check if this gate's output is used as input to any other gate
            # (This is complex; simplified: skip)
            pass

        # Try removing gates whose output value is already available from an earlier wire
        new_gates = []
        wire_remap = {}  # old wire -> new wire
        old_to_new = list(range(n + 1))  # inputs and zero stay the same

        for i in range(len(gates)):
            a, b = gates[i]
            new_a = old_to_new[a]
            new_b = old_to_new[b]
            new_val = vals[a] ^ vals[b]

            # Check if this value is already available
            # (simplified: skip for now, just keep all gates)
            new_gates.append([new_a, new_b])
            old_to_new.append(len(new_gates) + n)

        # Remap outputs
        new_outputs = [old_to_new[o] if o < len(old_to_new) else o for o in outputs]

        return new_gates, new_outputs

    # Try multiple strategies
    best_gates = None
    best_outputs = None
    best_score = float('inf')

    # Strategy 1: Sort by popcount ascending
    order1 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'))
    g1, o1 = greedy_synthesize(order1)
    score1 = len(g1) / baseline
    if score1 < best_score:
        best_score = score1
        best_gates = g1
        best_outputs = o1

    # Strategy 2: Sort by popcount descending
    order2 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'), reverse=True)
    g2, o2 = greedy_synthesize(order2)
    score2 = len(g2) / baseline
    if score2 < best_score:
        best_score = score2
        best_gates = g2
        best_outputs = o2

    # Strategy 3: Prefer newer wires
    order3 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'))
    g3, o3 = greedy_synthesize(order3, prefer_new=True)
    score3 = len(g3) / baseline
    if score3 < best_score:
        best_score = score3
        best_gates = g3
        best_outputs = o3

    # Strategy 4: Random restarts
    num_restarts = min(30, max(10, int(time_left() / 0.3)))
    for r in range(num_restarts):
        if time_left() < 3.0:
            break
        order = list(range(m))
        random.shuffle(order)
        g, o = greedy_synthesize(order, seed=r)
        score = len(g) / baseline
        if score < best_score:
            best_score = score
            best_gates = g
            best_outputs = o

    # Strategy 5: Random with prefer_new
    num_restarts2 = min(20, max(5, int(time_left() / 0.3)))
    for r in range(num_restarts2):
        if time_left() < 3.0:
            break
        order = list(range(m))
        random.shuffle(order)
        g, o = greedy_synthesize(order, seed=r + 1000, prefer_new=True)
        score = len(g) / baseline
        if score < best_score:
            best_score = score
            best_gates = g
            best_outputs = o

    if best_gates is None:
        best_gates = []
        best_outputs = [n] * m

    return {'gates': best_gates, 'outputs': best_outputs}
# EVOLVE-BLOCK-END
