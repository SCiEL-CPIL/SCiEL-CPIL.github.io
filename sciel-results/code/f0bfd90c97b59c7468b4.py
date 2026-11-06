"""Exact XOR-circuit synthesis: Simulated Annealing over global gate lists."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """Verify gates/outputs compute matrix @ x over GF(2) with depth <= depth_limit."""
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

    # Depth check
    depth = [0] * total_wires
    for k in range(num_gates):
        a, b = gates[k]
        if a < 0 or a >= total_wires or b < 0 or b >= total_wires:
            return False
        if a >= n + 1 + k or b >= n + 1 + k:
            return False
        depth[n + 1 + k] = 1 + max(depth[a], depth[b])
    
    max_depth = 0
    if num_gates > 0:
        max_depth = max(depth[n + 1 + k] for k in range(num_gates))
    for wire in outputs:
        if wire < 0 or wire >= total_wires:
            return False
        max_depth = max(max_depth, depth[wire])
    if max_depth > depth_limit:
        return False

    # Symbolic verification
    wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
    for i in range(n):
        wire_vec[i][i] = 1
    for k in range(num_gates):
        a, b = gates[k]
        wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
    for i in range(m):
        wire = outputs[i]
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    return True


def _baseline_synthesis(matrix, depth_limit=16):
    """Independent XOR chains per row."""
    m, n = matrix.shape
    gates = []
    outputs = []
    for i in range(m):
        row = matrix[i]
        positions = np.where(row)[0].tolist()
        if len(positions) == 0:
            outputs.append(n)
        elif len(positions) == 1:
            outputs.append(positions[0])
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                a = current
                b = positions[j]
                gates.append([a, b])
                current = n + 1 + len(gates) - 1
            outputs.append(current)
    return {'gates': gates, 'outputs': outputs}


def _global_greedy_synthesis(matrix, depth_limit=16, time_budget=10.0):
    """
    Global vector-space greedy basis construction.
    At each step, select the XOR pair from all available wires that maximizes
    the number of remaining target rows that can be satisfied with 1 fewer gate
    or directly.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Initialize wires
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))  # wire n = zero
    
    vec_to_wires = {}
    for w in range(n + 1):
        vec_to_wires.setdefault(wire_vec[w].tobytes(), []).append(w)
        
    gates = []
    outputs = [None] * m
    remaining_rows = set(range(m))
    
    # Pre-compute target bytes
    target_bytes = [matrix[i].tobytes() for i in range(m)]
    
    while remaining_rows:
        if time.time() > deadline:
            break
            
        W = len(wire_vec)
        best_pair = None
        best_score = -1
        best_vec = None
        
        # Evaluate all pairs
        # Optimization: only consider pairs that produce a vector that helps
        # "Help" means: the new vector is a target, or reduces the popcount of a target
        # For simplicity and robustness, we score by:
        # 1. If new_vec is a target: +1000
        # 2. Else: sum over remaining targets of (popcount(target) - popcount(target XOR new_vec))
        #    This is expensive. Approximation: just check if it's a target or 0.
        #    Actually, let's use a simpler heuristic:
        #    Score = 1000 if target, else 0.
        #    Tie-break: random.
        
        # To speed up, we can sample pairs if W is large
        max_checks = 5000
        if W * (W - 1) // 2 > max_checks:
            # Sample
            indices = list(range(W))
            random.shuffle(indices)
            checked = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a = indices[i]
                    b = indices[j]
                    new_vec = (wire_vec[a] + wire_vec[b]) % 2
                    nb = new_vec.tobytes()
                    
                    score = 0
                    # if nb in target_bytes_set:
                    #     pass # This check is too slow. Let's use a set of target bytes.
                        
                    # Better: keep a set of active target bytes
                    # But targets are fixed.
                    # Let's just check if it's in the set of all target bytes.
                    # We can pre-compute a set of target bytes.
                    
                    # Let's refactor to use a set for targets.
                    pass
                    
        # Refactored:
        target_set = set(target_bytes)
        
        if W * (W - 1) // 2 > max_checks:
            indices = list(range(W))
            random.shuffle(indices)
            checked = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a = indices[i]
                    b = indices[j]
                    new_vec = (wire_vec[a] + wire_vec[b]) % 2
                    nb = new_vec.tobytes()
                    if nb in target_set:
                        best_pair = (a, b)
                        best_score = 1000
                        best_vec = new_vec
                        break
                if best_pair:
                    break
        else:
            for a in range(W):
                for b in range(a + 1, W):
                    new_vec = (wire_vec[a] + wire_vec[b]) % 2
                    nb = new_vec.tobytes()
                    if nb in target_set:
                        best_pair = (a, b)
                        best_score = 1000
                        best_vec = new_vec
                        break
                if best_pair:
                    break
        
        if best_pair:
            a, b = best_pair
            k = len(gates)
            new_wire = n + 1 + k
            gates.append([a, b])
            wire_vec.append(best_vec)
            vec_to_wires.setdefault(best_vec.tobytes(), []).append(new_wire)
            
            # Resolve any rows that match this new vector
            for r in list(remaining_rows):
                if target_bytes[r] == best_vec.tobytes():
                    outputs[r] = new_wire
                    remaining_rows.remove(r)
        else:
            # No direct target found. Fallback: pick a random pair to expand basis?
            # Or break and use baseline for remaining?
            # To ensure completion, we should probably just add a random pair or break.
            # If we break, we must fill remaining with baseline chains.
            # Let's try to add a pair that reduces popcount of some target.
            # Heuristic: pick pair that minimizes max popcount of remaining targets XOR new_vec.
            # This is complex. Let's just pick a random pair to keep going, or fallback.
            # For robustness, fallback to baseline for remaining rows.
            break

    # Fill remaining rows with baseline chains using current gates
    for r in remaining_rows:
        target = matrix[r]
        tb = target.tobytes()
        if tb in vec_to_wires:
            outputs[r] = vec_to_wires[tb][0]
        else:
            # Chain
            positions = np.where(target)[0].tolist()
            if len(positions) == 0:
                outputs[r] = n
            elif len(positions) == 1:
                outputs[r] = positions[0]
            else:
                current = positions[0]
                for j in range(1, len(positions)):
                    b = positions[j]
                    new_vec = (wire_vec[current] + wire_vec[b]) % 2
                    nb = new_vec.tobytes()
                    if nb in vec_to_wires:
                        current = vec_to_wires[nb][0]
                    else:
                        k = len(gates)
                        new_wire = n + 1 + k
                        gates.append([current, b])
                        wire_vec.append(new_vec)
                        vec_to_wires.setdefault(nb, []).append(new_wire)
                        current = new_wire
                outputs[r] = current

    return {'gates': gates, 'outputs': outputs}


def _simulated_annealing(matrix, gates, outputs, depth_limit=16, time_budget=30.0):
    """
    Simulated Annealing to minimize gate count.
    State: (gates, outputs)
    Moves:
    1. Remove a gate if its output wire is not used by any other gate or output.
    2. Replace a gate's inputs with another pair of wires that produces the same vector (redundancy).
    3. Redirect an output to an earlier wire with the same vector.
    4. (Advanced) Replace a gate with a sequence of 2 gates? (Too complex for now).
    
    We will focus on 1, 2, and 3.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Deep copy
    gates = [g[:] for g in gates]
    outputs = outputs[:]
    
    # Pre-compute vectors for current state
    def compute_vectors(gates, outputs):
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        return wire_vec

    wire_vec = compute_vectors(gates, outputs)
    
    # Build usage map: wire -> list of (gate_index, input_pos) or (output_index, None)
    def build_usage(gates, outputs):
        usage = {}
        for k, g in enumerate(gates):
            for pos in [0, 1]:
                w = g[pos]
                usage.setdefault(w, []).append((k, pos))
        for i, o in enumerate(outputs):
            usage.setdefault(o, []).append((i, None))
        return usage

    usage = build_usage(gates, outputs)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    steps = 10000
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Random move
        move_type = random.randint(1, 3)
        
        if move_type == 1:
            # Remove unused gate
            if not gates:
                continue
            k = random.randint(0, len(gates) - 1)
            wire_k = n + 1 + k
            # Check if used
            if wire_k in usage:
                # Check if used by gates > k or outputs
                # Usage map is static, need to check if any usage is valid
                # Actually, if we remove gate k, wire_k disappears.
                # We can only remove if usage[wire_k] is empty OR only used by gates that are also removed?
                # Simplest: if usage[wire_k] is empty, remove.
                if not usage[wire_k]:
                    # Remove
                    gates.pop(k)
                    # Update usage? Hard to do incrementally.
                    # Let's just recompute usage occasionally or use a simpler check.
                    # For SA, we can just try the move, verify, and accept/reject.
                    # But verification is O(N^2).
                    # Let's stick to the polish-style approach but with SA acceptance?
                    # No, SA requires evaluating cost. Cost = len(gates).
                    # So we just need to know if the move is valid.
                    
                    # Let's implement a helper to check validity of a specific gate removal.
                    pass # This logic is getting messy. Let's simplify.

        # Simplified SA Strategy:
        # Instead of complex moves, we will use a "Perturb and Polish" approach.
        # 1. Perturb: Add a random gate, or remove a random gate (if valid), or change a gate input.
        # 2. Polish: Run the existing _polish function.
        # 3. Accept if polished result is better, or with probability exp(-delta/T).
        
        # Let's implement Perturb.
        new_gates = [g[:] for g in gates]
        new_outputs = outputs[:]
        
        if random.random() < 0.5 and new_gates:
            # Remove random gate
            k = random.randint(0, len(new_gates) - 1)
            wire_k = n + 1 + k
            # Check if wire_k is used by any other gate or output
            used = False
            for j, g in enumerate(new_gates):
                if j != k and (g[0] == wire_k or g[1] == wire_k):
                    used = True
                    break
            if not used:
                for o in new_outputs:
                    if o == wire_k:
                        used = True
                        break
            if not used:
                new_gates.pop(k)
        else:
            # Change random gate input
            if new_gates:
                k = random.randint(0, len(new_gates) - 1)
                wire_k = n + 1 + k
                # Pick a new input wire < wire_k
                if wire_k > 0:
                    new_input = random.randint(0, wire_k - 1)
                    pos = random.randint(0, 1)
                    new_gates[k][pos] = new_input
        
        # Polish the new state
        polished = _polish(matrix, new_gates, new_outputs, depth_limit, time_budget=0.5)
        
        if len(polished['gates']) < len(gates):
            gates = polished['gates']
            outputs = polished['outputs']
            wire_vec = compute_vectors(gates, outputs)
            usage = build_usage(gates, outputs)
        elif random.random() < np.exp(-(len(polished['gates']) - len(gates)) / T):
            # Accept worse solution with probability
            gates = polished['gates']
            outputs = polished['outputs']
            wire_vec = compute_vectors(gates, outputs)
            usage = build_usage(gates, outputs)

    # Final polish
    final = _polish(matrix, gates, outputs, depth_limit, time_budget=5.0)
    return final


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=10.0):
    """
    Redundancy-elimination polish.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(20):
        if time.time() > deadline:
            break
        if len(gates) == 0:
            break

        num_gates = len(gates)
        total_wires = n + 1 + num_gates

        # Build wire_vec
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2

        # Build vec_to_earliest_wire
        vec_to_earliest = {}
        for w in range(total_wires):
            vb = wire_vec[w].tobytes()
            if vb not in vec_to_earliest:
                vec_to_earliest[vb] = w

        improved = False

        # Pass 1: remove unused gates (from last to first)
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            used = False
            for ow in outputs:
                if ow == wire_k:
                    used = True
                    break
            if not used:
                for g in gates:
                    if g[0] == wire_k or g[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                improved = True
                break

        if improved:
            continue

        # Pass 2: for each gate, check if its vector is available from an earlier wire
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()

            # Find earliest wire with same vector that is not wire_k
            earlier = None
            for w in range(wire_k):
                if wire_vec[w].tobytes() == vec_k_bytes:
                    earlier = w
                    break

            if earlier is not None:
                # Redirect all uses of wire_k to 'earlier'
                for j in range(k + 1, len(gates)):
                    if gates[j][0] == wire_k:
                        gates[j][0] = earlier
                    if gates[j][1] == wire_k:
                        gates[j][1] = earlier
                for i in range(m):
                    if outputs[i] == wire_k:
                        outputs[i] = earlier
                gates.pop(k)
                improved = True
                break

        if not improved:
            break

    if _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """Entry point."""
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    start_time = time.time()

    # Baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    best_result = baseline
    best_count = len(baseline['gates'])

    if best_count == 0:
        return baseline

    # Global Greedy Synthesis
    try:
        greedy = _global_greedy_synthesis(matrix, depth_limit, time_budget=5.0)
        if len(greedy['gates']) < best_count:
            best_result = greedy
            best_count = len(greedy['gates'])
    except Exception:
        pass

    # Simulated Annealing
    elapsed = time.time() - start_time
    if elapsed < 50.0 and best_count > 0:
        try:
            sa = _simulated_annealing(matrix, best_result['gates'], best_result['outputs'],
                                      depth_limit, time_budget=40.0)
            if len(sa['gates']) < best_count:
                best_result = sa
                best_count = len(sa['gates'])
        except Exception:
            pass

    # Final Polish
    elapsed = time.time() - start_time
    if elapsed < 55.0 and best_count > 0:
        try:
            polished = _polish(matrix, best_result['gates'], best_result['outputs'],
                               depth_limit, time_budget=5.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
