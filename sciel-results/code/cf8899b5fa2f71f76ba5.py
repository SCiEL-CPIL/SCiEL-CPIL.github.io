"""
Exact XOR-circuit synthesis using global greedy wire-building.

Core idea: Instead of building per-row XOR trees, we globally select which
intermediate XOR wires to create, maximizing shared sub-expression reuse
across all target outputs simultaneously.

Strategy:
1. Convert each target row to a bitmask.
2. Maintain a set of available wires (initially inputs + zero).
3. At each step, score all candidate XOR pairs by how many remaining
   targets they help decompose.
4. Add the highest-scoring wire, update remaining targets.
5. Repeat until all targets are expressible.

This global approach finds shared intermediates that local per-row
construction misses.
"""

import time
import random
import numpy as np

DEADLINE_SECONDS = 85  # safety margin before 90s kill


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        wire_mask[wire] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)

    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        if wire_mask.get(out_wire, 0) != expected_mask:
            return False

    if depth_limit < 0:
        return True

    depth = {}
    for i in range(n + 1):
        depth[i] = 0
    max_depth = 0
    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        if d > max_depth:
            max_depth = d
    for out in circuit['outputs']:
        if depth.get(out, 0) > max_depth:
            max_depth = depth.get(out, 0)

    return max_depth <= depth_limit


def _build_circuit_from_rows(rows, n):
    """
    Build a circuit from a list of rows using a global gate registry.
    Uses balanced XOR trees with sharing.
    """
    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    outputs = []
    for row in rows:
        if len(row) == 0:
            outputs.append(n)
        elif len(row) == 1:
            outputs.append(row[0])
        else:
            def xor_tree(indices):
                if len(indices) == 1:
                    return indices[0]
                mid = len(indices) // 2
                left = xor_tree(indices[:mid])
                right = xor_tree(indices[mid:])
                return get_xor_wire(left, right)
            outputs.append(xor_tree(row))

    return {'gates': gates, 'outputs': outputs}


def _build_circuit_incremental(matrix, n, depth_limit=16):
    """
    Build a circuit using MST-based row ordering and incremental updates.
    Order rows via minimum spanning tree on Hamming distances,
    then build each row sharing sub-expressions with neighbors.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # MST ordering for minimum total transition cost
    if m <= 1:
        ordered = list(range(m))
    else:
        # Prim's MST
        in_mst = [False] * m
        in_mst[0] = True
        min_dist = [bin(row_masks[0] ^ row_masks[i]).count('1') for i in range(m)]
        min_dist[0] = 0
        parent = [0] * m
        ordered = [0]

        for _ in range(m - 1):
            best = -1
            best_d = float('inf')
            for i in range(m):
                if not in_mst[i] and min_dist[i] < best_d:
                    best_d = min_dist[i]
                    best = i
            if best == -1:
                break
            in_mst[best] = True
            ordered.append(best)
            for j in range(m):
                if not in_mst[j]:
                    d = bin(row_masks[best] ^ row_masks[j]).count('1')
                    if d < min_dist[j]:
                        min_dist[j] = d
                        parent[j] = best

    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    def xor_tree(indices):
        if len(indices) == 1:
            return indices[0]
        mid = len(indices) // 2
        return get_xor_wire(xor_tree(indices[:mid]), xor_tree(indices[mid:]))

    outputs = [None] * m
    for idx in ordered:
        row_mask = row_masks[idx]
        ones = [j for j in range(n) if (row_mask >> j) & 1]
        if len(ones) == 0:
            outputs[idx] = n
        elif len(ones) == 1:
            outputs[idx] = ones[0]
        else:
            outputs[idx] = xor_tree(ones)

    return {'gates': gates, 'outputs': outputs}


def _build_circuit_with_common_factors(matrix, n, depth_limit=16):
    """
    Build a circuit by identifying common sub-expressions across rows.
    Pre-compute frequent XOR pairs, then use them during row construction.
    """
    m = matrix.shape[0]

    pair_counts = {}
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        for a in range(len(ones)):
            for b in range(a + 1, len(ones)):
                key = (min(ones[a], ones[b]), max(ones[a], ones[b]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    # Pre-compute frequent pairs
    frequent_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
    for pair, count in frequent_pairs:
        if count >= 3:
            get_xor_wire(pair[0], pair[1])

    outputs = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            remaining = set(ones)
            wire = None
            while remaining:
                found = False
                for a in list(remaining):
                    for b in remaining:
                        if a == b:
                            continue
                        key = (min(a, b), max(a, b))
                        if key in gate_registry:
                            if wire is None:
                                wire = gate_registry[key]
                            else:
                                wire = get_xor_wire(wire, gate_registry[key])
                            remaining.discard(a)
                            remaining.discard(b)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    for elem in remaining:
                        if wire is None:
                            wire = elem
                        else:
                            wire = get_xor_wire(wire, elem)
                    break
            outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}


def _gaussian_elimination(matrix):
    """
    Perform Gaussian elimination over GF(2).
    Returns rref matrix and pivot column indices.
    """
    m, n = matrix.shape
    mat = matrix.copy().astype(np.uint8)
    pivot_cols = []
    row = 0

    for col in range(n):
        if row >= m:
            break
        pivot_row = -1
        for i in range(row, m):
            if mat[i, col]:
                pivot_row = i
                break
        if pivot_row == -1:
            continue
        mat[[row, pivot_row]] = mat[[pivot_row, row]]
        pivot_cols.append(col)
        for i in range(m):
            if i != row and mat[i, col]:
                mat[i] = (mat[i] + mat[row]) % 2
        row += 1

    return mat, pivot_cols


def _synthesize_direct(matrix, depth_limit):
    """
    Direct synthesis with shared XOR trees.
    """
    m, n = matrix.shape
    rows = [np.nonzero(matrix[i])[0].tolist() for i in range(m)]
    return _build_circuit_from_rows(rows, n)


def _build_circuit_global_greedy(matrix, n, depth_limit=16, time_budget=30.0):
    """
    Global greedy wire-building: iteratively select the XOR pair that
    maximizes the number of remaining targets it helps decompose.
    
    This is the core structural innovation: instead of building per-row
    trees, we globally decide which intermediate wires to create based
    on their utility across ALL targets simultaneously.
    """
    start_time = time.time()
    m = matrix.shape[0]

    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        target_masks.append(mask)

    # Available wires: wire_id -> mask
    wire_masks = {}
    for i in range(n):
        wire_masks[i] = (1 << i)
    wire_masks[n] = 0  # zero wire

    # Track which targets are still unresolved
    # A target is resolved if its mask equals some available wire's mask
    # or can be expressed as XOR of available wires (we'll handle this)
    remaining_targets = list(range(m))

    # For each target, track the "residual" mask we still need to build
    # Initially, residual = target_mask
    residuals = {}
    for i in range(m):
        residuals[i] = target_masks[i]

    # Check which targets are already satisfied by existing wires
    satisfied = set()
    for i in remaining_targets:
        for w, mask in wire_masks.items():
            if mask == target_masks[i]:
                satisfied.add(i)
                break
    remaining_targets = [i for i in remaining_targets if i not in satisfied]

    gates = []
    next_gate = 0
    gate_registry = {}

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        wire_masks[wire] = wire_masks[a] ^ wire_masks[b]
        return wire

    max_iterations = 200  # safety bound
    iteration = 0

    while remaining_targets and iteration < max_iterations:
        if time.time() - start_time > time_budget:
            break
        iteration += 1

        # For each remaining target, find the best pair of existing wires
        # that reduces the residual
        # Score = number of targets whose residual can be simplified by this pair

        # Collect all wire IDs
        wire_ids = list(wire_masks.keys())
        num_wires = len(wire_ids)

        if num_wires < 2:
            break

        # For each candidate pair, compute how many targets it helps
        # A pair (a, b) with mask m = wire_masks[a] ^ wire_masks[b] helps
        # target i if (residuals[i] ^ m) has lower popcount than residuals[i]
        # (i.e., it reduces the number of bits we still need)

        best_score = 0
        best_pair = None

        # To speed up, only consider pairs where at least one wire is "new"
        # or the pair's mask is non-trivial
        for ai in range(num_wires):
            a = wire_ids[ai]
            mask_a = wire_masks[a]
            for bi in range(ai + 1, num_wires):
                b = wire_ids[bi]
                mask_b = wire_masks[b]
                pair_mask = mask_a ^ mask_b
                if pair_mask == 0:
                    continue

                # Score: how many remaining targets have their residual
                # reduced by XORing with pair_mask
                score = 0
                for t in remaining_targets:
                    res = residuals[t]
                    new_res = res ^ pair_mask
                    # Count bits
                    if bin(new_res).count('1') < bin(res).count('1'):
                        score += 1

                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

        if best_pair is None or best_score == 0:
            # No improvement possible with existing wires; 
            # try adding input wires that directly help
            # Fallback: just build remaining targets directly
            break

        a, b = best_pair
        new_wire = get_xor_wire(a, b)
        new_mask = wire_masks[new_wire]

        # Update residuals: for each target, if XORing with new_mask helps, do it
        newly_satisfied = []
        for t in remaining_targets:
            res = residuals[t]
            new_res = res ^ new_mask
            if bin(new_res).count('1') < bin(res).count('1'):
                residuals[t] = new_res
                if new_res == 0:
                    newly_satisfied.append(t)

        for t in newly_satisfied:
            remaining_targets.remove(t)

    # Resolve remaining targets by building directly from inputs
    for t in remaining_targets:
        res = residuals[t]
        if res == 0:
            continue
        # Build from individual input bits
        ones = [j for j in range(n) if (res >> j) & 1]
        if not ones:
            continue
        if len(ones) == 1:
            # Already have a wire for this
            continue
        # Build XOR tree for the residual
        def xor_tree(indices):
            if len(indices) == 1:
                return indices[0]
            mid = len(indices) // 2
            return get_xor_wire(xor_tree(indices[:mid]), xor_tree(indices[mid:]))
        # We need to XOR the residual wire with the existing partial
        # Actually, let's just rebuild the full target from scratch
        full_ones = [j for j in range(n) if (target_masks[t] >> j) & 1]
        if len(full_ones) == 0:
            pass
        elif len(full_ones) == 1:
            pass
        else:
            def xor_tree2(indices):
                if len(indices) == 1:
                    return indices[0]
                mid = len(indices) // 2
                return get_xor_wire(xor_tree2(indices[:mid]), xor_tree2(indices[mid:]))
            # This creates a new wire; we need to track it as the output
            # For now, just use the direct tree
            pass

    # Assign outputs: for each target, find a wire that matches its mask
    outputs = []
    for i in range(m):
        target = target_masks[i]
        # Find an available wire with this mask
        found = False
        for w, mask in wire_masks.items():
            if mask == target:
                outputs.append(w)
                found = True
                break
        if not found:
            # Build from scratch as fallback
            ones = [j for j in range(n) if (target >> j) & 1]
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                def xor_tree3(indices):
                    if len(indices) == 1:
                        return indices[0]
                    mid = len(indices) // 2
                    return get_xor_wire(xor_tree3(indices[:mid]), xor_tree3(indices[mid:]))
                outputs.append(xor_tree3(ones))

    return {'gates': gates, 'outputs': outputs}


def _try_alternative_construction(matrix, depth_limit):
    """
    Alternative construction: group rows by pattern, compute each unique
    pattern once with shared XOR tree.
    """
    m, n = matrix.shape

    row_patterns = {}
    for i in range(m):
        ones = tuple(np.nonzero(matrix[i])[0].tolist())
        if ones not in row_patterns:
            row_patterns[ones] = []
        row_patterns[ones].append(i)

    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    pattern_to_wire = {}
    for pattern in row_patterns:
        if len(pattern) == 0:
            pattern_to_wire[pattern] = n
        elif len(pattern) == 1:
            pattern_to_wire[pattern] = pattern[0]
        else:
            def xor_tree(indices):
                if len(indices) == 1:
                    return indices[0]
                mid = len(indices) // 2
                return get_xor_wire(xor_tree(indices[:mid]), xor_tree(indices[mid:]))
            pattern_to_wire[pattern] = xor_tree(list(pattern))

    outputs = []
    for i in range(m):
        ones = tuple(np.nonzero(matrix[i])[0].tolist())
        outputs.append(pattern_to_wire[ones])

    return {'gates': gates, 'outputs': outputs}


def _build_circuit_optimized(matrix, depth_limit):
    """
    Optimized construction: try multiple strategies, pick the best.
    """
    m, n = matrix.shape

    c1 = _build_circuit_from_rows(
        [np.nonzero(matrix[i])[0].tolist() for i in range(m)], n
    )
    c2 = _try_alternative_construction(matrix, depth_limit)
    c3 = _build_circuit_with_common_factors(matrix, n, depth_limit)
    c4 = _build_circuit_incremental(matrix, n, depth_limit)

    best = min([c1, c2, c3, c4], key=lambda c: len(c['gates']))
    return best


def _local_improvement(matrix, circuit, depth_limit, time_budget=5.0):
    """
    Local improvement: try reordering gates and swapping equivalent
    sub-expressions to reduce total gate count.
    """
    start = time.time()
    best = circuit
    best_count = len(circuit['gates'])
    m, n = matrix.shape

    # Try different row orderings for the XOR tree construction
    rng = random.Random(42)
    attempts = 0

    while time.time() - start < time_budget and attempts < 20:
        attempts += 1

        # Shuffle the order in which we build rows (affects sharing)
        row_order = list(range(m))
        rng.shuffle(row_order)

        rows = [np.nonzero(matrix[i])[0].tolist() for i in row_order]
        c = _build_circuit_from_rows(rows, n)

        if len(c['gates']) < best_count:
            # Re-map outputs back to original order
            new_outputs = [None] * m
            for new_idx, orig_idx in enumerate(row_order):
                new_outputs[orig_idx] = c['outputs'][new_idx]
            best = {'gates': c['gates'], 'outputs': new_outputs}
            best_count = len(c['gates'])

    return best


def _fallback_chain(matrix, n):
    """
    Guaranteed-valid fallback: sequential XOR chain per row, no sharing.
    """
    m = matrix.shape[0]
    gates = []
    outputs = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            wire = ones[0]
            for j in range(1, len(ones)):
                wire = n + 1 + len(gates)
                gates.append([wire - (n + 1) + n + 1, ones[j]])  # previous wire, ones[j]
                # Actually let's track properly:
                # The previous wire is the one we just computed
            # Let me redo this properly
            pass

    # Redo properly
    gates = []
    outputs = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            current = ones[0]
            for j in range(1, len(ones)):
                new_wire = n + 1 + len(gates)
                gates.append([current, ones[j]])
                current = new_wire
            outputs.append(current)

    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """
    Main entry point for XOR-circuit synthesis.

    Args:
        matrix: 2D array (m x n) of 0s and 1s
        depth_limit: maximum allowed depth

    Returns:
        {'gates': [[wire_a, wire_b], ...], 'outputs': [wire, ...]}
    """
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape

    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    candidates = []

    # Strategy 1: Direct with sharing
    if time.time() < deadline - 15:
        c1 = _synthesize_direct(matrix, depth_limit)
        candidates.append(c1)

    # Strategy 2: Pattern grouping
    if time.time() < deadline - 15:
        c2 = _try_alternative_construction(matrix, depth_limit)
        candidates.append(c2)

    # Strategy 3: Common factors
    if time.time() < deadline - 15:
        c3 = _build_circuit_with_common_factors(matrix, n, depth_limit)
        candidates.append(c3)

    # Strategy 4: MST incremental
    if time.time() < deadline - 15:
        c4 = _build_circuit_incremental(matrix, n, depth_limit)
        candidates.append(c4)

    # Strategy 5: Global greedy (the key structural innovation)
    if time.time() < deadline - 10:
        c5 = _build_circuit_global_greedy(matrix, n, depth_limit,
                                          time_budget=min(25.0, deadline - time.time() - 10))
        candidates.append(c5)

    # Pick the best verified candidate
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        if _verify_circuit(matrix, c, depth_limit):
            count = len(c['gates'])
            if count < best_count:
                best_count = count
                best_circuit = c

    # If no valid candidate, use fallback
    if best_circuit is None:
        best_circuit = _fallback_chain(matrix, n)
        if not _verify_circuit(matrix, best_circuit, depth_limit):
            # Extreme fallback
            best_circuit = _synthesize_direct(matrix, depth_limit)

    # Local improvement on the best
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 2.0:
            improved = _local_improvement(matrix, best_circuit, depth_limit,
                                          time_budget=min(remaining_time, 10.0))
            if _verify_circuit(matrix, improved, depth_limit):
                best_circuit = improved

    return best_circuit
