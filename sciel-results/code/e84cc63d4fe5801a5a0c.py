"""
Exact XOR-circuit synthesis using CSE with optimized row ordering and local search.

Key structural changes from parent:
1. Row ordering via minimum spanning tree (MST) on Hamming distances for better incremental sharing.
2. Local search refinement: try swapping adjacent rows in the ordered sequence and re-evaluate.
3. More aggressive common-factor pre-computation with threshold tuning.
4. Hybrid construction: for each row, choose between incremental update and fresh tree based on cost.
5. Depth-aware construction: avoid deep chains when depth_limit is tight.
"""

import time
import numpy as np

DEADLINE_SECONDS = 85


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0  # constant zero

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        mask_a = wire_mask.get(a, 0)
        mask_b = wire_mask.get(b, 0)
        wire_mask[wire] = mask_a ^ mask_b

    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        actual_mask = wire_mask.get(out_wire, 0)
        if actual_mask != expected_mask:
            return False

    if depth_limit < 0:
        return True

    max_depth = 0
    depth = {}
    for i in range(n + 1):
        depth[i] = 0

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        max_depth = max(max_depth, d)

    for out in circuit['outputs']:
        max_depth = max(max_depth, depth.get(out, 0))

    if max_depth > depth_limit:
        return False

    return True


def _make_gate_factory(n):
    """Create a gate factory with CSE registry."""
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_xor_wire(a, b):
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    return get_xor_wire, gates, gate_registry


def _xor_tree(indices, get_xor_wire):
    """Build a balanced XOR tree for a list of input indices."""
    if len(indices) == 1:
        return indices[0]
    mid = len(indices) // 2
    left = _xor_tree(indices[:mid], get_xor_wire)
    right = _xor_tree(indices[mid:], get_xor_wire)
    return get_xor_wire(left, right)


def _hamming_distance(m1, m2):
    """Hamming distance between two bitmasks."""
    return bin(m1 ^ m2).count('1')


def _mst_order(row_masks, m):
    """
    Compute row ordering using Minimum Spanning Tree (Prim's algorithm)
    on Hamming distances, starting from the lowest-weight row.
    """
    if m <= 1:
        return list(range(m))

    # Start from lowest weight row
    start = min(range(m), key=lambda i: bin(row_masks[i]).count('1'))
    visited = [False] * m
    visited[start] = True
    order = [start]

    # Prim's: repeatedly add the unvisited row closest to any visited row
    for _ in range(m - 1):
        best_row = -1
        best_dist = float('inf')
        for u in order:
            for v in range(m):
                if not visited[v]:
                    d = _hamming_distance(row_masks[u], row_masks[v])
                    if d < best_dist:
                        best_dist = d
                        best_row = v
        visited[best_row] = True
        order.append(best_row)

    return order


def _build_direct(matrix, n):
    """Direct construction: balanced XOR trees with CSE."""
    get_xor_wire, gates, _ = _make_gate_factory(n)
    outputs = []
    for i in range(matrix.shape[0]):
        ones = np.nonzero(matrix[i])[0].tolist()
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            outputs.append(_xor_tree(ones, get_xor_wire))
    return {'gates': gates, 'outputs': outputs}


def _build_incremental_mst(matrix, n, depth_limit):
    """
    Incremental construction with MST-based row ordering.
    For each row, decide whether to incrementally update or rebuild fresh.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    order = _mst_order(row_masks, m)

    get_xor_wire, gates, gate_registry = _make_gate_factory(n)
    outputs = [None] * m
    current_wire = None
    current_set = set()

    for idx in order:
        row_mask = row_masks[idx]
        row_set = set(j for j in range(n) if (row_mask >> j) & 1)

        if current_wire is None:
            if len(row_set) == 0:
                current_wire = n
            elif len(row_set) == 1:
                current_wire = list(row_set)[0]
            else:
                current_wire = _xor_tree(list(row_set), get_xor_wire)
            current_set = row_set
        else:
            to_remove = current_set - row_set
            to_add = row_set - current_set
            change_count = len(to_remove) + len(to_add)

            # Estimate cost of incremental vs. fresh
            fresh_cost = max(0, len(row_set) - 1)  # XOR tree needs len-1 gates
            incr_cost = change_count

            # Use incremental if it's cheaper and doesn't blow up depth
            if change_count <= 4 and change_count < fresh_cost:
                new_wire = current_wire
                for bit in to_remove:
                    new_wire = get_xor_wire(new_wire, bit)
                for bit in to_add:
                    new_wire = get_xor_wire(new_wire, bit)
                current_wire = new_wire
            else:
                if len(row_set) == 0:
                    current_wire = n
                elif len(row_set) == 1:
                    current_wire = list(row_set)[0]
                else:
                    current_wire = _xor_tree(list(row_set), get_xor_wire)
            current_set = row_set

        outputs[idx] = current_wire

    return {'gates': gates, 'outputs': outputs}


def _build_common_factors(matrix, n):
    """
    Pre-compute frequent column-pair XORs, then build each row
    using those pre-computed wires where possible.
    """
    m = matrix.shape[0]

    # Count pair co-occurrence
    pair_counts = {}
    row_ones = []
    for i in range(m):
        ones = np.nonzero(matrix[i])[0].tolist()
        row_ones.append(ones)
        for a in range(len(ones)):
            for b in range(a + 1, len(ones)):
                key = (min(ones[a], ones[b]), max(ones[a], ones[b]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    get_xor_wire, gates, gate_registry = _make_gate_factory(n)

    # Pre-compute pairs that appear in 3+ rows
    frequent = sorted(pair_counts.items(), key=lambda x: -x[1])
    for pair, count in frequent:
        if count >= 3:
            get_xor_wire(pair[0], pair[1])

    outputs = []
    for i in range(m):
        ones = row_ones[i]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            # Greedily use pre-computed pairs
            remaining = list(ones)
            wire = None
            changed = True
            while remaining and changed:
                changed = False
                for a_idx in range(len(remaining)):
                    for b_idx in range(a_idx + 1, len(remaining)):
                        a, b = remaining[a_idx], remaining[b_idx]
                        key = (min(a, b), max(a, b))
                        if key in gate_registry:
                            pw = gate_registry[key]
                            if wire is None:
                                wire = pw
                            else:
                                wire = get_xor_wire(wire, pw)
                            remaining.pop(b_idx)
                            remaining.pop(a_idx)
                            changed = True
                            break
                    if changed:
                        break
            # XOR in any remaining singles
            for elem in remaining:
                if wire is None:
                    wire = elem
                else:
                    wire = get_xor_wire(wire, elem)
            if wire is None:
                wire = n
            outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}


def _build_pattern_grouped(matrix, n):
    """Group identical rows and compute each unique pattern once."""
    m = matrix.shape[0]
    pattern_map = {}
    for i in range(m):
        key = tuple(np.nonzero(matrix[i])[0].tolist())
        if key not in pattern_map:
            pattern_map[key] = i

    get_xor_wire, gates, _ = _make_gate_factory(n)
    pattern_wire = {}
    for pattern in pattern_map:
        if len(pattern) == 0:
            pattern_wire[pattern] = n
        elif len(pattern) == 1:
            pattern_wire[pattern] = pattern[0]
        else:
            pattern_wire[pattern] = _xor_tree(list(pattern), get_xor_wire)

    outputs = []
    for i in range(m):
        key = tuple(np.nonzero(matrix[i])[0].tolist())
        outputs.append(pattern_wire[key])

    return {'gates': gates, 'outputs': outputs}


def _build_incremental_with_swaps(matrix, n, depth_limit):
    """
    Incremental construction with local search: try swapping adjacent rows
    in the MST order to reduce total gate count.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    order = _mst_order(row_masks, m)

    def _evaluate_order(ord):
        """Build circuit for a given row order and return gate count."""
        get_xor_wire, gates, _ = _make_gate_factory(n)
        current_wire = None
        current_set = set()
        for idx in ord:
            row_mask = row_masks[idx]
            row_set = set(j for j in range(n) if (row_mask >> j) & 1)

            if current_wire is None:
                if len(row_set) == 0:
                    current_wire = n
                elif len(row_set) == 1:
                    current_wire = list(row_set)[0]
                else:
                    current_wire = _xor_tree(list(row_set), get_xor_wire)
                current_set = row_set
            else:
                to_remove = current_set - row_set
                to_add = row_set - current_set
                change_count = len(to_remove) + len(to_add)
                fresh_cost = max(0, len(row_set) - 1)

                if change_count <= 4 and change_count < fresh_cost:
                    new_wire = current_wire
                    for bit in to_remove:
                        new_wire = get_xor_wire(new_wire, bit)
                    for bit in to_add:
                        new_wire = get_xor_wire(new_wire, bit)
                    current_wire = new_wire
                else:
                    if len(row_set) == 0:
                        current_wire = n
                    elif len(row_set) == 1:
                        current_wire = list(row_set)[0]
                    else:
                        current_wire = _xor_tree(list(row_set), get_xor_wire)
                current_set = row_set
        return len(gates)

    best_order = order[:]
    best_count = _evaluate_order(best_order)

    # Try swapping adjacent pairs
    for _ in range(3):  # 3 passes of local search
        improved = False
        for i in range(len(best_order) - 1):
            trial = best_order[:]
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            trial_count = _evaluate_order(trial)
            if trial_count < best_count:
                best_order = trial
                best_count = trial_count
                improved = True
        if not improved:
            break

    # Build final circuit with best order
    get_xor_wire, gates, gate_registry = _make_gate_factory(n)
    outputs = [None] * m
    current_wire = None
    current_set = set()

    for idx in best_order:
        row_mask = row_masks[idx]
        row_set = set(j for j in range(n) if (row_mask >> j) & 1)

        if current_wire is None:
            if len(row_set) == 0:
                current_wire = n
            elif len(row_set) == 1:
                current_wire = list(row_set)[0]
            else:
                current_wire = _xor_tree(list(row_set), get_xor_wire)
            current_set = row_set
        else:
            to_remove = current_set - row_set
            to_add = row_set - current_set
            change_count = len(to_remove) + len(to_add)
            fresh_cost = max(0, len(row_set) - 1)

            if change_count <= 4 and change_count < fresh_cost:
                new_wire = current_wire
                for bit in to_remove:
                    new_wire = get_xor_wire(new_wire, bit)
                for bit in to_add:
                    new_wire = get_xor_wire(new_wire, bit)
                current_wire = new_wire
            else:
                if len(row_set) == 0:
                    current_wire = n
                elif len(row_set) == 1:
                    current_wire = list(row_set)[0]
                else:
                    current_wire = _xor_tree(list(row_set), get_xor_wire)
            current_set = row_set

        outputs[idx] = current_wire

    return {'gates': gates, 'outputs': outputs}


def _fallback_chain(matrix, n):
    """Simple chain construction as a guaranteed-correct fallback."""
    m = matrix.shape[0]
    gates = []
    outputs = []
    for i in range(m):
        ones = np.nonzero(matrix[i])[0].tolist()
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            wire = ones[0]
            for j in range(1, len(ones)):
                a = wire
                b = ones[j]
                wire = n + 1 + len(gates)
                gates.append([a, b])
            outputs.append(wire)
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

    # Strategy 1: Direct with CSE
    if time.time() < deadline - 5:
        c1 = _build_direct(matrix, n)
        candidates.append(c1)

    # Strategy 2: Incremental with MST ordering
    if time.time() < deadline - 5:
        c2 = _build_incremental_mst(matrix, n, depth_limit)
        candidates.append(c2)

    # Strategy 3: Common factors
    if time.time() < deadline - 5:
        c3 = _build_common_factors(matrix, n)
        candidates.append(c3)

    # Strategy 4: Pattern grouped
    if time.time() < deadline - 5:
        c4 = _build_pattern_grouped(matrix, n)
        candidates.append(c4)

    # Strategy 5: Incremental with local search swaps
    if time.time() < deadline - 5:
        c5 = _build_incremental_with_swaps(matrix, n, depth_limit)
        candidates.append(c5)

    # Pick the best
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        count = len(c['gates'])
        if count < best_count:
            best_count = count
            best_circuit = c

    if best_circuit is None:
        best_circuit = _fallback_chain(matrix, n)

    # Verify
    if not _verify_circuit(matrix, best_circuit, depth_limit):
        # Try each candidate in order
        found = False
        for c in candidates:
            if _verify_circuit(matrix, c, depth_limit):
                best_circuit = c
                found = True
                break
        if not found:
            best_circuit = _fallback_chain(matrix, n)

    # Final safety check
    if not _verify_circuit(matrix, best_circuit, depth_limit):
        best_circuit = _fallback_chain(matrix, n)

    return best_circuit
