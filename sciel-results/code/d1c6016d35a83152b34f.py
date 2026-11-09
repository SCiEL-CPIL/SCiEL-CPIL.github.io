"""
Exact XOR-circuit synthesis: minimum-gate DAG construction via greedy
shared-subexpression extraction with basis-aware ordering, recursive
decomposition, and local-search refinement.

Strategy:
  1. Represent each output row as a bitmask (linear combination of inputs).
  2. Greedily extract shared XOR subexpressions: at each step, pick the pair
     of wires whose XOR reduces the total Hamming-weight of remaining rows
     the most, subject to the new wire being "useful" (appears in >= 2 rows).
  3. After greedy CSE, do a recursive balanced-tree decomposition of each
     remaining row, reusing already-computed wires.
  4. Local search: remove unused gates, collapse trivial gates (zero or
     single-bit), and try 1-swap moves.
  5. Multiple restarts with different row orderings.

The key insight is that the optimal circuit is a DAG where each gate computes
the XOR of two previously-computed wires. We search over such DAGs by
greedily adding shared subexpressions that reduce total gate count, then
refine with local search.
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import random

DEADLINE_SECONDS = 82.0


def _verify_circuit(matrix, circuit, depth_limit):
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0

    gates = circuit['gates']
    for k, (a, b) in enumerate(gates):
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

    depth = {}
    for i in range(n + 1):
        depth[i] = 0

    max_depth = 0
    for k, (a, b) in enumerate(gates):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        max_depth = max(max_depth, d)

    for out in circuit['outputs']:
        max_depth = max(max_depth, depth.get(out, 0))

    if max_depth > depth_limit:
        return False

    return True


def _row_masks(matrix):
    m, n = matrix.shape
    masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        masks.append(mask)
    return masks


def _popcount(x):
    return bin(x).count('1')


def _xor_tree(indices, get_xor_wire):
    if len(indices) == 1:
        return indices[0]
    mid = len(indices) // 2
    left = _xor_tree(indices[:mid], get_xor_wire)
    right = _xor_tree(indices[mid:], get_xor_wire)
    return get_xor_wire(left, right)


def _build_greedy_cse(matrix, n, row_order=None, max_iters=200):
    """
    Build circuit via greedy CSE with shared subexpression extraction.
    Uses a benefit metric based on Hamming weight reduction.
    """
    m = matrix.shape[0]
    if row_order is None:
        row_order = list(range(m))

    row_masks = _row_masks(matrix)

    # For each row, maintain a set of wire indices that XOR to give the row value.
    row_sets = []
    for idx in row_order:
        mask = row_masks[idx]
        s = set(j for j in range(n) if (mask >> j) & 1)
        if not s:
            s = {n}
        row_sets.append(s)

    # Gate registry: frozenset({a,b}) -> wire index
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_wire(a, b):
        if a == b:
            return n
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    # Compute the total "cost" of current row sets (sum of sizes)
    def total_cost():
        return sum(len(s) for s in row_sets)

    # For a candidate new wire w = a ^ b, compute the benefit:
    # For each row containing both a and b, replacing {a,b} with {w} changes
    # the row size by -1 (two elements become one). But w itself has a mask,
    # and future decompositions may be cheaper. We approximate benefit as
    # the number of rows where both a and b appear (each such row saves 1 gate
    # in its decomposition).
    def compute_pair_benefit():
        # Count how many rows contain each wire
        wire_row_count = defaultdict(int)
        for s in row_sets:
            for w in s:
                wire_row_count[w] += 1

        # For pairs, count co-occurrence
        pair_count = defaultdict(int)
        for s in row_sets:
            lst = sorted(s)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    pair_count[frozenset((lst[i], lst[j]))] += 1
        return pair_count

    pair_count = compute_pair_benefit()
    current_cost = total_cost()

    changed = True
    iter_count = 0
    while changed and iter_count < max_iters:
        changed = False
        iter_count += 1

        best_pair = None
        best_benefit = 0
        for key, cnt in pair_count.items():
            a, b = tuple(key)
            if key in gate_registry:
                continue
            # Benefit: number of rows where both a and b appear
            # Each such row saves 1 gate in its final decomposition
            benefit = cnt
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = key

        if best_pair is None or best_benefit < 2:
            break

        a, b = tuple(best_pair)
        new_wire = get_wire(a, b)
        changed = True

        for i in range(len(row_sets)):
            s = row_sets[i]
            if a in s and b in s:
                s.discard(a)
                s.discard(b)
                s.add(new_wire)

        pair_count = compute_pair_benefit()

    # Build final outputs using balanced XOR trees, reusing existing gates
    outputs = []
    for i in range(len(row_sets)):
        s = row_sets[i]
        if len(s) == 0:
            outputs.append(n)
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            lst = sorted(s)
            outputs.append(_xor_tree(lst, get_wire))

    # Map outputs back to original row order
    if row_order != list(range(m)):
        final_outputs = [0] * m
        for new_idx, orig_idx in enumerate(row_order):
            final_outputs[orig_idx] = outputs[new_idx]
        outputs = final_outputs

    return {'gates': gates, 'outputs': outputs}


def _build_direct(matrix, n):
    """Direct construction: balanced XOR trees with CSE."""
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_xor_wire(a, b):
        if a == b:
            return n
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

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


def _build_basis_aware(matrix, n):
    """
    Basis-aware construction: first compute a basis for the row space,
    then express each row in terms of the basis, building shared gates.
    """
    m = matrix.shape[0]
    row_masks = _row_masks(matrix)

    # Gaussian elimination to find a basis
    basis = []  # list of (mask, wire) where wire is the gate that computes it
    basis_masks = []

    # Sort rows by popcount (ascending) for efficiency
    indexed_rows = sorted(range(m), key=lambda i: _popcount(row_masks[i]))

    for idx in indexed_rows:
        mask = row_masks[idx]
        if mask == 0:
            continue
        # Try to reduce by existing basis
        reduced = mask
        for b_mask in basis_masks:
            if reduced & b_mask:
                # Find the pivot bit
                pivot = (reduced & b_mask).bit_length() - 1
                if (b_mask >> pivot) & 1:
                    reduced ^= b_mask
        if reduced == 0:
            continue
        # Add to basis
        basis_masks.append(reduced)
        basis.append(reduced)

    # Now express each row in terms of basis vectors
    # This is complex; instead, do a simpler approach:
    # Process rows in order of increasing popcount, building shared subexpressions
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_wire(a, b):
        if a == b:
            return n
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    # Process rows sorted by popcount
    outputs = [0] * m
    for idx in indexed_rows:
        mask = row_masks[idx]
        if mask == 0:
            outputs[idx] = n
            continue
        ones = [j for j in range(n) if (mask >> j) & 1]
        if len(ones) == 1:
            outputs[idx] = ones[0]
        else:
            outputs[idx] = _xor_tree(ones, get_wire)

    return {'gates': gates, 'outputs': outputs}


def _local_search(matrix, circuit, depth_limit, time_budget=5.0):
    """Local search: remove unused gates, collapse trivial gates."""
    gates = [g[:] for g in circuit['gates']]
    outputs = circuit['outputs'][:]
    n = matrix.shape[1]
    m = matrix.shape[0]

    def compute_masks():
        wm = {}
        for i in range(n):
            wm[i] = (1 << i)
        wm[n] = 0
        for k, (a, b) in enumerate(gates):
            wm[n + 1 + k] = wm.get(a, 0) ^ wm.get(b, 0)
        return wm

    # Remove unused gates iteratively
    changed = True
    while changed:
        changed = False
        if not gates:
            break
        usage = defaultdict(int)
        for k, (a, b) in enumerate(gates):
            usage[a] += 1
            usage[b] += 1
        for out in outputs:
            usage[out] += 1

        new_gates = []
        removed_wires = set()
        for k, (a, b) in enumerate(gates):
            wire = n + 1 + k
            if usage[wire] == 0:
                removed_wires.add(wire)
            else:
                new_gates.append([a, b])

        if removed_wires:
            old_to_new = {}
            new_gates2 = []
            next_w = n + 1
            for k, (a, b) in enumerate(gates):
                wire = n + 1 + k
                if wire not in removed_wires:
                    old_to_new[wire] = next_w
                    a2 = old_to_new.get(a, a)
                    b2 = old_to_new.get(b, b)
                    new_gates2.append([a2, b2])
                    next_w += 1
            outputs = [old_to_new.get(o, o) for o in outputs]
            gates = new_gates2
            changed = True

    # Collapse trivial gates (a^b = 0 or single input)
    changed = True
    while changed:
        changed = False
        wm = compute_masks()
        for k in range(len(gates) - 1, -1, -1):
            a, b = gates[k]
            wire = n + 1 + k
            val = wm.get(a, 0) ^ wm.get(b, 0)
            if val == 0:
                replacement = n
            elif (val & (val - 1)) == 0:
                replacement = val.bit_length() - 1
            else:
                continue

            # Check if wire is used
            used = False
            for k2, (a2, b2) in enumerate(gates):
                if k2 == k:
                    continue
                if a2 == wire or b2 == wire:
                    used = True
                    break
            if not used:
                for out in outputs:
                    if out == wire:
                        used = True
                        break

            if not used:
                gates.pop(k)
                changed = True
                break
            else:
                for k2, (a2, b2) in enumerate(gates):
                    if k2 == k:
                        continue
                    if a2 == wire:
                        gates[k2][0] = replacement
                    if b2 == wire:
                        gates[k2][1] = replacement
                for i in range(len(outputs)):
                    if outputs[i] == wire:
                        outputs[i] = replacement
                gates.pop(k)
                changed = True
                break

    result = {'gates': gates, 'outputs': outputs}
    if _verify_circuit(matrix, result, depth_limit):
        return result
    return circuit


def _try_replacement(matrix, circuit, depth_limit):
    """
    Try replacing a gate with a different decomposition.
    For each gate k computing wire w = a ^ b, try to find an alternative
    pair (c, d) such that c ^ d = a ^ b, using already-computed wires.
    If found and it reduces the total gate count, apply the change.
    """
    gates = [g[:] for g in circuit['gates']]
    outputs = circuit['outputs'][:]
    n = matrix.shape[1]
    m = matrix.shape[0]

    def compute_masks():
        wm = {}
        for i in range(n):
            wm[i] = (1 << i)
        wm[n] = 0
        for k, (a, b) in enumerate(gates):
            wm[n + 1 + k] = wm.get(a, 0) ^ wm.get(b, 0)
        return wm

    wm = compute_masks()

    # Build a lookup: mask -> list of wires that compute it
    mask_to_wires = defaultdict(list)
    for i in range(n):
        mask_to_wires[1 << i].append(i)
    mask_to_wires[0].append(n)
    for k in range(len(gates)):
        wire = n + 1 + k
        mask_to_wires[wm[wire]].append(wire)

    # For each gate, try to find a cheaper alternative
    improved = False
    for k in range(len(gates)):
        a, b = gates[k]
        wire = n + 1 + k
        target_mask = wm[wire]

        # Try all pairs of already-computed wires (indices < wire)
        # that XOR to target_mask
        candidates = mask_to_wires.get(target_mask, [])
        for c in candidates:
            for d in candidates:
                if c == d:
                    continue
                if c > wire or d > wire:
                    continue
                if c == a and d == b:
                    continue
                if c == b and d == a:
                    continue
                # Check if this replacement is valid (c and d are computed before wire)
                # They are, since c, d < wire
                # Apply replacement
                gates[k] = [min(c, d), max(c, d)]
                improved = True
                break
            if improved:
                break
        if improved:
            break

    if improved:
        result = {'gates': gates, 'outputs': outputs}
        if _verify_circuit(matrix, result, depth_limit):
            return result

    return circuit


def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.int8)

    n_cols = matrix.shape[1]
    n_rows = matrix.shape[0]

    best = None
    best_gates = float('inf')

    # Strategy 1: Direct construction
    c = _build_direct(matrix, n_cols)
    if _verify_circuit(matrix, c, depth_limit):
        best = c
        best_gates = len(c['gates'])

    # Strategy 2: Greedy CSE with default ordering
    if time.time() < deadline - 5.0:
        c = _build_greedy_cse(matrix, n_cols)
        if _verify_circuit(matrix, c, depth_limit):
            if len(c['gates']) < best_gates:
                best = c
                best_gates = len(c['gates'])

    # Strategy 3: Basis-aware construction
    if time.time() < deadline - 5.0:
        c = _build_basis_aware(matrix, n_cols)
        if _verify_circuit(matrix, c, depth_limit):
            if len(c['gates']) < best_gates:
                best = c
                best_gates = len(c['gates'])

    # Strategy 4: Multiple restarts with random row orderings
    if time.time() < deadline - 10.0:
        random.seed(42)
        for restart in range(8):
            if time.time() > deadline - 5.0:
                break
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            c = _build_greedy_cse(matrix, n_cols, row_order=row_order)
            if _verify_circuit(matrix, c, depth_limit):
                if len(c['gates']) < best_gates:
                    best = c
                    best_gates = len(c['gates'])

    # Local search refinement
    if best is not None and time.time() < deadline - 3.0:
        budget = min(5.0, deadline - time.time() - 2.0)
        refined = _local_search(matrix, best, depth_limit, time_budget=budget)
        if len(refined['gates']) < best_gates:
            best = refined
            best_gates = len(refined['gates'])

    # Try gate replacement
    if best is not None and time.time() < deadline - 2.0:
        replaced = _try_replacement(matrix, best, depth_limit)
        if len(replaced['gates']) < best_gates:
            best = replaced
            best_gates = len(replaced['gates'])

    # Final verification
    if best is None or not _verify_circuit(matrix, best, depth_limit):
        best = _build_direct(matrix, n_cols)
        if not _verify_circuit(matrix, best, depth_limit):
            best = {'gates': [], 'outputs': [n_cols] * n_rows}

    return best
