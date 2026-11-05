import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
import heapq
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape

    zero_wire = n

    # Compute row weights and baseline
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))

    # Handle trivial cases
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}

    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)

    # Try ILP-based synthesis for small-to-medium instances
    ilp_result = _ilp_synthesis(target_masks, n, zero_wire, row_weights, deadline)
    if ilp_result is not None:
        gates, outputs = ilp_result
        # Verify the result
        if _verify_circuit(gates, outputs, mat, zero_wire):
            return {'gates': gates, 'outputs': outputs}

    # Fallback: greedy sharing approach
    greedy_result = _greedy_synthesis_v2(target_masks, n, zero_wire, row_weights, deadline)
    if greedy_result is not None:
        gates, outputs = greedy_result
        return {'gates': gates, 'outputs': outputs}

    # Fallback: simple per-row XOR
    gates = []
    outputs = []
    next_wire = n + 1
    for i in range(m):
        if row_weights[i] == 0:
            outputs.append(zero_wire)
            continue
        ones = np.where(mat[i] == 1)[0]
        if len(ones) == 1:
            outputs.append(int(ones[0]))
            continue
        current = int(ones[0])
        for j in range(1, len(ones)):
            other = int(ones[j])
            gates.append([current, other])
            current = next_wire
            next_wire += 1
        outputs.append(current)

    return {'gates': gates, 'outputs': outputs}


def _verify_circuit(gates, outputs, mat, zero_wire):
    """Verify that the circuit correctly computes the matrix over GF(2)."""
    m, n = mat.shape

    # Evaluate each wire's value as a bitmask of input contributions
    # wire_values[i] = bitmask representing which inputs XOR together to produce wire i
    num_wires = max(n + 1 + len(gates), max(outputs) + 1 if outputs else n + 1)
    wire_values = [0] * num_wires

    # Input wires
    for i in range(n):
        wire_values[i] = (1 << i)
    wire_values[n] = 0  # zero wire

    # Compute gate outputs
    for k, (w1, w2) in enumerate(gates):
        wire_idx = n + 1 + k
        wire_values[wire_idx] = wire_values[w1] ^ wire_values[w2]

    # Check outputs
    for i in range(m):
        out_wire = outputs[i]
        if out_wire >= len(wire_values):
            return False
        expected_mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                expected_mask |= (1 << j)
        if wire_values[out_wire] != expected_mask:
            return False

    return True


def _ilp_synthesis(target_masks, n, zero_wire, row_weights, deadline):
    """
    ILP-based synthesis using minimum-cost flow / set-cover formulation.
    
    For small n (<= 12), enumerate all possible 2-input XOR combinations
    and solve a set-cover ILP to find the minimum gate count.
    
    Variables: x_e for each potential gate e = (a, b) where a, b are
    available wires (inputs or previously computed).
    
    This is complex, so we use a simpler approach: 
    - Enumerate all masks that could be useful (targets + their submasks)
    - Build a graph where edges represent XOR operations
    - Find minimum-cost subset of edges that connects all targets to inputs
    """
    elapsed = lambda: time.time() - start_time
    if elapsed() > deadline:
        return None

    # For very small n, try exact DP
    if n <= 10:
        result = _exact_dp_small(target_masks, n, zero_wire, deadline)
        if result is not None:
            return result

    # For medium n, use greedy with look-ahead
    result = _greedy_lookahead(target_masks, n, zero_wire, row_weights, deadline)
    if result is not None:
        return result

    return None


def _exact_dp_small(target_masks, n, zero_wire, deadline):
    """
    Exact DP for very small n (<= 10).
    
    State: subset of targets already computed.
    Transition: add a new gate that produces a mask that helps compute remaining targets.
    
    Actually, the state should be the set of all masks currently available.
    But that's 2^(2^n) which is too large even for n=10.
    
    Alternative: since we only care about computing the specific target masks,
    we can use a different DP. Let's define:
    
    dp[S] = minimum number of gates needed to compute all targets in S,
    where S is a subset of the target indices.
    
    Transition: for each target t not in S, find the cheapest way to compute t
    using gates that may involve other targets in S.
    
    This is still complex. Let's use a simpler heuristic for small n:
    try all permutations of target processing order and pick the best.
    """
    elapsed = lambda: time.time() - start_time
    if elapsed() > deadline:
        return None

    m = len(target_masks)
    if m > 8:
        return None

    best_gates = None
    best_count = float('inf')

    # Try different processing orders (limited permutations)
    indices = list(range(m))
    num_orders = min(100, len(indices) ** 3)  # Cap attempts
    random.seed(42)

    for _ in range(num_orders):
        order = indices[:]
        random.shuffle(order)

        available = {}
        for i in range(n):
            available[1 << i] = i
        available[0] = zero_wire

        gates = []
        next_wire = n + 1
        target_wires = {}
        valid = True

        for idx in order:
            if elapsed() > deadline:
                valid = False
                break
            mask = target_masks[idx]

            if mask in available:
                target_wires[idx] = available[mask]
                continue

            # Try to find decomposition using 2 available wires
            found = False
            avail_items = list(available.items())
            for i1, (m1, w1) in enumerate(avail_items):
                for i2, (m2, w2) in enumerate(avail_items):
                    if i1 == i2:
                        continue
                    if m1 == 0 or m2 == 0:
                        continue
                    if m1 ^ m2 == mask:
                        gates.append([w1, w2])
                        target_wires[idx] = next_wire
                        next_wire += 1
                        available[mask] = next_wire - 1
                        found = True
                        break
                if found:
                    break

            if not found:
                # Try 1 available + 1 input
                for i in range(n):
                    input_mask = 1 << i
                    other_mask = mask ^ input_mask
                    if other_mask in available:
                        other_wire = available[other_mask]
                        gates.append([i, other_wire])
                        target_wires[idx] = next_wire
                        next_wire += 1
                        available[mask] = next_wire - 1
                        found = True
                        break

            if not found:
                # Fallback: direct computation
                ones = [j for j in range(n) if (mask >> j) & 1]
                if len(ones) == 0:
                    target_wires[idx] = zero_wire
                elif len(ones) == 1:
                    target_wires[idx] = ones[0]
                else:
                    current = ones[0]
                    for j in range(1, len(ones)):
                        gates.append([current, ones[j]])
                        current = next_wire
                        next_wire += 1
                    target_wires[idx] = current
                    available[mask] = current

        if valid and len(gates) < best_count:
            best_count = len(gates)
            best_gates = gates[:]
            best_outputs = [target_wires[i] for i in range(m)]

    if best_gates is not None:
        return best_gates, best_outputs

    return None


def _greedy_lookahead(target_masks, n, zero_wire, row_weights, deadline):
    """
    Greedy with look-ahead: at each step, choose the next target to compute
    that minimizes the estimated total cost, considering sharing opportunities.
    """
    elapsed = lambda: time.time() - start_time
    if elapsed() > deadline:
        return None

    m = len(target_masks)

    # Start with inputs and zero
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire

    gates = []
    next_wire = n + 1
    target_wires = [None] * m

    # Mark directly available targets
    for i in range(m):
        if target_masks[i] in available:
            target_wires[i] = available[target_masks[i]]

    # Greedily process remaining targets
    remaining = set(range(m))
    for i in range(m):
        if target_wires[i] is not None:
            remaining.discard(i)

    while remaining:
        if elapsed() > deadline:
            break

        best_idx = None
        best_cost = float('inf')
        best_decomp = None

        for idx in list(remaining):
            mask = target_masks[idx]

            # Estimate cost to compute this mask
            cost = _estimate_cost(mask, available, n)

            if cost < best_cost:
                best_cost = cost
                best_idx = idx
                best_decomp = _find_best_decomp(mask, available, n)

        if best_idx is None:
            break

        # Compute the best target
        mask = target_masks[best_idx]
        wire = None

        # Try to find decomposition
        found = False
        avail_items = list(available.items())

        # Try 2 available wires
        for m1, w1 in avail_items:
            if m1 == 0:
                continue
            other_mask = mask ^ m1
            if other_mask in available and other_mask != 0:
                w2 = available[other_mask]
                gates.append([w1, w2])
                wire = next_wire
                next_wire += 1
                available[mask] = wire
                found = True
                break

        if not found:
            # Try 1 input + 1 available
            for i in range(n):
                input_mask = 1 << i
                other_mask = mask ^ input_mask
                if other_mask in available:
                    w2 = available[other_mask]
                    gates.append([i, w2])
                    wire = next_wire
                    next_wire += 1
                    available[mask] = wire
                    found = True
                    break

        if not found:
            # Fallback: direct computation
            ones = [j for j in range(n) if (mask >> j) & 1]
            if len(ones) == 0:
                wire = zero_wire
            elif len(ones) == 1:
                wire = ones[0]
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                wire = current
                available[mask] = wire

        target_wires[best_idx] = wire
        remaining.discard(best_idx)

    # Verify all targets are computed
    if any(w is None for w in target_wires):
        return None

    return gates, target_wires


def _estimate_cost(mask, available, n):
    """Estimate the cost to compute mask from available wires."""
    if mask in available:
        return 0

    # Try 2 available wires
    avail_masks = set(available.keys())
    for m1 in avail_masks:
        if m1 == 0:
            continue
        if (mask ^ m1) in avail_masks:
            return 1

    # Try 1 input + 1 available
    for i in range(n):
        input_mask = 1 << i
        if (mask ^ input_mask) in avail_masks:
            return 1

    # Fallback: weight - 1
    return mask.bit_count() - 1


def _find_best_decomp(mask, available, n):
    """Find the best decomposition of mask using available wires."""
    # Placeholder for future use
    return None


def _greedy_synthesis_v2(target_masks, n, zero_wire, row_weights, deadline):
    """
    Improved greedy synthesis with better sharing detection.
    """
    elapsed = lambda: time.time() - start_time
    if elapsed() > deadline:
        return None

    m = len(target_masks)

    # Start with inputs and zero
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire

    gates = []
    next_wire = n + 1
    target_wires = [None] * m

    # Mark directly available targets
    for i in range(m):
        if target_masks[i] in available:
            target_wires[i] = available[target_masks[i]]

    # Process remaining targets in order of increasing weight
    remaining = [i for i in range(m) if target_wires[i] is None]
    remaining.sort(key=lambda i: row_weights[i])

    for idx in remaining:
        if elapsed() > deadline:
            break

        mask = target_masks[idx]

        # Try to find a decomposition
        wire = None
        found = False

        # Try: XOR of two available wires
        avail_items = list(available.items())
        for m1, w1 in avail_items:
            if m1 == 0:
                continue
            other_mask = mask ^ m1
            if other_mask in available:
                w2 = available[other_mask]
                if other_mask == 0:
                    wire = w1
                elif m1 == 0:
                    wire = w2
                else:
                    gates.append([w1, w2])
                    wire = next_wire
                    next_wire += 1
                    available[mask] = wire
                found = True
                break

        if not found:
            # Try: XOR of one input and one available wire
            for i in range(n):
                input_mask = 1 << i
                other_mask = mask ^ input_mask
                if other_mask in available:
                    w2 = available[other_mask]
                    if other_mask == 0:
                        wire = i
                    else:
                        gates.append([i, w2])
                        wire = next_wire
                        next_wire += 1
                        available[mask] = wire
                    found = True
                    break

        if not found:
            # Fallback: compute from inputs directly
            ones = [j for j in range(n) if (mask >> j) & 1]
            if len(ones) == 0:
                wire = zero_wire
            elif len(ones) == 1:
                wire = ones[0]
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                wire = current
                available[mask] = wire

        target_wires[idx] = wire

    return gates, target_wires
