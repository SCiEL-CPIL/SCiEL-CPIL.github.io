import time
import numpy as np
from itertools import combinations
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.uint8) % 2

    # Compute RREF with row tracking
    M = A.copy()
    T = np.eye(m, dtype=np.uint8)

    rank = 0
    pivot_cols = []

    for col in range(n):
        if time.time() > deadline:
            break
        pivot = -1
        for row in range(rank, m):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue

        M[[rank, pivot]] = M[[pivot, rank]]
        T[[rank, pivot]] = T[[pivot, rank]]

        for row in range(m):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                T[row] ^= T[rank]

        pivot_cols.append(col)
        rank += 1
        if rank == m:
            break

    # Compute T^{-1}
    aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
    for col in range(m):
        if time.time() > deadline:
            break
        pivot = -1
        for row in range(col, m):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(m):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    T_inv = aug[:, m:]

    # Each output j is XOR of basis rows i where T_inv[j, i] == 1
    # Each basis row i is XOR of input cols where M[i, col] == 1

    # Strategy: Joint synthesis with beam search over output ordering
    # For each output, we need to XOR a subset of input wires
    # We can share intermediate XORs

    # Precompute the input subsets for each output
    output_subsets = []
    for j in range(m):
        contributing = [i for i in range(rank) if T_inv[j, i] == 1]
        # Expand to input indices
        input_set = set()
        for i in contributing:
            for col in range(n):
                if M[i, col] == 1:
                    if col in input_set:
                        input_set.remove(col)
                    else:
                        input_set.add(col)
        output_subsets.append(frozenset(input_set))

    # Beam search: state is (available_wires_dict, gates_so_far, outputs_done)
    # available_wires_dict maps frozenset_of_inputs -> wire_number
    # We want to minimize total gates

    # Simpler approach: try multiple orderings of outputs, use greedy with memoization
    # and pick the best

    best_gates = None
    best_outputs = None
    best_count = float('inf')

    # Try several orderings
    orderings = []
    # Natural order
    orderings.append(list(range(m)))
    # Sort by popcount ascending (easy first, share more)
    orderings.append(sorted(range(m), key=lambda j: len(output_subsets[j])))
    # Sort by popcount descending
    orderings.append(sorted(range(m), key=lambda j: -len(output_subsets[j])))
    # Random permutations
    import random
    random.seed(42)
    for _ in range(20):
        perm = list(range(m))
        random.shuffle(perm)
        orderings.append(perm)

    for order in orderings:
        if time.time() > deadline:
            break

        gates = []
        wire_cache = {}  # frozenset -> wire
        outputs = [None] * m

        for j in order:
            if time.time() > deadline:
                break
            subset = output_subsets[j]
            if not subset:
                outputs[j] = n
                continue
            if len(subset) == 1:
                outputs[j] = next(iter(subset))
                continue

            # Build XOR tree with memoization
            elems = list(subset)
            # Use divide and conquer
            def xor_list(lst):
                if not lst:
                    return n
                if len(lst) == 1:
                    return lst[0]
                key = frozenset(lst)
                if key in wire_cache:
                    return wire_cache[key]
                mid = len(lst) // 2
                left = xor_list(lst[:mid])
                right = xor_list(lst[mid:])
                g = len(gates)
                gates.append([left, right])
                result = n + 1 + g
                wire_cache[key] = result
                return result

            outputs[j] = xor_list(elems)

        if time.time() > deadline:
            break

        if len(gates) < best_count:
            best_count = len(gates)
            best_gates = gates[:]
            best_outputs = outputs[:]

    # Also try: compute all basis rows first, then combine
    if time.time() < deadline:
        gates = []
        wire_cache = {}
        basis_wires = []

        def xor_list(lst):
            if not lst:
                return n
            if len(lst) == 1:
                return lst[0]
            key = frozenset(lst)
            if key in wire_cache:
                return wire_cache[key]
            mid = len(lst) // 2
            left = xor_list(lst[:mid])
            right = xor_list(lst[mid:])
            g = len(gates)
            gates.append([left, right])
            result = n + 1 + g
            wire_cache[key] = result
            return result

        for i in range(rank):
            cols = [j for j in range(n) if M[i, j] == 1]
            basis_wires.append(xor_list(cols))

        outputs2 = [None] * m
        for j in range(m):
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            if not contributing:
                outputs2[j] = n
            else:
                outputs2[j] = xor_list([basis_wires[i] for i in contributing])

        if len(gates) < best_count:
            best_count = len(gates)
            best_gates = gates[:]
            best_outputs = outputs2[:]

    if best_gates is None:
        # Fallback: simple linear
        gates = []
        outputs = []
        for j in range(m):
            ones = [col for col in range(n) if A[j, col] == 1]
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                current = ones[0]
                for k in range(1, len(ones)):
                    g = len(gates)
                    gates.append([current, ones[k]])
                    current = n + 1 + g
                outputs.append(current)
        best_gates = gates
        best_outputs = outputs

    return {'gates': best_gates, 'outputs': best_outputs}
