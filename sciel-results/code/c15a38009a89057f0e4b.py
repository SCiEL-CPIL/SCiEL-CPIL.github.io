import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0  # 54s safety margin out of 60s

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.uint8) % 2

    # Compute RREF with row tracking and Minimum Weight Pivot Selection
    M = A.copy()
    T = np.eye(m, dtype=np.uint8)

    rank = 0
    pivot_cols = []

    for col in range(n):
        if time.time() > deadline:
            break
        
        # Find candidate pivots in this column
        candidates = []
        for row in range(rank, m):
            if M[row, col] == 1:
                # Calculate weight of the row (number of 1s)
                weight = np.sum(M[row])
                candidates.append((weight, row))
        
        if not candidates:
            continue

        # Select pivot with minimum weight to keep matrix sparse
        candidates.sort(key=lambda x: x[0])
        pivot = candidates[0][1]

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
    # We only need the first 'rank' columns of T_inv if rank < m, 
    # but T is m x m. If rank < m, T is singular? 
    # No, T is built from row swaps and XORs. It is always invertible.
    # However, if rank < m, the last m-rank rows of M are 0.
    # The outputs corresponding to those rows are 0.
    
    # Standard inverse calculation
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

    # Build XOR trees efficiently
    gates = []
    wire_cache = {}  # frozenset of inputs -> wire number

    def xor_wires(wire_list):
        """XOR a list of wires, reusing cached intermediate results."""
        if not wire_list:
            return n
        if len(wire_list) == 1:
            return wire_list[0]

        key = frozenset(wire_list)
        if key in wire_cache:
            return wire_cache[key]

        # Use divide-and-conquer to build balanced XOR tree
        mid = len(wire_list) // 2
        left = xor_wires(wire_list[:mid])
        right = xor_wires(wire_list[mid:])

        g = len(gates)
        gates.append([left, right])
        result = n + 1 + g
        wire_cache[key] = result
        return result

    # Step 1: For each pivot row, compute the wire
    pivot_wires = []
    for i in range(rank):
        if time.time() > deadline:
            # Fallback: simple linear XOR
            ones = [j for j in range(n) if M[i, j] == 1]
            if not ones:
                pivot_wires.append(n)
            elif len(ones) == 1:
                pivot_wires.append(ones[0])
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    g = len(gates)
                    gates.append([current, ones[j]])
                    current = n + 1 + g
                pivot_wires.append(current)
        else:
            ones = [j for j in range(n) if M[i, j] == 1]
            pivot_wires.append(xor_wires(ones))

    # Step 2: Express each output as XOR of pivot wires
    outputs = []
    for j in range(m):
        if time.time() > deadline:
            # Fallback: simple linear XOR
            # Note: If rank < m, T_inv rows corresponding to non-pivot original rows
            # should be zero vectors if the system is consistent, but T is invertible.
            # Actually, if rank < m, some rows of M are 0.
            # The outputs for those rows are 0 (wire n).
            # T_inv maps original row index to pivot combination.
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            if not contributing:
                outputs.append(n)
            elif len(contributing) == 1:
                outputs.append(pivot_wires[contributing[0]])
            else:
                current = pivot_wires[contributing[0]]
                for i in range(1, len(contributing)):
                    g = len(gates)
                    gates.append([current, pivot_wires[contributing[i]]])
                    current = n + 1 + g
                outputs.append(current)
        else:
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            outputs.append(xor_wires([pivot_wires[i] for i in contributing]))

    return {'gates': gates, 'outputs': outputs}
