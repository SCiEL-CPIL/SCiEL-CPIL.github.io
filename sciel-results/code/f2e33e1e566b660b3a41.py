import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.uint8) % 2
    
    # RREF Fallback
    def rref_synthesis():
        M = A.copy()
        T = np.eye(m, dtype=np.uint8)
        rank = 0
        pivot_cols = []
        for col in range(n):
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
        
        aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
        for col in range(m):
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
        
        gates = []
        wire_cache = {}
        
        def xor_wires(wire_list):
            if not wire_list:
                return n
            if len(wire_list) == 1:
                return wire_list[0]
            key = frozenset(wire_list)
            if key in wire_cache:
                return wire_cache[key]
            mid = len(wire_list) // 2
            left = xor_wires(wire_list[:mid])
            right = xor_wires(wire_list[mid:])
            g = len(gates)
            gates.append([left, right])
            result = n + 1 + g
            wire_cache[key] = result
            return result
            
        pivot_wires = []
        for i in range(rank):
            ones = [j for j in range(n) if M[i, j] == 1]
            pivot_wires.append(xor_wires(ones))
            
        outputs = []
        for j in range(m):
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            outputs.append(xor_wires([pivot_wires[i] for i in contributing]))
            
        return {'gates': gates, 'outputs': outputs}

    # Greedy Synthesis
    def greedy_synthesis():
        row_masks = []
        for i in range(m):
            mask = 0
            for j in range(n):
                if A[i, j] == 1:
                    mask |= (1 << j)
            row_masks.append(mask)
            
        available_wires = []
        wire_to_id = {}
        for j in range(n):
            mask = 1 << j
            wire_id = j
            available_wires.append(mask)
            wire_to_id[mask] = wire_id
            
        zero_mask = 0
        zero_wire_id = n
        available_wires.append(zero_mask)
        wire_to_id[zero_mask] = zero_wire_id
        
        needed_masks = set(row_masks)
        gates = []
        next_wire_id = n + 1
        available_set = set(available_wires)
        
        max_iters = 5000
        iter_count = 0
        stagnation_count = 0
        
        while needed_masks and iter_count < max_iters:
            if time.time() > deadline:
                return None
            
            iter_count += 1
            
            # Check for already satisfied
            satisfied_now = [mask for mask in needed_masks if mask in wire_to_id]
            if satisfied_now:
                for mask in satisfied_now:
                    needed_masks.remove(mask)
                stagnation_count = 0
                continue
            
            # Try direct satisfaction
            found_direct = False
            best_pair = None
            best_mask = None
            
            for m_need in needed_masks:
                for a in available_wires:
                    b = m_need ^ a
                    if b in available_set:
                        found_direct = True
                        best_mask = m_need
                        best_pair = (a, b)
                        break
                if found_direct:
                    break
            
            if found_direct:
                a, b = best_pair
                wa = wire_to_id[a]
                wb = wire_to_id[b]
                gates.append([wa, wb])
                new_wire_id = next_wire_id
                next_wire_id += 1
                available_wires.append(best_mask)
                wire_to_id[best_mask] = new_wire_id
                available_set.add(best_mask)
                needed_masks.remove(best_mask)
                stagnation_count = 0
                continue
            
            # Stagnation
            stagnation_count += 1
            if stagnation_count > 50:
                # Try a different heuristic: pick pair with min popcount
                min_pop = n + 1
                best_pair_inter = None
                for i in range(len(available_wires)):
                    for j in range(i + 1, len(available_wires)):
                        a = available_wires[i]
                        b = available_wires[j]
                        c = a ^ b
                        if c == 0:
                            continue
                        pop = bin(c).count('1')
                        if pop < min_pop:
                            min_pop = pop
                            best_pair_inter = (a, b, c)
                
                if best_pair_inter:
                    a, b, c = best_pair_inter
                    wa = wire_to_id[a]
                    wb = wire_to_id[b]
                    gates.append([wa, wb])
                    new_wire_id = next_wire_id
                    next_wire_id += 1
                    available_wires.append(c)
                    wire_to_id[c] = new_wire_id
                    available_set.add(c)
                    stagnation_count = 0
                else:
                    return None
            else:
                # Pick a random pair to explore
                i = np.random.randint(0, len(available_wires))
                j = np.random.randint(0, len(available_wires))
                if i == j:
                    continue
                a = available_wires[i]
                b = available_wires[j]
                c = a ^ b
                if c == 0:
                    continue
                if c in wire_to_id:
                    continue
                wa = wire_to_id[a]
                wb = wire_to_id[b]
                gates.append([wa, wb])
                new_wire_id = next_wire_id
                next_wire_id += 1
                available_wires.append(c)
                wire_to_id[c] = new_wire_id
                available_set.add(c)

        if needed_masks:
            return None
            
        outputs = []
        for mask in row_masks:
            outputs.append(wire_to_id[mask])
            
        return {'gates': gates, 'outputs': outputs}

    # Try Greedy first
    result = greedy_synthesis()
    if result:
        return result
        
    # Fallback to RREF
    return rref_synthesis()
