"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}

    # Build conflict bitsets
    conflict_bits = [0] * n
    for i, node in enumerate(valid_nodes):
        for s in squares:
            j = s - node
            if 1 <= j <= N and j != node and j in valid_set:
                j_idx = node_to_idx[j]
                conflict_bits[i] |= (1 << j_idx)

    # Greedy min-degree independent set as initial lower bound
    remaining = (1 << n) - 1
    selected_bits = 0

    while remaining:
        best_node = -1
        best_degree = n + 1
        for i in range(n):
            if (remaining >> i) & 1:
                deg = bin(conflict_bits[i] & remaining).count('1')
                if deg < best_degree:
                    best_degree = deg
                    best_node = i
        selected_bits |= (1 << best_node)
        remaining &= ~(1 << best_node)
        remaining &= ~conflict_bits[best_node]

    best_size = bin(selected_bits).count('1')
    best_solution_bits = selected_bits

    # Local search: greedy add
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        for i in range(n):
            if (best_solution_bits >> i) & 1:
                continue
            if conflict_bits[i] & best_solution_bits:
                continue
            best_solution_bits |= (1 << i)
            best_size += 1
            improved = True

    # Aggressive multi-swap local search
    unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
    selected_list = [i for i in range(n) if (best_solution_bits >> i) & 1]

    for _ in range(200):
        if time.time() > deadline - 15.0:
            break
        improved = False
        # Try 1-swap: remove k selected, add 1 unselected
        for a in unselected:
            if (best_solution_bits >> a) & 1:
                continue
            conflicting = [b for b in selected_list if (conflict_bits[a] >> b) & 1]
            if not conflicting:
                # Can add directly
                best_solution_bits |= (1 << a)
                best_size += 1
                unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
                selected_list = [i for i in range(n) if (best_solution_bits >> i) & 1]
                improved = True
                break
            if len(conflicting) > 2:
                continue
            removed_bits = 0
            for b in conflicting:
                removed_bits |= (1 << b)
            candidate_bits = (best_solution_bits & ~removed_bits) | (1 << a)
            temp_bits = candidate_bits
            temp_size = best_size - len(conflicting) + 1
            # Try to add more nodes after the swap
            for c in unselected:
                if c == a or (removed_bits >> c) & 1:
                    continue
                if (temp_bits >> c) & 1:
                    continue
                if conflict_bits[c] & temp_bits:
                    continue
                temp_bits |= (1 << c)
                temp_size += 1
            if temp_size > best_size:
                best_solution_bits = temp_bits
                best_size = temp_size
                unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
                selected_list = [i for i in range(n) if (best_solution_bits >> i) & 1]
                improved = True
                break
        if improved:
            continue
        
        # Try 2-swap: remove 2 selected, add 2 unselected
        found_2swap = False
        for a in unselected:
            if (best_solution_bits >> a) & 1:
                continue
            conf_a = [b for b in selected_list if (conflict_bits[a] >> b) & 1]
            if len(conf_a) > 2:
                continue
            for c in unselected:
                if c == a or (best_solution_bits >> c) & 1:
                    continue
                conf_c = [b for b in selected_list if (conflict_bits[c] >> b) & 1]
                if len(conf_c) > 2:
                    continue
                # Combined conflicts
                all_conf = set(conf_a) | set(conf_c)
                if len(all_conf) > 3:
                    continue
                removed_bits = 0
                for b in all_conf:
                    removed_bits |= (1 << b)
                candidate_bits = (best_solution_bits & ~removed_bits) | (1 << a) | (1 << c)
                temp_bits = candidate_bits
                temp_size = best_size - len(all_conf) + 2
                # Try to add more
                for d in unselected:
                    if d == a or d == c or (removed_bits >> d) & 1:
                        continue
                    if (temp_bits >> d) & 1:
                        continue
                    if conflict_bits[d] & temp_bits:
                        continue
                    temp_bits |= (1 << d)
                    temp_size += 1
                if temp_size > best_size:
                    best_solution_bits = temp_bits
                    best_size = temp_size
                    unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
                    selected_list = [i for i in range(n) if (best_solution_bits >> i) & 1]
                    improved = True
                    found_2swap = True
                    break
            if found_2swap:
                break
        if not improved:
            break

    # Simulated annealing with multiple restarts
    if time.time() < deadline - 10.0:
        random.seed(42)
        
        for restart in range(5):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb the best solution for restarts > 0
            if restart == 0:
                current_bits = best_solution_bits
                current_size = best_size
            else:
                current_bits = best_solution_bits
                current_size = best_size
                # Remove some random selected nodes
                sel_list = [i for i in range(n) if (current_bits >> i) & 1]
                num_remove = max(1, current_size // 8)
                to_remove = random.sample(sel_list, min(num_remove, len(sel_list)))
                for b in to_remove:
                    current_bits &= ~(1 << b)
                    current_size -= 1
                # Try to add back greedily
                unsel = [i for i in range(n) if not ((current_bits >> i) & 1)]
                for a in unsel:
                    if not (conflict_bits[a] & current_bits):
                        current_bits |= (1 << a)
                        current_size += 1
            
            T = 1.0
            T_min = 0.01
            cooling = 0.9995
            
            unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
            selected_list = [i for i in range(n) if (current_bits >> i) & 1]
            
            while time.time() < deadline - 10.0 and T > T_min:
                # Pick a random move: add, remove, or swap
                move_type = random.random()
                
                if move_type < 0.35 and unselected_list:
                    # Try to add a random unselected node
                    a = random.choice(unselected_list)
                    if conflict_bits[a] & current_bits:
                        continue
                    new_bits = current_bits | (1 << a)
                    new_size = current_size + 1
                    delta = 1
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_bits = new_bits
                        current_size = new_size
                        if current_size > best_size:
                            best_solution_bits = current_bits
                            best_size = current_size
                        unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
                        selected_list = [i for i in range(n) if (current_bits >> i) & 1]
                elif move_type < 0.65 and selected_list:
                    # Try to remove a random selected node
                    b = random.choice(selected_list)
                    new_bits = current_bits & ~(1 << b)
                    new_size = current_size - 1
                    delta = -1
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_bits = new_bits
                        current_size = new_size
                        if current_size > best_size:
                            best_solution_bits = current_bits
                            best_size = current_size
                        unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
                        selected_list = [i for i in range(n) if (current_bits >> i) & 1]
                else:
                    # Try a swap: remove one selected, add one unselected
                    if not selected_list or not unselected_list:
                        continue
                    b = random.choice(selected_list)
                    a = random.choice(unselected_list)
                    # Check if a conflicts with current_bits minus b
                    temp_bits = current_bits & ~(1 << b)
                    if conflict_bits[a] & temp_bits:
                        continue
                    new_bits = temp_bits | (1 << a)
                    new_size = current_size  # swap, size stays same
                    delta = 0
                    # Accept with some probability to escape local optima
                    if random.random() < math.exp(delta / T):
                        current_bits = new_bits
                        current_size = new_size
                        if current_size > best_size:
                            best_solution_bits = current_bits
                            best_size = current_size
                        unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
                        selected_list = [i for i in range(n) if (current_bits >> i) & 1]
                
                T *= cooling
            
            # Greedy add after each SA run
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(n):
                    if (best_solution_bits >> i) & 1:
                        continue
                    if conflict_bits[i] & best_solution_bits:
                        continue
                    best_solution_bits |= (1 << i)
                    best_size += 1
                    improved = True

    # Extract result
    result = []
    for i in range(n):
        if (best_solution_bits >> i) & 1:
            result.append(valid_nodes[i])

    result.sort()

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                valid = False
                break
        if not valid:
            break

    if not valid:
        result = []
        for i in range(n):
            if (selected_bits >> i) & 1:
                result.append(valid_nodes[i])
        result.sort()

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
