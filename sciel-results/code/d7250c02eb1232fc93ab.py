import time
import numpy as np
from collections import deque
import heapq

def xor_baseline(matrix, depth_limit):
    """
    Fallback: Synthesize each row independently.
    """
    m, n = matrix.shape
    gates = []
    outputs = []
    
    # wire_map: mask -> wire_index
    # Inputs: 0..n-1
    # Zero: n
    # Gates start at n+1
    
    global_wire_map = {}
    for i in range(n):
        global_wire_map[1 << i] = i
    global_wire_map[0] = n
    
    next_wire_id = n + 1
    
    for i in range(m):
        target = 0
        for j in range(n):
            if matrix[i, j]:
                target |= (1 << j)
        
        if target in global_wire_map:
            outputs.append(global_wire_map[target])
            continue
        
        # Decompose target
        bits = []
        t = target
        while t:
            lsb = t & -t
            bits.append(lsb)
            t ^= lsb
        
        if not bits:
            outputs.append(n)
            continue
        
        if len(bits) == 1:
            idx = bits[0].bit_length() - 1
            outputs.append(idx)
            continue
        
        # Chain XOR
        current_wire = global_wire_map[bits[0]]
        current_mask = bits[0]
        
        for k in range(1, len(bits)):
            next_bit_mask = bits[k]
            next_bit_idx = next_bit_mask.bit_length() - 1
            
            in1 = current_wire
            in2 = next_bit_idx
            
            gates.append([in1, in2])
            
            new_wire = next_wire_id
            next_wire_id += 1
            
            new_mask = current_mask ^ next_bit_mask
            global_wire_map[new_mask] = new_wire
            
            current_wire = new_wire
            current_mask = new_mask
        
        outputs.append(current_wire)
    
    return {'gates': gates, 'outputs': outputs}

def _greedy_synthesize(rows, n):
    """
    Greedy construction to find a good upper bound.
    """
    targets = set(rows)
    targets.discard(0)
    
    available = set([1 << i for i in range(n)] + [0])
    gates = []
    wire_map = {1 << i: i for i in range(n)}
    wire_map[0] = n
    next_wire_id = n + 1
    
    while targets:
        best_gain = -1
        best_pair = None
        best_new_mask = None
        
        # Generate candidates
        avail_list = list(available)
        L = len(avail_list)
        
        for i in range(L):
            a = avail_list[i]
            for j in range(i + 1, L):
                b = avail_list[j]
                u = a ^ b
                if u in available:
                    continue
                
                # Calculate gain: how many targets does this help?
                # Simple metric: if u is a target, gain = 1.
                # Else, gain = 0 (or estimate based on popcount reduction?)
                gain = 0
                if u in targets:
                    gain = 100 # High priority to finish targets
                
                # Also, prefer u that is "simple" (low popcount) if not a target?
                # Or prefer u that shares bits with targets?
                # For now, just finish targets.
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
                    best_new_mask = u
        
        if best_pair is None:
            # No new wires can be generated? (Should not happen if n>0)
            break
            
        a, b = best_pair
        u = best_new_mask
        
        # Add gate
        gates.append([wire_map[a], wire_map[b]])
        wire_map[u] = next_wire_id
        next_wire_id += 1
        available.add(u)
        
        if u in targets:
            targets.remove(u)
            
        # If no target was hit, we might loop forever or generate useless wires.
        # We need a stopping condition or better heuristic.
        # If best_gain == 0, we are generating wires that don't directly complete a target.
        # This greedy is weak. We need a better one or just use baseline as UB.
        
    # If we didn't finish, fallback to baseline for the rest?
    # Actually, this greedy is too simple. Let's just use the baseline as the UB.
    # The Dijkstra will find the optimal if it's close.
    
    return len(gates), gates, wire_map

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    m, n = matrix.shape
    
    # Convert rows to bitmasks
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        rows.append(mask)
    
    unique_targets = set(rows)
    unique_targets.discard(0)
    
    if not unique_targets:
        return {'gates': [], 'outputs': [n] * m}
    
    # Initial state
    initial_available = frozenset([1 << i for i in range(n)] + [0])
    initial_wire_map = {1 << i: i for i in range(n)}
    initial_wire_map[0] = n
    
    # Get Upper Bound from Baseline
    baseline_res = xor_baseline(matrix, depth_limit)
    ub_gates = len(baseline_res['gates'])
    
    # Dijkstra / A*
    # State: (frozenset(available), gates_count)
    # We store the path to reconstruct.
    # To save memory, we store parent pointers.
    
    # Priority Queue: (heuristic + gates, gates, available_frozenset)
    # Heuristic: number of targets not in available
    def heuristic(available_fs):
        return len(unique_targets - available_fs)
    
    start_h = heuristic(initial_available)
    start_node = (start_h, 0, initial_available)
    
    pq = [start_node]
    
    # visited: available_fs -> min_gates
    visited = {initial_available: 0}
    
    # parent: available_fs -> (prev_available, gate_info)
    # gate_info: (mask_a, mask_b, new_mask)
    parent = {}
    
    best_solution = None
    best_cost = ub_gates
    
    while pq:
        if time.time() > deadline - 2.0:
            break
            
        h, g, avail_fs = heapq.heappop(pq)
        
        if g > visited.get(avail_fs, float('inf')):
            continue
            
        # Check if goal
        if unique_targets.issubset(avail_fs):
            if g < best_cost:
                best_cost = g
                # Reconstruct
                # We need to trace back
                # This is expensive. We can store the full path in the PQ?
                # No, memory.
                # Let's store the path in a separate dict or just return the baseline if Dijkstra is too slow to reconstruct.
                # Actually, for n=16, the path is short.
                # Let's store the list of gates in the node?
                # That blows up memory.
                
                # Alternative: Just use the baseline if we don't track parents properly.
                # Let's track parents.
                pass
            
            break # Found optimal (Dijkstra)
        
        # Generate neighbors
        avail_list = list(avail_fs)
        L = len(avail_list)
        
        for i in range(L):
            a = avail_list[i]
            for j in range(i + 1, L):
                b = avail_list[j]
                u = a ^ b
                if u in avail_fs:
                    continue
                
                new_avail_fs = avail_fs | {u}
                new_g = g + 1
                
                if new_g >= best_cost:
                    continue
                    
                if new_g < visited.get(new_avail_fs, float('inf')):
                    visited[new_avail_fs] = new_g
                    parent[new_avail_fs] = (avail_fs, (a, b, u))
                    new_h = heuristic(new_avail_fs)
                    heapq.heappush(pq, (new_h + new_g, new_g, new_avail_fs))
    
    # If we found a better solution, reconstruct it
    if best_cost < ub_gates:
        # Reconstruct path
        # We need to find the goal state in visited
        # Actually, we broke out of the loop when we popped a goal.
        # But we didn't store which goal state it was.
        # Let's modify the loop to store the best goal state.
        
        pass # This logic is flawed. Let's simplify.
        
    # Given the complexity of reconstruction and time limits, 
    # and the fact that baseline is a strong UB, 
    # we will just return the baseline if Dijkstra doesn't find a significantly better one quickly.
    # However, the task asks for improvement.
    
    # Let's try a simpler Beam Search with the baseline as UB.
    
    return baseline_res

# The above synthesize is a stub. Let's write the real one.

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    m, n = matrix.shape
    
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        rows.append(mask)
    
    unique_targets = set(rows)
    unique_targets.discard(0)
    
    if not unique_targets:
        return {'gates': [], 'outputs': [n] * m}
    
    initial_available = frozenset([1 << i for i in range(n)] + [0])
    
    # Baseline
    baseline_res = xor_baseline(matrix, depth_limit)
    ub_gates = len(baseline_res['gates'])
    
    # Beam Search
    beam_width = 200
    beam = [(0, initial_available, [])] # (gates_count, available_fs, gates_list)
    
    best_solution = None
    best_cost = ub_gates
    
    for depth in range(depth_limit):
        if time.time() > deadline - 2.0:
            break
            
        candidates = []
        
        for g_count, avail_fs, gates_list in beam:
            if g_count >= best_cost:
                continue
                
            avail_list = list(avail_fs)
            L = len(avail_list)
            
            for i in range(L):
                a = avail_list[i]
                for j in range(i + 1, L):
                    b = avail_list[j]
                    u = a ^ b
                    if u in avail_fs:
                        continue
                    
                    new_avail_fs = avail_fs | {u}
                    new_g_count = g_count + 1
                    
                    if new_g_count >= best_cost:
                        continue
                    
                    # Heuristic: prefer states with more targets covered
                    # Score: (g_count, -targets_covered)
                    targets_covered = len(unique_targets & new_avail_fs)
                    score = (new_g_count, -targets_covered)
                    
                    candidates.append((score, new_avail_fs, gates_list + [(a, b, u)]))
            
        if not candidates:
            break
            
        # Sort and dedup
        candidates.sort(key=lambda x: x[0])
        
        new_beam = []
        seen = set()
        
        for score, avail_fs, gates_list in candidates:
            if avail_fs in seen:
                continue
            seen.add(avail_fs)
            new_beam.append((len(gates_list), avail_fs, gates_list))
            
            if unique_targets.issubset(avail_fs):
                if len(gates_list) < best_cost:
                    best_cost = len(gates_list)
                    best_solution = (gates_list, avail_fs)
            
            if len(new_beam) >= beam_width:
                break
        
        beam = new_beam
    
    if best_solution:
        gates_list, final_avail = best_solution
        # Reconstruct wire map
        wire_map = {1 << i: i for i in range(n)}
        wire_map[0] = n
        next_wire = n + 1
        circuit_gates = []
        for a, b, u in gates_list:
            circuit_gates.append([wire_map[a], wire_map[b]])
            wire_map[u] = next_wire
            next_wire += 1
        
        outputs = [wire_map[r] for r in rows]
        return {'gates': circuit_gates, 'outputs': outputs}
    else:
        return baseline_res
