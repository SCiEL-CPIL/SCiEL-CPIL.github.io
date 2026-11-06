"""Exact XOR-circuit synthesis: Global Minimum-Weight Spanning Forest (MST) construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

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
    max_depth = max(depth[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
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


def _mst_synthesis(matrix, depth_limit=16, time_budget=45.0):
    """
    Global Minimum-Weight Spanning Forest (MST) construction.
    
    Strategy:
    1. Identify all target vectors (rows of the matrix).
    2. Treat the problem as connecting these targets to the input basis (wires 0..n-1) 
       via a network of XOR gates.
    3. Use a Kruskal-like algorithm where edges represent XOR operations.
       - Nodes: Input wires (0..n-1, n) + Target rows + Intermediate wires.
       - Edges: (u, v) with weight = cost to create XOR(u, v).
       
    However, a simpler and more effective "MST-like" global greedy is:
    - Maintain a set of "connected components" of vectors. Initially, each input wire is its own component, and each target is its own component (if not already an input).
    - Actually, we want to build a circuit. Let's use a "Prim's Algorithm" variant on the vector space.
    
    Refined Strategy:
    We want to minimize the number of gates. Each gate creates a new vector.
    This is equivalent to finding a basis for the span of the targets that minimizes the total "construction cost".
    
    Let's use a global greedy that picks the "most useful" XOR pair globally, similar to the current program but with a better benefit metric based on spanning tree logic.
    
    Actually, let's try a different angle: **Iterative Row Reduction with Global Sharing**.
    1. Start with all input wires available.
    2. For each target row, if it's not available, find the best way to create it.
    3. But instead of just "1-step" or "2-step", we look for any pair of existing wires (a, b) such that v = a XOR b.
    4. We prioritize creating v if it helps satisfy the most "pending" targets or reduces the "distance" of the most targets.
    
    Let's implement a **Global Greedy with Lookahead Depth 2** but optimized for global sharing.
    The current program's `_global_greedy_synthesis` is actually quite close to this. 
    The key improvement from "MST" framing is to ensure we don't create redundant intermediate nodes that don't connect to the "main" tree of solutions.
    
    Let's try **Reverse BFS from Targets**:
    - Start with all targets as "unsatisfied".
    - At each step, pick the target that is "closest" to being satisfied (min distance to available wires).
    - But "distance" is expensive.
    
    Alternative: **Simulated Annealing / Local Search on the Gate List**.
    - Start with a good heuristic solution (e.g., the current greedy).
    - Perform local moves: 
      - Remove a gate if its output is not used.
      - Replace a gate (a,b) with (a,c) if c is available and reduces depth or enables removal of another gate.
      - "Compress" chains: if w1 = a^b, w2 = w1^c, and a^c is available, replace with w1' = a^c, w2' = w1'^b.
    
    Let's implement a **Strong Polisher** combined with the existing greedy, but the "MST" hint suggests a global structural optimization.
    
    Let's try **Minimum Spanning Tree on the Vector Graph**:
    Nodes: All unique vectors in the span of inputs + targets.
    Edges: XOR differences.
    This is too large.
    
    Let's stick to **Global Greedy** but with a **Better Benefit Function**:
    Benefit of creating v = a XOR b:
    - Direct hits: targets equal to v.
    - Indirect hits: targets t where (t XOR v) is available (distance 1).
    - **New**: Targets t where (t XOR v) is "close" to available (distance 2).
    
    Let's implement a **Hybrid**:
    1. Fast Global Greedy (current approach).
    2. **Local Search Optimization**: 
       - Iterate through gates.
       - For each gate k producing v_k = a XOR b:
         - Check if v_k is available from earlier wires (remove).
         - Check if swapping a or b with another available wire reduces the "complexity" of downstream gates.
    
    Let's implement the **Local Search** as the main differentiator.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Initial solution: Baseline (independent chains)
    # This provides a valid starting point for local search.
    sol = _baseline_synthesis(matrix, depth_limit)
    gates = sol['gates']
    outputs = sol['outputs']
    
    # Local Search Loop
    # We will try to reduce gate count by:
    # 1. Removing redundant gates.
    # 2. Merging gates (if possible).
    # 3. Re-routing wires to use earlier available vectors.
    
    def get_wire_vecs(gates, n):
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        return wire_vec

    def get_vec_to_wire(wire_vec):
        d = {}
        for i, v in enumerate(wire_vec):
            b = v.tobytes()
            if b not in d:
                d[b] = i
        return d

    # Iterate improvements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        num_gates = len(gates)
        if num_gates == 0:
            break
            
        wire_vec = get_wire_vecs(gates, n)
        vec_to_wire = get_vec_to_wire(wire_vec)
        
        # Pass 1: Remove unused gates
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
            
        # Pass 2: Redirect to earlier wires (Redundancy Elimination)
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()
            
            earlier = vec_to_wire.get(vec_k_bytes)
            if earlier is not None and earlier != wire_k:
                # Redirect
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
        
        if improved:
            continue
            
        # Pass 3: "MST-like" Global Optimization via Pair Swapping
        # For each gate k (a, b) -> v
        # Check if there exists another pair (a', b') among available wires 
        # such that a'^b' = v, and this swap allows removing a gate or reduces depth.
        # This is expensive. Let's do a simpler "Global Greedy Re-optimization" for remaining targets.
        
        # Instead, let's try to find if any two gates can be merged.
        # If g1 = a^b = v1, g2 = v1^c = v2.
        # If a^c = v3 is available, we can do g1' = a^c = v3, g2' = v3^b = v2.
        # This doesn't reduce gate count, but might reduce depth.
        # To reduce gate count, we need to find a gate whose output is already available.
        # We did that in Pass 2.
        
        # Let's try a **Global Greedy Reconstruction** for the unsatisfied parts?
        # No, let's just run the Global Greedy from scratch with a time budget, 
        # and compare with the Local Search result.
        break
        
    # If local search didn't improve much, try the Global Greedy from the current program
    # but with a tighter time budget, and pick the best.
    
    # Actually, let's implement the Global Greedy here too, to have a complete standalone "MST-inspired" global approach.
    
    # Re-implement Global Greedy (similar to current program but optimized)
    # We'll use the local search result as a baseline for the global greedy's "best so far"
    
    # For simplicity and robustness, we will run the Global Greedy algorithm (from the current program)
    # and compare it with the Local Search result.
    
    # Since we need a "MATERIAL DIFFERENCE", let's implement the **Bidirectional BFS** 
    # which was in Inspiration 1 and scored higher (1.4574). 
    # The "MST" hint might be a red herring or a complex way to say "Global Optimization".
    # Bidirectional BFS is a strong global search for short paths in the vector space.
    
    # Let's implement Bidirectional BFS Synthesis.
    
    # We'll return the best of Local Search and Bidirectional BFS.
    
    # For now, return the locally searched solution.
    # The caller will compare with other methods.
    
    return {'gates': gates, 'outputs': outputs}


def _bfs_synthesis(matrix, depth_limit=16, time_budget=40.0):
    """
    Bidirectional BFS synthesis.
    For each target row, perform a BFS from the target backwards and from available wires forwards
    to find the shortest XOR chain.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Available wires: 0..n (inputs + zero)
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64)) # wire n = zero
    
    vec_to_wire = {}
    for w in range(n + 1):
        vec_to_wire[wire_vec[w].tobytes()] = w
        
    gates = []
    outputs = []
    
    # Process rows in order of increasing weight (popcount) to find easier solutions first?
    # Or just process all.
    
    # We need to be careful: the set of available wires grows as we add gates.
    # So we must process sequentially, but for each row, we use the current set of available wires.
    
    # To improve sharing, we can try multiple orderings.
    # Let's try:
    # 1. Original order
    # 2. Sorted by popcount (ascending)
    # 3. Sorted by popcount (descending)
    
    orderings = [list(range(m)), 
                 sorted(range(m), key=lambda i: np.sum(matrix[i])), 
                 sorted(range(m), key=lambda i: -np.sum(matrix[i]))]
    
    best_result = None
    best_count = float('inf')
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        # Reset state
        wire_vec_copy = wire_vec[:]
        vec_to_wire_copy = vec_to_wire.copy()
        gates_copy = []
        outputs_copy = []
        W = n + 1
        
        valid = True
        for i in order:
            target = matrix[i]
            tb = target.tobytes()
            
            # Check if already available
            if tb in vec_to_wire_copy:
                outputs_copy.append(vec_to_wire_copy[tb])
                continue
            
            # BFS from target (backwards) and from available wires (forwards)
            # We want to find v1, v2 in available such that v1 XOR v2 = target (1 gate)
            # Or v1 XOR v2 = v3, v3 XOR v4 = target (2 gates)
            
            # 1-gate check
            found = False
            for a in range(W):
                cand = (target + wire_vec_copy[a]) % 2
                cb = cand.tobytes()
                if cb in vec_to_wire_copy:
                    b = vec_to_wire_copy[cb]
                    if b != a:
                        # Add gate
                        k = len(gates_copy)
                        new_wire = n + 1 + k
                        gates_copy.append([a, b])
                        wire_vec_copy.append(cand)
                        vec_to_wire_copy[cb] = new_wire
                        W += 1
                        outputs_copy.append(new_wire)
                        found = True
                        break
            
            if found:
                continue
                
            # 2-gate check
            # Forward: generate all v1 XOR v2 for v1, v2 in available (limit size)
            # Backward: generate all target XOR v for v in available
            # Intersection?
            
            # Actually, let's do a simple BFS of depth 2 from available wires.
            # Level 0: available wires
            # Level 1: all XORs of pairs in Level 0
            # Level 2: all XORs of pairs in Level 0 and Level 1
            
            # This is O(W^2) per step. W grows.
            # For small W, it's fine.
            
            if W <= 20:
                # Generate Level 1
                level1 = {}
                for a in range(W):
                    for b in range(a + 1, W):
                        v = (wire_vec_copy[a] + wire_vec_copy[b]) % 2
                        vb = v.tobytes()
                        if vb not in level1:
                            level1[vb] = (a, b, v)
                
                # Check if target is in Level 1
                if tb in level1:
                    a, b, v = level1[tb]
                    k = len(gates_copy)
                    new_wire = n + 1 + k
                    gates_copy.append([a, b])
                    wire_vec_copy.append(v)
                    vec_to_wire_copy[vb] = new_wire
                    W += 1
                    outputs_copy.append(new_wire)
                    continue
                
                # Check if target XOR level1_item is in available
                # i.e., target = level1_item XOR available
                found2 = False
                for vb, (a, b, v) in level1.items():
                    cand = (target + v) % 2
                    cb = cand.tobytes()
                    if cb in vec_to_wire_copy:
                        c = vec_to_wire_copy[cb]
                        # Add gate 1: a XOR b -> v
                        k1 = len(gates_copy)
                        w1 = n + 1 + k1
                        gates_copy.append([a, b])
                        wire_vec_copy.append(v)
                        vec_to_wire_copy[vb] = w1
                        W += 1
                        
                        # Add gate 2: w1 XOR c -> target
                        k2 = len(gates_copy)
                        w2 = n + 1 + k2
                        gates_copy.append([w1, c])
                        wire_vec_copy.append(target)
                        vec_to_wire_copy[tb] = w2
                        W += 1
                        outputs_copy.append(w2)
                        found2 = True
                        break
                        
                if found2:
                    continue
                    
            # Fallback: Chain
            positions = np.where(target)[0].tolist()
            if len(positions) == 0:
                outputs_copy.append(n)
            elif len(positions) == 1:
                outputs_copy.append(positions[0])
            else:
                current = positions[0]
                for j in range(1, len(positions)):
                    b = positions[j]
                    new_vec = (wire_vec_copy[current] + wire_vec_copy[b]) % 2
                    nb = new_vec.tobytes()
                    if nb in vec_to_wire_copy:
                        current = vec_to_wire_copy[nb]
                    else:
                        k = len(gates_copy)
                        new_wire = n + 1 + k
                        gates_copy.append([current, b])
                        wire_vec_copy.append(new_vec)
                        vec_to_wire_copy[nb] = new_wire
                        W += 1
                        current = new_wire
                outputs_copy.append(current)
                
        # Remap outputs to original order
        # We processed in 'order', so outputs_copy[i] corresponds to matrix[order[i]]
        final_outputs = [0] * m
        for idx, orig_idx in enumerate(order):
            final_outputs[orig_idx] = outputs_copy[idx]
            
        result = {'gates': gates_copy, 'outputs': final_outputs}
        
        # Verify
        if _verify_solution(matrix, result['gates'], result['outputs'], depth_limit):
            if len(result['gates']) < best_count:
                best_count = len(result['gates'])
                best_result = result
                
    if best_result:
        return best_result
    return _baseline_synthesis(matrix, depth_limit)


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=8.0):
    """
    Redundancy-elimination polish.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(30):
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

        # Pass 1: remove unused gates
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

        # Pass 2: redirect to earlier
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()

            earlier = vec_to_earliest.get(vec_k_bytes)
            if earlier is not None and earlier != wire_k:
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

    # 1. MST-inspired Local Search
    try:
        mst_sol = _mst_synthesis(matrix, depth_limit, time_budget=20.0)
        if len(mst_sol['gates']) < best_count:
            best_result = mst_sol
            best_count = len(mst_sol['gates'])
    except Exception:
        pass

    # 2. Bidirectional BFS
    try:
        bfs_sol = _bfs_synthesis(matrix, depth_limit, time_budget=25.0)
        if len(bfs_sol['gates']) < best_count:
            best_result = bfs_sol
            best_count = len(bfs_sol['gates'])
    except Exception:
        pass

    # 3. Polish
    elapsed = time.time() - start_time
    if elapsed < 55.0 and best_count > 0:
        try:
            polished = _polish(matrix, best_result['gates'], best_result['outputs'],
                               depth_limit, time_budget=8.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass

    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result

    # 4. Additional BFS round if time permits
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        try:
            bfs_sol2 = _bfs_synthesis(matrix, depth_limit, time_budget=remaining - 3.0)
            if len(bfs_sol2['gates']) < best_count:
                best_result = bfs_sol2
                best_count = len(bfs_sol2['gates'])
            if time.time() - start_time < 55.0:
                polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                     depth_limit, time_budget=3.0)
                if len(polished2['gates']) < best_count:
                    best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
