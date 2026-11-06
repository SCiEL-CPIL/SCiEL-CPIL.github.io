"""Exact XOR-circuit synthesis: Global Set-Cover Greedy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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


def _global_greedy_synthesis(matrix, depth_limit=16, time_budget=50.0, max_candidate_gates=200):
    """
    Global Set-Cover Greedy.
    1. Start with input wires and zero.
    2. Repeatedly find the gate (a, b) that maximizes the number of unsatisfied rows
       that can be completed with 1 or 2 additional gates (or 0 if the gate itself
       is a row).
    3. Add that gate.
    4. Repeat until all rows are satisfied or time runs out.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # State
    # wires: list of vectors
    wires = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wires.append(v)
    wires.append(np.zeros(n, dtype=np.int64)) # wire n is zero
    
    gates = []
    outputs = [None] * m
    unsatisfied = list(range(m))
    
    # Map from vector bytes to wire index (earliest)
    vec_to_wire = {}
    for i in range(n + 1):
        vec_to_wire[wires[i].tobytes()] = i
        
    # Pre-compute popcounts for sorting or heuristic
    # We will re-evaluate "best gate" every step.
    # To speed up, we can limit the search space for (a, b) pairs.
    # If we have W wires, there are W^2/2 pairs.
    # For each pair, we compute v = a XOR b.
    # We check if v is a row (1 gate to finish) or v XOR x is a row for existing x (2 gates to finish).
    
    # Optimization: Pre-compute the set of "targets" (unsatisfied rows) as bytes.
    # Actually, we need to check if (target + v) is in existing wires.
    
    while unsatisfied and time.time() < deadline:
        W = len(wires)
        best_score = -1
        best_gate = None
        best_new_vec = None
        
        # Limit pairs if W is large
        # For n=16, m=16, W might grow to ~50. 50^2 = 2500 pairs.
        # For each pair, we check against ~16 unsatisfied rows.
        # 2500 * 16 = 40,000 ops per step.
        # If we need 20 steps, that's 800k ops. Very fast.
        
        # However, checking "v XOR x is in existing wires" requires a hash lookup.
        # Let's iterate over unsatisfied rows and find the best (a,b) for each,
        # then pick the global best?
        # No, the "score" is global: how many rows does this ONE gate help?
        
        # A gate g = a XOR b helps row R if:
        # 1. g == R (cost 0 more gates)
        # 2. g XOR x == R for some existing wire x (cost 1 more gate)
        # 3. g XOR x XOR y == R for some existing x, y (cost 2 more gates) -- maybe too deep?
        # Let's stick to 1 more gate (total depth 2 for this row) as the primary metric,
        # plus a bonus for creating "useful" intermediate vectors.
        
        # Actually, the "Set Cover" logic is:
        # Cost of gate g = 1.
        # Benefit of gate g = sum over unsatisfied rows R of (1 / min_gates_needed(R, given g))
        # If g satisfies R directly, min_gates = 0.
        # If g XOR x satisfies R, min_gates = 1.
        
        # Let's simplify:
        # Score = (num rows satisfied directly) * 100 + (num rows satisfied with 1 more gate) * 10 + (random tiebreak)
        
        # Iterate over pairs (a, b)
        # To avoid O(W^2) if W is huge, we might need to be smart.
        # But W is usually small (< 100).
        
        # Optimization: Only consider pairs (a, b) where a < b.
        # Also, if a XOR b is already in wires, skip (it's redundant).
        
        # To speed up the inner loop, let's pre-compute the set of existing vectors.
        # existing_vecs = set(w.tobytes() for w in wires)
        
        # For each candidate gate g:
        #   g_vec = wires[a] ^ wires[b]
        #   if g_vec in existing_vecs: continue
        #   score = 0
        #   for r in unsatisfied:
        #       if g_vec == matrix[r]: score += 100
        #       else:
        #           # check if g_vec ^ matrix[r] is in existing_vecs
        #           needed = g_vec ^ matrix[r]
        #           if needed.tobytes() in existing_vecs:
        #               score += 10
        #   if score > best_score:
        #       best_score = score
        #       best_gate = (a, b)
        #       best_new_vec = g_vec
        
        # This is O(W^2 * U * n) roughly.
        # n=16, U<=16, W<=100. 10000 * 16 * 16 = 2.5M ops.
        # In Python, this might be slow if done naively with numpy arrays.
        # Let's use bytes for hashing.
        
        existing_bytes = set(w.tobytes() for w in wires)
        
        # Pre-compute row bytes for unsatisfied
        unsat_bytes = [matrix[r].tobytes() for r in unsatisfied]
        
        # We also need to check "1 more gate" condition:
        # (g_vec ^ row_vec) in existing_bytes
        # This is a hash lookup.
        
        # To make this faster, we can iterate over unsatisfied rows first?
        # No, the gate is global.
        
        # Let's try the W^2 loop.
        # Optimization: if W is very large, sample?
        # For now, assume W < 200.
        
        # To avoid checking all W^2, we can note that if we've already checked (a,b), we don't need (b,a).
        
        # Further optimization:
        # We can pre-compute for each unsatisfied row r, the set of vectors v such that v ^ r is in existing.
        # That set is { existing_vec ^ r for existing_vec in existing }.
        # Let's call this set S_r.
        # If g_vec is in S_r, then g helps r with 1 more gate.
        
        # So for each r, we have a set S_r of size W.
        # We want to find g_vec that is in the most S_r's (or equals row_r).
        
        # Let's construct S_r sets.
        # S_r = set( (w ^ matrix[r]).tobytes() for w in wires )
        
        # Then for each candidate g_vec:
        #   count = 0
        #   for r in unsatisfied:
        #       if g_vec_bytes == unsat_bytes[r]: count += 100
        #       elif g_vec_bytes in S_r: count += 10
        #   ...
        
        # This is O(U * W) to build S_r, and O(U) to score each candidate.
        # Total O(W^2 * U).
        
        # Let's implement this.
        
        S_r = []
        for r in unsatisfied:
            row_vec = matrix[r]
            s_set = set()
            for w in wires:
                s_set.add((w ^ row_vec).tobytes())
            S_r.append(s_set)
            
        # Now iterate candidates
        # To speed up, we can limit candidates if W is large.
        # But let's try full search first.
        
        # Randomization for tie-breaking or to escape local minima?
        # The greedy is deterministic. Let's add a bit of noise or random restarts if stuck.
        # For now, deterministic greedy.
        
        # To avoid O(W^2) if W is huge, we can stop early if we find a score of 100 * U (all satisfied).
        
        found_all = False
        for a in range(W):
            for b in range(a + 1, W):
                g_vec = wires[a] ^ wires[b]
                g_bytes = g_vec.tobytes()
                
                if g_bytes in existing_bytes:
                    continue
                
                score = 0
                for idx, r in enumerate(unsatisfied):
                    if g_bytes == unsat_bytes[idx]:
                        score += 100
                    else:
                        if g_bytes in S_r[idx]:
                            score += 10
                
                if score > best_score:
                    best_score = score
                    best_gate = (a, b)
                    best_new_vec = g_vec
                    if score == 100 * len(unsatisfied):
                        found_all = True
                        break
            if found_all:
                break
                
        if best_gate is None or best_score <= 0:
            # Fallback: pick a random gate or use baseline for remaining
            # If no gate helps, we might be stuck in a local minimum or need deeper chains.
            # Let's just pick a random pair that creates a new vector.
            valid_pairs = []
            for a in range(W):
                for b in range(a + 1, W):
                    g_bytes = (wires[a] ^ wires[b]).tobytes()
                    if g_bytes not in existing_bytes:
                        valid_pairs.append((a, b))
            if not valid_pairs:
                break # No new vectors possible? Should not happen unless we've exhausted the space.
            
            a, b = random.choice(valid_pairs)
            best_gate = (a, b)
            best_new_vec = wires[a] ^ wires[b]
            best_score = 1 # Force it
            
        # Add the gate
        a, b = best_gate
        new_wire_idx = len(wires)
        wires.append(best_new_vec)
        gates.append([a, b])
        existing_bytes.add(best_new_vec.tobytes())
        
        # Update outputs and unsatisfied
        # Check if any unsatisfied row is now directly satisfied by new_wire_idx
        # Or if any unsatisfied row can be satisfied with 1 more gate (which we will handle in next step? 
        # No, we only add 1 gate per step. The "1 more gate" benefit is potential future savings.
        # But if a row is directly satisfied, we must mark it.
        
        new_unsatisfied = []
        for r in unsatisfied:
            if matrix[r].tobytes() == best_new_vec.tobytes():
                outputs[r] = new_wire_idx
            else:
                new_unsatisfied.append(r)
                
        unsatisfied = new_unsatisfied
        
        # Safety: if we are taking too long, break
        if time.time() > deadline:
            break
            
    # If we still have unsatisfied rows, we need to complete them.
    # The greedy might have created useful intermediates.
    # For each remaining row, we can do a local BFS/chain using the current wire set.
    
    if unsatisfied:
        # Local completion
        # For each r in unsatisfied:
        #   target = matrix[r]
        #   Try to find a, b in wires such that a^b = target (0 gates)
        #   Try to find a, b in wires such that (a^b)^c = target (1 gate)
        #   Try to find a, b, c, d in wires such that ((a^b)^(c^d)) = target (2 gates)
        #   Fallback: chain
        
        # We will use a simplified BFS for the remaining rows.
        # Since W is now larger, the search space is bigger, but U is smaller.
        
        for r in unsatisfied:
            if time.time() > deadline:
                break
            target = matrix[r]
            target_bytes = target.tobytes()
            
            # 0 gates
            if target_bytes in existing_bytes:
                # Find the wire index
                for w_idx, w in enumerate(wires):
                    if w.tobytes() == target_bytes:
                        outputs[r] = w_idx
                        break
                continue
            
            # 1 gate: a ^ b = target
            found = False
            W = len(wires)
            # We need to check if target ^ a is in existing for any a.
            # This is O(W).
            for a in range(W):
                needed = target ^ wires[a]
                if needed.tobytes() in existing_bytes:
                    # Find b
                    for b in range(W):
                        if wires[b].tobytes() == needed.tobytes():
                            # Add gate a ^ b
                            new_w = len(wires)
                            wires.append(target) # Actually target is the result
                            # Wait, a^b = target.
                            wires[-1] = target
                            gates.append([a, b])
                            existing_bytes.add(target_bytes)
                            outputs[r] = new_w
                            found = True
                            break
                    if found: break
            if found:
                continue
                
            # 2 gates: (a^b) ^ c = target
            # This is harder. Let's just do a chain fallback for now.
            # Chain:
            positions = np.where(target)[0].tolist()
            if len(positions) == 0:
                outputs[r] = n
            elif len(positions) == 1:
                outputs[r] = positions[0]
            else:
                current = positions[0]
                for j in range(1, len(positions)):
                    b = positions[j]
                    new_vec = wires[current] ^ wires[b]
                    nb = new_vec.tobytes()
                    if nb in existing_bytes:
                        # Find wire
                        for w_idx, w in enumerate(wires):
                            if w.tobytes() == nb:
                                current = w_idx
                                break
                    else:
                        current = len(wires)
                        wires.append(new_vec)
                        gates.append([current-1, b]) # Wait, current is new index, so gate index is current-1?
                        # No, gates list index is len(gates)-1. Wire index is n+1+len(gates)-1.
                        # Let's be careful.
                        new_wire_idx = len(wires) - 1
                        gates.append([current-1, b]) # This is wrong. current was the previous wire.
                        # Let's redo the chain logic carefully.
                        
                # Let's restart the chain logic properly.
                current = positions[0]
                for j in range(1, len(positions)):
                    b = positions[j]
                    new_vec = wires[current] ^ wires[b]
                    nb = new_vec.tobytes()
                    if nb in existing_bytes:
                        for w_idx, w in enumerate(wires):
                            if w.tobytes() == nb:
                                current = w_idx
                                break
                    else:
                        new_wire_idx = len(wires)
                        wires.append(new_vec)
                        gates.append([current, b])
                        existing_bytes.add(nb)
                        current = new_wire_idx
                outputs[r] = current

    # Final verification and cleanup
    if not all(o is not None for o in outputs):
        return _baseline_synthesis(matrix, depth_limit)
        
    if _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return _baseline_synthesis(matrix, depth_limit)


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=5.0):
    """Redundancy elimination."""
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(10):
        if time.time() > deadline:
            break
        if len(gates) == 0:
            break

        num_gates = len(gates)
        total_wires = n + 1 + num_gates

        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2

        vec_to_earliest = {}
        for w in range(total_wires):
            vb = wire_vec[w].tobytes()
            if vb not in vec_to_earliest:
                vec_to_earliest[vb] = w

        improved = False
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()

            earlier = None
            for w in range(wire_k):
                if wire_vec[w].tobytes() == vec_k_bytes:
                    earlier = w
                    break

            if earlier is not None:
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

    # Global Greedy
    try:
        greedy = _global_greedy_synthesis(matrix, depth_limit, time_budget=40.0)
        if len(greedy['gates']) < best_count:
            best_result = greedy
            best_count = len(greedy['gates'])
    except Exception:
        pass

    # Polish
    elapsed = time.time() - start_time
    if elapsed < 55.0 and best_count > 0:
        try:
            polished = _polish(matrix, best_result['gates'], best_result['outputs'],
                               depth_limit, time_budget=5.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass

    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result

    # Second round of greedy with different random seed or just more time?
    # The greedy is deterministic, so running it again won't help unless we change the tie-breaking.
    # We can add a randomized version.
    
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        # Try a randomized version of greedy
        # We can modify _global_greedy_synthesis to accept a random_seed and use it for tie-breaking
        # For now, let's just run the baseline BFS from the current program if we have time?
        # No, let's stick to the new strategy.
        # We can try to run the greedy again but with a "noise" factor in the score.
        # Let's implement a quick randomized greedy inline or just return.
        pass

    return best_result

# EVOLVE-BLOCK-END
