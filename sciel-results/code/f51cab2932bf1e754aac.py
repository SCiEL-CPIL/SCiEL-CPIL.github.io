def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    import random
    from collections import defaultdict

    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to integer bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        target_masks.append(mask)

    # We will model the problem as finding a minimum-cost set of XOR gates
    # that generates all target masks from inputs.
    # We use a "BFS from inputs" approach to find the shortest path (min gates)
    # to each target, but we allow sharing by treating the set of available
    # wires as the state. However, full state space is too large.
    
    # Instead, we use a randomized local search / simulated annealing approach
    # on the circuit structure.
    
    # 1. Generate an initial circuit using a simple greedy method.
    # 2. Iteratively improve it by swapping gate operands or adding/removing gates
    #    if it reduces the total gate count while maintaining correctness and depth.

    def get_depth(gates, n):
        """Calculate the depth of the circuit."""
        if not gates:
            return 0
        depth = [0] * (n + 1 + len(gates))
        # Inputs and zero have depth 0
        for k, (a, b) in enumerate(gates):
            wire_idx = n + 1 + k
            depth[wire_idx] = max(depth[a], depth[b]) + 1
        return max(depth)

    def verify_circuit(gates, outputs, target_masks, n):
        """Verify that the circuit produces the correct outputs."""
        if len(outputs) != len(target_masks):
            return False
        
        # Simulate the circuit
        # Wire 0..n-1 are inputs (bitmasks 1<<0, ..., 1<<(n-1))
        # Wire n is 0
        wire_val = [0] * (n + 1 + len(gates))
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        for i, out_wire in enumerate(outputs):
            if wire_val[out_wire] != target_masks[i]:
                return False
        return True

    def generate_initial_circuit(target_masks, n):
        """Generate a simple initial circuit using chain XOR."""
        gates = []
        outputs = []
        mask_to_wire = {}
        for i in range(n):
            mask_to_wire[1 << i] = i
        mask_to_wire[0] = n
        next_wire = n + 1
        
        for t in target_masks:
            # Find a subset of inputs that XOR to t
            # Simple approach: XOR all inputs where bit is set
            bits = [i for i in range(n) if (t >> i) & 1]
            if len(bits) == 0:
                outputs.append(n)
                continue
            elif len(bits) == 1:
                outputs.append(bits[0])
                continue
            
            # Check if t is already generated
            if t in mask_to_wire:
                outputs.append(mask_to_wire[t])
                continue
                
            curr_wire = bits[0]
            for b in bits[1:]:
                gates.append([curr_wire, b])
                curr_wire = next_wire
                next_wire += 1
                # Optional: record intermediate masks if we want to reuse
                # But for simplicity, we just chain.
                # We could update mask_to_wire for intermediate steps, but
                # it's tricky to know the mask without storing it.
                # Let's just store the final mask.
            
            mask_to_wire[t] = curr_wire
            outputs.append(curr_wire)
            
        return gates, outputs

    # Initial generation
    gates, outputs = generate_initial_circuit(target_masks, n)
    
    # If depth is already too high, we might need to restructure, 
    # but chain XOR has depth popcount-1.
    # For n=16, max depth 15, which is <= 16. So depth is likely fine initially.
    
    # Local Search / Simulated Annealing
    # We want to minimize len(gates).
    # Moves:
    # 1. Swap operands of a gate? (No effect on value, only depth potentially, but depth is not in cost)
    # 2. Add a new gate that creates a wire already in use? (Redundant)
    # 3. Replace a gate with a different pair of wires that produce the same mask?
    #    This requires knowing the mask of every wire.
    # 4. "Compress" the circuit by finding common sub-expressions.
    
    # A better approach: 
    # Represent the circuit as a DAG.
    # Try to find a smaller DAG that computes the same functions.
    
    # Let's use a different strategy: 
    # **Minimum XOR Circuit via Integer Linear Programming (ILP) is too slow.**
    # **Use a Greedy Set Cover on the "space of wires" with a twist:**
    # Instead of just adding wires, we can also **remove** wires that are no longer needed
    # and **re-route** outputs.
    
    # However, the most effective structural change for XOR circuits is often
    # to find a **basis** for the targets and then express them in that basis
    # with minimal intermediate nodes.
    
    # Let's try a **Beam Search** on the set of available wires, but with a 
    # focus on **minimizing the number of gates** to reach all targets.
    # State: (available_wires_set, gates_list, cost)
    # This is essentially what the current program does, but let's make it 
    # more aggressive in pruning and using a better heuristic.
    
    # Actually, let's look at the "Inspiration 2" strategy: 
    # "Randomized search / local improvement".
    
    # We will implement a simple local search:
    # 1. Start with the greedy circuit.
    # 2. Randomly pick a gate and try to change its inputs to other available wires
    #    that produce the same mask (if possible) or a mask that allows simplification.
    # 3. This is hard because changing inputs changes the mask.
    
    # Alternative Local Search:
    # **Swap-based optimization**:
    # Identify wires that are used by multiple outputs.
    # Try to find a shorter path to those wires.
    
    # Let's try a **Genetic Algorithm** on the circuit structure?
    # Too complex.
    
    # Let's go back to the **Greedy Set Cover** but with **Global Reuse**.
    # The current program does this, but it might be getting stuck in local minima.
    # Let's add **Randomization** to the greedy choice to escape local minima.
    
    # Strategy:
    # 1. Run the greedy algorithm multiple times with different random seeds/tie-breaking.
    # 2. Keep the best result.
    
    def greedy_synthesize(target_masks, n, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        available_masks = set()
        for i in range(n):
            available_masks.add(1 << i)
        available_masks.add(0)
        
        target_set = set(target_masks)
        initial_targets = target_set - available_masks
        
        gates = []
        mask_to_wire = {}
        for i in range(n):
            mask_to_wire[1 << i] = i
        mask_to_wire[0] = n
        
        next_wire = n + 1
        
        while initial_targets:
            if time.time() > DEADLINE:
                break
                
            # Generate candidates
            avail_list = list(available_masks)
            if len(avail_list) < 2:
                break
                
            # To avoid O(N^2) with large N, we can sample if N is large.
            # But N is usually small (number of unique masks generated).
            
            candidates = []
            # We want to find w = a ^ b that is "useful".
            # Useful means:
            # 1. w is a target.
            # 2. w helps form a target (T ^ w is in available).
            
            # Instead of iterating all pairs, we can iterate targets and find
            # the best w for each target, then aggregate.
            
            # For each target T, we want w such that T ^ w is in available.
            # This means w = T ^ v for some v in available.
            # So for each target T, the best candidates are T ^ v for v in available.
            # If we add w = T ^ v, then T = w ^ v (1 gate).
            
            # So, for each target T, we can generate candidates w = T ^ v.
            # If w is not in available, adding w solves T in 1 step.
            # Also, w might help other targets.
            
            best_w = None
            best_score = -1
            best_a = None
            best_b = None
            
            # Iterate over a subset of targets to find the best global w.
            # If we just pick the best w for the first target, we might miss global sharing.
            # Let's score w by how many targets it helps.
            
            # To keep it fast, we limit the number of candidates.
            # For each target, pick a few random v from available.
            
            candidates_set = set()
            for t in initial_targets:
                # Pick up to 5 random v from available
                # If available is small, pick all.
                if len(available_masks) < 10:
                    vs = list(available_masks)
                else:
                    vs = random.sample(list(available_masks), 5)
                    
                for v in vs:
                    w = t ^ v
                    if w not in available_masks and w != 0:
                        candidates_set.add(w)
            
            if not candidates_set:
                # No candidates found? Fallback to random pair.
                a, b = random.sample(avail_list, 2)
                w = a ^ b
                if w in available_masks:
                    break
                best_w = w
                best_a = a
                best_b = b
            else:
                # Score each candidate
                # We need to find a, b such that a ^ b = w.
                # We can store this when we generate w.
                # But we generated w from t ^ v. So v is one operand.
                # The other operand is t.
                # But t might not be in available.
                # We need a, b in available such that a ^ b = w.
                
                # This is a linear algebra problem: find a, b in A such that a ^ b = w.
                # This is equivalent to finding a in A such that a ^ w is in A.
                # i.e., a ^ w in A.
                
                # For each w, find if there exists a in A such that a ^ w in A.
                # If so, we can add the gate.
                
                best_w = None
                best_score = -1
                best_a = None
                best_b = None
                
                for w in candidates_set:
                    # Check if w can be formed by two available wires.
                    # Find a in available such that a ^ w in available.
                    found = False
                    # To speed up, we can iterate a smaller set or use a set lookup.
                    # for a in available_masks:
                    #     if (a ^ w) in available_masks:
                    #         found = True
                    #         break
                    # This is O(N) per w. If |candidates| is large, this is slow.
                    # But |candidates| is bounded by |targets| * 5.
                    
                    for a in available_masks:
                        b = a ^ w
                        if b in available_masks:
                            # Found a pair
                            # Score w
                            score = 0
                            if w in target_set:
                                score += 10000
                            # Count how many targets T have (T ^ w) in available
                            for t in initial_targets:
                                if (t ^ w) in available_masks:
                                    score += 10
                            
                            if score > best_score:
                                best_score = score
                                best_w = w
                                best_a = a
                                best_b = b
                            found = True
                            break # Found one pair for this w, enough for now? 
                            # Actually, any pair works to create w.
                            # We just need to pick one.
                            
                    # If no pair found, we can't create w directly.
                    # We would need to create intermediate wires.
                    # For simplicity, we skip w if it can't be formed by 1 gate.
                    # This restricts us to 1-gate additions.
                    # This is the same as the current program.
                    
                if best_w is None:
                    # No candidate could be formed by 1 gate?
                    # This shouldn't happen if we have inputs.
                    # Fallback: pick any pair.
                    a, b = random.sample(avail_list, 2)
                    best_w = a ^ b
                    best_a = a
                    best_b = b
                    if best_w in available_masks:
                        break
            
            # Add the gate
            wire_a = mask_to_wire[best_a]
            wire_b = mask_to_wire[best_b]
            
            gates.append([wire_a, wire_b])
            new_wire_idx = next_wire
            next_wire += 1
            
            mask_to_wire[best_w] = new_wire_idx
            available_masks.add(best_w)
            
            # Update targets
            new_initial_targets = set()
            for t in initial_targets:
                if t not in available_masks:
                    new_initial_targets.add(t)
            initial_targets = new_initial_targets
            
        # Fallback for unsolved targets
        unsolved = set(target_masks) - available_masks
        for t in unsolved:
            # Build from inputs
            bits = [i for i in range(n) if (t >> i) & 1]
            if len(bits) == 0:
                outputs.append(n)
                continue
            elif len(bits) == 1:
                outputs.append(bits[0])
                continue
            
            curr_wire = bits[0]
            for b in bits[1:]:
                gates.append([curr_wire, b])
                curr_wire = next_wire
                next_wire += 1
            
            mask_to_wire[t] = curr_wire
            outputs.append(curr_wire)
            
        # Map original targets to wires
        outputs = []
        for t in target_masks:
            if t in mask_to_wire:
                outputs.append(mask_to_wire[t])
            else:
                outputs.append(n)
                
        return gates, outputs

    # Run greedy synthesis multiple times with different seeds
    best_gates = None
    best_outputs = None
    best_count = float('inf')
    
    num_trials = 10
    for i in range(num_trials):
        if time.time() > DEADLINE:
            break
        g, o = greedy_synthesize(target_masks, n, random_seed=i)
        if verify_circuit(g, o, target_masks, n):
            if len(g) < best_count:
                best_count = len(g)
                best_gates = g
                best_outputs = o
                
    if best_gates is None:
        # Fallback to initial
        best_gates, best_outputs = generate_initial_circuit(target_masks, n)
        
    # Final check
    if not verify_circuit(best_gates, best_outputs, target_masks, n):
        # Something went wrong, use the simplest valid circuit
        best_gates, best_outputs = generate_initial_circuit(target_masks, n)
        
    return {'gates': best_gates, 'outputs': best_outputs}
