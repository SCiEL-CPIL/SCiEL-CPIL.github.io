import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish well before.
    # We use 85s as the hard stop for the main loop to allow for final verification.
    deadline = start_time + 85.0
    
    n = 512
    k = 4
    
    # Precompute all 4-term arithmetic progressions (4-APs) in [1, n]
    # Each AP is a tuple (a, b, c, d)
    # We store them as lists of indices for fast lookup
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_list.append((a, b, c, e))
            
    # For each element x, precompute which APs it belongs to.
    # ap_indices[x] is a list of indices into ap_list
    ap_indices = [[] for _ in range(n + 1)]
    for i, (a, b, c, e) in enumerate(ap_list):
        ap_indices[a].append(i)
        ap_indices[b].append(i)
        ap_indices[c].append(i)
        ap_indices[e].append(i)
        
    # Helper to check if a set (list of bools or set of ints) is valid
    # We represent the current state as a list of booleans for fast access
    # But for SA, we need to track conflicts efficiently.
    
    def count_conflicts(state):
        """
        state: list of bools, index 1..n
        Returns number of 4-APs fully contained in state.
        """
        conflicts = 0
        for i, (a, b, c, e) in enumerate(ap_list):
            if state[a] and state[b] and state[c] and state[e]:
                conflicts += 1
        return conflicts

    def generate_cantor_seed():
        """
        Generate a seed based on base-3 digits (Cantor set).
        Numbers with no '2' in base-3 representation are 3-free (and thus 4-free).
        """
        state = [False] * (n + 1)
        for i in range(1, n + 1):
            temp = i
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                state[i] = True
        return state

    def generate_random_seed(density=0.3):
        state = [False] * (n + 1)
        for i in range(1, n + 1):
            if random.random() < density:
                state[i] = True
        return state

    def generate_greedy_seed():
        """
        Simple greedy: add if no conflict.
        """
        state = [False] * (n + 1)
        for i in range(1, n + 1):
            # Check if adding i creates a conflict
            # We can do a fast check by looking at ap_indices[i]
            conflict = False
            for ap_idx in ap_indices[i]:
                a, b, c, e = ap_list[ap_idx]
                # If the other 3 are present, adding i creates a conflict
                # But wait, we are adding i. So we check if the others are in state.
                others = [x for x in (a, b, c, e) if x != i]
                if all(state[x] for x in others):
                    conflict = True
                    break
            if not conflict:
                state[i] = True
        return state

    def simulated_annealing(initial_state, start_time, deadline, target_conflicts=0):
        """
        Runs SA to minimize conflicts, then maximizes size among zero-conflict states.
        Actually, the objective is to maximize size subject to conflicts=0.
        A common trick is to minimize E = conflicts * BIG + (n - size).
        Or just minimize conflicts, and if 0, try to increase size.
        
        Let's use Energy E = conflicts * 1000 + (n - size).
        We want to minimize E.
        """
        BIG = 1000
        state = initial_state[:]
        current_size = sum(state[1:])
        current_conflicts = count_conflicts(state)
        current_energy = current_conflicts * BIG + (n - current_size)
        
        best_state = state[:]
        best_size = current_size
        best_energy = current_energy
        
        T = 100.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < deadline - 5.0:
            # Generate a neighbor
            # Move types:
            # 1. Flip a bit (add or remove)
            # 2. Swap (remove one, add another) - preserves size, helps escape local optima for size
            
            if random.random() < 0.8:
                # Flip
                x = random.randint(1, n)
                new_state = state[:]
                new_state[x] = not new_state[x]
                
                # Calculate delta energy efficiently?
                # count_conflicts is O(APs). APs ~ 512^2 / 3 ~ 87k.
                # 87k operations per step is too slow for SA in Python if we do many steps.
                # We need to be smarter or limit steps.
                
                # Let's estimate steps. 85s / 87k ops/step ~ 1000 steps? That's too few.
                # We need a faster conflict count update.
                
                # Let's optimize count_conflicts using the precomputed ap_indices.
                # Actually, maintaining a count of "partial APs" is complex.
                # Let's try a different move set or faster check.
                
                # Alternative: Use the bitset approach from inspirations but with SA logic?
                # The current program uses bitsets. Bit operations are fast.
                # Let's stick to the list of bools but optimize the conflict count.
                
                # Actually, 87k APs is not that many. 
                # 85 seconds / 87k = ~1000 steps is indeed too low for SA.
                # We need to reduce the cost of evaluation.
                
                # Let's precompute for each x, the APs it participates in.
                # When we flip x, only APs involving x can change their conflict status.
                # Number of APs involving x is roughly O(n) ~ 500.
                # So delta conflict calculation is O(500).
                # 85s / 500 = ~100,000 steps. This is feasible.
                
                # Let's implement delta conflict calculation.
                
                # We need to know the current conflict status of each AP.
                # Or just count conflicts among the affected APs.
                
                # Let's restructure:
                # Maintain state.
                # To evaluate new_state after flipping x:
                # 1. Identify all APs containing x.
                # 2. For each such AP, check if it was a conflict before, and is it a conflict now?
                #    - If it was conflict (all 4 in state) and now not (x removed), conflict -1.
                #    - If it was not conflict and now is (x added, others already in), conflict +1.
                #    - Otherwise no change.
                
                # This requires knowing if the AP was a conflict.
                # We can just check the other 3 elements in the AP.
                
                delta_conflicts = 0
                for ap_idx in ap_indices[x]:
                    a, b, c, e = ap_list[ap_idx]
                    others = [y for y in (a, b, c, e) if y != x]
                    
                    # Check if others are all in the OLD state
                    others_in_old = all(state[y] for y in others)
                    # Check if others are all in the NEW state (same as old, since x is the only change)
                    # Wait, if we flip x, the status of the AP depends on x and the others.
                    # Old conflict: state[x] and others_in_old
                    # New conflict: new_state[x] and others_in_old (others didn't change)
                    
                    old_was_conflict = state[x] and others_in_old
                    new_is_conflict = new_state[x] and others_in_old
                    
                    if old_was_conflict and not new_is_conflict:
                        delta_conflicts -= 1
                    elif not old_was_conflict and new_is_conflict:
                        delta_conflicts += 1
                    
                new_size = current_size + (1 if new_state[x] else -1)
                new_conflicts = current_conflicts + delta_conflicts
                new_energy = new_conflicts * BIG + (n - new_size)
                
                delta_E = new_energy - current_energy
                
                # Acceptance
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    state = new_state
                    current_size = new_size
                    current_conflicts = new_conflicts
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = state[:]
                        best_size = current_size
                        
            else:
                # Swap move: remove y, add x
                # This preserves size, so size term in energy is constant.
                # Only conflicts change.
                y = random.randint(1, n)
                if not state[y]:
                    # y is not in set, can't remove. Try flip instead or skip.
                    # Let's just do a flip if y is not in set.
                    new_state = state[:]
                    new_state[y] = True
                    # ... same logic as flip ...
                    # For simplicity, let's just stick to flips for now. 
                    # Swaps are harder to compute delta for efficiently without two delta calcs.
                    pass
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_state, best_size

    # Multi-start strategy
    best_global_state = None
    best_global_size = 0
    
    seeds = []
    # 1. Cantor set
    seeds.append(generate_cantor_seed())
    # 2. Greedy
    seeds.append(generate_greedy_seed())
    # 3. Random dense
    seeds.append(generate_random_seed(0.5))
    # 4. Random sparse
    seeds.append(generate_random_seed(0.2))
    
    # Allocate time to each seed
    # We have ~85s. 4 seeds. ~20s each?
    # Or run them in parallel? No, single thread.
    # Let's run each for a fixed time slice, keeping the best.
    
    time_per_seed = (deadline - start_time - 5.0) / len(seeds)
    
    for i, seed in enumerate(seeds):
        # Calculate initial size
        seed_size = sum(seed[1:])
        if seed_size > best_global_size:
            best_global_size = seed_size
            best_global_state = seed[:]
            
        # Run SA
        # We pass the deadline for this specific seed
        seed_deadline = start_time + 5.0 + (i + 1) * time_per_seed
        # Ensure we don't go over the global deadline
        if seed_deadline > deadline - 5.0:
            seed_deadline = deadline - 5.0
            
        # SA function needs to be modified to take a specific deadline
        # My current SA uses global deadline. Let's refactor slightly or just pass a local deadline.
        
        # Let's create a local SA runner
        def run_sa_local(state, local_deadline):
            BIG = 1000
            cur_state = state[:]
            cur_size = sum(cur_state[1:])
            cur_conflicts = count_conflicts(cur_state)
            cur_energy = cur_conflicts * BIG + (n - cur_size)
            
            best_local_state = cur_state[:]
            best_local_size = cur_size
            best_local_energy = cur_energy
            
            T = 100.0
            T_min = 0.1
            cooling = 0.9995
            
            while time.time() < local_deadline:
                x = random.randint(1, n)
                new_state = cur_state[:]
                new_state[x] = not new_state[x]
                
                delta_conflicts = 0
                for ap_idx in ap_indices[x]:
                    a, b, c, e = ap_list[ap_idx]
                    others = [y for y in (a, b, c, e) if y != x]
                    others_in = all(cur_state[y] for y in others)
                    
                    old_conf = cur_state[x] and others_in
                    new_conf = new_state[x] and others_in
                    
                    if old_conf and not new_conf:
                        delta_conflicts -= 1
                    elif not old_conf and new_conf:
                        delta_conflicts += 1
                        
                new_size = cur_size + (1 if new_state[x] else -1)
                new_conflicts = cur_conflicts + delta_conflicts
                new_energy = new_conflicts * BIG + (n - new_size)
                
                delta_E = new_energy - cur_energy
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    cur_state = new_state
                    cur_size = new_size
                    cur_conflicts = new_conflicts
                    cur_energy = new_energy
                    
                    if cur_energy < best_local_energy:
                        best_local_energy = cur_energy
                        best_local_state = cur_state[:]
                        best_local_size = cur_size
                        
                T *= cooling
                if T < T_min:
                    T = T_min
                    
            return best_local_state, best_local_size

        local_best_state, local_best_size = run_sa_local(seed, seed_deadline)
        
        if local_best_size > best_global_size:
            best_global_size = local_best_size
            best_global_state = local_best_state[:]

    # Final verification
    if best_global_state is None:
        # Fallback
        best_global_state = generate_greedy_seed()
        best_global_size = sum(best_global_state[1:])
        
    result = [i for i in range(1, n + 1) if best_global_state[i]]
    
    # Double check validity
    S_set = set(result)
    valid = True
    for a in S_set:
        for d in range(1, (n - a) // 3 + 1):
            if (a+d) in S_set and (a+2*d) in S_set and (a+3*d) in S_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: greedy
        S_mask = 0
        # Reuse greedy logic from seed program or simple greedy
        state = [False] * (n + 1)
        for i in range(1, n + 1):
            conflict = False
            for ap_idx in ap_indices[i]:
                a, b, c, e = ap_list[ap_idx]
                others = [x for x in (a, b, c, e) if x != i]
                if all(state[x] for x in others):
                    conflict = True
                    break
            if not conflict:
                state[i] = True
        result = [i for i in range(1, n + 1) if state[i]]
        
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
