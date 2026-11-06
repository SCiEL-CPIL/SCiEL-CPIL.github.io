import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute APs and incidence structure
    ap_indices = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Build reverse index: for each index i, which APs contain it?
    aps_containing = [[] for _ in range(N)]
    for j, terms in enumerate(ap_indices):
        for i in terms:
            aps_containing[i].append(j)
    
    # Convert to numpy arrays for faster access if needed, but lists are fine for small counts
    # Precompute lengths? Not strictly needed for this strategy.

    def compute_initial_sums(signs):
        S = np.zeros(num_aps, dtype=np.int32)
        for j, terms in enumerate(ap_indices):
            S[j] = np.sum(signs[terms])
        return S

    def get_hard_max(S):
        return int(np.max(np.abs(S)))

    def get_secondary_metric(S, H):
        # Minimize the number of APs at the maximum, then at H-1, etc.
        # Using sum of squares of abs(S) for S >= H-1 is a good proxy for "tightness"
        abs_S = np.abs(S)
        # Focus on the "tail" of the distribution near the max
        mask = abs_S >= (H - 1)
        if np.any(mask):
            return np.sum(abs_S[mask] ** 2)
        return 0.0

    def polish(signs, S, rng, time_limit):
        """Aggressive local search polish"""
        H = get_hard_max(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try flipping each index in random order
            order = list(range(N))
            rng.shuffle(order)
            for i in order:
                old_sign = signs[i]
                delta_s = -2.0 * old_sign
                affected = aps_containing[i]
                S[affected] += delta_s
                
                new_H = get_hard_max(S)
                if new_H < H:
                    signs[i] = -old_sign
                    H = new_H
                    improved = True
                elif new_H == H:
                    # Check secondary metric
                    old_sec = get_secondary_metric(S - delta_s, H) # Restore old state to compare? 
                    # Actually, we already modified S. Let's compute sec for new state.
                    new_sec = get_secondary_metric(S, H)
                    # We need old_sec. Let's store it before modifying S in the loop? 
                    # Better: compute old_sec before modification.
                    # To avoid double computation, we can just accept if new_sec < old_sec.
                    # Let's restructure: compute old_sec outside.
                    pass # This logic is slightly flawed in-place. Let's fix in main loop.
                else:
                    S[affected] -= delta_s
                    continue
                
                if improved:
                    break # Restart order if H improved? Or just continue?
                    # Restarting is safer to ensure we catch all benefits of new H.
        return signs, S, H

    # Improved Polish:
    def polish_v2(signs, S, rng, time_limit):
        H = get_hard_max(S)
        old_sec = get_secondary_metric(S, H)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            order = list(range(N))
            rng.shuffle(order)
            for i in order:
                old_sign = signs[i]
                delta_s = -2.0 * old_sign
                affected = aps_containing[i]
                
                # Save old state for affected APs to allow easy rollback
                old_vals = S[affected].copy()
                
                S[affected] += delta_s
                
                new_H = get_hard_max(S)
                if new_H < H:
                    signs[i] = -old_sign
                    H = new_H
                    old_sec = get_secondary_metric(S, H)
                    improved = True
                    break # Restart
                elif new_H == H:
                    new_sec = get_secondary_metric(S, H)
                    if new_sec < old_sec:
                        signs[i] = -old_sign
                        old_sec = new_sec
                        improved = True
                        # Don't break, try to improve further within same H
                    else:
                        # Rollback
                        S[affected] = old_vals
                else:
                    # Rollback
                    S[affected] = old_vals
        return signs, S, H

    best_signs = None
    best_hard_max = N
    best_sec = 0.0

    def thue_morse():
        return np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)], dtype=np.int8)

    def random_init(rng):
        return np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)

    # 2. Generate diverse initial solutions
    candidates = []
    candidates.append(thue_morse())
    for s in [42, 123, 456, 789, 1000, 2000]:
        rng = random.Random(s)
        candidates.append(random_init(rng))

    for signs in candidates:
        S = compute_initial_sums(signs)
        hmax = get_hard_max(S)
        sec = get_secondary_metric(S, hmax)
        if hmax < best_hard_max or (hmax == best_hard_max and sec < best_sec):
            best_hard_max = hmax
            best_sec = sec
            best_signs = signs.copy()
        if time.time() > deadline - 10:
            break

    # 3. Tabu Search
    # Parameters
    tabu_len = 50
    tabu_list = [0] * tabu_len
    tabu_counter = [0] * tabu_len
    tabu_idx = 0
    
    rng = random.Random(42)
    signs = best_signs.copy()
    S = compute_initial_sums(signs)
    current_H = get_hard_max(S)
    current_sec = get_secondary_metric(S, current_H)
    
    no_improve_steps = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5:
        # Pick random index
        i = rng.randint(0, N - 1)
        
        # Check Tabu
        is_tabu = tabu_counter[tabu_idx] > 0 # This is not quite right, need to check if i is in tabu
        # Simpler Tabu: store the last `tabu_len` flipped indices
        # Let's use a set for O(1) lookup, or a list of (index, expiration)
        
        # Let's switch to a simpler Tabu implementation:
        # tabu_set = set()
        # Each step:
        #   if i in tabu_set: check aspiration
        #   else: accept/reject
        #   add i to tabu_set, remove oldest
        
        # Re-implementing Tabu loop properly
        if not hasattr(construct, '_tabu_set'):
            construct._tabu_set = set()
            construct._tabu_queue = []
        
        tabu_set = construct._tabu_set
        tabu_queue = construct._tabu_queue
        
        aspiration = False
        if i in tabu_set:
            # Aspiration: accept if it improves global best
            # We don't know new_H yet. We can only check aspiration after computing new_H.
            pass
        
        old_sign = signs[i]
        delta_s = -2.0 * old_sign
        affected = aps_containing[i]
        
        old_vals = S[affected].copy()
        S[affected] += delta_s
        
        new_H = get_hard_max(S)
        
        accept = False
        if new_H < best_hard_max:
            accept = True
            aspiration = True
        elif new_H == best_hard_max:
            new_sec = get_secondary_metric(S, new_H)
            if new_sec < best_sec:
                accept = True
                aspiration = True
            elif new_H == current_H:
                # Stay in same H, try to improve current state
                if new_sec < current_sec:
                    accept = True
        elif new_H < current_H:
            # Improved current state, but not global best
            accept = True
        
        # Tabu Check
        if i in tabu_set and not aspiration:
            accept = False
            
        if accept:
            signs[i] = -old_sign
            current_H = new_H
            current_sec = get_secondary_metric(S, new_H)
            
            if current_H < best_hard_max or (current_H == best_hard_max and current_sec < best_sec):
                best_hard_max = current_H
                best_sec = current_sec
                best_signs = signs.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            # Update Tabu
            tabu_set.add(i)
            tabu_queue.append(i)
            if len(tabu_queue) > tabu_len:
                old_i = tabu_queue.pop(0)
                tabu_set.remove(old_i)
                
        else:
            S[affected] = old_vals
            no_improve_steps += 1
            
        # Periodic Polish if stuck
        if no_improve_steps > max_no_improve and time.time() < deadline - 10:
            signs, S, current_H = polish_v2(signs, S, rng, time.time() + 2)
            current_sec = get_secondary_metric(S, current_H)
            if current_H < best_hard_max or (current_H == best_hard_max and current_sec < best_sec):
                best_hard_max = current_H
                best_sec = current_sec
                best_signs = signs.copy()
            no_improve_steps = 0
            # Reset tabu
            construct._tabu_set.clear()
            construct._tabu_queue.clear()

    # 4. Final Polish
    if best_signs is not None and time.time() < deadline - 2:
        best_signs, S, _ = polish_v2(best_signs.copy(), compute_initial_sums(best_signs), random.Random(999), deadline - 1)
        final_hmax = get_hard_max(S)
        if final_hmax < best_hard_max:
            best_hard_max = final_hmax

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
