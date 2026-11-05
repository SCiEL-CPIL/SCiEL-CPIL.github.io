import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    base = 51975
    # Generate divisors
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Filter out 1? The problem says m > 1.
    # 51975 is odd, so all divisors are odd.
    
    # Helper to calculate score
    def get_score(counts):
        return int(np.sum(counts == 0))
    
    # Helper to update counts for a move
    # Returns delta_score
    
    def simulate_change(counts, m, old_r, new_r):
        # old_r::m loses 1, new_r::m gains 1
        # Loss: indices in old_r::m with count 1
        # Gain: indices in new_r::m with count 0
        # Note: if old_r == new_r, delta is 0.
        if old_r == new_r:
            return 0
        loss = int(np.sum(counts[old_r::m] == 1))
        gain = int(np.sum(counts[new_r::m] == 0))
        return loss - gain

    def apply_change(counts, m, old_r, new_r):
        if old_r == new_r:
            return
        counts[old_r::m] -= 1
        counts[new_r::m] += 1

    def revert_change(counts, m, old_r, new_r):
        if old_r == new_r:
            return
        counts[new_r::m] -= 1
        counts[old_r::m] += 1

    def simulate_add(counts, m, r):
        # Gain: indices in r::m with count 0
        gain = int(np.sum(counts[r::m] == 0))
        return -gain # Score decreases by gain

    def apply_add(counts, m, r):
        counts[r::m] += 1

    def simulate_remove(counts, m, r):
        # Loss: indices in r::m with count 1
        loss = int(np.sum(counts[r::m] == 1))
        return loss # Score increases by loss

    def apply_remove(counts, m, r):
        counts[r::m] -= 1

    def greedy_init(order):
        counts = np.zeros(base, dtype=np.int8)
        classes = []
        used = set()
        for m in order:
            if time.time() > deadline - 5:
                break
            best_r = 0
            best_gain = -1
            # Sample residues
            if m <= 50:
                candidates = range(m)
            else:
                candidates = [random.randint(0, m-1) for _ in range(20)]
                candidates.append(0)
            
            for r in candidates:
                gain = int(np.sum(counts[r::m] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m, best_r])
                apply_add(counts, m, best_r)
                used.add(m)
        return classes, counts

    def simulated_annealing(classes, counts, T_init, T_final, alpha, max_iters):
        current_score = get_score(counts)
        best_score = current_score
        best_classes = [c[:] for c in classes]
        best_counts = counts.copy()
        
        used_moduli = set(m for m, _ in classes)
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        T = T_init
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            if T < T_final:
                break
                
            # Choose a move
            move_type = random.random()
            accepted = False
            delta = 0
            move_info = None
            
            if move_type < 0.6 and classes:
                # Change residue
                idx = random.randint(0, len(classes) - 1)
                m, old_r = classes[idx]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                delta = simulate_change(counts, m, old_r, new_r)
                move_info = ('change', idx, m, old_r, new_r)
                
            elif move_type < 0.8 and unused_moduli:
                # Add class
                m = random.choice(unused_moduli)
                r = random.randint(0, m - 1)
                delta = simulate_add(counts, m, r)
                move_info = ('add', m, r)
                
            else:
                # Remove class
                if not classes:
                    continue
                idx = random.randint(0, len(classes) - 1)
                m, r = classes[idx]
                delta = simulate_remove(counts, m, r)
                move_info = ('remove', idx, m, r)
                
            if move_info is None:
                continue
                
            # Acceptance criterion
            if delta < 0:
                accepted = True
            else:
                # Metropolis criterion
                if random.random() < np.exp(-delta / T):
                    accepted = True
            
            if accepted:
                # Apply move
                if move_info[0] == 'change':
                    _, idx, m, old_r, new_r = move_info
                    apply_change(counts, m, old_r, new_r)
                    classes[idx][1] = new_r
                elif move_info[0] == 'add':
                    _, m, r = move_info
                    apply_add(counts, m, r)
                    classes.append([m, r])
                    used_moduli.add(m)
                    unused_moduli.remove(m)
                elif move_info[0] == 'remove':
                    _, idx, m, r = move_info
                    apply_remove(counts, m, r)
                    classes.pop(idx)
                    used_moduli.remove(m)
                    unused_moduli.append(m)
                    
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                    best_counts = counts.copy()
                    
            T *= alpha
            
        return best_classes, best_counts, best_score

    best_classes = []
    best_score = base
    best_counts = None

    # Phase 1: Greedy Initialization + SA
    # Try a few different greedy orders
    orders = [
        sorted(divisors),
        sorted(divisors, reverse=True),
        divisors[:],
        divisors[:]
    ]
    
    for i, order in enumerate(orders):
        if i > 0:
            random.shuffle(order)
            
        if time.time() > deadline - 60:
            break
            
        classes, counts = greedy_init(order)
        sc = get_score(counts)
        
        # Run SA
        # Parameters: T_init, T_final, alpha, max_iters
        # max_iters should be adjusted based on time remaining
        time_left = deadline - time.time()
        max_iters = int(time_left * 1000) # Rough estimate
        T_init = 10.0
        T_final = 0.01
        alpha = 0.999
        
        classes_sa, counts_sa, sc_sa = simulated_annealing(classes, counts, T_init, T_final, alpha, max_iters)
        
        if sc_sa < best_score:
            best_score = sc_sa
            best_classes = classes_sa[:]
            best_counts = counts_sa.copy()
            
    # Phase 2: Final Polish (Hill Climbing on best result)
    # Sometimes SA stops at a non-optimal local min due to cooling.
    # Do a quick hill climb to ensure we are at a local optimum.
    if best_classes and time.time() < deadline - 10:
        # Simple hill climb
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try changing residues
            for i in range(len(best_classes)):
                m, r = best_classes[i]
                best_r = r
                best_sc = best_score
                if m <= 20:
                    candidates = range(m)
                else:
                    candidates = [random.randint(0, m-1) for _ in range(10)]
                    candidates.append(r)
                
                for new_r in candidates:
                    if new_r == r: continue
                    delta = simulate_change(best_counts, m, r, new_r)
                    if delta < 0:
                        # Apply
                        apply_change(best_counts, m, r, new_r)
                        best_classes[i][1] = new_r
                        best_score += delta
                        best_sc = best_score
                        best_r = new_r
                        improved = True
                        break
                # If we found an improvement, break to restart loop with new state
                if improved:
                    break
            # Try adding
            used = set(m for m, _ in best_classes)
            unused = [m for m in divisors if m not in used]
            if unused:
                random.shuffle(unused)
                for m in unused:
                    if time.time() > deadline - 5:
                        break
                    best_r = 0
                    best_gain = 0
                    if m <= 20:
                        candidates = range(m)
                    else:
                        candidates = [random.randint(0, m-1) for _ in range(10)]
                    for r in candidates:
                        gain = int(np.sum(best_counts[r::m] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    if best_gain > 0:
                        apply_add(best_counts, m, best_r)
                        best_classes.append([m, best_r])
                        best_score -= best_gain
                        improved = True
                        break
            # Try removing
            if len(best_classes) > 1:
                for i in range(len(best_classes)):
                    m, r = best_classes[i]
                    loss = int(np.sum(best_counts[r::m] == 1))
                    if loss == 0:
                        apply_remove(best_counts, m, r)
                        best_classes.pop(i)
                        # Score unchanged
                        improved = True
                        break

    if not best_classes:
        best_classes = [[3, 0]]
        
    return best_classes
