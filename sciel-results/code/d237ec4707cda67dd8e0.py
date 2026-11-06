import time
import random

def construct():
    N = 51975
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Precompute all divisors > 1 of 51975
    divisors = []
    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    if N not in divisors:
        divisors.append(N)
    divisors.sort()
    
    # Helper: compute score (uncovered count) using bytearray
    def calc_score(classes):
        covered = bytearray(N)
        for m, r in classes:
            for x in range(r, N, m):
                covered[x] = 1
        return covered.count(0)
    
    # Helper: greedy construction with given order
    def greedy(order):
        covered = bytearray(N)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = 0
                for x in range(r, N, m):
                    if covered[x] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            for x in range(best_r, N, m):
                covered[x] = 1
        return classes
    
    # Helper: local search - coordinate descent on residues and swaps
    def local_search(classes, deadline):
        classes = [c[:] for c in classes]
        current_score = calc_score(classes)
        improved = True
        passes = 0
        max_passes = 20
        
        while improved and passes < max_passes and time.time() < deadline - 5:
            improved = False
            passes += 1
            used_moduli = set(m for m, r in classes)
            
            # Phase 1: Try changing residue for each class
            for i in range(len(classes)):
                if time.time() > deadline - 5:
                    break
                m_old, r_old = classes[i]
                
                # Compute covered without class i
                covered_wo = bytearray(N)
                for j, (m_j, r_j) in enumerate(classes):
                    if j != i:
                        for x in range(r_j, N, m_j):
                            covered_wo[x] = 1
                
                uncov_wo = covered_wo.count(0)
                # Old unique coverage
                old_unique = 0
                for x in range(r_old, N, m_old):
                    if covered_wo[x] == 0:
                        old_unique += 1
                
                best_r_new = r_old
                best_gain = old_unique
                
                for r_new in range(m_old):
                    gain = 0
                    for x in range(r_new, N, m_old):
                        if covered_wo[x] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r_new = r_new
                
                if best_r_new != r_old:
                    classes[i][1] = best_r_new
                    current_score -= (best_gain - old_unique)
                    improved = True
            
            # Phase 2: Try swapping a class with an unused modulus
            if time.time() < deadline - 10:
                unused_moduli = [m for m in divisors if m not in used_moduli]
                for i in range(len(classes)):
                    if time.time() > deadline - 5:
                        break
                    m_old, r_old = classes[i]
                    
                    # Compute covered without class i
                    covered_wo = bytearray(N)
                    for j, (m_j, r_j) in enumerate(classes):
                        if j != i:
                            for x in range(r_j, N, m_j):
                                covered_wo[x] = 1
                    
                    uncov_wo = covered_wo.count(0)
                    old_unique = 0
                    for x in range(r_old, N, m_old):
                        if covered_wo[x] == 0:
                            old_unique += 1
                    
                    best_swap = None
                    best_swap_gain = old_unique
                    
                    for m_new in unused_moduli:
                        best_r_new = 0
                        best_gain_new = -1
                        for r_new in range(m_new):
                            gain = 0
                            for x in range(r_new, N, m_new):
                                if covered_wo[x] == 0:
                                    gain += 1
                            if gain > best_gain_new:
                                best_gain_new = gain
                                best_r_new = r_new
                        
                        if best_gain_new > best_swap_gain:
                            best_swap_gain = best_gain_new
                            best_swap = (m_new, best_r_new)
                    
                    if best_swap and best_swap_gain > old_unique:
                        classes[i] = [best_swap[0], best_swap[1]]
                        current_score -= (best_swap_gain - old_unique)
                        improved = True
                        # Update used_moduli
                        used_moduli.discard(m_old)
                        used_moduli.add(best_swap[0])
                        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        return classes, current_score
    
    # Final polish: try all single-residue changes and single swaps one more time
    def final_polish(classes, deadline):
        classes = [c[:] for c in classes]
        current_score = calc_score(classes)
        
        # Try all residue changes
        for i in range(len(classes)):
            if time.time() > deadline - 2:
                break
            m, r = classes[i]
            covered_wo = bytearray(N)
            for j, (m_j, r_j) in enumerate(classes):
                if j != i:
                    for x in range(r_j, N, m_j):
                        covered_wo[x] = 1
            uncov_wo = covered_wo.count(0)
            old_unique = 0
            for x in range(r, N, m):
                if covered_wo[x] == 0:
                    old_unique += 1
            best_r = r
            best_gain = old_unique
            for r_new in range(m):
                gain = 0
                for x in range(r_new, N, m):
                    if covered_wo[x] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r_new
            if best_r != r:
                classes[i][1] = best_r
                current_score -= (best_gain - old_unique)
        
        # Try all swaps
        if time.time() < deadline - 2:
            used_moduli = set(m for m, r in classes)
            unused_moduli = [m for m in divisors if m not in used_moduli]
            for i in range(len(classes)):
                if time.time() > deadline - 2:
                    break
                m_old, r_old = classes[i]
                covered_wo = bytearray(N)
                for j, (m_j, r_j) in enumerate(classes):
                    if j != i:
                        for x in range(r_j, N, m_j):
                            covered_wo[x] = 1
                uncov_wo = covered_wo.count(0)
                old_unique = 0
                for x in range(r_old, N, m_old):
                    if covered_wo[x] == 0:
                        old_unique += 1
                best_swap = None
                best_swap_gain = old_unique
                for m_new in unused_moduli:
                    best_r_new = 0
                    best_gain_new = -1
                    for r_new in range(m_new):
                        gain = 0
                        for x in range(r_new, N, m_new):
                            if covered_wo[x] == 0:
                                gain += 1
                        if gain > best_gain_new:
                            best_gain_new = gain
                            best_r_new = r_new
                    if best_gain_new > best_swap_gain:
                        best_swap_gain = best_gain_new
                        best_swap = (m_new, best_r_new)
                if best_swap and best_swap_gain > old_unique:
                    classes[i] = [best_swap[0], best_swap[1]]
                    current_score -= (best_swap_gain - old_unique)
        
        return classes, current_score
    
    # Main loop
    best_classes = []
    best_score = N
    
    # Strategy 1: Greedy with increasing order
    classes_inc = greedy(divisors)
    score_inc = calc_score(classes_inc)
    if score_inc < best_score:
        best_score = score_inc
        best_classes = classes_inc
    
    # Strategy 2: Greedy with decreasing order
    classes_dec = greedy(divisors[::-1])
    score_dec = calc_score(classes_dec)
    if score_dec < best_score:
        best_score = score_dec
        best_classes = classes_dec
    
    # Strategy 3: Random restarts with greedy + local search
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline - 15:
        num_restarts += 1
        order = divisors[:]
        random.shuffle(order)
        classes = greedy(order)
        score = calc_score(classes)
        
        # Local search
        if time.time() < deadline - 10:
            classes, score = local_search(classes, deadline)
        
        if score < best_score:
            best_score = score
            best_classes = classes
            if best_score == 0:
                break
    
    # Final polish on the best solution
    if time.time() < deadline - 3 and best_classes:
        best_classes, best_score = final_polish(best_classes, deadline)
    
    return best_classes
