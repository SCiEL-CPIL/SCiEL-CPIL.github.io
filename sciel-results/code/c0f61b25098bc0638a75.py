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
    
    # Precompute bit masks for each (m, r) pair
    # mask[m][r] is a Python int with bits set for x where x % m == r
    masks = {}
    for m in divisors:
        masks[m] = []
        for r in range(m):
            mask = 0
            for x in range(r, N, m):
                mask |= (1 << x)
            masks[m].append(mask)
    
    all_ones = (1 << N) - 1
    
    def calc_score(classes):
        covered = 0
        for m, r in classes:
            covered |= masks[m][r]
        return bin(all_ones ^ covered).count('1')
    
    def greedy_set_cover():
        """Greedy set cover: repeatedly pick the class covering most uncovered residues."""
        uncovered = all_ones
        classes = []
        used_moduli = set()
        
        while uncovered and time.time() < deadline - 5:
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in divisors:
                if m in used_moduli:
                    continue
                for r in range(m):
                    gain = bin(uncovered & masks[m][r]).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            
            if best_m is None:
                break
            
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            uncovered &= ~masks[best_m][best_r]
        
        return classes, bin(uncovered).count('1')
    
    def local_search(classes, deadline):
        """Local search using bitset operations for speed."""
        classes = [c[:] for c in classes]
        current_score = calc_score(classes)
        improved = True
        passes = 0
        max_passes = 30
        
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
                covered_wo = 0
                for j, (m_j, r_j) in enumerate(classes):
                    if j != i:
                        covered_wo |= masks[m_j][r_j]
                
                uncov_wo = all_ones ^ covered_wo
                old_unique = bin(uncov_wo & masks[m_old][r_old]).count('1')
                
                best_r_new = r_old
                best_gain = old_unique
                
                for r_new in range(m_old):
                    gain = bin(uncov_wo & masks[m_old][r_new]).count('1')
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
                    covered_wo = 0
                    for j, (m_j, r_j) in enumerate(classes):
                        if j != i:
                            covered_wo |= masks[m_j][r_j]
                    
                    uncov_wo = all_ones ^ covered_wo
                    old_unique = bin(uncov_wo & masks[m_old][r_old]).count('1')
                    
                    best_swap = None
                    best_swap_gain = old_unique
                    
                    for m_new in unused_moduli:
                        best_r_new = 0
                        best_gain_new = -1
                        for r_new in range(m_new):
                            gain = bin(uncov_wo & masks[m_new][r_new]).count('1')
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
                        used_moduli.discard(m_old)
                        used_moduli.add(best_swap[0])
                        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        return classes, current_score
    
    def final_polish(classes, deadline):
        """Final high-precision polish pass."""
        classes = [c[:] for c in classes]
        current_score = calc_score(classes)
        
        # Try all residue changes
        for i in range(len(classes)):
            if time.time() > deadline - 2:
                break
            m, r = classes[i]
            covered_wo = 0
            for j, (m_j, r_j) in enumerate(classes):
                if j != i:
                    covered_wo |= masks[m_j][r_j]
            uncov_wo = all_ones ^ covered_wo
            old_unique = bin(uncov_wo & masks[m][r]).count('1')
            best_r = r
            best_gain = old_unique
            for r_new in range(m):
                gain = bin(uncov_wo & masks[m][r_new]).count('1')
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
                covered_wo = 0
                for j, (m_j, r_j) in enumerate(classes):
                    if j != i:
                        covered_wo |= masks[m_j][r_j]
                uncov_wo = all_ones ^ covered_wo
                old_unique = bin(uncov_wo & masks[m_old][r_old]).count('1')
                best_swap = None
                best_swap_gain = old_unique
                for m_new in unused_moduli:
                    best_r_new = 0
                    best_gain_new = -1
                    for r_new in range(m_new):
                        gain = bin(uncov_wo & masks[m_new][r_new]).count('1')
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
    
    # Strategy 1: Pure greedy set cover
    classes_greedy, score_greedy = greedy_set_cover()
    if score_greedy < best_score:
        best_score = score_greedy
        best_classes = classes_greedy
    
    # Strategy 2: Random restarts with greedy + local search
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline - 15:
        num_restarts += 1
        
        # Start from greedy set cover result, then perturb
        classes, score = greedy_set_cover()
        
        # Randomly remove some classes and re-greedy
        if len(classes) > 3:
            classes_copy = [c[:] for c in classes]
            # Remove 1-3 random classes
            num_remove = min(3, len(classes_copy) - 1)
            indices_to_remove = random.sample(range(len(classes_copy)), num_remove)
            indices_to_remove.sort(reverse=True)
            for idx in indices_to_remove:
                classes_copy.pop(idx)
            classes = classes_copy
        
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
