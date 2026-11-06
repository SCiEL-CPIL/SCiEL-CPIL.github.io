import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Generate divisors of 51975 (odd divisors > 1)
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    
    # Precompute residue indices for each modulus
    residue_indices = {}
    for m in divisors:
        residues = []
        for r in range(m):
            residues.append(np.arange(r, N, m))
        residue_indices[m] = residues

    def compute_uncovered(covered):
        return np.count_nonzero(~covered)

    def find_best_residue(m, covered):
        """Find the residue r for modulus m that covers the most uncovered residues."""
        max_count = 0
        best_r = 0
        for r in range(m):
            c = np.count_nonzero(~covered[residue_indices[m][r]])
            if c > max_count:
                max_count = c
                best_r = r
        return best_r, max_count

    def greedy_initialization(order=None):
        """Greedy set cover initialization."""
        covered = np.zeros(N, dtype=bool)
        classes = []
        if order is None:
            available_moduli = divisors.copy()
        else:
            available_moduli = order.copy()
        
        while available_moduli:
            if time.time() > deadline:
                break
                
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in available_moduli:
                max_count = 0
                best_r_m = 0
                for r in range(m):
                    c = np.count_nonzero(~covered[residue_indices[m][r]])
                    if c > max_count:
                        max_count = c
                        best_r_m = r
                
                if max_count > best_gain:
                    best_gain = max_count
                    best_m = m
                    best_r = best_r_m
            
            if best_m is None or best_gain == 0:
                break
                
            classes.append([best_m, best_r])
            covered[best_r::best_m] = True
            available_moduli.remove(best_m)
            
            if np.all(covered):
                break

        return classes, covered

    def coordinate_descent(classes, covered, max_iter=10):
        """Coordinate descent: for each selected m, try to find a better r."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            current_uncovered = compute_uncovered(covered)
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                m = classes[i][0]
                r_old = classes[i][1]
                
                # Temporarily remove the current residue class
                covered[r_old::m] = False
                
                # Find best new r for this modulus
                r_new, gain_new = find_best_residue(m, covered)
                
                # Apply new r
                covered[r_new::m] = True
                uncovered_after_add = compute_uncovered(covered)
                
                # Check if improvement
                if uncovered_after_add < current_uncovered:
                    classes[i][1] = r_new
                    current_uncovered = uncovered_after_add
                    improved = True
                else:
                    # Revert to original
                    covered[r_new::m] = False
                    covered[r_old::m] = True
                
                if current_uncovered == 0:
                    return classes, covered

        return classes, covered

    def swap_search(classes, covered, unselected_moduli, max_swaps=50):
        """Try to swap a selected modulus with an unselected one."""
        selected_moduli = {c[0] for c in classes}
        improved = False
        current_uncovered = compute_uncovered(covered)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            best_swap_gain = 0
            best_m_out = None
            best_m_in = None
            best_r_in = None
            best_idx = None
            
            for i, (m_out, r_out) in enumerate(classes):
                if time.time() > deadline:
                    break
                
                # Temporarily remove m_out
                covered[r_out::m_out] = False
                new_uncovered = compute_uncovered(covered)
                lost = new_uncovered - current_uncovered
                
                # Try each unselected modulus
                for m_in in unselected_moduli:
                    r_in, gain_in = find_best_residue(m_in, covered)
                    net_gain = gain_in - lost
                    if net_gain > best_swap_gain:
                        best_swap_gain = net_gain
                        best_m_out = m_out
                        best_m_in = m_in
                        best_r_in = r_in
                        best_idx = i
                
                # Revert
                covered[r_out::m_out] = True
            
            if best_m_out is not None and best_swap_gain > 0:
                # Perform swap
                classes[best_idx][0] = best_m_in
                classes[best_idx][1] = best_r_in
                covered[best_r_in::best_m_in] = True
                
                # Update moduli sets
                selected_moduli.remove(best_m_out)
                selected_moduli.add(best_m_in)
                unselected_moduli.remove(best_m_in)
                unselected_moduli.append(best_m_out)
                
                current_uncovered = compute_uncovered(covered)
                improved = True
                if current_uncovered == 0:
                    return classes, covered, True
            else:
                break

        return classes, covered, improved

    def add_unselected(classes, covered, unselected_moduli):
        """Add unselected moduli if they help."""
        improved = False
        current_uncovered = compute_uncovered(covered)
        
        for m in unselected_moduli[:]:
            if time.time() > deadline:
                break
            r, gain = find_best_residue(m, covered)
            if gain > 0:
                classes.append([m, r])
                covered[r::m] = True
                unselected_moduli.remove(m)
                improved = True
                current_uncovered = compute_uncovered(covered)
                if current_uncovered == 0:
                    return classes, covered, True

        return classes, covered, improved

    def local_search(classes, covered, max_rounds=20):
        """Combined local search: coordinate descent, swaps, and additions."""
        selected_moduli = {c[0] for c in classes}
        unselected_moduli = [m for m in divisors if m not in selected_moduli]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Coordinate descent
            classes, covered = coordinate_descent(classes, covered, max_iter=5)
            if compute_uncovered(covered) == 0:
                return classes, covered, True
            
            # Swap search
            classes, covered, swapped = swap_search(classes, covered, unselected_moduli, max_swaps=10)
            if compute_uncovered(covered) == 0:
                return classes, covered, True
            
            # Add unselected
            classes, covered, added = add_unselected(classes, covered, unselected_moduli)
            if compute_uncovered(covered) == 0:
                return classes, covered, True
            
            # Update unselected list
            selected_moduli = {c[0] for c in classes}
            unselected_moduli = [m for m in divisors if m not in selected_moduli]
            
            if not swapped and not added:
                break

        return classes, covered, compute_uncovered(covered) == 0

    # Main algorithm with multiple restarts
    best_classes = None
    best_uncovered = N  # Worst case: nothing covered
    
    # First run: standard greedy
    classes, covered = greedy_initialization()
    classes, covered, _ = local_search(classes, covered, max_rounds=15)
    uncovered = compute_uncovered(covered)
    
    if uncovered < best_uncovered:
        best_uncovered = uncovered
        best_classes = [c[:] for c in classes]
    
    if best_uncovered == 0:
        return best_classes
    
    # Try different initial orders based on modulus size
    # Smaller moduli first (cover more residues per class)
    divisors_sorted_asc = sorted(divisors)
    classes, covered = greedy_initialization(order=divisors_sorted_asc)
    classes, covered, _ = local_search(classes, covered, max_rounds=10)
    uncovered = compute_uncovered(covered)
    
    if uncovered < best_uncovered:
        best_uncovered = uncovered
        best_classes = [c[:] for c in classes]
    
    if best_uncovered == 0:
        return best_classes
    
    # Larger moduli first
    divisors_sorted_desc = sorted(divisors, reverse=True)
    classes, covered = greedy_initialization(order=divisors_sorted_desc)
    classes, covered, _ = local_search(classes, covered, max_rounds=10)
    uncovered = compute_uncovered(covered)
    
    if uncovered < best_uncovered:
        best_uncovered = uncovered
        best_classes = [c[:] for c in classes]
    
    if best_uncovered == 0:
        return best_classes
    
    # Randomized restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random order with some bias toward smaller moduli
        random_order = divisors.copy()
        random.shuffle(random_order)
        # Bias: put smaller moduli earlier with some probability
        for i in range(len(random_order)):
            if random.random() < 0.5:
                # Find smallest available and swap with some probability
                min_idx = min(range(i, len(random_order)), key=lambda j: random_order[j])
                if random.random() < 0.3:
                    random_order[i], random_order[min_idx] = random_order[min_idx], random_order[i]
        
        classes, covered = greedy_initialization(order=random_order)
        classes, covered, _ = local_search(classes, covered, max_rounds=8)
        uncovered = compute_uncovered(covered)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [c[:] for c in classes]
            if best_uncovered == 0:
                return best_classes
    
    # Final polish: more aggressive local search on best solution
    if best_classes is not None and best_uncovered > 0:
        classes = [c[:] for c in best_classes]
        covered = np.zeros(N, dtype=bool)
        for m, r in classes:
            covered[r::m] = True
        
        classes, covered, _ = local_search(classes, covered, max_rounds=30)
        uncovered = compute_uncovered(covered)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = classes
    
    return best_classes
