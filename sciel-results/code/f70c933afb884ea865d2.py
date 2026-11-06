import time
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Generate divisors of 51975 (odd divisors > 1)
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    
    # Precompute bitmask for each (m, r) pair
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask = 0
            for pos in range(r, N, m):
                mask |= (1 << pos)
            masks[m][r] = mask
    
    def greedy_initialization(order=None):
        """Greedy set cover using bitmasks."""
        uncovered = (1 << N) - 1
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
                for r in range(m):
                    gain = (masks[m][r] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            
            if best_m is None or best_gain == 0:
                break
                
            classes.append([best_m, best_r])
            uncovered &= ~masks[best_m][best_r]
            available_moduli.remove(best_m)
            
            if uncovered == 0:
                break

        return classes, uncovered
    
    def coordinate_descent(classes, uncovered, max_iter=10):
        """Coordinate descent using bitmasks."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                m = classes[i][0]
                r_old = classes[i][1]
                
                base_uncovered = uncovered | masks[m][r_old]
                base_count = base_uncovered.bit_count()
                
                best_new_r = r_old
                best_new_count = base_count
                
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    new_uncovered = base_uncovered & ~masks[m][r_new]
                    new_count = new_uncovered.bit_count()
                    if new_count < best_new_count:
                        best_new_count = new_count
                        best_new_r = r_new
                
                if best_new_r != r_old:
                    classes[i][1] = best_new_r
                    uncovered = base_uncovered & ~masks[m][best_new_r]
                    improved = True
                
                if uncovered == 0:
                    return classes, uncovered

        return classes, uncovered
    
    def swap_search(classes, uncovered, unselected_moduli, max_swaps=10):
        """Try to swap a selected modulus with an unselected one."""
        improved = False
        
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
                
                base_uncovered = uncovered | masks[m_out][r_out]
                base_count = base_uncovered.bit_count()
                
                for m_in in unselected_moduli:
                    for r_in in range(m_in):
                        new_uncovered = base_uncovered & ~masks[m_in][r_in]
                        new_count = new_uncovered.bit_count()
                        gain = base_count - new_count
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_m_out = m_out
                            best_m_in = m_in
                            best_r_in = r_in
                            best_idx = i
            
            if best_m_out is not None and best_swap_gain > 0:
                classes[best_idx][0] = best_m_in
                classes[best_idx][1] = best_r_in
                
                # Recompute uncovered
                uncovered = (1 << N) - 1
                for m, r in classes:
                    uncovered &= ~masks[m][r]
                
                unselected_moduli.remove(best_m_in)
                unselected_moduli.append(best_m_out)
                
                improved = True
                if uncovered == 0:
                    return classes, uncovered, True
            else:
                break

        return classes, uncovered, improved
    
    def add_unselected(classes, uncovered, unselected_moduli):
        """Add unselected moduli if they help."""
        improved = False
        
        for m in unselected_moduli[:]:
            if time.time() > deadline:
                break
            best_r = None
            best_gain = 0
            for r in range(m):
                gain = (masks[m][r] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m, best_r])
                uncovered &= ~masks[m][best_r]
                unselected_moduli.remove(m)
                improved = True
                if uncovered == 0:
                    return classes, uncovered, True

        return classes, uncovered, improved
    
    def local_search(classes, uncovered, max_rounds=20):
        """Combined local search."""
        selected_moduli = {c[0] for c in classes}
        unselected_moduli = [m for m in divisors if m not in selected_moduli]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            classes, uncovered = coordinate_descent(classes, uncovered, max_iter=5)
            if uncovered == 0:
                return classes, uncovered, True
            
            classes, uncovered, swapped = swap_search(classes, uncovered, unselected_moduli, max_swaps=10)
            if uncovered == 0:
                return classes, uncovered, True
            
            classes, uncovered, added = add_unselected(classes, uncovered, unselected_moduli)
            if uncovered == 0:
                return classes, uncovered, True
            
            selected_moduli = {c[0] for c in classes}
            unselected_moduli = [m for m in divisors if m not in selected_moduli]
            
            if not swapped and not added:
                break

        return classes, uncovered, uncovered == 0
    
    best_classes = None
    best_uncovered_count = N
    
    # Standard greedy
    classes, uncovered = greedy_initialization()
    classes, uncovered, _ = local_search(classes, uncovered, max_rounds=15)
    uncovered_count = uncovered.bit_count()
    
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = [c[:] for c in classes]
    
    if best_uncovered_count == 0:
        return best_classes
    
    # Sorted ascending
    divisors_sorted_asc = sorted(divisors)
    classes, uncovered = greedy_initialization(order=divisors_sorted_asc)
    classes, uncovered, _ = local_search(classes, uncovered, max_rounds=10)
    uncovered_count = uncovered.bit_count()
    
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = [c[:] for c in classes]
    
    if best_uncovered_count == 0:
        return best_classes
    
    # Sorted descending
    divisors_sorted_desc = sorted(divisors, reverse=True)
    classes, uncovered = greedy_initialization(order=divisors_sorted_desc)
    classes, uncovered, _ = local_search(classes, uncovered, max_rounds=10)
    uncovered_count = uncovered.bit_count()
    
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = [c[:] for c in classes]
    
    if best_uncovered_count == 0:
        return best_classes
    
    # Randomized restarts
    for restart in range(8):
        if time.time() > deadline:
            break
        
        random_order = divisors.copy()
        random.shuffle(random_order)
        
        classes, uncovered = greedy_initialization(order=random_order)
        classes, uncovered, _ = local_search(classes, uncovered, max_rounds=8)
        uncovered_count = uncovered.bit_count()
        
        if uncovered_count < best_uncovered_count:
            best_uncovered_count = uncovered_count
            best_classes = [c[:] for c in classes]
            if best_uncovered_count == 0:
                return best_classes
    
    # Final polish
    if best_classes is not None and best_uncovered_count > 0:
        classes = [c[:] for c in best_classes]
        uncovered = (1 << N) - 1
        for m, r in classes:
            uncovered &= ~masks[m][r]
        
        classes, uncovered, _ = local_search(classes, uncovered, max_rounds=30)
        uncovered_count = uncovered.bit_count()
        
        if uncovered_count < best_uncovered_count:
            best_uncovered_count = uncovered_count
            best_classes = classes
    
    return best_classes
