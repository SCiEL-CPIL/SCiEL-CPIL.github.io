"""
Odd covering systems with moduli dividing 51975.
Uses vectorized greedy maximum coverage with efficient local search refinement.
"""
import time
import numpy as np

BASE = 51975  # 3^3 * 5^2 * 7 * 11

# Precompute all divisors > 1 of BASE
_DIVISORS = None

def _get_divisors():
    global _DIVISORS
    if _DIVISORS is None:
        def gen_divisors(primes, exponents):
            if not primes:
                return [1]
            result = []
            for d in gen_divisors(primes[1:], exponents[1:]):
                for e in range(exponents[0] + 1):
                    result.append(d * primes[0] ** e)
            return result
        
        all_divs = gen_divisors([3, 5, 7, 11], [3, 2, 1, 1])
        _DIVISORS = sorted([d for d in all_divs if d > 1])
    return _DIVISORS

def _vectorized_best_residue(m, uncovered_arr):
    """
    Find the residue r that covers the most uncovered residues for modulus m.
    Uses vectorized numpy operations.
    """
    n_full = BASE // m
    remainder = BASE % m
    
    # Reshape uncovered_arr to (n_full, m) if possible
    # The first n_full*m elements can be reshaped
    if n_full > 0:
        # Create a 2D array where row i contains elements at positions i*m to (i+1)*m - 1
        # uncovered_arr[i*m + r] for i in range(n_full), r in range(m)
        main_part = uncovered_arr[:n_full * m].reshape(n_full, m)
        sums = np.sum(main_part, axis=0)
        
        # Handle the remainder: elements from n_full*m to BASE-1
        if remainder > 0:
            rem_part = uncovered_arr[n_full * m:n_full * m + remainder]
            sums[:remainder] += rem_part
    
    # Find the best residue
    best_r = int(np.argmax(sums))
    best_count = int(sums[best_r])
    
    return best_r, best_count

def _rebuild_uncovered(selected_classes):
    """Rebuild the uncovered array from a list of selected classes."""
    unc = np.ones(BASE, dtype=np.uint8)
    for m, r in selected_classes:
        unc[r::m] = 0
    return unc

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    divisors = _get_divisors()
    n_divs = len(divisors)
    divisors_sorted = sorted(divisors)
    
    # Greedy construction using vectorized best residue finding
    uncovered_arr = np.ones(BASE, dtype=np.uint8)
    selected_classes = []
    used_moduli = set()
    
    for step in range(n_divs):
        if time.time() > deadline:
            break
        
        best_m = None
        best_r = None
        best_gain = -1
        
        # Try each unused modulus, using vectorized computation
        for m in divisors_sorted:
            if m in used_moduli:
                continue
            
            r, gain = _vectorized_best_residue(m, uncovered_arr)
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = r
        
        if best_m is None or best_gain <= 0:
            break
        
        # Add this class
        used_moduli.add(best_m)
        selected_classes.append((best_m, best_r))
        uncovered_arr[best_r::best_m] = 0
    
    current_score = int(np.sum(uncovered_arr))
    
    # Local search: try to improve by changing residues
    # Use an efficient approach: track coverage counts
    improved = True
    max_local_iters = 30
    local_iter = 0
    
    while improved and local_iter < max_local_iters and time.time() < deadline:
        improved = False
        local_iter += 1
        
        for i in range(len(selected_classes)):
            if time.time() > deadline:
                break
            
            m, r_old = selected_classes[i]
            
            # Compute uncovered without this class efficiently
            # We need to temporarily remove class i and find the best r for m
            # Then decide if it improves things
            
            # Build uncovered without class i
            unc_without = np.ones(BASE, dtype=np.uint8)
            for j in range(len(selected_classes)):
                if j == i:
                    continue
                mj, rj = selected_classes[j]
                unc_without[rj::mj] = 0
            
            # Find best residue for m given unc_without
            best_r, best_gain = _vectorized_best_residue(m, unc_without)
            
            if best_r != r_old:
                # Check if this actually improves the total score
                new_unc = unc_without.copy()
                new_unc[best_r::m] = 0
                new_score = int(np.sum(new_unc))
                
                if new_score < current_score:
                    selected_classes[i] = (m, best_r)
                    current_score = new_score
                    uncovered_arr = new_unc
                    improved = True
    
    # Phase 2: Try adding unused moduli
    for m in divisors_sorted:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        best_r, best_gain = _vectorized_best_residue(m, uncovered_arr)
        
        if best_gain > 0:
            new_unc = uncovered_arr.copy()
            new_unc[best_r::m] = 0
            new_score = int(np.sum(new_unc))
            if new_score < current_score:
                used_moduli.add(m)
                selected_classes.append((m, best_r))
                current_score = new_score
                uncovered_arr = new_unc
    
    # Phase 3: More local search after additions
    improved = True
    local_iter = 0
    while improved and local_iter < max_local_iters and time.time() < deadline:
        improved = False
        local_iter += 1
        
        for i in range(len(selected_classes)):
            if time.time() > deadline:
                break
            
            m, r_old = selected_classes[i]
            
            unc_without = np.ones(BASE, dtype=np.uint8)
            for j in range(len(selected_classes)):
                if j == i:
                    continue
                mj, rj = selected_classes[j]
                unc_without[rj::mj] = 0
            
            best_r, best_gain = _vectorized_best_residue(m, unc_without)
            
            if best_r != r_old:
                new_unc = unc_without.copy()
                new_unc[best_r::m] = 0
                new_score = int(np.sum(new_unc))
                
                if new_score < current_score:
                    selected_classes[i] = (m, best_r)
                    current_score = new_score
                    uncovered_arr = new_unc
                    improved = True
    
    # Final polish: try swap operations (replace one class with another)
    if time.time() < deadline:
        for i in range(len(selected_classes)):
            if time.time() > deadline:
                break
            
            m_i, r_i = selected_classes[i]
            
            # Try removing class i and adding a different unused modulus
            unc_without_i = np.ones(BASE, dtype=np.uint8)
            for j in range(len(selected_classes)):
                if j == i:
                    continue
                mj, rj = selected_classes[j]
                unc_without_i[rj::mj] = 0
            
            best_new_m = None
            best_new_r = None
            best_new_score = current_score
            
            for m_new in divisors_sorted:
                if m_new in used_moduli:
                    continue
                
                best_r_new, _ = _vectorized_best_residue(m_new, unc_without_i)
                new_unc = unc_without_i.copy()
                new_unc[best_r_new::m_new] = 0
                new_score = int(np.sum(new_unc))
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_m = m_new
                    best_new_r = best_r_new
            
            if best_new_m is not None and best_new_score < current_score:
                # Perform the swap
                selected_classes[i] = (best_new_m, best_new_r)
                used_moduli.discard(m_i)
                used_moduli.add(best_new_m)
                current_score = best_new_score
                uncovered_arr = _rebuild_uncovered(selected_classes)
    
    # Convert to required format
    result = [[m, r] for m, r in selected_classes]
    return result
