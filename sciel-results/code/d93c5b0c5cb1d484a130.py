"""
Optimized construction of a 4-AP-free subset of [1,256] maximizing 3-AP count.
Uses simulated annealing with smart move generation.
"""
import time
import random
import math
from itertools import combinations

N = 256

# Precompute 4-AP conflict lists for fast validity checks
def _precompute_4ap_conflicts():
    """For each element, precompute which other elements would form a 4-AP with it."""
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b, c, e = a + d, a + 2 * d, a + 3 * d
            # a is first: conflicts with b,c,e as second,third,fourth
            conflicts[a].update([b, c, e])
            conflicts[b].add(a)
            conflicts[c].add(a)
            conflicts[e].add(a)
            # Also mark that {b,c,e} must not all be present if a is added
            # We handle this differently - conflicts[x] means "if x is in set, these cannot all be in set"
    return conflicts

# Precompute which triples form 4-APs for fast checking
def _precompute_4ap_triples():
    """Return list of (a,d) for all 4-APs."""
    triples = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            triples.append((a, d))
    return triples

_4AP_TRIPLES = _precompute_4ap_triples()

def _count_3aps(selected):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 in selected."""
    count = 0
    s = set(selected)
    for a in selected:
        for d in range(1, (N - a) // 2 + 1):
            if (a + d) in s and (a + 2 * d) in s:
                count += 1
    return count

def _has_4ap(selected):
    """Check if selected contains a nonconstant 4-term AP."""
    s = set(selected)
    for a in selected:
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            if (a + d) in s and (a + 2 * d) in s and (a + 3 * d) in s:
                return True
    return False

def _check_4ap_after_add_fast(selected_set, elem):
    """Fast check if adding elem creates a 4-AP using precomputed triples."""
    s = selected_set
    for (a, d) in _4AP_TRIPLES:
        b, c, e = a + d, a + 2 * d, a + 3 * d
        # Check if exactly one of {a,b,c,e} is elem and the other three are in s
        if elem == a:
            if b in s and c in s and e in s:
                return True
        elif elem == b:
            if a in s and c in s and e in s:
                return True
        elif elem == c:
            if a in s and b in s and e in s:
                return True
        elif elem == e:
            if a in s and b in s and c in s:
                return True
    return False

def _greedy_init():
    """Greedy construction: add elements that maximize marginal 3-AP gain while staying 4-AP-free."""
    selected = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        best_elem = None
        best_gain = -1
        
        for elem in remaining:
            # Check if adding elem creates a 4-AP
            if _check_4ap_after_add_fast(selected, elem):
                continue
            
            # Count 3-APs involving elem
            gain = 0
            s = selected
            # elem as middle: a, elem, a+2d => a = elem - d, a+2d = elem + d
            for d in range(1, N):
                if (elem - d) in s and (elem + d) in s:
                    gain += 1
            # elem as first: elem, elem+d, elem+2d
            for d in range(1, (N - elem) // 2 + 1):
                if (elem + d) in s and (elem + 2 * d) in s:
                    gain += 1
            # elem as last: elem-2d, elem-d, elem
            for d in range(1, elem // 2 + 1):
                if (elem - 2 * d) in s and (elem - d) in s:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_elem = elem
        
        if best_elem is not None:
            selected.add(best_elem)
            remaining.remove(best_elem)
        else:
            break
    
    return sorted(selected)

def _count_3aps_fast(selected_set):
    """Faster 3-AP count using set membership."""
    count = 0
    for a in selected_set:
        for d in range(1, (N - a) // 2 + 1):
            if (a + d) in selected_set and (a + 2 * d) in selected_set:
                count += 1
    return count

def _check_4ap_after_add(selected_set, elem):
    """Check if adding elem to selected_set creates a 4-AP."""
    s = selected_set
    # elem as first: elem, elem+d, elem+2d, elem+3d
    for d in range(1, (N - elem) // 3 + 1):
        if (elem + d) in s and (elem + 2 * d) in s and (elem + 3 * d) in s:
            return True
    # elem as second: elem-d, elem, elem+d, elem+2d
    for d in range(1, min(elem - 1, (N - elem) // 2) + 1):
        if (elem - d) in s and (elem + d) in s and (elem + 2 * d) in s:
            return True
    # elem as third: elem-2d, elem-d, elem, elem+d
    for d in range(1, min(elem // 2, N - elem) + 1):
        if (elem - 2 * d) in s and (elem - d) in s and (elem + d) in s:
            return True
    # elem as fourth: elem-3d, elem-2d, elem-d, elem
    for d in range(1, (elem - 1) // 3 + 1):
        if (elem - 3 * d) in s and (elem - 2 * d) in s and (elem - d) in s:
            return True
    return False

def _check_4ap_after_remove(selected_set, elem):
    """Removing an element cannot create a new 4-AP, so always valid."""
    return True

def _count_3aps_delta_add(selected_set, elem):
    """Count new 3-APs created by adding elem."""
    s = selected_set
    gain = 0
    # elem as middle: (elem-d, elem, elem+d)
    for d in range(1, N):
        if (elem - d) in s and (elem + d) in s:
            gain += 1
    # elem as first: (elem, elem+d, elem+2d)
    for d in range(1, (N - elem) // 2 + 1):
        if (elem + d) in s and (elem + 2 * d) in s:
            gain += 1
    # elem as last: (elem-2d, elem-d, elem)
    for d in range(1, elem // 2 + 1):
        if (elem - 2 * d) in s and (elem - d) in s:
            gain += 1
    return gain

def _count_3aps_delta_remove(selected_set, elem):
    """Count 3-APs lost by removing elem."""
    s = selected_set
    loss = 0
    # elem as middle
    for d in range(1, N):
        if (elem - d) in s and (elem + d) in s:
            loss += 1
    # elem as first
    for d in range(1, (N - elem) // 2 + 1):
        if (elem + d) in s and (elem + 2 * d) in s:
            loss += 1
    # elem as last
    for d in range(1, elem // 2 + 1):
        if (elem - 2 * d) in s and (elem - d) in s:
            loss += 1
    return loss

def _swap_check_4ap(selected_set, remove_elem, add_elem):
    """Check if swapping remove_elem for add_elem keeps the set 4-AP-free."""
    # Remove remove_elem, then check if adding add_elem creates a 4-AP
    s = selected_set - {remove_elem}
    return _check_4ap_after_add(s, add_elem)

def _swap_3ap_delta(selected_set, remove_elem, add_elem):
    """Net change in 3-AP count when swapping remove_elem for add_elem."""
    s = selected_set
    loss = _count_3aps_delta_remove(s, remove_elem)
    # For gain, we need to compute on s - {remove_elem} + {add_elem}
    s2 = s - {remove_elem}
    gain = _count_3aps_delta_add(s2, add_elem)
    return gain - loss

def _check_4ap_after_add(current_set, elem):
    """Check if adding elem to current_set creates a 4-AP."""
    s = current_set
    # elem as first: elem, elem+d, elem+2d, elem+3d
    for d in range(1, (N - elem) // 3 + 1):
        if (elem + d) in s and (elem + 2 * d) in s and (elem + 3 * d) in s:
            return True
    # elem as second: elem-d, elem, elem+d, elem+2d
    for d in range(1, min(elem - 1, (N - elem) // 2) + 1):
        if (elem - d) in s and (elem + d) in s and (elem + 2 * d) in s:
            return True
    # elem as third: elem-2d, elem-d, elem, elem+d
    for d in range(1, min(elem // 2, N - elem) + 1):
        if (elem - 2 * d) in s and (elem - d) in s and (elem + d) in s:
            return True
    # elem as fourth: elem-3d, elem-2d, elem-d, elem
    for d in range(1, (elem - 1) // 3 + 1):
        if (elem - 3 * d) in s and (elem - 2 * d) in s and (elem - d) in s:
            return True
    return False

def _check_4ap_after_remove_add(selected_set, remove_elem, add_elem):
    """Check if swapping remove_elem for add_elem keeps the set 4-AP-free."""
    s = selected_set - {remove_elem}
    return _check_4ap_after_add(s, add_elem)

def _count_3aps_delta_add(selected_set, elem):
    """Count new 3-APs created by adding elem."""
    s = selected_set
    gain = 0
    # elem as middle: (elem-d, elem, elem+d)
    for d in range(1, N):
        if (elem - d) in s and (elem + d) in s:
            gain += 1
    # elem as first: (elem, elem+d, elem+2d)
    for d in range(1, (N - elem) // 2 + 1):
        if (elem + d) in s and (elem + 2 * d) in s:
            gain += 1
    # elem as last: (elem-2d, elem-d, elem)
    for d in range(1, elem // 2 + 1):
        if (elem - 2 * d) in s and (elem - d) in s:
            gain += 1
    return gain

def _count_3aps_delta_remove(selected_set, elem):
    """Count 3-APs lost by removing elem."""
    s = selected_set
    loss = 0
    # elem as middle
    for d in range(1, N):
        if (elem - d) in s and (elem + d) in s:
            loss += 1
    # elem as first
    for d in range(1, (N - elem) // 2 + 1):
        if (elem + d) in s and (elem + 2 * d) in s:
            loss += 1
    # elem as last
    for d in range(1, elem // 2 + 1):
        if (elem - 2 * d) in s and (elem - d) in s:
            loss += 1
    return loss

def _swap_3ap_delta(selected_set, remove_elem, add_elem):
    """Net change in 3-AP count when swapping remove_elem for add_elem."""
    s = selected_set
    loss = _count_3aps_delta_remove(s, remove_elem)
    s2 = s - {remove_elem}
    gain = _count_3aps_delta_add(s2, add_elem)
    return gain - loss

def _two_for_one_delta(selected_set, add_elem1, add_elem2):
    """Net 3-AP gain from adding two elements simultaneously (accounts for APs between them)."""
    s = selected_set
    gain1 = _count_3aps_delta_add(s, add_elem1)
    gain2 = _count_3aps_delta_add(s, add_elem2)
    # Check if add_elem1 and add_elem2 form a 3-AP together
    bonus = 0
    a, b = add_elem1, add_elem2
    if a > b:
        a, b = b, a
    d = (b - a)
    if d > 0:
        # Check (a, b, a+2d) or (b-d, a, b) etc.
        # a,b as first two: need a+2d in s
        if a + 2 * d in s and a + 2 * d not in (a, b):
            bonus += 1
        # a,b as last two: need b-2d in s
        if b - 2 * d in s and b - 2 * d not in (a, b) and b - 2 * d >= 1:
            bonus += 1
        # a,b as middle pair: need a-d and b+d in s (i.e., a-d, a, b, b+d but that's 4-AP)
        # Actually for 3-AP: (a-d, a, b) requires b = a + d which means d = b-a, so a-d, a, b
        # That's already counted if a-d in s
        # And (a, b, b+d) requires b+d in s
        if b + d in s and b + d not in (a, b) and b + d <= N:
            bonus += 1
        if a - d in s and a - d not in (a, b) and a - d >= 1:
            bonus += 1
    return gain1 + gain2 + bonus

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    random.seed(42)
    
    # Phase 1: Greedy initialization
    best = _greedy_init()
    best_set = set(best)
    best_score = _count_3aps(best_set)
    
    # Phase 2: Simulated annealing with add/remove/swap moves
    current = list(best)
    current_set = set(current)
    current_score = best_score
    
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    
    iterations = 0
    max_iters = 2000000
    
    while time.time() < deadline and iterations < max_iters and T > T_min:
        iterations += 1
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.35:
            # Add move
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            elem = random.choice(candidates)
            
            if _check_4ap_after_add(current_set, elem):
                continue
            
            gain = _count_3aps_delta_add(current_set, elem)
            delta = gain
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set.add(elem)
                current.append(elem)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best = sorted(current_set)
        
        elif move_type < 0.55:
            # Remove move
            if not current_set:
                continue
            elem = random.choice(list(current_set))
            
            loss = _count_3aps_delta_remove(current_set, elem)
            delta = -loss
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set.remove(elem)
                current.remove(elem)
                current_score += delta
        
        else:
            # Swap move
            if not current_set:
                continue
            remove_elem = random.choice(list(current_set))
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            add_elem = random.choice(candidates)
            
            if _check_4ap_after_remove_add(current_set, remove_elem, add_elem):
                continue
            
            delta = _swap_3ap_delta(current_set, remove_elem, add_elem)
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_set.remove(remove_elem)
                current_set.add(add_elem)
                current.remove(remove_elem)
                current.append(add_elem)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best = sorted(current_set)
        
        T *= cooling
    
    # Phase 3: Aggressive local refinement
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # Try single adds
            best_add = None
            best_add_gain = 0
            for elem in range(1, N + 1):
                if elem in best_set:
                    continue
                if _check_4ap_after_add(best_set, elem):
                    continue
                gain = _count_3aps_delta_add(best_set, elem)
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add = elem
            if best_add and best_add_gain > 0:
                best_set.add(best_add)
                best_score += best_add_gain
                best = sorted(best_set)
                improved = True
            
            # Try single swaps
            best_swap_delta = 0
            best_swap = None
            for remove_elem in list(best_set):
                for add_elem in range(1, N + 1):
                    if add_elem in best_set:
                        continue
                    if _check_4ap_after_remove_add(best_set, remove_elem, add_elem):
                        continue
                    delta = _swap_3ap_delta(best_set, remove_elem, add_elem)
                    if delta > best_swap_delta:
                        best_swap_delta = delta
                        best_swap = (remove_elem, add_elem)
            
            if best_swap and best_swap_delta > 0:
                rm, ad = best_swap
                best_set.remove(rm)
                best_set.add(ad)
                best_score += best_swap_delta
                best = sorted(best_set)
                improved = True
            
            # Try 2-for-1 exchanges (remove 1, add 2)
            if time.time() < deadline - 1:
                best_2f1_delta = 0
                best_2f1 = None
                for remove_elem in list(best_set):
                    s_without = best_set - {remove_elem}
                    loss = _count_3aps_delta_remove(best_set, remove_elem)
                    # Try adding two elements
                    for a in range(1, N + 1):
                        if a in best_set:
                            continue
                        if _check_4ap_after_add(s_without, a):
                            continue
                        for b in range(a + 1, N + 1):
                            if b in best_set:
                                continue
                            # Check 4-AP with both added
                            s_temp = s_without | {a, b}
                            # Quick check: does adding both create 4-AP?
                            valid = True
                            for (ap_a, d) in _4AP_TRIPLES:
                                b1, c1, e1 = ap_a + d, ap_a + 2 * d, ap_a + 3 * d
                                # Check if any of a,b is in this 4-AP and the other 3 are in s_temp
                                for elem_check in [a, b]:
                                    if elem_check == ap_a and b1 in s_temp and c1 in s_temp and e1 in s_temp:
                                        valid = False
                                        break
                                    if elem_check == b1 and ap_a in s_temp and c1 in s_temp and e1 in s_temp:
                                        valid = False
                                        break
                                    if elem_check == c1 and ap_a in s_temp and b1 in s_temp and e1 in s_temp:
                                        valid = False
                                        break
                                    if elem_check == e1 and ap_a in s_temp and b1 in s_temp and c1 in s_temp:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            
                            gain1 = _count_3aps_delta_add(s_without, a)
                            gain2 = _count_3aps_delta_add(s_without | {a}, b)
                            delta = gain1 + gain2 - loss
                            if delta > best_2f1_delta:
                                best_2f1_delta = delta
                                best_2f1 = (remove_elem, a, b)
                
                if best_2f1 and best_2f1_delta > 0:
                    rm, a, b = best_2f1
                    best_set.remove(rm)
                    best_set.add(a)
                    best_set.add(b)
                    best_score += best_2f1_delta
                    best = sorted(best_set)
                    improved = True
    
    return sorted(best_set)
