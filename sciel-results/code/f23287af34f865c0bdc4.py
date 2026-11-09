"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 192
MAX_K = 48
DEADLINE_MARGIN = 5.0  # seconds to keep before the 90s kill


def _is_valid_set(selected):
    """Full validation of a set. Used only for final check."""
    sel = list(selected)
    n = len(sel)
    if n < 4:
        return True
    for i in range(n):
        a = sel[i]
        others = [v for v in sel if v != a]
        if len(others) >= 3:
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
    return True

def _valid_addition_fast(x, selected_list):
    """
    Check if adding x to selected_list keeps the set valid.
    Optimized for speed.
    """
    sel = selected_list
    n = len(sel)
    
    # Condition 1: x must not divide sum of any 3 distinct others in sel
    if n >= 3:
        for t in combinations(sel, 3):
            if sum(t) % x == 0:
                return False
    
    # Condition 2: For each a in sel, (x + u + v) % a != 0
    # where u, v are in sel \ {a}
    # We can optimize this by precomputing sums? No, just loop.
    for a in sel:
        # We need pairs (u, v) from sel such that u != v, u != a, v != a.
        # And (x + u + v) % a != 0.
        # Iterate u in sel
        for u in sel:
            if u == a:
                continue
            # Iterate v in sel
            for v in sel:
                if v == a or v == u:
                    continue
                if (x + u + v) % a == 0:
                    return False
    return True

def _greedy_removal(pool_start, pool_end):
    """Start with all elements in [pool_start, pool_end], remove those causing violations."""
    selected = list(range(pool_start, pool_end + 1))
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            a = selected[i]
            others = [v for v in selected if v != a]
            if len(others) >= 3:
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        selected.pop(i)
                        changed = True
                        break
            if changed:
                break
    return selected

def _local_search(selected, time_limit):
    """
    Improved local search with swaps and remove-1-add-2.
    """
    selected = list(selected)
    selected_set = set(selected)
    candidates = [x for x in range(1, N + 1) if x not in selected_set]
    candidates.sort(reverse=True)
    
    deadline = time.time() + time_limit
    best_size = len(selected)
    best_selected = list(selected)
    last_improve_time = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # Step 1: Try to add any candidate
        for x in candidates:
            if time.time() >= deadline:
                break
            if x in selected_set:
                continue
            if _valid_addition_fast(x, selected):
                selected.append(x)
                selected_set.add(x)
                improved = True
                last_improve_time = time.time()
                candidates.remove(x)
                break
        
        if improved:
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = list(selected)
            continue
        
        # Step 2: Try Swap (Remove one, Add one)
        if time.time() - last_improve_time > 0.5:
            swap_done = False
            for _ in range(10):
                if time.time() >= deadline:
                    break
                if len(selected) == 0 or not candidates:
                    break
                idx = random.randint(0, len(selected) - 1)
                to_remove = selected[idx]
                sel_without = selected[:idx] + selected[idx+1:]
                sel_without_set = set(sel_without)
                
                x = random.choice(candidates)
                if x in sel_without_set:
                    continue
                if _valid_addition_fast(x, sel_without):
                    selected = sel_without + [x]
                    selected_set = sel_without_set | {x}
                    candidates.remove(x)
                    if to_remove not in candidates:
                        candidates.append(to_remove)
                    swap_done = True
                    break
            
            if swap_done:
                last_improve_time = time.time()
                continue
        
        # Step 3: Try Remove-1-Add-2
        if not improved and time.time() - last_improve_time > 1.0:
            for i in range(len(selected)):
                if time.time() >= deadline:
                    break
                a = selected[i]
                sel_without_a = selected[:i] + selected[i+1:]
                sel_without_a_set = set(sel_without_a)
                
                added = []
                for x in candidates:
                    if time.time() >= deadline:
                        break
                    if x in sel_without_a_set or x == a:
                        continue
                    if _valid_addition_fast(x, sel_without_a):
                        temp_sel = sel_without_a + [x]
                        temp_set = sel_without_a_set | {x}
                        
                        for y in candidates:
                            if time.time() >= deadline:
                                break
                            if y in temp_set or y == a or y == x:
                                continue
                            if _valid_addition_fast(y, temp_sel):
                                added = [x, y]
                                break
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    selected = sel_without_a + added
                    selected_set = sel_without_a_set | set(added)
                    for x in added:
                        if x in candidates:
                            candidates.remove(x)
                    if a not in candidates:
                        candidates.append(a)
                    
                    improved = True
                    last_improve_time = time.time()
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = list(selected)
                    break
        
        # If stuck for a long time, shuffle candidates to try different order
        if not improved and time.time() - last_improve_time > 2.0:
            random.shuffle(candidates)
            last_improve_time = time.time()

    return best_selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 60s limit, keep 5s margin
    
    # Phase 1: Greedy removal from dense upper pool
    best = _greedy_removal(120, 192)
    best_size = len(best)
    
    # Phase 2: Try different pool starts if time permits
    if time.time() < deadline - 15:
        for ps in [100, 80, 60]:
            if time.time() >= deadline - 15:
                break
            candidate = _greedy_removal(ps, 192)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    # Phase 3: Local search to add more elements
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        best = _local_search(best, remaining)
        best_size = len(best)
        
    # Final validation
    if not _is_valid_set(best):
        best = _greedy_removal(144, 192)
        
    return best
# EVOLVE-BLOCK-END
