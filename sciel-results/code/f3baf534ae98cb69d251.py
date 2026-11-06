import time
import random

N = 256
MAX_SUM = 2 * N  # 512
# We use bitmasks for sums: sum index s in [2, 512], so we need bits 0..511 (512 bits)
# Python ints can handle arbitrary precision, so 1 << s works for s up to 512

def _saturate_greedy(sel, sum_mask, order):
    """Saturate the set by adding elements in the given order if possible."""
    sel_set = set(sel)
    changed = True
    while changed:
        changed = False
        for x in order:
            if x in sel_set:
                continue
            # Check if x can be added
            ok = True
            # Check 2x
            if (sum_mask >> (2 * x)) & 1:
                ok = False
            else:
                # Check x + a for a in sel
                for a in sel:
                    if (sum_mask >> (x + a)) & 1:
                        ok = False
                        break
            if ok:
                sel.append(x)
                sel_set.add(x)
                sum_mask |= 1 << (2 * x)
                for a in sel[:-1]:
                    sum_mask |= 1 << (x + a)
                changed = True
    return sel, sum_mask

def _is_valid_sidon(sel):
    """Check if sel is a valid Sidon set."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(sel):
    """Check if sel is inclusion-maximal (no missing element can be added)."""
    if not _is_valid_sidon(sel):
        return False
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        can_add = True
        for a in sel:
            if x + a in sums:
                can_add = False
                break
        if can_add:
            return False
    return True

def _build_with_greedy_max_block(deadline, seed=None):
    """
    Build a maximal Sidon set by greedily picking the element that blocks the most
    other elements, then saturate.
    """
    rng = random.Random(seed)
    sel = []
    sum_mask = 0
    remaining = set(range(1, N + 1))
    
    while remaining and time.time() < deadline:
        best_x = None
        best_block = -1
        
        # Sample candidates if remaining is large
        if len(remaining) > 60:
            candidates = rng.sample(list(remaining), 60)
        else:
            candidates = list(remaining)
        
        for x in candidates:
            # Check if x can be added
            ok = True
            if (sum_mask >> (2 * x)) & 1:
                ok = False
            else:
                for a in sel:
                    if (sum_mask >> (x + a)) & 1:
                        ok = False
                        break
            
            if not ok:
                continue
            
            # Count how many remaining elements would be blocked
            block = 0
            for y in remaining:
                if y == x:
                    continue
                # y is blocked if after adding x, y cannot be added
                # y cannot be added if 2y in sums, or y+a in sums for some a in sel∪{x}
                blocked = False
                # Check 2y
                if (sum_mask >> (2 * y)) & 1:
                    blocked = True
                else:
                    # Check y+a for a in sel
                    for a in sel:
                        if (sum_mask >> (y + a)) & 1:
                            blocked = True
                            break
                    # Check y+x
                    if not blocked and (sum_mask >> (y + x)) & 1:
                        blocked = True
                    # Also need to check if y+x creates a collision with existing sums
                    # Actually, y is blocked if adding y to sel∪{x} would violate Sidon
                    # This means y+a in sums for some a in sel∪{x} OR 2y in sums
                    # We already checked 2y and y+a for a in sel, and y+x
                    # But we also need: y+x might equal 2a for some a, or y+a might equal x+a' etc.
                    # Wait, the check is: after adding x, can y still be added?
                    # y can be added to sel∪{x} iff for all a in sel∪{x}, y+a not in sums∪new_sums, and 2y not in sums∪new_sums
                    # new_sums = {x+a for a in sel} ∪ {2x}
                    # So y is blocked if:
                    # 2y in sums OR 2y in new_sums (i.e., 2y = x+a for some a, or 2y=2x)
                    # OR y+a in sums for some a in sel
                    # OR y+a in new_sums for some a in sel (i.e., y+a = x+a' => y=a' impossible, or y+a=2x => y=2x-a)
                    # OR y+x in sums
                    # OR y+x in new_sums (i.e., y+x = x+a => y=a impossible, or y+x=2x => y=x already excluded)
                
                if blocked:
                    block += 1
            
            if block > best_block:
                best_block = block
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        sel.append(best_x)
        sum_mask |= 1 << (2 * best_x)
        for a in sel[:-1]:
            sum_mask |= 1 << (best_x + a)
        remaining.discard(best_x)
    
    # Saturate in ascending order
    order = list(range(1, N + 1))
    sel, sum_mask = _saturate_greedy(sel, sum_mask, order)
    
    return sel

def _local_search_reduce(sel, deadline, max_iters=30):
    """Try removing elements and re-saturating to reduce size."""
    best = list(sel)
    best_size = len(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try removing each element
        for i in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i+1:]
            
            # Check if candidate is still Sidon
            sums = set()
            valid = True
            for a_idx in range(len(candidate)):
                a = candidate[a_idx]
                for b_idx in range(a_idx, len(candidate)):
                    s = a + candidate[b_idx]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Convert to mask
            sum_mask = 0
            for s in sums:
                sum_mask |= 1 << s
            
            # Saturate candidate
            candidate, sum_mask = _saturate_greedy(candidate, sum_mask, range(1, N + 1))
            
            if len(candidate) < best_size:
                best = candidate
                best_size = len(best)
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Multiple randomized constructions
    for trial in range(150):
        if time.time() > deadline:
            break
        sel = _build_with_greedy_max_block(deadline, seed=trial)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
        
        # Try local search on this result
        if len(sel) <= best_size:
            improved = _local_search_reduce(sel, deadline, max_iters=15)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    # Phase 2: More aggressive local search on the best
    if best_set is not None and time.time() < deadline - 10:
        improved = _local_search_reduce(best_set, deadline, max_iters=50)
        if len(improved) < best_size:
            best_set = improved
    
    # Fallback: simple greedy ascending
    if best_set is None:
        sel = []
        sum_mask = 0
        for x in range(1, N + 1):
            ok = True
            if (sum_mask >> (2 * x)) & 1:
                ok = False
            else:
                for a in sel:
                    if (sum_mask >> (x + a)) & 1:
                        ok = False
                        break
            if ok:
                sel.append(x)
                sum_mask |= 1 << (2 * x)
                for a in sel[:-1]:
                    sum_mask |= 1 << (x + a)
        best_set = sel
    
    return best_set
