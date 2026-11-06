import time
import random

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def get_mask(sel):
        """Compute bitmask of all sums a+b for a,b in sel, a<=b."""
        m = 0
        for i in range(len(sel)):
            a = sel[i]
            m |= 1 << (2 * a)
            for j in range(i + 1, len(sel)):
                b = sel[j]
                m |= 1 << (a + b)
        return m

    def can_add(x, sel, sum_mask):
        """Check if x can be added to sel (Sidon property maintained)."""
        if (sum_mask >> (2 * x)) & 1:
            return False
        for a in sel:
            if (sum_mask >> (x + a)) & 1:
                return False
        return True

    def add_element(x, sel, sel_set, sum_mask):
        """Add x to sel, update sel_set and sum_mask."""
        sel.append(x)
        sel_set.add(x)
        for a in sel[:-1]:
            sum_mask |= 1 << (x + a)
        sum_mask |= 1 << (2 * x)
        return sel, sel_set, sum_mask

    def is_maximal(sel, sum_mask, sel_set):
        """Check if sel is inclusion-maximal."""
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            # Check if y can be added
            if (sum_mask >> (2 * y)) & 1:
                continue
            conflict = False
            for a in sel:
                if (sum_mask >> (y + a)) & 1:
                    conflict = True
                    break
            if not conflict:
                return False
        return True

    def saturate(sel, sel_set, sum_mask):
        """Add all possible elements to make the set maximal."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                if can_add(x, sel, sum_mask):
                    sel, sel_set, sum_mask = add_element(x, sel, sel_set, sum_mask)
                    changed = True
        return sel, sel_set, sum_mask

    def greedy_build(seed):
        """Greedy construction with blocking heuristic."""
        rng = random.Random(seed)
        sel = []
        sel_set = set()
        sum_mask = 0
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        
        while remaining and time.time() < deadline:
            best_x = -1
            best_score = -1
            
            # Sample candidates
            candidates = remaining if len(remaining) < 40 else rng.sample(remaining, 40)
            
            for x in candidates:
                if x in sel_set:
                    continue
                if not can_add(x, sel, sum_mask):
                    continue
                
                # Score: how many remaining elements does x block?
                block_count = 0
                for y in remaining:
                    if y == x:
                        continue
                    # y is blocked if (y+x) in sum_mask OR 2y in sum_mask
                    if (sum_mask >> (y + x)) & 1:
                        block_count += 1
                    elif (sum_mask >> (2 * y)) & 1 and 2 * y != 2 * x:
                        block_count += 0.5  # Partial credit for pre-existing blocks
                
                if block_count > best_score:
                    best_score = block_count
                    best_x = x
            
            if best_x == -1:
                # Try full remaining
                for x in remaining:
                    if x in sel_set:
                        continue
                    if can_add(x, sel, sum_mask):
                        best_x = x
                        break
            
            if best_x == -1:
                break
            
            sel, sel_set, sum_mask = add_element(best_x, sel, sel_set, sum_mask)
            remaining.remove(best_x)
        
        # Saturate
        sel, sel_set, sum_mask = saturate(sel, sel_set, sum_mask)
        return sel

    def local_search_reduce(sel, time_budget=5.0):
        """Try to reduce size via 2-remove-1-add moves."""
        start = time.time()
        current = list(sel)
        current_size = len(current)
        
        while time.time() < start + time_budget:
            improved = False
            current_set = set(current)
            current_mask = get_mask(current)
            elems = list(current)
            
            for i in range(len(elems)):
                if time.time() > start + time_budget:
                    break
                for j in range(i + 1, len(elems)):
                    if time.time() > start + time_budget:
                        break
                    x1 = elems[i]
                    x2 = elems[j]
                    
                    # Remove x1, x2
                    rem_sel = [e for e in current if e != x1 and e != x2]
                    rem_set = set(rem_sel)
                    rem_mask = get_mask(rem_sel)
                    
                    # Try adding one element y
                    for y in range(1, N + 1):
                        if y in rem_set:
                            continue
                        if not can_add(y, rem_sel, rem_mask):
                            continue
                        
                        new_sel = rem_sel + [y]
                        new_set = rem_set | {y}
                        new_mask = rem_mask
                        for a in rem_sel:
                            new_mask |= 1 << (y + a)
                        new_mask |= 1 << (2 * y)
                        
                        # Check maximality
                        if is_maximal(new_sel, new_mask, new_set):
                            current = new_sel
                            current_size = len(current)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try perturbation
                if len(current) < 2:
                    break
                x_idx = random.randint(0, len(current) - 1)
                x = current[x_idx]
                rem_sel = current[:x_idx] + current[x_idx + 1:]
                rem_set = set(rem_sel)
                rem_mask = get_mask(rem_sel)
                
                found = False
                for _ in range(10):
                    y = random.randint(1, N)
                    if y in rem_set:
                        continue
                    if not can_add(y, rem_sel, rem_mask):
                        continue
                    
                    new_sel = rem_sel + [y]
                    new_set = rem_set | {y}
                    new_mask = rem_mask
                    for a in rem_sel:
                        new_mask |= 1 << (y + a)
                    new_mask |= 1 << (2 * y)
                    
                    # Saturate
                    new_sel, new_set, new_mask = saturate(new_sel, new_set, new_mask)
                    
                    if len(new_sel) < current_size:
                        current = new_sel
                        current_size = len(current)
                        found = True
                        break
                
                if not found:
                    break
        
        return current

    best_set = None
    best_size = float('inf')
    
    # Phase 1: Generate candidates with greedy
    num_trials = 30
    for i in range(num_trials):
        if time.time() > deadline - 15:
            break
        sel = greedy_build(i)
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
    
    if best_set is None:
        best_set = greedy_build(0)
        best_size = len(best_set)

    # Phase 2: Local search reduction
    current = list(best_set)
    current_size = len(current)
    
    while time.time() < deadline - 3:
        improved = False
        current_set = set(current)
        current_mask = get_mask(current)
        elems = list(current)
        
        for i in range(len(elems)):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, len(elems)):
                if time.time() > deadline - 3:
                    break
                x1 = elems[i]
                x2 = elems[j]
                
                rem_sel = [e for e in current if e != x1 and e != x2]
                rem_set = set(rem_sel)
                rem_mask = get_mask(rem_sel)
                
                for y in range(1, N + 1):
                    if y in rem_set:
                        continue
                    if not can_add(y, rem_sel, rem_mask):
                        continue
                    
                    new_sel = rem_sel + [y]
                    new_set = rem_set | {y}
                    new_mask = rem_mask
                    for a in rem_sel:
                        new_mask |= 1 << (y + a)
                    new_mask |= 1 << (2 * y)
                    
                    if is_maximal(new_sel, new_mask, new_set):
                        current = new_sel
                        current_size = len(current)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Perturbation
            if len(current) < 2:
                break
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            rem_sel = current[:x_idx] + current[x_idx + 1:]
            rem_set = set(rem_sel)
            rem_mask = get_mask(rem_sel)
            
            found = False
            for _ in range(15):
                y = random.randint(1, N)
                if y in rem_set:
                    continue
                if not can_add(y, rem_sel, rem_mask):
                    continue
                
                new_sel = rem_sel + [y]
                new_set = rem_set | {y}
                new_mask = rem_mask
                for a in rem_sel:
                    new_mask |= 1 << (y + a)
                new_mask |= 1 << (2 * y)
                
                new_sel, new_set, new_mask = saturate(new_sel, new_set, new_mask)
                
                if len(new_sel) < current_size:
                    current = new_sel
                    current_size = len(current)
                    found = True
                    break
            
            if not found:
                break

    return current
