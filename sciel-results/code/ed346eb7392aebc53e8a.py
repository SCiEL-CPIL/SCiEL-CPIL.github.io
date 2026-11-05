import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    # Best solution found
    best_set = None
    best_max_val = float('inf')
    
    def get_diffs(elements):
        """Calculate set of positive differences and count duplicates."""
        diffs = []
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                d = elements[j] - elements[i]
                if d > 0:
                    diffs.append(d)
        return diffs

    def evaluate(elements):
        """
        Returns (is_valid, max_val, penalty)
        penalty is 0 if valid, otherwise > 0
        """
        if not elements:
            return False, 0, 1000000
        if 0 not in elements:
            return False, max(elements), 1000000
        
        # Check size constraint
        if len(elements) > 32:
            return False, max(elements), 1000000 + (len(elements) - 32) * 1000
            
        diffs = get_diffs(elements)
        diff_set = set(diffs)
        
        # Check Sidon property (all diffs distinct)
        if len(diffs) != len(diff_set):
            # Count duplicates
            from collections import Counter
            counts = Counter(diffs)
            dup_count = sum(c - 1 for c in counts.values() if c > 1)
            return False, max(elements), 1000000 + dup_count * 100
            
        # Check coverage of 1..30
        missing = target_diffs - diff_set
        if missing:
            return False, max(elements), 1000000 + len(missing) * 1000
            
        # Valid
        return True, max(elements), 0

    def update_best(elements):
        nonlocal best_set, best_max_val
        is_valid, max_val, _ = evaluate(elements)
        if is_valid and max_val < best_max_val:
            best_max_val = max_val
            best_set = set(elements)

    # Initial greedy construction to get a valid solution quickly
    # Start with 0
    current = {0}
    # Try to add elements to cover 1..30
    # We'll do a simple greedy pass
    for _ in range(10): # 10 attempts at greedy init
        a = [0]
        diffs = set()
        valid = True
        # Sort targets
        targets = list(range(1, 31))
        random.shuffle(targets)
        for d in targets:
            if d in diffs:
                continue
            found = False
            # Try to extend existing elements
            candidates = []
            for x in a:
                # x + d
                y1 = x + d
                if y1 not in a:
                    candidates.append(y1)
                # x - d
                y2 = x - d
                if y2 >= 0 and y2 not in a:
                    candidates.append(y2)
            
            random.shuffle(candidates)
            for y in candidates:
                # Check if adding y maintains Sidon
                new_a = a + [y]
                new_diffs = []
                for i in range(len(new_a)):
                    for j in range(i+1, len(new_a)):
                        nd = new_a[j] - new_a[i]
                        if nd > 0:
                            new_diffs.append(nd)
                if len(new_diffs) == len(set(new_diffs)):
                    a = new_a
                    diffs.update(new_diffs)
                    found = True
                    break
            
            if not found:
                # Add a random large element? Or break?
                # If we can't cover d, the set might not be valid for this randomization
                # Let's just break and restart init
                valid = False
                break
        
        if valid and target_diffs.issubset(diffs):
            update_best(a)
            break # Found a valid init, stop greedy inits
            
    if best_set is None:
        # Fallback: Random init
        for _ in range(5):
            size = random.randint(15, 30)
            elements = [0] + random.sample(range(1, 2000), size-1)
            update_best(elements)

    if best_set is None:
        # Last resort: return seed-like solution if we failed completely
        # But we should try harder. Let's do a simple random search for a bit
        for _ in range(1000):
            if time.time() > deadline:
                break
            size = random.randint(10, 30)
            elements = [0] + random.sample(range(1, 1000), size-1)
            update_best(elements)
            
    if best_set is None:
        # If we still don't have a valid solution, return a hardcoded one?
        # The seed program always returns a valid one.
        # We can just return the result of the seed logic if it's fast, but we are rewriting.
        # Let's ensure we have *some* valid set.
        # A known small Sidon set covering 1..30?
        # We'll just return [0] + list(range(1, 31)) ? No, that's not Sidon.
        # We'll use the greedy logic from the seed as a backup if it's fast.
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    # Local Search / Simulated Annealing
    # Start from the best valid set found
    current_elements = list(best_set)
    current_max = best_max_val
    
    # We will try to reduce the max element
    # Moves:
    # 1. Try to move an element to a smaller value
    # 2. Try to remove an element (if size > min_needed)
    # 3. Try to add an element (if it helps reduce max? Unlikely directly, but might allow removing a large one)
    
    # To be safe and fast, we will use a simple hill-climbing with random restarts
    # Objective: Minimize max(A) subject to validity
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # If we are restarting, pick a random valid neighborhood or perturb best
        if restart == 0:
            elements = list(best_set)
        else:
            # Perturb best set
            elements = list(best_set)
            # Remove 1-2 random elements (not 0)
            for _ in range(random.randint(1, 2)):
                if len(elements) > 1:
                    idx = random.randint(1, len(elements)-1) # Don't remove 0 at index 0? Ensure 0 is in set
                    # Find index of 0
                    if 0 in elements:
                        elements.pop(idx)
                    else:
                        elements.pop(random.randint(0, len(elements)-1))
            # Add 1-2 random elements
            for _ in range(random.randint(1, 2)):
                val = random.randint(1, current_max)
                if val not in elements:
                    elements.append(val)
        
        # Local Search
        stuck = False
        for step in range(10000):
            if time.time() > deadline:
                return sorted(best_set)
                
            if len(elements) > 32:
                # Remove random non-zero
                if len(elements) > 1:
                    # Keep 0
                    non_zero = [e for e in elements if e != 0]
                    if non_zero:
                        elements.remove(random.choice(non_zero))
                continue
                
            # Try a move
            move_type = random.choice(['move', 'add', 'remove'])
            
            if move_type == 'move':
                if len(elements) > 1:
                    # Pick a non-zero element to move
                    non_zero = [e for e in elements if e != 0]
                    if non_zero:
                        old_val = random.choice(non_zero)
                        new_val = random.randint(1, current_max + 100)
                        if new_val not in elements:
                            elements.remove(old_val)
                            elements.append(new_val)
                            is_valid, new_max, _ = evaluate(elements)
                            if is_valid:
                                if new_max < current_max:
                                    current_max = new_max
                                    update_best(elements)
                                # Even if not better, accept if valid to explore?
                                # For now, greedy: only accept if valid.
                                # To escape local minima, we might need to accept worse valid states.
                                # But "worse" means larger max.
                                # Let's just accept any valid state for a while?
                                # No, we want to minimize.
                            else:
                                # Revert
                                elements.remove(new_val)
                                elements.append(old_val)
                                stuck = True
                            continue
            elif move_type == 'add':
                if len(elements) < 32:
                    new_val = random.randint(1, current_max + 100)
                    if new_val not in elements:
                        elements.append(new_val)
                        is_valid, new_max, _ = evaluate(elements)
                        if is_valid:
                            if new_max < current_max:
                                current_max = new_max
                                update_best(elements)
                        else:
                            elements.remove(new_val)
                            stuck = True
                        continue
            elif move_type == 'remove':
                if len(elements) > 1:
                    non_zero = [e for e in elements if e != 0]
                    if non_zero:
                        old_val = random.choice(non_zero)
                        elements.remove(old_val)
                        is_valid, new_max, _ = evaluate(elements)
                        if is_valid:
                            if new_max < current_max:
                                current_max = new_max
                                update_best(elements)
                        else:
                            elements.append(old_val)
                            stuck = True
                        continue
            
            if stuck:
                break
                
    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
