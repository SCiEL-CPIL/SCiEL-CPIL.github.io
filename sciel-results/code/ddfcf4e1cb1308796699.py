import time
import random

# EVOLVE-BLOCK-START

START_TIME = time.time()
DEADLINE = 58.0

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _is_valid(elements):
    n = len(elements)
    if n == 0:
        return True
    B = [0] * (n + 1)
    B[0] = 1
    for elem in elements:
        for k in range(n, 0, -1):
            B[k] |= (B[k - 1] << elem)
    seen = 0
    for k in range(n + 1):
        if B[k] & seen:
            return False
        seen |= B[k]
    return True

def _check_addition_fast(B, n, new_elem):
    new_n = n + 1
    new_B = [0] * (new_n + 1)
    for k in range(n + 1):
        new_B[k] = B[k]
        
    Union_All = 0
    for k in range(n + 1):
        Union_All |= B[k]
        
    for k in range(n + 1):
        if B[k] == 0:
            continue
        new_sums = B[k] << new_elem
        target_t = k + 1
        conflict_mask = Union_All ^ B[target_t]
        if new_sums & conflict_mask:
            return None
        new_B[target_t] |= new_sums
        
    return new_B

def _greedy_build(order, max_size=60):
    elements = []
    B = [1]
    n = 0
    for c in order:
        if n >= max_size:
            break
        if _time_left() < 0.5:
            break
        new_B = _check_addition_fast(B, n, c)
        if new_B is not None:
            elements.append(c)
            B = new_B
            n += 1
    return elements, B

def construct():
    best_elements = []
    best_size = 0
    best_B = None

    # 1. Ascending
    if _time_left() > 1.0:
        e, B = _greedy_build(list(range(1, 257)))
        if len(e) > best_size:
            best_size, best_elements, best_B = len(e), e, B

    # 2. Descending
    if _time_left() > 1.0:
        e, B = _greedy_build(list(range(256, 0, -1)))
        if len(e) > best_size:
            best_size, best_elements, best_B = len(e), e, B

    # 3. Randomized
    rng = random.Random(42)
    while _time_left() > 5.0:
        order = list(range(1, 257))
        rng.shuffle(order)
        e, B = _greedy_build(order)
        if len(e) > best_size:
            best_size, best_elements, best_B = len(e), e, B

    # 4. Polish: Try to add more
    if best_elements and _time_left() > 1.0:
        used = set(best_elements)
        cur_e = list(best_elements)
        cur_B = best_B
        cur_n = best_size
        
        # Try adding in ascending order
        for c in range(1, 257):
            if c in used:
                continue
            if _time_left() < 0.5:
                break
            new_B = _check_addition_fast(cur_B, cur_n, c)
            if new_B is not None:
                cur_e.append(c)
                cur_B = new_B
                cur_n += 1
                used.add(c)
        
        if len(cur_e) > best_size:
            best_size, best_elements, best_B = len(cur_e), cur_e, cur_B

    # 5. Local Search: Swaps (Limited)
    if best_elements and _time_left() > 2.0:
        used = set(best_elements)
        # Try a few random swaps
        for _ in range(50):
            if _time_left() < 0.5:
                break
            
            # Pick a random index to replace
            i = random.randint(0, len(best_elements) - 1)
            orig_val = best_elements[i]
            
            # Pick a random unused value smaller than orig_val? Or just any unused?
            # Smaller values might fit better in "gaps".
            candidates = [c for c in range(1, orig_val) if c not in used]
            if not candidates:
                continue
                
            new_val = random.choice(candidates)
            
            # Test the swap
            test_elements = best_elements[:i] + [new_val] + best_elements[i+1:]
            
            # Quick check: Is it valid?
            if not _is_valid(test_elements):
                continue
            
            # If valid, try to extend
            n_test = len(test_elements)
            B_test = [0] * (n_test + 1)
            B_test[0] = 1
            for elem in test_elements:
                for k in range(n_test, 0, -1):
                    B_test[k] |= (B_test[k - 1] << elem)
            
            test_used = set(test_elements)
            # Try to add one more element
            added = False
            for c in range(1, 257):
                if c in test_used:
                    continue
                new_B_test = _check_addition_fast(B_test, n_test, c)
                if new_B_test is not None:
                    test_elements.append(c)
                    B_test = new_B_test
                    n_test += 1
                    test_used.add(c)
                    added = True
                    break # Just take the first one
            
            if len(test_elements) > len(best_elements):
                best_elements = test_elements
                best_B = B_test
                best_size = len(best_elements)
                used = set(best_elements)

    return [int(x) for x in best_elements]

# EVOLVE-BLOCK-END
