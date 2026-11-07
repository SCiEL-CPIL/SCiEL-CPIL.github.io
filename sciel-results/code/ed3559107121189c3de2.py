import time
import random

def _is_valid_mask(mask):
    """Check if mask represents a valid set (no forbidden triple).
    A triple a,b,c (distinct) is forbidden if a+b, a+c, b+c are all in S.
    We check: for all distinct a<b<c in S, if (a+b) in S and (a+c) in S and (b+c) in S -> invalid.
    """
    elems = [i for i in range(1, 129) if (mask >> (i-1)) & 1]
    S = set(elems)
    for i in range(len(elems)):
        a = elems[i]
        for j in range(i+1, len(elems)):
            b = elems[j]
            for k in range(j+1, len(elems)):
                c = elems[k]
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    return False
    return True

def _is_valid_mask_fast(mask, S_set):
    """Fast validity check using precomputed set."""
    for i in range(1, 129):
        if not ((mask >> (i-1)) & 1):
            continue
        a = i
        for b in range(a+1, 129):
            if not ((mask >> (b-1)) & 1):
                continue
            for c in range(b+1, 129):
                if not ((mask >> (c-1)) & 1):
                    continue
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _mask_to_set(mask):
    return [i for i in range(1, 129) if (mask >> (i-1)) & 1]

def _set_to_mask(S):
    mask = 0
    for v in S:
        mask |= (1 << (v-1))
    return mask

def construct():
    start = time.time()
    deadline = 85.0  # safety margin
    
    best_mask = 0
    best_size = 0
    
    # Precompute all valid masks? Too many. Instead, use greedy + local search.
    
    def greedy_random():
        """Randomized greedy: pick random order, add if valid."""
        order = list(range(1, 129))
        random.shuffle(order)
        mask = 0
        S = set()
        for x in order:
            # Check if adding x creates a forbidden triple
            # New triples involving x: (x, a, b) where a,b in S and x+a, x+b, a+b in S∪{x}
            valid = True
            # Triples where x is the largest: a, b in S, x in S
            # Need a+b in S∪{x}, a+x in S∪{x}, b+x in S∪{x}
            # Since x is new, a+x > x so a+x must be in S (if a+x != x, which it isn't since a>0)
            # Actually a+x could be <= 128 or > 128
            # Simpler: just check all triples with x
            for i in range(len(S)):
                a = list(S)[i]
                for j in range(i+1, len(S)):
                    b = list(S)[j]
                    # Triple (x, a, b) - need distinct, so x != a, x != b, a != b (a,b distinct from S)
                    # Check if x+a, x+b, a+b are all in S ∪ {x}
                    if x + a in S or x + a == x:  # x+a == x means a=0, impossible
                        if x + b in S or x + b == x:
                            if a + b in S or a + b == x:
                                valid = False
                                break
                if not valid:
                    break
            if valid:
                mask |= (1 << (x-1))
                S.add(x)
        return mask, len(S)
    
    def local_search(mask, S):
        """Try to improve by adding elements or swapping."""
        # Try adding
        for x in range(1, 129):
            if (mask >> (x-1)) & 1:
                continue
            # Check validity
            S_new = S | {x}
            # Check triples involving x
            valid = True
            S_list = sorted(S_new)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    for k in range(j+1, len(S_list)):
                        c = S_list[k]
                        if (a+b) in S_new and (a+c) in S_new and (b+c) in S_new:
                            if a == x or b == x or c == x:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                mask |= (1 << (x-1))
                S.add(x)
                return True
        return False
    
    # Phase 1: Multiple randomized greedy restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        mask, size = greedy_random()
        if size > best_size:
            best_mask = mask
            best_size = size
    
    # Phase 2: Local search on best
    S_best = set(_mask_to_set(best_mask))
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try adding elements in random order
        candidates = [x for x in range(1, 129) if not (best_mask >> (x-1)) & 1]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            # Check if adding x is valid
            S_test = S_best | {x}
            S_list = sorted(S_test)
            valid = True
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    for k in range(j+1, len(S_list)):
                        c = S_list[k]
                        if (a+b) in S_test and (a+c) in S_test and (b+c) in S_test:
                            if a == x or b == x or c == x:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_mask |= (1 << (x-1))
                S_best.add(x)
                best_size += 1
                improved = True
    
    # Phase 3: Try swaps (remove one, add one) to escape local optima
    if time.time() < deadline - 3.0:
        S_list = sorted(S_best)
        for _ in range(50):
            if time.time() > deadline - 3.0:
                break
            # Pick random element to remove
            if not S_list:
                break
            rem = random.choice(S_list)
            # Try adding random elements
            adds = [x for x in range(1, 129) if x not in S_best]
            random.shuffle(adds)
            found = False
            for add in adds[:20]:
                if time.time() > deadline - 3.0:
                    break
                S_test = (S_best - {rem}) | {add}
                S_list_test = sorted(S_test)
                valid = True
                for i in range(len(S_list_test)):
                    a = S_list_test[i]
                    for j in range(i+1, len(S_list_test)):
                        b = S_list_test[j]
                        for k in range(j+1, len(S_list_test)):
                            c = S_list_test[k]
                            if (a+b) in S_test and (a+c) in S_test and (b+c) in S_test:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_mask = _set_to_mask(S_test)
                    S_best = S_test
                    S_list = sorted(S_best)
                    found = True
                    break
            if found:
                break
    
    # Final validation
    if not _is_valid_mask_fast(best_mask, set(_mask_to_set(best_mask))):
        # Fallback to a known valid set
        S = set(range(65, 129))
        best_mask = _set_to_mask(S)
    
    result = _mask_to_set(best_mask)
    return sorted(result)
