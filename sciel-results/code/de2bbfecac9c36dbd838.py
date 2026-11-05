"""Optimized construction for max 3-AP, no 4-AP subset of [1,256]."""
import time
import random
import sys

N = 256
DEADLINE_MARGIN = 10  # seconds before the 90s limit

# Precompute all 4-term APs
FAP = []  # list of (a, b, c, d) tuples
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            FAP.append((a, b, c, e))

# For each element, precompute which 4-APs it participates in
# And for 3-AP counting, precompute pairs
FAP_COUNT = len(FAP)

# Precompute 3-AP count: for each pair (a, c) with a < c and (c-a) even, the middle b
# Actually let's just count 3-APs directly from the set

def count_3aps(selected):
    """Count distinct 3-term APs with positive difference in selected set."""
    count = 0
    sel = set(selected)
    for a in range(1, N + 1):
        if a not in sel:
            continue
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = b + d
            if b in sel and c in sel:
                count += 1
    return count

def has_4ap(selected):
    """Check if selected set has any nonconstant 4-term AP."""
    sel = set(selected)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return True
    return False

def greedy_construct():
    """Greedy: add elements that maximize 3-AP contribution while avoiding 4-APs."""
    selected = set()
    for x in range(1, N + 1):
        candidate = selected | {x}
        if not has_4ap(candidate):
            selected.add(x)
    return sorted(selected)

def greedy_reverse_construct():
    """Start with all elements, remove those in 4-APs to maximize remaining 3-APs."""
    selected = set(range(1, N + 1))
    changed = True
    while changed:
        changed = False
        # Find all 4-APs in selected
        violating = []
        for (a, b, c, e) in FAP:
            if a in selected and b in selected and c in selected and e in selected:
                violating.append((a, b, c, e))
        if not violating:
            break
        # For each element, count how many 4-APs it's in
        from collections import Counter
        elem_4ap_count = Counter()
        for (a, b, c, e) in violating:
            for elem in (a, b, c, e):
                elem_4ap_count[elem] += 1
        # Remove the element that appears in the most 4-APs
        worst_elem = max(elem_4ap_count, key=lambda x: elem_4ap_count[x])
        selected.remove(worst_elem)
        changed = True
    return sorted(selected)

def local_search(initial, time_limit=5.0):
    """Local search: try adding/removing elements to improve 3-AP count."""
    start = time.time()
    selected = set(initial)
    current_score = count_3aps(selected)
    
    best_selected = set(selected)
    best_score = current_score
    
    # Precompute for speed
    while time.time() - start < time_limit:
        improved = False
        # Try adding elements
        for x in range(1, N + 1):
            if x in selected:
                continue
            candidate = selected | {x}
            if has_4ap(candidate):
                continue
            new_score = count_3aps(candidate)
            if new_score > current_score:
                selected = candidate
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
                break
        if not improved:
            # Try removing elements
            for x in list(selected):
                candidate = selected - {x}
                # Removing can't create 4-APs, so always valid
                new_score = count_3aps(candidate)
                if new_score > current_score:
                    selected = candidate
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(selected)
                    break
        if not improved:
            # Try swaps
            for x in range(1, N + 1):
                if x in selected:
                    continue
                for y in list(selected):
                    candidate = (selected - {y}) | {x}
                    if has_4ap(candidate):
                        continue
                    new_score = count_3aps(candidate)
                    if new_score > current_score:
                        selected = candidate
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = set(selected)
                        break
                if improved:
                    break
            if not improved:
                break
    
    return sorted(best_selected), best_score

def construct():
    start_time = time.time()
    
    # Try multiple strategies
    best_result = []
    best_score = -1
    
    # Strategy 1: Greedy forward
    if time.time() - start_time < DEADLINE_MARGIN:
        r1 = greedy_construct()
        s1 = count_3aps(r1)
        if s1 > best_score:
            best_score = s1
            best_result = r1
        # Local search from this
        if time.time() - start_time < DEADLINE_MARGIN:
            r1_ls, s1_ls = local_search(r1, time_limit=min(3.0, DEADLINE_MARGIN - (time.time() - start_time)))
            if s1_ls > best_score:
                best_score = s1_ls
                best_result = r1_ls
    
    # Strategy 2: Greedy reverse
    if time.time() - start_time < DEADLINE_MARGIN:
        r2 = greedy_reverse_construct()
        s2 = count_3aps(r2)
        if s2 > best_score:
            best_score = s2
            best_result = r2
        # Local search from this
        if time.time() - start_time < DEADLINE_MARGIN:
            r2_ls, s2_ls = local_search(r2, time_limit=min(3.0, DEADLINE_MARGIN - (time.time() - start_time)))
            if s2_ls > best_score:
                best_score = s2_ls
                best_result = r2_ls
    
    # Strategy 3: Randomized greedy with multiple restarts
    if time.time() - start_time < DEADLINE_MARGIN:
        for trial in range(20):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            random.seed(trial)
            order = list(range(1, N + 1))
            random.shuffle(order)
            selected = set()
            for x in order:
                candidate = selected | {x}
                if not has_4ap(candidate):
                    selected.add(x)
            r3 = sorted(selected)
            s3 = count_3aps(r3)
            if s3 > best_score:
                best_score = s3
                best_result = r3
    
    # Final local search on best
    if best_result and time.time() - start_time < DEADLINE_MARGIN:
        final_ls, final_s = local_search(best_result, time_limit=min(10.0, DEADLINE_MARGIN - (time.time() - start_time)))
        if final_s > best_score:
            best_score = final_s
            best_result = final_ls
    
    if not best_result:
        # Fallback: just return a small valid set
        best_result = [1]
    
    return best_result

# Verify no 4-APs in the result
def validate(result):
    sel = set(result)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return False
    return True

if __name__ == "__main__":
    result = construct()
    score = count_3aps(result)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")
