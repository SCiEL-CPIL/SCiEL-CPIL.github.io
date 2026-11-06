"""Improved program: difference-partitioning with greedy construction and local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit (evaluator kills at 90s)
    
    n = 256
    max_size = 64
    
    def check_sidon(s):
        """Check if set s is Sidon: all pair sums (including a+a) are unique."""
        if len(s) <= 1:
            return True
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def get_diffs(s):
        """Get nonzero differences of set s."""
        if len(s) <= 1:
            return set()
        diffs = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d = s[j] - s[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def can_add_to_sidon(s, x):
        """Check if x can be added to Sidon set s."""
        for a in s:
            sm = a + x
            for b in s:
                if b != a and a + b == sm:
                    return False
                if b == a and b + b == sm:
                    # a+a == a+x means a==x, but x not in s (we check uniqueness)
                    pass
        # More careful check:
        existing_sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                existing_sums.add(s[i] + s[j])
        for a in s:
            sm = a + x
            if sm in existing_sums:
                return False
            existing_sums.add(sm)
        # Also check x+x
        sm = x + x
        if sm in existing_sums:
            return False
        return True
    
    def greedy_construct(diffs_pool, elements, max_elems):
        """Greedy construct a Sidon set from elements using only diffs in diffs_pool."""
        s = []
        used_diffs = set()
        for x in elements:
            if len(s) >= max_elems:
                break
            # Check Sidon property
            if not can_add_to_sidon(s, x):
                continue
            # Check differences
            new_diffs = set()
            for a in s:
                d1 = x - a
                d2 = a - x
                new_diffs.add(d1)
                new_diffs.add(d2)
            # Filter out zero
            new_diffs.discard(0)
            # Check if any new diff conflicts with used_diffs or is not in pool
            conflict = False
            for d in new_diffs:
                if d in used_diffs:
                    conflict = True
                    break
                if diffs_pool is not None and d not in diffs_pool:
                    conflict = True
                    break
            if conflict:
                continue
            s.append(x)
            used_diffs |= new_diffs
        return s, used_diffs
    
    def local_search(s, elements, diffs_pool, other_diffs, max_elems, iterations=100):
        """Local search to improve Sidon set size."""
        s = list(s)
        used_diffs = get_diffs(s)
        improved = True
        it = 0
        while improved and it < iterations:
            improved = False
            it += 1
            # Try removing one element and adding two
            for i in range(len(s)):
                s_without = s[:i] + s[i+1:]
                # Try adding elements
                candidates = [x for x in elements if x not in s_without]
                random.shuffle(candidates)
                added = 0
                for x in candidates:
                    if len(s_without) + added >= max_elems:
                        break
                    if not can_add_to_sidon(s_without, x):
                        continue
                    new_diffs = set()
                    for a in s_without:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    new_diffs.discard(0)
                    conflict = False
                    for d in new_diffs:
                        if d in used_diffs:
                            conflict = True
                            break
                        if d in other_diffs:
                            conflict = True
                            break
                        if diffs_pool is not None and d not in diffs_pool:
                            conflict = True
                            break
                    if conflict:
                        continue
                    s_without.append(x)
                    used_diffs |= new_diffs
                    added += 1
                if len(s_without) > len(s):
                    s = s_without
                    used_diffs = get_diffs(s)
                    improved = True
        return s, used_diffs
    
    # Strategy 1: Simple greedy with difference partitioning
    # Partition differences: positive vs negative, or by magnitude
    all_diffs = set(range(1, n))  # 1 to 255
    
    # Try different partitions
    best_A = []
    best_B = []
    best_total = 0
    
    # Strategy: assign differences by parity or by range
    partitions = [
        (set(d for d in range(1, n) if d % 2 == 1), set(d for d in range(1, n) if d % 2 == 0)),
        (set(d for d in range(1, n//2+1)), set(d for d in range(n//2+1, n))),
        (set(d for d in range(1, n) if d % 3 != 0), set(d for d in range(1, n) if d % 3 == 0)),
        (set(d for d in range(1, n) if d <= 128), set(d for d in range(1, n) if d > 128)),
    ]
    
    for diff_pool_A, diff_pool_B in partitions:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Greedy for A
        elements_A = list(range(1, n+1))
        random.shuffle(elements_A)
        A, diffs_A = greedy_construct(diff_pool_A, elements_A, max_size)
        
        # Greedy for B, avoiding diffs_A
        other_diffs = diffs_A
        elements_B = list(range(1, n+1))
        random.shuffle(elements_B)
        B, diffs_B = greedy_construct(diff_pool_B, elements_B, max_size)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A
            best_B = B
    
    # Strategy 2: More systematic - try to maximize each set independently
    # with careful difference management
    if time.time() - start_time < time_limit - 10:
        # Try constructing A first with all diffs, then B with remaining
        for seed in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            random.seed(seed)
            
            # Build A greedily
            elements = list(range(1, n+1))
            random.shuffle(elements)
            A = []
            diffs_A = set()
            for x in elements:
                if len(A) >= max_size:
                    break
                if not can_add_to_sidon(A, x):
                    continue
                new_diffs = set()
                for a in A:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                new_diffs.discard(0)
                if new_diffs & diffs_A:
                    continue
                A.append(x)
                diffs_A |= new_diffs
            
            # Build B with remaining diffs
            B = []
            diffs_B = set()
            for x in elements:
                if len(B) >= max_size:
                    break
                if not can_add_to_sidon(B, x):
                    continue
                new_diffs = set()
                for b in B:
                    new_diffs.add(x - b)
                    new_diffs.add(b - x)
                new_diffs.discard(0)
                if new_diffs & diffs_A:  # Must avoid A's diffs
                    continue
                if new_diffs & diffs_B:
                    continue
                B.append(x)
                diffs_B |= new_diffs
            
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = A
                best_B = B
    
    # Local search improvement
    if time.time() - start_time < time_limit - 5:
        all_elements = set(range(1, n+1))
        diffs_A = get_diffs(best_A)
        diffs_B = get_diffs(best_B)
        
        # Try to improve A
        A_improved, _ = local_search(best_A, list(all_elements), None, diffs_B, max_size, iterations=50)
        if len(A_improved) > len(best_A):
            best_A = A_improved
            diffs_A = get_diffs(best_A)
        
        # Try to improve B
        B_improved, _ = local_search(best_B, list(all_elements), None, diffs_A, max_size, iterations=50)
        if len(B_improved) > len(best_B):
            best_B = B_improved
            diffs_B = get_diffs(best_B)
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation
    # Check Sidon property
    for s in (best_A, best_B):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                assert sm not in sums, f"Duplicate sum {sm} in {s}"
                sums.add(sm)
    
    # Check disjoint differences
    diffs_A = get_diffs(best_A)
    diffs_B = get_diffs(best_B)
    assert not (diffs_A & diffs_B), "Differences not disjoint"
    
    # Cap sizes
    best_A = best_A[:max_size]
    best_B = best_B[:max_size]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
