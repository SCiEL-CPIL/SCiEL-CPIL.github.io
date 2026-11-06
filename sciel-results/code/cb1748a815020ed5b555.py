# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A|+|B|.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_a = set()
    best_b = set()
    best_score = 0
    
    def check_sidon(s):
        """Check if set s is a Sidon set (unique pair sums including a+a)."""
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                s_val = lst[i] + lst[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def get_diff_set(s):
        """Get the set of nonzero differences (both positive and negative) of set s."""
        lst = sorted(s)
        diffs = set()
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i != j:
                    d = lst[i] - lst[j]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def try_add_a(x, a_set, b_set, a_sums, b_diffs):
        """Try to add x to A. Returns (new_a_set, new_a_sums) if valid, else None."""
        # Check Sidon condition for A
        for y in a_set:
            s = x + y
            if s in a_sums:
                return None
            if x == y:
                # x+x must not conflict with existing sums
                pass
        
        # More careful check
        new_sums = set(a_sums)
        for y in a_set:
            s_val = x + y
            if s_val in new_sums:
                return None
            new_sums.add(s_val)
        
        # Check difference disjointness with B
        for y in a_set:
            d1 = x - y
            d2 = y - x
            if d1 != 0 and d1 in b_diffs:
                return None
            if d2 != 0 and d2 in b_diffs:
                return None
        
        return a_set | {x}, new_sums
    
    def try_add_b(x, a_set, b_set, a_diffs, b_sums):
        """Try to add x to B. Returns (new_b_set, new_b_sums) if valid, else None."""
        new_sums = set(b_sums)
        for y in b_set:
            s_val = x + y
            if s_val in new_sums:
                return None
            new_sums.add(s_val)
        
        for y in b_set:
            d1 = x - y
            d2 = y - x
            if d1 != 0 and d1 in a_diffs:
                return None
            if d2 != 0 and d2 in a_diffs:
                return None
        
        return b_set | {x}, new_sums
    
    def greedy_construct(order_a, order_b):
        """Greedy construction with given orders."""
        a_set = set()
        b_set = set()
        a_sums = set()
        b_sums = set()
        a_diffs = set()
        b_diffs = set()
        
        # Build A first
        for x in order_a:
            result = try_add_a(x, a_set, b_set, a_sums, b_diffs)
            if result:
                a_set, a_sums = result
                # Update a_diffs
                for y in a_set:
                    if y != x:
                        a_diffs.add(x - y)
                        a_diffs.add(y - x)
        
        # Build B
        for x in order_b:
            result = try_add_b(x, a_set, b_set, a_diffs, b_sums)
            if result:
                b_set, b_sums = result
                for y in b_set:
                    if y != x:
                        b_diffs.add(x - y)
                        b_diffs.add(y - x)
        
        return a_set, b_set
    
    def greedy_interleaved(order_a, order_b):
        """Interleaved greedy construction."""
        a_set = set()
        b_set = set()
        a_sums = set()
        b_sums = set()
        a_diffs = set()
        b_diffs = set()
        
        # Interleave: try to add from A and B in alternating fashion
        ia, ib = 0, 0
        while ia < len(order_a) or ib < len(order_b):
            if ia < len(order_a) and (ib >= len(order_b) or ia % 2 == 0):
                if ia < len(order_a):
                    x = order_a[ia]
                    result = try_add_a(x, a_set, b_set, a_sums, b_diffs)
                    if result:
                        a_set, a_sums = result
                        for y in a_set:
                            if y != x:
                                a_diffs.add(x - y)
                                a_diffs.add(y - x)
                    ia += 1
            else:
                if ib < len(order_b):
                    x = order_b[ib]
                    result = try_add_b(x, a_set, b_set, a_diffs, b_sums)
                    if result:
                        b_set, b_sums = result
                        for y in b_set:
                            if y != x:
                                b_diffs.add(x - y)
                                b_diffs.add(y - x)
                    ib += 1
        
        return a_set, b_set
    
    # Try multiple random orderings
    base_order = list(range(1, 257))
    
    # First, try a deterministic approach: small numbers first
    a, b = greedy_construct(base_order, base_order)
    if len(a) + len(b) > best_score:
        best_score = len(a) + len(b)
        best_a, best_b = a, b
    
    # Try reversed order
    rev_order = list(range(256, 0, -1))
    a, b = greedy_construct(rev_order, rev_order)
    if len(a) + len(b) > best_score:
        best_score = len(a) + len(b)
        best_a, best_b = a, b
    
    # Random restarts
    rng = random.Random(42)
    max_iterations = 200
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Random perturbation of order
        order_a = base_order[:]
        order_b = base_order[:]
        
        # Randomly shuffle portions
        if rng.random() < 0.5:
            rng.shuffle(order_a)
        if rng.random() < 0.5:
            rng.shuffle(order_b)
        
        # Sometimes use interleaved
        if rng.random() < 0.3:
            a, b = greedy_interleaved(order_a, order_b)
        else:
            a, b = greedy_construct(order_a, order_b)
        
        if len(a) + len(b) > best_score:
            best_score = len(a) + len(b)
            best_a, best_b = a, b
            # If we found a good solution, try to improve locally
            if best_score >= 30:
                break
    
    # Local improvement: try to add more elements
    if time.time() < deadline - 5:
        a_set = set(best_a)
        b_set = set(best_b)
        a_sums = set()
        b_sums = set()
        a_diffs = get_diff_set(a_set)
        b_diffs = get_diff_set(b_set)
        
        for x in sorted(a_set):
            for y in a_set:
                a_sums.add(x + y)
        for x in sorted(b_set):
            for y in b_set:
                b_sums.add(x + y)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, 257):
                if x not in a_set:
                    result = try_add_a(x, a_set, b_set, a_sums, b_diffs)
                    if result:
                        a_set, a_sums = result
                        for y in a_set:
                            if y != x:
                                a_diffs.add(x - y)
                                a_diffs.add(y - x)
                        improved = True
                        break
                if time.time() > deadline - 2:
                    break
            if time.time() > deadline - 2:
                break
            for x in range(1, 257):
                if x not in b_set:
                    result = try_add_b(x, a_set, b_set, a_diffs, b_sums)
                    if result:
                        b_set, b_sums = result
                        for y in b_set:
                            if y != x:
                                b_diffs.add(x - y)
                                b_diffs.add(y - x)
                        improved = True
                        break
            if time.time() > deadline - 2:
                break
        
        if len(a_set) + len(b_set) > best_score:
            best_score = len(a_set) + len(b_set)
            best_a, best_b = a_set, b_set
    
    # Ensure nonempty
    if not best_a:
        best_a = {1}
    if not best_b:
        best_b = {2}
    
    # Final validation
    if not check_sidon(best_a) or not check_sidon(best_b):
        # Fallback to simple solution
        best_a = {1, 2, 4, 8, 16, 32, 64, 128}
        best_b = {3, 6, 12, 24, 48, 96, 192}
        if not check_sidon(best_a):
            best_a = {1, 2, 4, 8, 16, 32, 64, 128}
        if not check_sidon(best_b):
            best_b = {3, 6, 12, 24, 48, 96, 192}
    
    da = get_diff_set(best_a)
    db = get_diff_set(best_b)
    if da & db:
        # Fallback
        best_a = {1, 2, 4, 8, 16, 32, 64, 128}
        best_b = {3, 6, 12, 24, 48, 96, 192}
    
    return [list(best_a), list(best_b)]
# EVOLVE-BLOCK-END
