"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    import time as _time
    deadline = _time.time() + 130  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_mis_conflict_graph(elements, time_budget):
        """
        Find a large sum-free subset using:
        1. Build conflict graph where edge (u,v) exists if u+v is in the set
        2. Iteratively remove highest-conflict nodes
        3. Greedy independent set on residual
        4. Local search improvement
        """
        n = len(elements)
        elem_set = set(elements)
        elem_index = {e: i for i, e in enumerate(elements)}
        
        # Build conflict adjacency: for each pair (i,j), if elements[i]+elements[j] is in set,
        # then including both i and j is forbidden (since elements[i]+elements[j] would need to be excluded)
        # Actually, the constraint is: no a, b, c in subset with a+b=c.
        # So if we include a and b, we cannot include a+b (if a+b is in the original set).
        # This is a 3-uniform hypergraph independent set problem.
        
        # For greedy approach: build a conflict graph where u~v if u+v is in the set.
        # Then an independent set in this graph is sum-free? No, that's not quite right.
        # If u+v=w and w is in the set, then {u,v,w} violates. But {u,v} alone is fine.
        # The issue is we need no triple.
        
        # Better: think of it as - for each element c, the pairs (a,b) with a+b=c are conflicts.
        # We want to choose a subset with no such triple.
        
        # Strategy: 
        # 1. Start with all elements
        # 2. While there's a violating triple, remove the element with highest "conflict degree"
        # 3. Then try to add back removed elements if they don't create conflicts
        
        current = list(elements)
        current_set = set(current)
        
        def find_violations(s):
            """Find all violating triples in set s."""
            violations = []
            s_list = list(s)
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(len(s_list)):
                    b = s_list[j]
                    c = a + b
                    if c in s:
                        violations.append((a, b, c))
            return violations
        
        def conflict_degree(s, elem):
            """Count how many triples involve this element."""
            count = 0
            s_list = list(s)
            for a in s_list:
                if a + elem in s:
                    count += 1
            for b in s_list:
                if elem + b in s and b != a:  # avoid double counting
                    pass
            # Simpler: count pairs (x,y) in s such that x+y=elem or elem+y=x or x+elem=y
            # i.e., elem is involved as a, b, or c
            as_c = 0
            as_a_or_b = 0
            for x in s_list:
                if x != elem and (x + elem) in s:
                    as_a_or_b += 1
            for x in s_list:
                for y in s_list:
                    if x + y == elem:
                        as_c += 1
            return as_a_or_b + as_c
        
        # Iterative removal of high-conflict elements
        removed = set()
        remaining = set(elements)
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if _time.time() > deadline:
                break
            # Find a violating triple
            found = False
            rem_list = list(remaining)
            for a in rem_list:
                for b in rem_list:
                    c = a + b
                    if c in remaining:
                        found = True
                        # Choose which to remove: the one with highest conflict degree
                        candidates = [a, b, c]
                        best_to_remove = None
                        best_deg = -1
                        for cand in candidates:
                            if cand in remaining:
                                deg = conflict_degree(remaining, cand)
                                if deg > best_deg:
                                    best_deg = deg
                                    best_to_remove = cand
                        remaining.discard(best_to_remove)
                        removed.add(best_to_remove)
                        break
                if found:
                    break
            if not found:
                break
        
        # Now try to add back removed elements
        for elem in list(removed):
            if _time.time() > deadline:
                break
            test_set = remaining | {elem}
            # Check if adding elem creates a violation
            valid = True
            for a in test_set:
                if a != elem:
                    if a + elem in test_set:
                        valid = False
                        break
                    if elem + a in test_set:
                        valid = False
                        break
            # Also check if elem is the sum
            if valid:
                for a in test_set:
                    for b in test_set:
                        if a != elem and b != elem and a + b == elem:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                remaining.add(elem)
        
        # Local search: try swapping
        result = list(remaining)
        result_set = set(result)
        
        # Try to add any element not in result
        for elem in elements:
            if elem in result_set:
                continue
            if _time.time() > deadline:
                break
            test_set = result_set | {elem}
            valid = True
            # Check all triples involving elem
            for a in test_set:
                for b in test_set:
                    if a + b == elem or elem + b == a or a + elem == b:
                        # Check if this creates a new violation
                        if a in result_set and b in result_set and (a + b) in test_set:
                            if a + b == elem:
                                # elem is the sum, a and b already in set - violation
                                valid = False
                                break
                        if a in result_set and b in result_set and (a + b) in result_set:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                result_set.add(elem)
                result.append(elem)
        
        return sorted(result_set)

    def greedy_sum_free(elements, time_budget):
        """
        Greedy approach: sort elements and build subset ensuring sum-free property.
        Use the odd/even split as a baseline, then try to improve.
        """
        # Start with all odd elements (sum-free)
        odds = [x for x in elements if x % 2 != 0]
        evens = [x for x in elements if x % 2 == 0]
        
        # Also consider large magnitude elements
        # Elements with |x| > max/2 can't be expressed as sum of two smaller elements in range
        max_abs = max(abs(x) for x in elements) if elements else 0
        
        # Strategy: start with odds, try to add evens if they don't create conflicts
        result = set(odds)
        
        # Try to add evens
        for e in sorted(evens, key=lambda x: -abs(x)):
            if _time.time() > deadline:
                break
            # Check if adding e creates a violation
            valid = True
            for a in result:
                if a + e in result:
                    valid = False
                    break
                if e + a in result:
                    valid = False
                    break
            if valid:
                # Check if e is sum of two in result
                for a in result:
                    for b in result:
                        if a + b == e:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                result.add(e)
        
        # Try to add remaining odds that were excluded? No, all odds are included.
        # Try local search: remove an element to allow adding two others
        result_list = list(result)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            for i in range(len(result_list)):
                for j in range(i+1, len(result_list)):
                    if _time.time() > deadline:
                        break
                    a, b = result_list[i], result_list[j]
                    # Try removing a and b, see if we can add more
                    test = result - {a, b}
                    # Try to add elements not in test
                    added = 0
                    for elem in elements:
                        if elem in test:
                            continue
                        valid = True
                        for x in test:
                            if x + elem in test or elem + x in test:
                                valid = False
                                break
                        if valid:
                            for x in test:
                                for y in test:
                                    if x + y == elem:
                                        valid = False
                                        break
                                if not valid:
                                    break
                        if valid:
                            test.add(elem)
                            added += 1
                    if len(test) > len(result):
                        result = test
                        result_list = list(result)
                        improved = True
                        break
                if improved:
                    break
        
        return sorted(result)

    def solve_instance(elements, time_budget):
        """Solve a single instance, trying multiple strategies."""
        if not elements:
            return []
        
        best = []
        best_size = 0
        
        # Strategy 1: Odd numbers
        odds = [x for x in elements if x % 2 != 0]
        if len(odds) > best_size:
            best = odds
            best_size = len(odds)
        
        # Strategy 2: Large magnitude elements
        max_abs = max(abs(x) for x in elements)
        half_max = max_abs // 2
        large = [x for x in elements if abs(x) > half_max]
        # Check if large is sum-free
        if is_sum_free(large) and len(large) > best_size:
            best = large
            best_size = len(large)
        
        # Strategy 3: Greedy with local search
        greedy_result = greedy_sum_free(elements, time_budget)
        if len(greedy_result) > best_size:
            best = greedy_result
            best_size = len(greedy_result)
        
        # Strategy 4: Conflict graph removal
        if _time.time() < deadline - 5:
            conflict_result = solve_mis_conflict_graph(elements, time_budget)
            if len(conflict_result) > best_size:
                best = conflict_result
                best_size = len(conflict_result)
        
        # Verify and return
        if not is_sum_free(best):
            # Fallback to odds
            best = odds
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        remaining_time = deadline - _time.time()
        if remaining_time <= 0:
            # Emergency fallback
            results.append([x for x in elements if x % 2 != 0])
            continue
        result = solve_instance(elements, remaining_time)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
