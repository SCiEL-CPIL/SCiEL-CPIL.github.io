import time

def construct():
    deadline = time.time() + 80  # safety margin
    
    n = 180
    # Pre-compute all forbidden triples (a, b, c) with a < b < c
    # where 1/a = 1/b + 1/c, i.e., (b-a)(c-a) = a^2
    forbidden = []  # list of (a, b, c) tuples
    for a in range(1, n + 1):
        a2 = a * a
        # d = b - a, must divide a^2, d >= 1
        for d in range(1, a2 + 1):
            if a2 % d != 0:
                continue
            e = a2 // d  # c - a
            b = a + d
            c = a + e
            if b > n or c > n:
                continue
            if b == c:
                continue
            if b == a or c == a:
                continue
            # Ensure distinct and sorted
            if b < c:
                forbidden.append((a, b, c))
            else:
                forbidden.append((a, c, b))
    
    # Remove duplicates
    forbidden = list(set(forbidden))
    
    # Greedy: start with all elements, remove those in most conflicts
    in_set = set(range(1, n + 1))
    
    # Count conflicts per element
    def count_conflicts(s):
        """Count number of forbidden triples fully in set s."""
        count = 0
        for (a, b, c) in forbidden:
            if a in s and b in s and c in s:
                count += 1
        return count
    
    # Greedy removal: repeatedly remove the element in the most active conflicts
    # But first, let's try a simpler approach: start with a large interval and extend
    
    # Actually, let's try: start with [k, 180] for various k, then try to add smaller numbers
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Try to extend downward from 90
    # For each number from 90 down to 1, check if adding it creates conflicts
    # with existing elements
    current_set = set(range(91, 181))
    
    for a in range(90, 0, -1):
        if time.time() > deadline - 5:
            break
        # Check if adding a creates any conflict
        can_add = True
        for (x, y, z) in forbidden:
            if a in (x, y, z):
                other1 = [v for v in (x, y, z) if v != a]
                if len(other1) == 2 and other1[0] in current_set and other1[1] in current_set:
                    can_add = False
                    break
        if can_add:
            current_set.add(a)
    
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)
    
    # Now try to add more numbers that we skipped
    for a in range(90, 0, -1):
        if time.time() > deadline - 3:
            break
        if a in best_set:
            continue
        can_add = True
        for (x, y, z) in forbidden:
            if a in (x, y, z):
                others = [v for v in (x, y, z) if v != a]
                if len(others) == 2 and others[0] in best_set and others[1] in best_set:
                    can_add = False
                    break
        if can_add:
            best_set.add(a)
    
    return sorted(best_set)
