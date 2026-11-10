"""Rewritten program: Maximum Independent Set via bitset branch-and-bound."""
import time
from itertools import combinations

def _build_conflict_graph():
    """Build conflict graph: edge (a,b) if some selected set containing both
    would violate the 'no element divides sum of 3 distinct others' rule.
    
    Two integers a,b conflict if there exist distinct x,y (x,y != a,b) such that
    a | (b+x+y) or b | (a+x+y).
    """
    n = 192
    # conflict_mask[i] is a bitset of integers j that conflict with i
    conflict_mask = [0] * (n + 1)
    
    # Precompute all triples' sums for efficiency
    # For each pair (a,b), check if there exist x,y != a,b with a|(b+x+y)
    # b+x+y ranges from b+1+2 to b+(n-1)+(n-2) (excluding a)
    # Actually x,y are distinct from a and b, and x != y
    # Sum = b + x + y where x,y in [1,n], x!=a, x!=b, y!=a, y!=b, x!=y
    # So x+y ranges over sums of two distinct numbers from [1,n] \ {a,b}
    
    # Precompute all possible sums of two distinct numbers
    # sum_two[i][j] = i+j for i<j
    # We need: for given a, b, does there exist x,y distinct from a,b and from each other
    # such that (b + x + y) % a == 0?
    
    # For efficiency: for each a, precompute which "s" values (x+y) make (b+s) % a == 0
    # s ranges from 1+2=3 to (n-1)+(n-2) but excluding pairs involving a or b
    
    # Let's precompute for each a: the set of s values (x+y, x!=y, x,y in 1..n) 
    # such that s % a gives a specific residue. Actually we need (b+s)%a==0, i.e., s%a == (-b)%a
    
    # For each a, compute achievable residues of (x+y) mod a where x,y distinct, x,y != a
    # Then for pair (a,b): conflict if (-b) mod a is achievable as (x+y) mod a with x,y distinct from a,b
    
    # This is complex. Let me use a simpler but still efficient approach:
    # For each pair (a,b), iterate over possible x values and check if a valid y exists.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            # Check if a | (b + x + y) for some distinct x,y not in {a,b}
            # b+x+y ≡ 0 (mod a) => x+y ≡ -b (mod a)
            target_mod_a = (-b) % a
            # Check if there exist x,y distinct, not in {a,b}, x!=y, with (x+y) % a == target_mod_a
            # Also check b | (a + x + y): x+y ≡ -a (mod b)
            target_mod_b = (-a) % b
            
            found_a = False
            found_b = False
            
            # For a: find x,y in [1,n]\{a,b}, x<y, (x+y)%a == target_mod_a
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                # Need y > x, y != a, y != b, (x+y) % a == target_mod_a
                # y ≡ (target_mod_a - x) % a (mod a)
                rem = (target_mod_a - x) % a
                # y in [x+1, n], y != a, y != b, y % a == rem
                # y = rem + k*a for some k
                if rem == 0:
                    start_y = a  # smallest positive y with y%a==0
                else:
                    start_y = rem
                # Find smallest y >= max(x+1, start_y) with y%a==rem
                lo = max(x + 1, start_y)
                if lo % a != rem:
                    lo += a - (lo % a)
                if lo <= n:
                    if lo != a and lo != b:
                        found_a = True
                    elif lo + a <= n:
                        # Check lo+a
                        y2 = lo + a
                        if y2 != a and y2 != b:
                            found_a = True
                    # Also check if there's another valid y (in case lo==a or lo==b but lo+a valid)
                    # Already handled above. Let's just check a couple candidates.
                    if not found_a:
                        # Try lo and lo+a and lo+2a
                        for y in [lo, lo + a, lo + 2 * a]:
                            if y > x and y <= n and y != a and y != b:
                                found_a = True
                                break
            
            # For b: find x,y in [1,n]\{a,b}, x<y, (x+y)%b == target_mod_b
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                rem = (target_mod_b - x) % b
                if rem == 0:
                    start_y = b
                else:
                    start_y = rem
                lo = max(x + 1, start_y)
                if lo % b != rem:
                    lo += b - (lo % b)
                if lo <= n:
                    if lo != a and lo != b:
                        found_b = True
                    elif lo + b <= n:
                        y2 = lo + b
                        if y2 != a and y2 != b:
                            found_b = True
                    if not found_b:
                        for y in [lo, lo + b, lo + 2 * b]:
                            if y > x and y <= n and y != a and y != b:
                                found_b = True
                                break
            
            if found_a or found_b:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    return conflict_mask

def _max_independent_set(conflict_mask, n, deadline):
    """Find maximum independent set in graph with n vertices (1-indexed) using
    branch-and-bound with bitsets."""
    # Vertices 1..n, conflict_mask[i] has bit j set if i conflicts with j
    # Independent set: no two vertices in the set have an edge between them
    
    best_size = 0
    best_set = []
    
    # Use bitsets for the candidate pool
    # available: bitmask of vertices still available
    # current: list of vertices in current independent set
    
    def popcount(x):
        return bin(x).count('1')
    
    def lsb(x):
        return x & (-x)
    
    def next_bit(x):
        """Return index of least significant set bit (1-indexed)"""
        if x == 0:
            return -1
        # Count trailing zeros
        t = x & (-x)
        # t is power of 2, index is log2(t)
        idx = 0
        while t > 1:
            t >>= 1
            idx += 1
        return idx + 1  # 1-indexed
    
    def branch(available, current_size, current_set, depth):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound: current_size + popcount(available)
        if current_size + popcount(available) <= best_size:
            return
        
        # Greedy coloring upper bound on the subgraph induced by available
        # Color the available vertices; max independent set <= number of colors
        # Simple greedy coloring
        uncolored = available
        color_classes = []
        while uncolored:
            color_class = 0
            remaining = uncolored
            while remaining:
                v = next_bit(remaining)
                # Can we add v to current color class? v must not conflict with any in color_class
                can_add = True
                temp = color_class
                while temp:
                    u = next_bit(temp)
                    if conflict_mask[v] & (1 << u):
                        can_add = False
                        break
                    temp &= temp - 1
                if can_add:
                    color_class |= (1 << v)
                    remaining &= ~(1 << v)
                else:
                    remaining &= ~(1 << v)
            if color_class == 0:
                break
            color_classes.append(color_class)
            uncolored &= ~color_class
        
        num_colors = len(color_classes)
        if current_size + num_colors <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in available subgraph
        v = -1
        max_deg = -1
        temp = available
        while temp:
            u = next_bit(temp)
            deg = popcount(conflict_mask[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u
            temp &= temp - 1
        
        if v == -1:
            return
        
        # Branch 1: include v
        new_available = available & ~(1 << v) & ~conflict_mask[v]
        current_set.append(v)
        branch(new_available, current_size + 1, current_set, depth + 1)
        current_set.pop()
        
        # Branch 2: exclude v
        new_available2 = available & ~(1 << v)
        branch(new_available2, current_size, current_set, depth + 1)
    
    # Start with all vertices available
    all_available = 0
    for i in range(1, n + 1):
        all_available |= (1 << i)
    
    branch(all_available, 0, [], 0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    n = 192
    conflict_mask = _build_conflict_graph()
    
    # Verify no self-conflicts
    for i in range(1, n + 1):
        conflict_mask[i] &= ~(1 << i)
    
    # Get maximum independent set
    result = _max_independent_set(conflict_mask, n, deadline)
    
    # Verify the result is valid
    def is_valid(selected):
        sel_set = set(selected)
        for a in selected:
            others = [v for v in selected if v != a]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
        return True
    
    if not is_valid(result):
        # Fallback to greedy
        selected = []
        for x in range(n, 0, -1):
            candidate = selected + [x]
            valid = True
            for a in candidate:
                others = [v for v in candidate if v != a]
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(x)
        result = selected
    
    return result
