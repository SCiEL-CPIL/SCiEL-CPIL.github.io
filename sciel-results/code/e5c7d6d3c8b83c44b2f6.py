"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_N = 256
_DEADLINE_FRAC = 0.85  # safety margin: stop at 85% of 90s = ~76.5s

def _build_conflict_bitsets():
    """
    For each x in 1..N, build a bitset (Python int) of all y in 1..N
    such that {x, y} is part of a nonconstant 3-AP.
    A 3-AP is (a, b, c) with a < b < c and b - a == c - b, i.e., a + c = 2b.
    For a given x, the conflicts are:
      - x as the middle: a + c = 2x, a < x < c  => a in [1, x-1], c = 2x - a in [x+1, N]
      - x as the first:  x + c = 2b, b > x      => b in [x+1, N], c = 2b - x in [x+1, N]
        Wait, that's not right. Let me reconsider.
      
      A 3-AP (a, b, c) with a < b < c, b = (a+c)/2.
      For element x:
        Case 1: x = a (first term). Then b and c exist with b - a = c - b => c = 2b - a = 2b - x.
                b ranges from x+1 to N, and c = 2b - x must be <= N and > b.
                So b in [x+1, floor((N+x)/2)], c = 2b - x.
                Conflicts for x: all b and c in these triples.
        Case 2: x = b (middle term). Then a + c = 2x, a < x < c.
                a in [1, x-1], c = 2x - a in [x+1, N].
                Conflicts for x: all a and c in these triples.
        Case 3: x = c (last term). Then a + c = 2b => a = 2b - x, with a < b < x.
                b in [ceil((x+1)/2), x-1], a = 2b - x in [1, b-1].
                Conflicts for x: all a and b in these triples.
      
      But for the conflict graph, we just need: for each x, the set of all y != x
      such that x and y appear together in some 3-AP.
      
      Actually, let's think differently. Two elements x and y are in conflict
      if there exists a third element z such that {x, y, z} forms a 3-AP.
      The three elements form a 3-AP iff they are {a, a+d, a+2d} for some d >= 1.
      So x and y conflict if:
        - y - x = d and x - d >= 1 (x is middle, y is last, first is x-d)
          OR y - x = 2d and x + d <= N (x is first, y is last, middle is x+d)
          OR x - y = d and y + d <= N (y is middle, x is last, third is y+d)
          OR x - y = 2d and y - d >= 1 (y is first, x is last, middle is y+d)
      
      More simply: x and y (x < y) conflict if:
        (y - x) is even and (x + (y-x)//2) is in [1, N]  -> they are endpoints of a 3-AP
        OR there exists z < x such that x - z == y - x  -> z, x, y is a 3-AP, so z = 2x - y >= 1
        OR there exists z > y such that z - y == y - x  -> x, y, z is a 3-AP, so z = 2y - x <= N
        OR there exists z between x and y... wait, if x < z < y and z-x == y-z, then z = (x+y)/2, 
          which requires x+y even. That's the same as the first case.
      
      Let me just enumerate all 3-APs and build the conflict sets.
    """
    # conflict[x] is a set of y's that conflict with x
    conflict = [set() for _ in range(N + 1)]
    
    # Enumerate all 3-APs: (a, a+d, a+2d) with a >= 1, d >= 1, a+2d <= N
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            c = a + 2 * d
            if c > N:
                break
            b = a + d
            # The three elements are a, b, c
            # Each pair conflicts
            conflict[a].add(b)
            conflict[a].add(c)
            conflict[b].add(a)
            conflict[b].add(c)
            conflict[c].add(a)
            conflict[c].add(b)
    
    # Convert to bitsets for fast operations
    # bitset[x] is a Python int where bit y is set if y conflicts with x
    bitsets = [0] * (N + 1)
    for x in range(1, N + 1):
        bits = 0
        for y in conflict[x]:
            bits |= (1 << y)
        bitsets[x] = bits
    
    return bitsets

def _greedy_initial():
    """Greedy: scan 1..N, add if no conflict with existing."""
    selected = set()
    for x in range(1, N + 1):
        # Check if x conflicts with any selected element
        # A conflict exists if there's y in selected such that x and y are in a 3-AP
        # We can check: for each y in selected, is x in conflict[y]?
        # But that's O(|selected|) per x. Let's use a different approach.
        # Actually, we can maintain a "blocked" bitset.
        pass
    
    # Simpler: just check directly
    selected_list = []
    for x in range(1, N + 1):
        ok = True
        for y in selected_list:
            # Check if x and y are in a 3-AP
            # They are if there exists z such that {x,y,z} is a 3-AP
            # Without loss, assume x < y (or handle both)
            a, b = (x, y) if x < y else (y, x)
            diff = b - a
            # Case 1: a, b are endpoints: middle = a + diff/2, requires diff even
            if diff % 2 == 0:
                mid = a + diff // 2
                if 1 <= mid <= N and mid != a and mid != b:
                    ok = False
                    break
            # Case 2: a, b are first two: third = b + diff, requires b + diff <= N
            if b + diff <= N:
                ok = False
                break
            # Case 3: a, b are last two: first = a - diff, requires a - diff >= 1
            if a - diff >= 1:
                ok = False
                break
        if ok:
            selected_list.append(x)
    
    return set(selected_list)

def _reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    s = Fraction(0, 1)
    for x in selected:
        s += Fraction(1, x)
    return s

def _is_valid(selected):
    """Check that selected has no 3-AP."""
    sel = set(selected)
    for a in sel:
        for b in sel:
            if a >= b:
                continue
            c = 2 * b - a
            if c in sel and c != a and c != b:
                return False
    return True

def _local_search(selected, bitsets, deadline):
    """
    Local search: try to remove one element and add one or more others.
    Focus on: remove x, then greedily add all elements that become valid.
    Accept if the net reciprocal sum improves.
    """
    current_sum = _reciprocal_sum(selected)
    best_sum = current_sum
    best_set = set(selected)
    start_time = time.time()
    
    # Precompute reciprocal values
    recips = [Fraction(0, 1)] + [Fraction(1, x) for x in range(1, N + 1)]
    
    # Convert selected to a set for fast lookup
    sel = set(selected)
    
    # Build a "conflict count" for each element not in sel:
    # How many elements in sel does it conflict with?
    # If 0, it can be added.
    
    while time.time() < deadline:
        improved = False
        
        # For each element x in sel, consider removing it
        # Then check which elements outside sel can be added
        # Compute the change in reciprocal sum
        
        # To speed up, precompute for each x in sel:
        #   removal_cost = 1/x
        #   gain = sum of 1/y for all y not in sel that have no conflicts with (sel - {x})
        
        # This is expensive. Let's do a simpler version:
        # For each x in sel (in order of increasing 1/x, i.e., increasing x),
        #   remove x, then try to add each y not in sel in order of decreasing 1/y
        #   (i.e., increasing y), accepting if valid.
        
        # Even simpler: for each x in sel, remove it, then greedily add all valid y.
        
        # Let's iterate over x in sel
        for x in list(sel):
            if time.time() > deadline:
                break
            
            # Remove x
            sel.remove(x)
            
            # Now try to add all y not in sel, in order of decreasing 1/y (increasing y)
            added = []
            for y in range(1, N + 1):
                if y in sel:
                    continue
                # Check if y can be added to current sel
                # y can be added if it has no conflict with any element in sel
                # i.e., for all z in sel, y is not in conflict[z]
                # Using bitsets: check if any z in sel has bit y set in bitsets[z]
                can_add = True
                for z in sel:
                    if bitsets[z] & (1 << y):
                        can_add = False
                        break
                if can_add:
                    sel.add(y)
                    added.append(y)
            
            # Compute new sum
            new_sum = current_sum - recips[x] + sum(recips[y] for y in added)
            
            if new_sum > best_sum:
                best_sum = new_sum
                best_set = set(sel)
                improved = True
            else:
                # Revert: remove the added elements, re-add x
                for y in added:
                    sel.remove(y)
                sel.add(x)
        
        if not improved:
            # No improvement found in this pass. Try a different strategy:
            # Random restarts or perturbations.
            # For now, break out to avoid infinite loop.
            # But let's try a few random removals.
            random_improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                if len(best_set) < 2:
                    break
                # Randomly remove an element from best_set
                x = random.choice(list(best_set))
                test_set = set(best_set)
                test_set.remove(x)
                
                # Greedily add valid elements
                added = []
                for y in range(1, N + 1):
                    if y in test_set:
                        continue
                    can_add = True
                    for z in test_set:
                        if bitsets[z] & (1 << y):
                            can_add = False
                            break
                    if can_add:
                        test_set.add(y)
                        added.append(y)
                
                new_sum = best_sum - recips[x] + sum(recips[y] for y in added)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_set = test_set
                    random_improved = True
            
            if not random_improved:
                break
    
    return list(best_set), best_sum

def construct():
    start_time = time.time()
    deadline = start_time + 90 * _DEADLINE_FRAC
    
    # Build conflict bitsets
    bitsets = _build_conflict_bitsets()
    
    # Initial greedy solution
    initial = _greedy_initial()
    
    # Local search
    result, best_sum = _local_search(initial, bitsets, deadline)
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    # Final validity check
    if not _is_valid(result):
        # Fallback to greedy
        result = list(_greedy_initial())
    
    return result
# EVOLVE-BLOCK-END
