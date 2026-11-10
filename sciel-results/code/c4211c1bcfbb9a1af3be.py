import time
import random
import math
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    N = 256
    ALL = list(range(1, N + 1))
    
    # Precompute reciprocal values as floats for fast comparison
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    
    # Bitmask representation for fast membership
    # Use a Python int as a bitset: bit i set means i is in the set
    def make_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m
    
    def in_mask(mask, x):
        return (mask >> x) & 1
    
    def has_ap3(mask, x):
        """Check if adding x to the set (represented by mask) creates a 3-term AP."""
        # x as middle: a, x, c with a + c = 2x, a < x < c
        # x as first: x, b, c with 2b = x + c => b = (x+c)/2, c > x
        # x as last: a, b, x with 2b = a + x => b = (a+x)/2, a < x
        # Check all three cases
        # Case 1: x is middle. For each a in mask with a < x, check if c = 2x - a is in mask
        a = 1
        while a < x:
            if in_mask(mask, a):
                c = 2 * x - a
                if c <= N and in_mask(mask, c):
                    return True
            a += 1
        # Case 2: x is first. For each c in mask with c > x, check if b = (x+c)/2 is in mask
        c = x + 1
        while c <= N:
            if in_mask(mask, c):
                s = x + c
                if s % 2 == 0:
                    b = s // 2
                    if in_mask(mask, b):
                        return True
            c += 1
        # Case 3: x is last. For each a in mask with a < x, check if b = (a+x)/2 is in mask
        a = 1
        while a < x:
            if in_mask(mask, a):
                s = a + x
                if s % 2 == 0:
                    b = s // 2
                    if in_mask(mask, b):
                        return True
            a += 1
        return False
    
    def greedy_fill(order, forbidden_mask=0, initial_mask=0):
        """Greedy fill in the given order, skipping forbidden and already-present elements."""
        mask = initial_mask
        for x in order:
            if in_mask(mask, x) or in_mask(forbidden_mask, x):
                continue
            if not has_ap3(mask, x):
                mask |= (1 << x)
        return mask
    
    def mask_sum(mask):
        """Compute sum of reciprocals for the mask."""
        s = 0.0
        for i in range(1, N + 1):
            if in_mask(mask, i):
                s += recip[i]
        return s
    
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if in_mask(mask, i)]
    
    def is_valid_mask(mask):
        """Verify no 3-term AP exists in the mask."""
        for i in range(1, N + 1):
            if in_mask(mask, i):
                for j in range(i + 1, N + 1):
                    if in_mask(mask, j):
                        k = 2 * j - i
                        if k <= N and in_mask(mask, k):
                            return False
        return True
    
    # Multiple greedy strategies
    best_mask = 0
    best_sum = 0.0
    
    # Strategy 1: ascending order (prioritize small numbers)
    m = greedy_fill(ALL)
    s = mask_sum(m)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Strategy 2: descending order
    m = greedy_fill(list(reversed(ALL)))
    s = mask_sum(m)
    if s > best_sum:
        best_sum = s
        best_mask = m
    
    # Strategy 3: priority by reciprocal value (same as ascending for 1/x)
    # Strategy 4: random restarts with random order
    random.seed(42)
    for _ in range(5):
        order = ALL[:]
        random.shuffle(order)
        m = greedy_fill(order)
        s = mask_sum(m)
        if s > best_sum:
            best_sum = s
            best_mask = m
    
    # Verify best is valid
    if not is_valid_mask(best_mask):
        # Fallback: use ascending greedy
        best_mask = greedy_fill(ALL)
        best_sum = mask_sum(best_mask)
    
    # Local search: 1-swap moves (remove one element, add another)
    # and 2-swap moves (remove two, add two)
    current_mask = best_mask
    current_sum = best_sum
    
    def try_1_swap(mask, s):
        """Try all 1-swap moves: remove x from mask, add y not in mask."""
        best_local_mask = mask
        best_local_sum = s
        elements_in = [i for i in range(1, N + 1) if in_mask(mask, i)]
        elements_out = [i for i in range(1, N + 1) if not in_mask(mask, i)]
        
        # Sort elements_out by reciprocal value descending to try best candidates first
        elements_out.sort(key=lambda x: -recip[x])
        
        for x in elements_in:
            if time.time() - start_time > DEADLINE:
                return best_local_mask, best_local_sum
            # Try removing x
            new_mask = mask & ~(1 << x)
            # Try adding each y not in mask
            for y in elements_out:
                if y == x:
                    continue
                # Check if y can be added to new_mask
                if not has_ap3(new_mask, y):
                    candidate_mask = new_mask | (1 << y)
                    candidate_sum = s - recip[x] + recip[y]
                    if candidate_sum > best_local_sum:
                        best_local_sum = candidate_sum
                        best_local_mask = candidate_mask
        return best_local_mask, best_local_sum
    
    def try_2_swap(mask, s):
        """Try 2-swap moves: remove two elements, add two others."""
        best_local_mask = mask
        best_local_sum = s
        elements_in = [i for i in range(1, N + 1) if in_mask(mask, i)]
        elements_out = [i for i in range(1, N + 1) if not in_mask(mask, i)]
        elements_out.sort(key=lambda x: -recip[x])
        
        # Only try promising 2-swaps: remove two large elements, add two small ones
        # Limit the search space
        if len(elements_in) < 2:
            return best_local_mask, best_local_sum
        
        # Try removing pairs of larger elements
        for i in range(len(elements_in)):
            if time.time() - start_time > DEADLINE:
                return best_local_mask, best_local_sum
            x1 = elements_in[i]
            for j in range(i + 1, len(elements_in)):
                x2 = elements_in[j]
                new_mask = mask & ~(1 << x1) & ~(1 << x2)
                # Try adding pairs of small elements
                for y1 in elements_out[:30]:
                    if y1 == x1 or y1 == x2:
                        continue
                    if not has_ap3(new_mask, y1):
                        intermediate_mask = new_mask | (1 << y1)
                        for y2 in elements_out[:30]:
                            if y2 == x1 or y2 == x2 or y2 == y1:
                                continue
                            if not has_ap3(intermediate_mask, y2):
                                candidate_mask = intermediate_mask | (1 << y2)
                                candidate_sum = s - recip[x1] - recip[x2] + recip[y1] + recip[y2]
                                if candidate_sum > best_local_sum:
                                    best_local_sum = candidate_sum
                                    best_local_mask = candidate_mask
        return best_local_mask, best_local_sum
    
    # Iterated local search
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        # Try 1-swap
        new_mask, new_sum = try_1_swap(current_mask, current_sum)
        if new_sum > current_sum:
            current_mask = new_mask
            current_sum = new_sum
            improved = True
            continue
        # Try 2-swap
        new_mask, new_sum = try_2_swap(current_mask, current_sum)
        if new_sum > current_sum:
            current_mask = new_mask
            current_sum = new_sum
            improved = True
            continue
        # If no improvement, do a random perturbation and restart
        if time.time() - start_time > DEADLINE - 10:
            break
        # Random perturbation: remove 2-3 random elements, re-greedy
        elements_in = [i for i in range(1, N + 1) if in_mask(current_mask, i)]
        if len(elements_in) < 3:
            break
        k = random.randint(2, 3)
        to_remove = random.sample(elements_in, k)
        forbidden = make_mask(to_remove)
        initial = current_mask
        for x in to_remove:
            initial &= ~(1 << x)
        new_mask = greedy_fill(ALL, forbidden, initial)
        new_sum = mask_sum(new_mask)
        if new_sum > current_sum:
            current_mask = new_mask
            current_sum = new_sum
            improved = True
    
    # Final check: if we found something better than initial best, use it
    if current_sum > best_sum:
        best_mask = current_mask
        best_sum = current_sum
    
    # === Branch-and-bound exact solver ===
    # Build conflict graph: edges connect pairs that form a 3-AP with a third element
    # For MWIS on this graph, we use branch-and-bound with greedy upper bound
    
    # Precompute conflicts: for each x, list of y that conflict with x (i.e., there exists z such that {x,y,z} is a 3-AP)
    conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c <= N and c != a and c != b:
                # a, b, c form a 3-AP
                if c not in conflicts[a]:
                    conflicts[a].append(c)
                if c not in conflicts[b]:
                    conflicts[b].append(c)
                if c not in conflicts[c]:
                    conflicts[c].append(a)
                if c not in conflicts[c]:
                    conflicts[c].append(b)
    
    # Remove duplicates and sort
    for i in range(1, N + 1):
        conflicts[i] = sorted(set(conflicts[i]))
    
    # Branch and bound for Maximum Weight Independent Set
    # We process vertices in order of decreasing weight (reciprocal)
    # Upper bound: sum of weights of all remaining vertices that don't conflict with chosen set
    
    # Sort vertices by weight descending
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: -recip[x])
    
    # Conflict sets as bitmasks for fast operations
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in conflicts[i]:
            conflict_mask[i] |= (1 << j)
    
    best_bb_mask = best_mask
    best_bb_sum = best_sum
    nodes_visited = 0
    max_nodes = 500000  # Limit B&B to avoid timeout
    
    def bb_recursive(available_mask, chosen_mask, current_weight, depth):
        """Branch and bound for MWIS."""
        nonlocal best_bb_mask, best_bb_sum, nodes_visited
        
        if time.time() - start_time > DEADLINE - 5 or nodes_visited > max_nodes:
            return
        
        nodes_visited += 1
        
        # If no available vertices, update best if improved
        if available_mask == 0:
            if current_weight > best_bb_sum:
                best_bb_sum = current_weight
                best_bb_mask = chosen_mask
            return
        
        # Upper bound: sum of weights of all available vertices
        # Quick bound: if current_weight + sum of all available <= best, prune
        ub = current_weight
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            ub += recip[idx]
            m ^= lsb
        
        if ub <= best_bb_sum:
            return
        
        # Choose vertex with highest weight from available
        # Find the available vertex with max weight
        best_v = -1
        best_w = -1.0
        m = available_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            if recip[idx] > best_w:
                best_w = recip[idx]
                best_v = idx
            m ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        # Remove best_v and all its conflicts from available
        new_available = available_mask & ~(1 << best_v) & ~conflict_mask[best_v]
        bb_recursive(new_available, chosen_mask | (1 << best_v), current_weight + recip[best_v], depth + 1)
        
        # Branch 2: exclude best_v
        new_available2 = available_mask & ~(1 << best_v)
        bb_recursive(new_available2, chosen_mask, current_weight, depth + 1)
    
    # Start B&B with all vertices available
    all_mask = 0
    for i in range(1, N + 1):
        all_mask |= (1 << i)
    
    bb_recursive(all_mask, 0, 0.0, 0)
    
    # Use B&B result if better
    if best_bb_sum > best_sum:
        best_mask = best_bb_mask
        best_sum = best_bb_sum
    
    # Verify final result
    if not is_valid_mask(best_mask):
        best_mask = greedy_fill(ALL)
        best_sum = mask_sum(best_mask)
    
    return list(mask_to_list(best_mask))
