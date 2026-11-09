import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s/90s limit
    N = 128
    # Bit i represents integer i+1 (1-indexed), bit 0 = value 1, bit 127 = value 128
    
    # Precompute conflict masks
    # conflict_mask[i] = bitmask of all vertices j such that there exist distinct k,l
    # where {i,j,k} forms a forbidden triple with i,j,k all distinct
    # Actually, let's think differently:
    # A triple (a,b,c) is forbidden if a,b,c are distinct and a+b, a+c, b+c are all in the set.
    # For the independent set formulation: vertices are 1..128.
    # Hyperedge: for each triple (a,b,c) with a<b<c, if a+b, a+c, b+c are all <= 128,
    # then {a,b,c,a+b,a+c,b+c} is a forbidden 6-set (but we only care about the 3 elements a,b,c
    # being all selected when their sums are also selected).
    
    # Better approach: For each pair (i,j) with i<j, define the "forbidden partner" set.
    # If i and j are both selected, then for any k distinct from i,j, we need to ensure
    # NOT (i+k in set AND j+k in set) OR k is not selected.
    # Actually the constraint is: no three distinct a,b,c with all three sums in the set.
    
    # Let me use a different formulation:
    # For each vertex v, precompute which pairs (u,w) would create a violation if v,u,w are all selected.
    # A violation occurs when for distinct a,b,c: a+b, a+c, b+c all in S.
    
    # Alternative: For each vertex v, compute conflict_mask[v] = set of vertices u such that
    # there exists w where {v,u,w} is a "dangerous" triple. But this depends on the sums being in S too.
    
    # Let me think about it more carefully. The constraint is:
    # For all distinct a,b,c in S: NOT (a+b in S AND a+c in S AND b+c in S)
    
    # Equivalently: For all distinct a,b,c: if a+b in S AND a+c in S AND b+c in S, then at least one of a,b,c is not in S.
    
    # For the independent set approach with bitmasks:
    # When considering adding vertex x to current set C, we need to check:
    # For all pairs (y,z) in C with y!=z, x!=y, x!=z:
    #   NOT (x+y in C∪{x} AND x+z in C∪{x} AND y+z in C∪{x})
    
    # This is complex. Let me use a precomputation approach:
    # For each x, precompute pairs_mask[x] = bitmask of all y such that there exists z where
    # {x,y,z} would be a forbidden triple (i.e., x+y, x+z, y+z are all <= 128).
    # But we also need the sums to actually be in the set.
    
    # Simpler approach: For each x, precompute the set of "witness" pairs.
    # When adding x to C, for each pair (y,z) in C:
    #   Check if x+y, x+z, y+z are all in C∪{x}
    
    # For O(1) checks, I'll precompute for each x a bitmask of all y such that
    # for some z, the triple {x,y,z} has all sums <= 128. Then during search,
    # I check if both y and z are in the current set AND all three sums are in the current set.
    
    # Let me just use a direct but optimized approach with bitmasks for membership.
    
    def time_check():
        return time.time() > deadline
    
    # Precompute: for each x in 1..128, and for each y in 1..128, y!=x,
    # store the set of z values such that {x,y,z} are all distinct and x+y, x+z, y+z are all in 1..128.
    # This tells us which triples involving x and y are "potentially forbidden".
    
    # Actually, let me precompute differently. For each x, precompute:
    # potential_conflicts[x] = list of (y, z) pairs where x,y,z are distinct and 
    # x+y <= 128, x+z <= 128, y+z <= 128
    # When x is added to C, we need to verify that for no (y,z) in potential_conflicts[x]
    # are both y and z in C AND x+y in C∪{x} AND x+z in C∪{x} AND y+z in C∪{x}
    
    # For efficiency, let me precompute for each x a bitmask of all y such that there exists z
    # with the above property. But the z dependency makes this tricky.
    
    # Let me use a simpler but effective approach: greedy with multiple orderings + local search.
    
    def is_valid_set(S_bits):
        """Check if a set (as bitmask) is valid. S_bits is an int where bit i means value i+1 is in the set."""
        # Get list of elements
        elems = [i+1 for i in range(128) if (S_bits >> i) & 1]
        elem_set = set(elems)
        for idx_a in range(len(elems)):
            a = elems[idx_a]
            for idx_b in range(idx_a+1, len(elems)):
                b = elems[idx_b]
                for idx_c in range(idx_b+1, len(elems)):
                    c = elems[idx_c]
                    if (a+b) in elem_set and (a+c) in elem_set and (b+c) in elem_set:
                        return False
        return True
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def greedy_build(order):
        """Build a set greedily following the given order."""
        S_bits = 0
        S_set = set()
        for x in order:
            if time_check():
                break
            # Try adding x
            # Check if adding x creates a violation
            valid = True
            for y in S_set:
                if y == x:
                    continue
                for z in S_set:
                    if z == x or z == y:
                        continue
                    # Check if {x,y,z} with sums all in S∪{x}
                    xy = x + y
                    xz = x + z
                    yz = y + z
                    # xy in S∪{x} means xy == x or xy in S_set
                    # But xy = x+y > x since y>0, so xy != x
                    # Similarly xz != x, yz != x (since all positive and distinct)
                    # So we need xy in S_set, xz in S_set, yz in S_set
                    if xy in S_set and xz in S_set and yz in S_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S_bits |= (1 << (x-1))
                S_set.add(x)
        return S_bits, len(S_set)
    
    def local_search(S_bits, S_set, max_iters=10000):
        """Try to improve the set via swaps and additions."""
        current_size = len(S_set)
        best_bits = S_bits
        best_set = S_set.copy()
        best_size = current_size
        
        for iteration in range(max_iters):
            if time_check():
                break
            
            improved = False
            
            # Try adding each element not in the set
            for x in range(1, 129):
                if x in best_set:
                    continue
                # Check if x can be added
                valid = True
                for y in best_set:
                    for z in best_set:
                        if z <= y:  # avoid duplicates, y<z
                            continue
                        if y == x or z == x:
                            continue
                        xy = x + y
                        xz = x + z
                        yz = y + z
                        if xy in best_set and xz in best_set and yz in best_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_set.add(x)
                    best_bits |= (1 << (x-1))
                    best_size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add another
            for x_remove in list(best_set):
                if time_check():
                    return best_bits, best_set, best_size
                for x_add in range(1, 129):
                    if x_add in best_set:
                        continue
                    # Try swapping x_remove with x_add
                    new_set = best_set - {x_remove}
                    new_set.add(x_add)
                    
                    # Check validity of new_set
                    valid = True
                    elems = list(new_set)
                    elem_set = new_set
                    for idx_a in range(len(elems)):
                        a = elems[idx_a]
                        for idx_b in range(idx_a+1, len(elems)):
                            b = elems[idx_b]
                            for idx_c in range(idx_b+1, len(elems)):
                                c = elems[idx_c]
                                if (a+b) in elem_set and (a+c) in elem_set and (b+c) in elem_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        best_set = new_set
                        best_bits = 0
                        for v in new_set:
                            best_bits |= (1 << (v-1))
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_bits, best_set, best_size
    
    # Multi-restart greedy with different orderings
    best_result = 0
    best_bits = 0
    
    # Ordering 1: increasing
    order1 = list(range(1, 129))
    bits1, size1 = greedy_build(order1)
    if size1 > best_result:
        best_result = size1
        best_bits = bits1
    
    # Ordering 2: decreasing
    order2 = list(range(128, 0, -1))
    bits2, size2 = greedy_build(order2)
    if size2 > best_result:
        best_result = size2
        best_bits = bits2
    
    # Ordering 3: random
    for restart in range(20):
        if time_check():
            break
        random.seed(restart)
        order3 = list(range(1, 129))
        random.shuffle(order3)
        bits3, size3 = greedy_build(order3)
        if size3 > best_result:
            best_result = size3
            best_bits = bits3
    
    # Local search from best
    best_set = set(i+1 for i in range(128) if (best_bits >> i) & 1)
    best_bits, best_set, best_size = local_search(best_bits, best_set)
    
    # Try local search from other good solutions
    for restart in range(5):
        if time_check():
            break
        random.seed(100 + restart)
        order = list(range(1, 129))
        random.shuffle(order)
        bits, size = greedy_build(order)
        if size >= best_result - 2:
            s_set = set(i+1 for i in range(128) if (bits >> i) & 1)
            bits, s_set, size = local_search(bits, s_set)
            if size > best_size:
                best_bits = bits
                best_set = s_set
                best_size = size
    
    result = sorted(best_set)
    
    # Convert to ensure Python int
    return [int(x) for x in result]
