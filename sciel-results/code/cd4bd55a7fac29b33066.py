import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs
    all_3aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                all_3aps.append((a, b, c))
    
    n_3aps = len(all_3aps)
    
    # Build conflict graph: two 3-APs conflict if they can be extended to a 4-AP
    # Two 3-APs (a1,b1,c1) and (a2,b2,c2) conflict if their union contains a 4-AP
    # Actually, we need: if we include both 3-APs, does that force a 4-AP?
    # A 4-AP is (x, x+d, x+2d, x+3d). It contains two overlapping 3-APs:
    # (x, x+d, x+2d) and (x+d, x+2d, x+3d)
    # So two 3-APs conflict if they are the two halves of some 4-AP.
    
    # Build mapping from 3-AP to its index
    ap_index = {}
    for i, (a, b, c) in enumerate(all_3aps):
        ap_index[(a, b, c)] = i
    
    # For each 4-AP, find the two 3-APs it contains and add conflict edge
    conflict = [set() for _ in range(n_3aps)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # First 3-AP: (a, b, c)
            # Second 3-AP: (b, c, e)
            ap1 = (a, b, c)
            ap2 = (b, c, e)
            if ap1 in ap_index and ap2 in ap_index:
                i1 = ap_index[ap1]
                i2 = ap_index[ap2]
                conflict[i1].add(i2)
                conflict[i2].add(i1)
    
    # Also, two 3-APs might conflict if they share elements in a way that creates a 4-AP
    # More generally: if we have 3-APs (a,b,c) and (a',b',c'), and there exists a 4-AP
    # that contains elements from both, they conflict.
    # The most common case is the overlapping halves of a 4-AP as above.
    # But also: (a, a+d, a+2d) and (a-d, a, a+d) if a-d >= 1
    # Let me be more systematic: for each pair of 3-APs, check if their union contains a 4-AP.
    # This is expensive, so let's use the 4-AP enumeration approach.
    
    # Actually, the conflict is: including both 3-APs means all 6 (or fewer, if they overlap)
    # elements are in S. A 4-AP is forbidden if all 4 of its elements are in S.
    # So two 3-APs conflict if there exists a 4-AP whose 4 elements are all contained in
    # the union of the two 3-APs' elements.
    
    # Let me redo this more carefully.
    # For each 4-AP (w, x, y, z), the 3-APs that are subsets of {w,x,y,z} are:
    # (w,x,y), (x,y,z), and also (w,x,z) is not a 3-AP unless d=0.
    # Actually (w,x,y) has diff d, (x,y,z) has diff d. 
    # Are there other 3-APs within {w,x,y,z}? 
    # (w,x,z): diffs are d and 2d, not equal. Not a 3-AP.
    # (w,y,z): diffs are 2d and d, not equal. Not a 3-AP.
    # So only (w,x,y) and (x,y,z) are 3-APs within a 4-AP.
    
    # So the conflict graph edges are exactly: for each 4-AP, connect its two 3-AP halves.
    # But wait - there could be other conflicts. If two 3-APs share 2 elements, and those
    # 2 elements plus 2 more form a 4-AP... Let me think.
    
    # Example: 3-AP1 = (1,2,3), 3-AP2 = (2,3,4). Union = {1,2,3,4}. Contains 4-AP (1,2,3,4). Conflict.
    # 3-AP1 = (1,3,5), 3-AP2 = (3,5,7). Union = {1,3,5,7}. Contains 4-AP (1,3,5,7). Conflict.
    # 3-AP1 = (1,2,3), 3-AP2 = (1,4,7). Union = {1,2,3,4,7}. Does it contain a 4-AP? 
    #   (1,2,3,4) - yes! So they conflict.
    
    # So I need to check all pairs. Let me do it differently.
    # For each 4-AP, mark which 3-APs are "covered" by it. Any two 3-APs that are both
    # subsets of the same 4-AP's element set conflict.
    
    # But also, two 3-APs can conflict even if no single 4-AP contains all their elements,
    # if the union of their elements contains a 4-AP.
    
    # Let me just build the conflict graph by checking each pair of 3-APs.
    # n_3aps is about 256*128/2 ~ 16384. Pairs would be ~134M which is too many.
    
    # Better approach: for each element, list the 3-APs containing it. For each pair of 3-APs
    # sharing an element, check if their union contains a 4-AP.
    
    # Actually, let me use a different strategy. The conflict is: 
    # S has no 4-AP. We want to maximize 3-APs in S.
    # This is equivalent to: choose a set of 3-APs such that no 4-AP is fully contained in 
    # the union of chosen 3-APs' elements.
    
    # Let me use the element-based approach instead. Build S, then count 3-APs.
    # Use branch and bound on elements.
    
    # Precompute for each element, which 4-APs it's in
    fours_by_elem = defaultdict(list)
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_4aps.append((a, b, c, e))
            fours_by_elem[a].append((b, c, e))
            fours_by_elem[b].append((a, c, e))
            fours_by_elem[c].append((a, b, e))
            fours_by_elem[e].append((a, b, c))
    
    threes_by_elem = defaultdict(list)
    for i, (a, b, c) in enumerate(all_3aps):
        threes_by_elem[a].append((b, c, i))
        threes_by_elem[b].append((a, c, i))
        threes_by_elem[c].append((a, b, i))
    
    def creates_4ap(S, x):
        for (o1, o2, o3) in fours_by_elem[x]:
            if S[o1] and S[o2] and S[o3]:
                return True
        return False
    
    def count_3aps(S):
        count = 0
        for a, b, c in all_3aps:
            if S[a] and S[b] and S[c]:
                count += 1
        return count
    
    def gain_3aps(S, x):
        count = 0
        for (o1, o2, idx) in threes_by_elem[x]:
            if S[o1] and S[o2]:
                count += 1
        return count
    
    # Greedy construction
    def greedy_construct():
        S = [False] * (N + 1)
        S_set = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_gain = -1
            for x in remaining:
                if creates_4ap(S, x):
                    continue
                g = gain_3aps(S, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x is None:
                break
                
            S[best_x] = True
            S_set.add(best_x)
            remaining.remove(best_x)
            
        return S, S_set
    
    # Simulated Annealing
    def simulated_annealing(S_init, S_set_init, time_budget=15):
        S = S_init[:]
        S_set = set(S_set_init)
        
        current_3ap = count_3aps(S)
        
        best_S = S[:]
        best_S_set = set(S_set)
        best_3ap = current_3ap
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.005
        
        all_elements = list(range(1, N + 1))
        
        while time.time() < start_time + time_budget and T > min_T:
            r = random.random()
            if r < 0.5:
                move_type = 2
            elif r < 0.8:
                move_type = 0
            else:
                move_type = 1
            
            if move_type == 0:
                candidates = [x for x in all_elements if not S[x]]
                if not candidates:
                    T *= cooling_rate
                    continue
                x = random.choice(candidates)
                
                if creates_4ap(S, x):
                    T *= cooling_rate
                    continue
                    
                gain = gain_3aps(S, x)
                delta = gain
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[x] = True
                    S_set.add(x)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                        
            elif move_type == 1:
                if not S_set:
                    T *= cooling_rate
                    continue
                x = random.choice(list(S_set))
                
                loss = gain_3aps(S, x)
                delta = -loss
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[x] = False
                    S_set.discard(x)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                        
            else:
                if not S_set:
                    T *= cooling_rate
                    continue
                x = random.choice(list(S_set))
                candidates = [y for y in all_elements if not S[y]]
                if not candidates:
                    T *= cooling_rate
                    continue
                y = random.choice(candidates)
                
                S[x] = False
                S_set.discard(x)
                
                if creates_4ap(S, y):
                    S[x] = True
                    S_set.add(x)
                    T *= cooling_rate
                    continue
                
                loss_x = gain_3aps(S, x)
                gain_y = gain_3aps(S, y)
                delta = gain_y - loss_x
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    S[y] = True
                    S_set.add(y)
                    current_3ap += delta
                    
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_S = S[:]
                        best_S_set = set(S_set)
                else:
                    S[x] = True
                    S_set.add(x)
                    
            T *= cooling_rate
            
        return best_S, best_S_set, best_3ap

    # Branch and bound on 3-AP selection
    # We select a subset of 3-APs. The constraint is that no 4-AP is fully covered.
    # Upper bound: remaining 3-APs + current count
    # This is essentially a maximum independent set on the conflict graph of 3-APs.
    
    # Build conflict graph properly: two 3-APs conflict if their union contains a 4-AP
    # For efficiency, use the fact that a 4-AP (a, a+d, a+2d, a+3d) contains exactly
    # two 3-APs: (a,a+d,a+2d) and (a+d,a+2d,a+3d)
    # But two 3-APs can also conflict if they share elements and their union has a 4-AP
    # that's not just the obvious halves.
    
    # Let me build it by: for each 4-AP, the two 3-AP halves conflict.
    # Then also check: if 3-AP1 and 3-AP2 share 2 elements, check if union has 4-AP.
    
    # Actually, the simplest correct approach: for each pair of 3-APs that share at least
    # one element, check if their union contains any 4-AP. But this is still expensive.
    
    # Alternative: use the element-based B&B with good bounds.
    
    # Let's try a different approach: maximum independent set on 3-AP conflict graph
    # using branch and bound with coloring upper bound.
    
    # First, build the conflict graph
    # Two 3-APs conflict if there exists a 4-AP all of whose elements are in the union
    # of the two 3-APs' elements.
    
    # For each 4-AP (a,b,c,e), the 3-APs contained in {a,b,c,e} are (a,b,c) and (b,c,e).
    # So these two conflict. But also, if another 3-AP (x,y,z) has {x,y,z} subset of {a,b,c,e},
    # then it also conflicts with both. But the only 3-APs in {a,b,c,e} are (a,b,c) and (b,c,e).
    
    # However, two 3-APs can conflict without being in the same 4-AP. Example:
    # (1,2,3) and (3,4,5): union = {1,2,3,4,5}. Contains 4-AP (1,2,3,4) and (2,3,4,5).
    # So they conflict.
    
    # So I need a more general check. Let me build it as follows:
    # For each element, list 3-APs containing it. For each pair of 3-APs sharing an element,
    # check if their union contains a 4-AP.
    
    # This is still potentially expensive. Let me limit to 3-APs that share at least 2 elements
    # or are "close" in some sense.
    
    # Actually, let me just use the element-based approach with SA, which already works.
    # The B&B on 3-APs is complex to implement correctly.
    
    # Let me focus on improving the SA with better initialization and moves.
    
    best_overall_3ap = 0
    best_overall_S = []
    
    # First: pure greedy
    if time.time() < deadline - 60:
        S_greedy, S_set_greedy = greedy_construct()
        best_S, best_S_set, best_3ap = simulated_annealing(S_greedy, S_set_greedy, time_budget=20)
        
        if best_3ap > best_overall_3ap:
            best_overall_3ap = best_3ap
            best_overall_S = best_S_set
    
    # Then: multiple randomized restarts
    num_restarts = 10
    time_per_restart = 6
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        S = [False] * (N + 1)
        S_set = set()
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        for x in remaining[:]:
            if creates_4ap(S, x):
                continue
            S[x] = True
            S_set.add(x)
        
        best_S, best_S_set, best_3ap = simulated_annealing(S, S_set, time_budget=time_per_restart)
        
        if best_3ap > best_overall_3ap:
            best_overall_3ap = best_3ap
            best_overall_S = best_S_set
    
    # Final polish
    if best_overall_S and time.time() < deadline - 3:
        S_final = [False] * (N + 1)
        for x in best_overall_S:
            S_final[x] = True
        best_S, best_S_set, best_3ap = simulated_annealing(S_final, set(best_overall_S), time_budget=3)
        
        if best_3ap > best_overall_3ap:
            best_overall_3ap = best_3ap
            best_overall_S = best_S_set
    
    if not best_overall_S:
        return [1]
        
    return sorted(best_overall_S)
