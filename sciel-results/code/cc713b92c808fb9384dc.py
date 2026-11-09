import time
import random
import math

N = 256
TIME_LIMIT = 40.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i

    # Precompute conflict bitsets: conflict[x] is a bitset of numbers that form a 3-AP with x
    # Two numbers a, b conflict if there exists c such that {a,b,c} is a non-constant 3-AP
    # i.e., c = 2a - b, c = 2b - a, or a+b = 2c (c = (a+b)/2)
    # Actually: a and b are in a 3-AP if there exists c != a, c != b with {a,b,c} forming AP.
    # The three cases: a is middle (c1, a, c2), b is middle, or c is middle.
    # Simpler: a and b conflict if 2a-b in [1,N] and 2a-b != b, OR 2b-a in [1,N] and 2b-a != a,
    # OR (a+b) is even and (a+b)//2 in [1,N] and (a+b)//2 != a and (a+b)//2 != b.
    # Wait - actually for a 3-term AP {x,y,z} with x<y<z, we have y = (x+z)/2.
    # So a and b conflict if:
    #   - There exists c > max(a,b) such that min(a,b), max(a,b), c is AP: c = 2*max - min, c in [1,N]
    #   - There exists c < min(a,b) such that c, min(a,b), max(a,b) is AP: c = 2*min - max, c in [1,N]
    #   - c = (a+b)/2 is in [1,N] and c != a and c != b (c is the middle term)
    #   Actually all three reduce to: the set {a,b} can be extended to a 3-AP.
    #   A 3-AP containing a and b exists iff:
    #     2a - b in [1,N] and 2a-b != b (a is middle: 2a-b, a, b)
    #     2b - a in [1,N] and 2b-a != a (b is middle: a, b, 2b-a)
    #     (a+b) even and (a+b)//2 in [1,N] and (a+b)//2 != a and (a+b)//2 != b (c is middle)
    #   But wait: if 2a-b = a, that means a=b, which can't happen. If 2a-b = b, that means a=b.
    #   So the conditions simplify to:
    #     2a-b in [1,N] (and automatically != b since a!=b)
    #     2b-a in [1,N]
    #     (a+b) even and (a+b)//2 in [1,N] and (a+b)//2 != a and (a+b)//2 != b
    #   The last: (a+b)//2 != a iff b != a (true), (a+b)//2 != b iff a != b (true).
    #   So: conflict if 2a-b in [1,N] OR 2b-a in [1,N] OR ((a+b) even AND (a+b)//2 in [1,N])
    
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c1 = 2 * a - b  # a is middle
            c2 = 2 * b - a  # b is middle
            if c1 >= 1 and c1 <= N and c1 != b:
                # conflict
                conflict[a] |= (1 << (b - 1))
                conflict[b] |= (1 << (a - 1))
            elif c2 >= 1 and c2 <= N and c2 != a:
                conflict[a] |= (1 << (b - 1))
                conflict[b] |= (1 << (a - 1))
            else:
                # Check if (a+b) is even and midpoint in range
                if (a + b) % 2 == 0:
                    mid = (a + b) // 2
                    if 1 <= mid <= N and mid != a and mid != b:
                        conflict[a] |= (1 << (b - 1))
                        conflict[b] |= (1 << (a - 1))

    # Fast can_add using bitsets
    def can_add(bitset, x):
        # Check if x conflicts with any element in bitset
        return (conflict[x] & bitset) == 0

    # Greedy coloring for bound computation
    # Returns a list of (color, weight) where weight is sum of reciprocals in that color class
    # The number of colors is an upper bound on the size of any independent set in the subgraph
    # For weighted case: sum of max weights per color is an upper bound on max weight independent set
    
    def greedy_color_bound(candidates, current_bitset):
        """
        candidates: list of element indices (1-indexed) to consider
        current_bitset: the bitset of already-selected elements (for filtering)
        Returns an upper bound on the additional weight achievable.
        """
        if not candidates:
            return 0.0
        
        # Build the induced subgraph on candidates
        # We'll do greedy coloring: repeatedly find a maximal independent set, assign it a color
        # For weighted coloring: each color class has weight = sum of reciprocals of its members
        # Upper bound = sum of color class weights (since any independent set picks at most one per color)
        
        # Actually, the standard bound is: number of colors >= size of max independent set
        # For weighted: we want a bound on max weight independent set
        # Greedy coloring gives: sum over colors of (max weight in color) <= ... no that's not right either.
        # 
        # The correct weighted bound: if we color the graph with k colors, then any independent set
        # contains at most one vertex from each color class. So the max weight independent set has
        # weight at most sum over colors of (max weight in that color class).
        # But that's a loose bound. A tighter one: the max weight independent set weight <= sum of color class weights
        # where each color class weight is the sum of all vertices in it... no.
        #
        # Actually the standard approach: color the graph greedily. The number of colors is an upper bound
        # on the clique number... wait no, it's an upper bound on the independence number.
        # For maximum weight independent set: the bound is that the total weight of any independent set
        # is at most the sum of the maximum weights in each color class.
        # 
        # Hmm, let me think more carefully. If we have a proper coloring with colors 1..k, then
        # any independent set picks at most one vertex from each color class. So:
        #   max_weight_independent_set <= sum_{c=1}^{k} max_{v in color_c} w(v)
        # This is valid.
        
        # For efficiency, let's do a simple greedy coloring and compute the bound.
        remaining = set(candidates)
        bound = 0.0
        
        while remaining:
            # Find a maximal independent set from remaining
            color_class = []
            color_bitset = 0
            for x in list(remaining):
                # x can be added to color class if it doesn't conflict with any in color_class
                if (conflict[x] & color_bitset) == 0:
                    color_class.append(x)
                    color_bitset |= (1 << (x - 1))
                    remaining.discard(x)
            
            # The max weight in this color class
            max_w = max(recip[x] for x in color_class)
            bound += max_w
        
        return bound

    # Better: use a tighter bound based on the actual sum of all candidate weights
    # The simplest valid bound: sum of all weights in candidates
    # But that's very loose. Let's use the coloring bound.
    
    # Actually, for performance, let's use a simpler but still effective approach:
    # Sort candidates by weight descending. Use a greedy independent set as a lower bound
    # and the sum of remaining as an upper bound, with pruning.
    
    # Let me restructure: use branch and bound with the following:
    # - Order vertices by weight descending
    # - At each step, decide include or exclude
    # - Bound: current_weight + sum of weights of all remaining vertices that could still be added
    #   (optimistic: ignore conflicts among remaining)
    # - This is valid but loose. To tighten, use coloring.
    
    # For N=256, a pure B&B might be too slow. Let's use a hybrid:
    # 1. Get a good initial solution via greedy + local search
    # 2. Use B&B with strong bounds to prove optimality or find better solution
    
    # Phase 1: Greedy initialization (by reciprocal descending)
    order = list(range(1, N + 1))
    order.sort(key=lambda x: recip[x], reverse=True)  # 1, 2, 3, ...
    
    current_bitset = 0
    current_score = 0.0
    for x in order:
        if can_add(current_bitset, x):
            current_bitset |= (1 << (x - 1))
            current_score += recip[x]
    
    best_bitset = current_bitset
    best_score = current_score

    # Phase 2: Local search to improve
    def get_elements(bitset):
        return [i for i in range(1, N + 1) if bitset & (1 << (i - 1))]
    
    def get_non_elements(bitset):
        return [i for i in range(1, N + 1) if not (bitset & (1 << (i - 1)))]
    
    # Simulated annealing with fast conflict checks
    temp = 0.03
    cooling = 0.9995
    min_temp = 1e-6
    sa_end = start_time + 15.0
    
    while time.time() < sa_end:
        if temp < min_temp:
            temp = 0.01
            current_bitset = best_bitset
            current_score = best_score
        
        r = random.random()
        
        if r < 0.35:
            # Add a random non-element
            non_els = get_non_elements(current_bitset)
            if non_els:
                x = random.choice(non_els)
                if can_add(current_bitset, x):
                    current_bitset |= (1 << (x - 1))
                    current_score += recip[x]
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset
        elif r < 0.55:
            # Remove a random element
            els = get_elements(current_bitset)
            if els:
                x = random.choice(els)
                delta = -recip[x]
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_bitset &= ~(1 << (x - 1))
                    current_score += delta
        else:
            # Swap
            els = get_elements(current_bitset)
            non_els = get_non_elements(current_bitset)
            if els and non_els:
                x_rem = random.choice(els)
                x_add = random.choice(non_els)
                if x_add < x_rem:  # prefer adding smaller (higher weight)
                    trial = current_bitset & ~(1 << (x_rem - 1))
                    if can_add(trial, x_add):
                        delta = recip[x_add] - recip[x_rem]
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_bitset = trial | (1 << (x_add - 1))
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_bitset = current_bitset
        
        temp *= cooling
    
    # Phase 3: Branch and Bound for exact optimization
    # Use the current best as a lower bound
    # Process vertices in order of decreasing weight
    # Bound: current_score + sum of weights of unprocessed vertices (loose but fast)
    # Tighter: use coloring bound on the remaining candidates
    
    # Precompute suffix sums for the loose bound
    suffix = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recip[order[i]]
    
    # B&B with recursion (iterative to avoid stack issues)
    # State: (bitset, score, idx) where idx is index into order
    # We'll use a stack-based DFS
    
    # To make B&B effective, we need good bounds. Let's use:
    # - Loose bound: score + suffix[idx]
    # - If this exceeds best_score, we continue
    # - For tighter pruning, we can check if the remaining vertices form an independent set
    #   (if they do, we can add all of them)
    
    # Actually, let's use a smarter approach:
    # At each node, compute the set of remaining vertices that are compatible with current_bitset
    # Then use a greedy coloring bound on that set
    
    # For N=256, this might still be slow. Let's limit B&B depth or use time budget.
    
    bb_end = start_time + TIME_LIMIT - 2.0
    
    # Use iterative deepening or just a time-limited B&B
    # Let's do a simple recursive B&B with time checks
    
    def bb(bitset, score, idx):
        nonlocal best_bitset, best_score
        
        if time.time() > bb_end:
            return
        
        if idx >= N:
            if score > best_score:
                best_score = score
                best_bitset = bitset
            return
        
        # Loose bound
        if score + suffix[idx] <= best_score:
            return
        
        x = order[idx]
        
        # Try including x
        if can_add(bitset, x):
            new_bitset = bitset | (1 << (x - 1))
            bb(new_bitset, score + recip[x], idx + 1)
        
        # Try excluding x
        bb(bitset, score, idx + 1)
    
    # The above B&B is too slow for N=256 without strong bounds.
    # Let's instead use a more targeted approach:
    # Start from the best solution found so far, and try to improve it
    # by considering swaps with unselected elements.
    
    # Phase 3: Targeted improvement - try all single swaps and double swaps
    current_best_els = set(get_elements(best_bitset))
    current_best_non = set(range(1, N + 1)) - current_best_els
    
    # Try all single swaps
    improved = True
    while improved:
        improved = False
        for x_add in sorted(current_best_non, key=lambda x: recip[x], reverse=True):
            for x_rem in sorted(current_best_els, key=lambda x: recip[x]):
                if recip[x_add] <= recip[x_rem]:
                    break  # no point continuing (we want higher weight)
                trial = (current_best_els - {x_rem}) | {x_add}
                # Check if trial is 3-AP free
                valid = True
                for a in trial:
                    for b in trial:
                        if a >= b:
                            continue
                        c = 2 * b - a
                        if c > b and c in trial:
                            valid = False
                            break
                        c = 2 * a - b
                        if c < a and c in trial:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_score = best_score + recip[x_add] - recip[x_rem]
                    best_score = new_score
                    best_bitset = 0
                    for el in trial:
                        best_bitset |= (1 << (el - 1))
                    current_best_els = trial
                    current_best_non = set(range(1, N + 1)) - current_best_els
                    improved = True
                    break
            if improved:
                break
    
    # Phase 4: Final local search polish
    final_end = start_time + TIME_LIMIT - 1.0
    temp = 0.01
    cooling = 0.999
    
    current_bitset = best_bitset
    current_score = best_score
    
    while time.time() < final_end:
        r = random.random()
        
        if r < 0.4:
            non_els = get_non_elements(current_bitset)
            if non_els:
                # Prefer small numbers (high reciprocal)
                x = random.choice(non_els[:min(50, len(non_els))])
                if can_add(current_bitset, x):
                    current_bitset |= (1 << (x - 1))
                    current_score += recip[x]
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset
        elif r < 0.6:
            els = get_elements(current_bitset)
            if els:
                x = random.choice(els)
                delta = -recip[x]
                if random.random() < math.exp(delta / temp):
                    current_bitset &= ~(1 << (x - 1))
                    current_score += delta
        else:
            els = get_elements(current_bitset)
            non_els = get_non_elements(current_bitset)
            if els and non_els:
                # Prefer removing large, adding small
                x_rem = random.choice(els)
                small_non = [x for x in non_els if x < x_rem]
                if small_non:
                    x_add = random.choice(small_non)
                    trial = current_bitset & ~(1 << (x_rem - 1))
                    if can_add(trial, x_add):
                        delta = recip[x_add] - recip[x_rem]
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_bitset = trial | (1 << (x_add - 1))
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_bitset = current_bitset
        
        temp *= cooling
    
    # Final verification
    result = get_elements(best_bitset)
    
    # Verify no 3-AP
    s = set(result)
    for a in result:
        for b in result:
            if a >= b:
                continue
            c = 2 * b - a
            if c > b and c in s:
                # Fallback to greedy
                fb = []
                fb_set = set()
                for x in range(1, N + 1):
                    fb_set.add(x)
                    ok = True
                    for a2 in fb_set:
                        for b2 in fb_set:
                            if a2 >= b2:
                                continue
                            c2 = 2 * b2 - a2
                            if c2 > b2 and c2 in fb_set:
                                ok = False
                                break
                            c2 = 2 * a2 - b2
                            if c2 < a2 and c2 in fb_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        fb.append(x)
                    else:
                        fb_set.remove(x)
                return fb
            c = 2 * a - b
            if c < a and c in s:
                fb = []
                fb_set = set()
                for x in range(1, N + 1):
                    fb_set.add(x)
                    ok = True
                    for a2 in fb_set:
                        for b2 in fb_set:
                            if a2 >= b2:
                                continue
                            c2 = 2 * b2 - a2
                            if c2 > b2 and c2 in fb_set:
                                ok = False
                                break
                            c2 = 2 * a2 - b2
                            if c2 < a2 and c2 in fb_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        fb.append(x)
                    else:
                        fb_set.remove(x)
                return fb
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
