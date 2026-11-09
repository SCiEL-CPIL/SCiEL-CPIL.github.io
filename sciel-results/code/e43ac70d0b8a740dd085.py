import time
import random

def construct():
    deadline = time.time() + 55.0
    N = 100
    T = 200
    universe = list(range(1, N + 1))
    
    def is_valid(s):
        bits = 1
        for x in s:
            bits |= (bits << x)
        return ((bits >> T) & 1) == 0

    def greedy(order):
        selected = []
        bits = 1
        for x in order:
            if x <= T:
                if (bits >> (T - x)) & 1:
                    continue
            selected.append(x)
            bits |= (bits << x)
        return selected

    def try_add(s, bits, x):
        if x <= T and ((bits >> (T - x)) & 1):
            return False
        return True

    def local_search(s):
        s = list(s)
        bits = 1
        for x in s:
            bits |= (bits << x)
        
        sset = set(s)
        candidates = [x for x in range(N, 0, -1) if x not in sset]
        
        changed = True
        while changed:
            changed = False
            # 1. Try to add single elements
            for i in range(len(candidates)):
                if time.time() > deadline:
                    return s
                x = candidates[i]
                if try_add(s, bits, x):
                    s.append(x)
                    sset.add(x)
                    bits |= (bits << x)
                    candidates.pop(i)
                    changed = True
                    break # Restart loop as bits changed
            
            if changed:
                continue
                
            # 2. Try to remove 1, add 2 (net gain +1)
            if time.time() < deadline - 5.0:
                found_swap = False
                for y in s:
                    if time.time() > deadline:
                        return s
                    # Compute bits without y
                    bits_wo_y = 1
                    for x in s:
                        if x != y:
                            bits_wo_y |= (bits_wo_y << x)
                    
                    # Pre-filter candidates that are addable to bits_wo_y
                    addable = []
                    for x in candidates:
                        if try_add(s, bits_wo_y, x):
                            addable.append(x)
                    
                    if len(addable) < 2:
                        continue
                        
                    # Try pairs in addable
                    for i in range(len(addable)):
                        x1 = addable[i]
                        bits_after_x1 = bits_wo_y | (bits_wo_y << x1)
                        for j in range(i + 1, len(addable)):
                            x2 = addable[j]
                            if try_add(s, bits_after_x1, x2):
                                # Valid swap found
                                s.remove(y)
                                sset.remove(y)
                                s.append(x1)
                                sset.add(x1)
                                s.append(x2)
                                sset.add(x2)
                                candidates.remove(x1)
                                candidates.remove(x2)
                                # Update bits
                                bits = bits_wo_y | (bits_wo_y << x1) | ((bits_wo_y | (bits_wo_y << x1)) << x2)
                                changed = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break
                            
        return s

    def graph_independent_set():
        """
        Model as maximum independent set:
        - Vertices: integers 1-100
        - Edge between a,b if {a,b} can be extended to a subset summing to 200
          (i.e., there exist other elements c1,...,ck in [1,100]\{a,b} such that
           a + b + c1 + ... + ck = 200)
        - Greedy: repeatedly pick vertex with lowest degree among remaining
        
        Key insight: if a+b > T, no edge (can't reach T with positive additions).
        If a+b <= T, we need to check if T-(a+b) is achievable from [1,100]\{a,b}.
        Since [1,100] can achieve any sum from 1 to 5050, and T-(a+b) >= 1,
        most pairs with a+b < T will have edges.
        
        Actually, let's think more carefully. The constraint is: no subset of the
        final set sums to exactly 200. So if we include both a and b, we must ensure
        that 200 - a - b is NOT achievable from the other selected elements.
        
        For the graph approach: two elements conflict if including both makes it
        "dangerous" - i.e., there's a risk that some subset including both sums to 200.
        But this is complex. Let's use a simpler heuristic:
        
        Elements x and y conflict if x + y <= T (since then T-x-y >= 0, and if we
        have enough other small elements, we might complete the sum).
        
        Actually, the real constraint is dynamic. Let's use a different approach:
        Build the graph where edge (a,b) exists if a + b <= T. Then find a
        maximum independent set in this graph using greedy min-degree.
        
        Wait, that's not quite right either. Let me reconsider.
        
        The actual problem: select max subset S of [1,100] such that no subset of S sums to 200.
        
        A necessary condition: for any two elements a,b in S with a+b <= 200,
        we cannot have all elements needed to complete the sum to 200 also in S.
        
        This is a complex constraint. The graph approach with edges for "potential
        completion" is tricky.
        
        Alternative: Use the fact that if we select only large elements (> T/2 = 100),
        then no two can sum to 200 (since min pair sum > 200). But [101,100] is empty.
        
        If we select elements > T/2, i.e., [101, 100]... wait, T=200, so elements > 100
        are [101, 100] which is empty. Elements >= 101 don't exist in [1,100].
        
        Elements in [101, 200] would be safe from pairwise sums, but our universe is [1,100].
        
        Let me think differently. The key observation: if we take all elements > T/2,
        no subset of them can sum to T (since each is > T/2, so any two sum > T).
        But in [1,100], elements > 100 don't exist.
        
        What about elements in [T/2 + 1, T] = [101, 200]? Still none in [1,100].
        
        So we need a different strategy. Let's use the greedy with min-degree in a
        conflict graph where two elements conflict if they can "complete" a sum-200
        subset together with some other elements.
        
        Simpler approach: two elements a,b conflict if a + b <= T AND (T - a - b)
        can be formed from elements in [1,100]\{a,b}. Since [1,100] can form any sum
        from 1 to 5050, and T-a-b is between 1 and T-2=198, most pairs with a+b<T
        will conflict.
        
        This means the conflict graph is very dense, and the max independent set is
        small. That's not helpful.
        
        Let me try a different angle: the original greedy (descending) works well because
        it picks large elements first, which are less likely to participate in sum-200
        subsets. Let's enhance this with a smarter local search.
        """
        # Use a different strategy: start with high elements, then carefully add low ones
        selected = []
        bits = 1
        
        # Phase 1: Add elements from 100 down to 51 (these are "safe" in pairs)
        for x in range(N, 50, -1):
            if x <= T and (bits >> (T - x)) & 1:
                continue
            selected.append(x)
            bits |= (bits << x)
        
        # Phase 2: Try to add elements from 50 down to 1
        for x in range(50, 0, -1):
            if x <= T and (bits >> (T - x)) & 1:
                continue
            selected.append(x)
            bits |= (bits << x)
        
        return selected

    def hybrid_greedy():
        """
        Hybrid: use graph-based min-degree selection.
        Build a conflict graph where edge (a,b) means "including both a and b
        makes it harder to avoid sum T". Specifically, a and b conflict if
        a + b <= T (since then T-a-b >= 0, and we might complete the sum).
        
        Then greedily select vertices with minimum degree in the remaining graph.
        """
        # Build conflict graph: edge (a,b) if a + b <= T
        # (This is a conservative conflict: if a+b > T, they can't both be in a sum-T subset
        #  unless there are negative elements, which there aren't)
        
        # Actually, even if a+b > T, they could be in a sum-T subset if combined with
        # other elements? No, because all elements are positive. If a+b > T, then any subset
        # containing both has sum > T. So no conflict.
        
        # If a+b <= T, then a+b could be part of a sum-T subset if T-a-b can be formed
        # from other selected elements. So conflict exists.
        
        # For the greedy: pick vertex with min degree first
        remaining = set(universe)
        selected = []
        
        # Precompute degrees
        degree = {}
        for a in universe:
            deg = 0
            for b in universe:
                if a != b and a + b <= T:
                    deg += 1
            degree[a] = deg
        
        while remaining and time.time() < deadline:
            # Pick vertex with min degree among remaining
            min_deg_vertex = min(remaining, key=lambda v: degree[v])
            selected.append(min_deg_vertex)
            remaining.remove(min_deg_vertex)
            
            # Update degrees: remove edges from min_deg_vertex
            for v in remaining:
                if min_deg_vertex + v <= T:
                    degree[v] -= 1
        
        return selected

    def min_degree_greedy_with_validation():
        """
        Min-degree greedy, but validate at the end and repair if needed.
        """
        remaining = set(universe)
        selected = []
        
        # Degree: number of other elements b such that a + b <= T
        degree = {}
        for a in universe:
            deg = sum(1 for b in universe if b != a and a + b <= T)
            degree[a] = deg
        
        while remaining:
            if time.time() > deadline:
                break
            min_deg_vertex = min(remaining, key=lambda v: degree[v])
            selected.append(min_deg_vertex)
            remaining.remove(min_deg_vertex)
            
            for v in remaining:
                if min_deg_vertex + v <= T:
                    degree[v] -= 1
        
        # Validate and repair
        # Check if selected is valid
        bits = 1
        for x in selected:
            bits |= (bits << x)
        
        if (bits >> T) & 1:
            # Invalid, need to remove elements
            # Greedily remove elements that are part of sum-T subsets
            # This is complex, so just fall back to simple greedy
            return greedy(list(range(N, 0, -1)))
        
        return selected

    best = []
    best_len = 0
    
    # Try multiple strategies
    strategies = [
        lambda: greedy(list(range(N, 0, -1))),
        lambda: greedy(list(range(1, N + 1))),
        graph_independent_set,
        min_degree_greedy_with_validation,
    ]
    
    # Add random orders
    for _ in range(5):
        rand_order = universe[:]
        random.shuffle(rand_order)
        strategies.append(lambda ro=rand_order: greedy(ro))
    
    for strat in strategies:
        if time.time() > deadline:
            break
        s = strat()
        s = local_search(s)
        if len(s) > best_len:
            best = s
            best_len = len(s)
            
    if best:
        best = local_search(best)
    else:
        best = list(range(1, 70))
        
    return best
