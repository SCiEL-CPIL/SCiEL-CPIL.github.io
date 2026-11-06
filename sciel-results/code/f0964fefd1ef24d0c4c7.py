"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_coloring(adj, P, colors):
        """
        Greedy coloring of the induced subgraph on P.
        Returns the number of colors used (upper bound on MIS size).
        adj: list of sets, adj[i] contains j if i and j are connected.
        P: list of vertices to color.
        colors: list where colors[i] is the color of vertex i (0 if uncolored).
        """
        color_num = 0
        # Sort P by degree in the subgraph P (descending) for better coloring
        # Degree in P: number of neighbors in P
        # To speed up, we can just use the original degree or compute quickly.
        # For small P, computing degree is fine.
        P_set = set(P)
        P_sorted = sorted(P, key=lambda v: len(adj[v] & P_set), reverse=True)
        
        for v in P_sorted:
            # Find the smallest color not used by neighbors in P
            used_colors = set()
            for u in adj[v]:
                if u in P_set and colors[u] > 0:
                    used_colors.add(colors[u])
            
            c = 1
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c > color_num:
                color_num = c
        return color_num

    def branch_and_bound(elements, max_time=30.0):
        """
        Finds a Maximum Independent Set in the conflict graph.
        Vertices: elements.
        Edge (u, v) if u and v cannot be in the same sum-free set.
        This happens if:
        1. u + v is in elements (then u+v, u, v form a triple if u+v is also chosen? No, wait.)
        
        Let's re-evaluate the conflict graph definition.
        We want a set S such that no a, b, c in S satisfy a+b=c.
        This is equivalent to: for any a, b in S, a+b is NOT in S.
        
        If we define a graph G where vertices are elements, and an edge exists between u and v
        if u and v "conflict", what is the conflict?
        
        If we pick u and v, we must ensure that u+v is not in S.
        This is not a simple pairwise constraint on u and v alone, because it depends on whether u+v is in S.
        However, if u+v is in the *universe* of elements, then we have a constraint: we cannot pick {u, v, u+v} all together.
        This is a 3-uniform hypergraph constraint (no hyperedge fully contained in S).
        
        Standard MIS is for 2-uniform hypergraphs (graphs).
        However, we can reduce this to a graph problem by observing:
        If we fix the set S, the condition is: for all a, b in S, a+b not in S.
        
        Consider the "sum graph" where edges connect a and b if a+b is in the universe.
        If we pick an independent set in this graph, say S_ind, then for any a, b in S_ind, a+b is NOT in the universe.
        Therefore, a+b is definitely not in S_ind. So S_ind is sum-free.
        Thus, MIS of the sum graph is a valid sum-free subset.
        
        Is it the maximum?
        Not necessarily. We might be able to include a pair (a, b) where a+b IS in the universe, as long as we don't include a+b.
        Example: Universe {1, 2, 3}. Sum graph edges: (1,2) because 1+2=3. (1,1) self-loop? No, distinct.
        Independent sets in sum graph: {1, 3}, {2, 3}, {1, 2} is NOT independent because 1+2=3 is in universe? 
        Wait, if edge is (u,v) if u+v in Universe, then (1,2) is an edge.
        Max Independent Set of sum graph on {1,2,3}:
        Edges: (1,2). 
        Independent sets: {1,3}, {2,3}, {1}, {2}, {3}, {}.
        Max size is 2.
        Is {1,3} sum-free? 1+1=2 (not in {1,3}), 1+3=4 (not in {1,3}), 3+3=6 (not in {1,3}). Yes.
        Is {2,3} sum-free? 2+2=4, 2+3=5, 3+3=6. Yes.
        
        What if we had Universe {1, 2, 3, 4}?
        Sum graph edges:
        1+1=2 (self? no, distinct elements in set, but a=b allowed in sum-free def. 
        "no a + b = c with a, b, c in it (a = b allowed)".
        So 1+1=2 is a violation if 1, 1, 2 are in S. Since 1 is in S, 2 is in S, it's a violation.
        So if 1 and 2 are in S, it's invalid.
        So (1, 2) is a conflict pair.
        
        General Conflict:
        Two elements x and y conflict if:
        1. x + y is in the universe (so we can't have x, y, x+y all in S).
           Wait, if x+y is in universe, we can still have x and y in S, provided x+y is NOT in S.
           So x and y do NOT inherently conflict just because x+y is in universe.
           They conflict with x+y.
           
        This means the problem is NOT a simple graph MIS. It is a 3-SAT-like or Hypergraph Independent Set problem.
        
        However, we can solve it using a branch and bound on the elements, maintaining the set S.
        Upper bound: 
        If we have a partial set S, and remaining candidates P.
        We want to bound the maximum number of elements we can add from P to S.
        
        A simple upper bound: |P|.
        A better bound: 
        Consider the constraints imposed by S on P.
        If x in S, then for any y in P:
        - y + x cannot be in S (already true, x in S, y in P, so y+x not in S is guaranteed if y+x not in S? No, y+x could be in P or S).
        Actually, if x in S and y in P, we must ensure:
        1. y + y not in S (if y in S eventually)
        2. y + x not in S
        3. x + x not in S (already checked when x was added)
        
        This is complex. Let's use a simpler heuristic based on the "Sum Graph" MIS but with a repair/refinement step, or use a SAT solver approach?
        No, standard library only.
        
        Let's stick to the "Sum Graph MIS" as a strong lower bound and initial solution, and then try to improve it by "swapping" elements that were excluded due to sum graph edges but might fit if the third element is excluded.
        
        Actually, the "Sum Graph" approach yields a valid sum-free set. The maximum sum-free set might be larger.
        Example: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.
        Sum graph edges: (1,2)->3, (1,3)->4, etc.
        MIS of sum graph is likely around n/2 or n/3.
        
        Let's try a different angle: 
        The problem is equivalent to finding the largest subset S such that S is disjoint from S+S.
        
        Let's implement a Branch and Bound that explores the inclusion/exclusion of elements.
        State: (index i, current_set S, current_size).
        Upper bound: current_size + (n - i).
        Pruning: If upper_bound <= best_size, prune.
        
        To make it faster, we need a tighter upper bound.
        We can compute the maximum independent set of the "conflict graph" of the remaining elements P, assuming S is empty? No, S affects conflicts.
        
        Let's use the following heuristic for the upper bound:
        The remaining elements P can be partitioned into groups based on their "sums".
        This is getting too complicated for a quick implementation.
        
        Alternative: Use the Spectral method to get a good initial guess, then use Local Search with "2-swap" (remove 1, add 1) and "3-swap" (remove 2, add 2 or remove 1 add 2) to escape local optima.
        The previous program did 1-swap. 2-swap might be better.
        
        Let's refine the Local Search to include 2-for-2 swaps and 1-for-2 swaps (remove 1, add 2).
        
        Strategy:
        1. Greedy construction (sorted by abs value descending).
        2. Local Search:
           - Try adding any element that fits.
           - If stuck, try 1-for-1 swap.
           - If stuck, try 1-for-2 swap (remove x from S, add y, z from Remaining if they fit with S\{x}).
           - If stuck, try 2-for-1 swap (remove x, y from S, add z from Remaining).
        
        This is a standard improvement for set packing problems.
        
        Let's implement this.
        """
        n = len(elements)
        if n <= 10:
            # Brute force for small n
            best = []
            best_set = set()
            for mask in range(1 << n):
                S = [elements[i] for i in range(n) if (mask >> i) & 1]
                if is_sum_free(S):
                    if len(S) > len(best):
                        best = S
                        best_set = set(S)
            return best
        
        # Greedy Initialization
        sorted_elems = sorted(elements, key=lambda x: abs(x), reverse=True)
        S = []
        S_set = set()
        for x in sorted_elems:
            conflict = False
            for a in S:
                if (x - a) in S_set or (a + x) in S_set:
                    conflict = True
                    break
            if (2 * x) in S_set:
                conflict = True
            if not conflict:
                S.append(x)
                S_set.add(x)
                
        remaining = [x for x in elements if x not in S_set]
        rng_local = Random(42)
        rng_local.shuffle(remaining)
        
        def can_add(x, S_list, S_set_local):
            for a in S_list:
                if (x - a) in S_set_local or (a + x) in S_set_local:
                    return False
            if (2 * x) in S_set_local:
                return False
            return True
            
        def try_add(x, S_list, S_set_local, remaining_list):
            if can_add(x, S_list, S_set_local):
                S_list.append(x)
                S_set_local.add(x)
                if x in remaining_list:
                    remaining_list.remove(x)
                return True
            return False
            
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Pass 1: Add any single element that fits
            i = 0
            while i < len(remaining):
                if time.time() > max_time: break
                x = remaining[i]
                if can_add(x, S, S_set):
                    S.append(x)
                    S_set.add(x)
                    remaining.pop(i)
                    improved = True
                    # Restart to check new possibilities
                    break
                else:
                    i += 1
            
            if not improved:
                # Pass 2: 1-for-1 Swap
                # Remove x from S, add y from Remaining
                found_swap = False
                # To speed up, limit the number of swaps tried per iteration
                for _ in range(50):
                    if time.time() > max_time: break
                    if not S or not remaining: break
                    i = rng_local.randint(0, len(S) - 1)
                    j = rng_local.randint(0, len(remaining) - 1)
                    x_remove = S[i]
                    x_add = remaining[j]
                    
                    # Tentatively remove
                    S.remove(x_remove)
                    S_set.remove(x_remove)
                    
                    if can_add(x_add, S, S_set):
                        S.append(x_add)
                        S_set.add(x_add)
                        remaining.pop(j)
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Revert
                        S.append(x_remove)
                        S_set.add(x_remove)
                        
                if found_swap:
                    continue
                    
                # Pass 3: 1-for-2 Swap (Remove 1 from S, Add 2 from Remaining)
                # This allows us to "trade" one element for two, increasing size.
                found_1_2 = False
                for _ in range(20): # Limit attempts
                    if time.time() > max_time: break
                    if not S or len(remaining) < 2: break
                    i = rng_local.randint(0, len(S) - 1)
                    x_remove = S[i]
                    
                    S.remove(x_remove)
                    S_set.remove(x_remove)
                    
                    # Try to find two elements in remaining that can be added
                    added = []
                    r_copy = remaining[:] # Shallow copy for iteration
                    rng_local.shuffle(r_copy)
                    
                    for y in r_copy:
                        if can_add(y, S, S_set):
                            S.append(y)
                            S_set.add(y)
                            added.append(y)
                            if len(added) == 2:
                                break
                                
                    if len(added) == 2:
                        # Remove added from remaining
                        for y in added:
                            remaining.remove(y)
                        improved = True
                        found_1_2 = True
                        break
                    else:
                        # Revert
                        for y in added:
                            S.remove(y)
                            S_set.remove(y)
                        S.append(x_remove)
                        S_set.add(x_remove)
                        
                if found_1_2:
                    continue
                    
                # Pass 4: 2-for-1 Swap (Remove 2 from S, Add 1 from Remaining)
                # This is less likely to improve size, but might enable further additions later?
                # Actually, this reduces size immediately. Only useful if it unlocks many additions later.
                # Skip for now to save time, focus on size-increasing moves.
                
                # Pass 5: Random Restart (if stuck for too long)
                # If we are stuck, maybe the current S is bad. 
                # But restarting loses progress. 
                # Instead, let's try a "shaking" move: remove a random subset of S and re-greedy.
                # This is expensive. 
                
                # If no move found, we are in a local optimum.
                # To escape, we can try a random 2-for-2 swap that maintains size.
                # Or just accept it.
                
                # Let's try one 2-for-2 swap that maintains size but changes composition.
                # This might allow a 1-for-2 later.
                for _ in range(10):
                    if time.time() > max_time: break
                    if len(S) < 2 or len(remaining) < 2: break
                    i1 = rng_local.randint(0, len(S) - 1)
                    i2 = rng_local.randint(0, len(S) - 1)
                    if i1 == i2: continue
                    j1 = rng_local.randint(0, len(remaining) - 1)
                    j2 = rng_local.randint(0, len(remaining) - 1)
                    if j1 == j2: continue
                    
                    x1 = S[i1]
                    x2 = S[i2]
                    y1 = remaining[j1]
                    y2 = remaining[j2]
                    
                    # Remove x1, x2
                    S.remove(x1)
                    S_set.remove(x1)
                    S.remove(x2)
                    S_set.remove(x2)
                    
                    if can_add(y1, S, S_set) and can_add(y2, S, S_set):
                        # Check if y1 and y2 conflict with each other
                        if can_add(y2, S + [y1], S_set | {y1}):
                            S.append(y1)
                            S_set.add(y1)
                            S.append(y2)
                            S_set.add(y2)
                            remaining.remove(y1)
                            remaining.remove(y2)
                            # Size unchanged, but composition changed.
                            # Don't set improved=True to avoid infinite loop of 2-for-2
                            # But we should break to re-evaluate 1-for-2 possibilities.
                            break
                        else:
                            # Revert
                            S.append(x1)
                            S_set.add(x1)
                            S.append(x2)
                            S_set.add(x2)
                    else:
                        S.append(x1)
                        S_set.add(x1)
                        S.append(x2)
                        S_set.add(x2)
                        
        return S

    def solve_instance(elements, max_time=30.0):
        # Run multiple restarts with different random seeds/orderings?
        # The greedy order is fixed (abs value). 
        # We can randomize the initial order.
        best_solution = []
        best_size = 0
        
        # Try a few different initial orderings
        orderings = [
            lambda x: abs(x),
            lambda x: -abs(x),
            lambda x: x,
            lambda x: -x,
            lambda x: Random(1).random() # Random order
        ]
        
        # We only have time for a few runs if n is large.
        # For n=240, one run of local search might take a while.
        # Let's just do one good run with the best heuristic (abs desc) and the improved local search.
        
        solution = branch_and_bound(elements, max_time)
        
        # Verify
        if is_sum_free(solution):
            return solution
        else:
            # Fallback to a simple valid set
            return [elements[0]]

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        # Allocate time based on size
        # n=60: fast, n=120: medium, n=240: slow
        # Local search cost is O(N^2) per check.
        # Let's give 20s, 30s, 50s respectively.
        time_budgets = {60: 20.0, 120: 30.0, 240: 50.0}
        tb = time_budgets.get(size, 30.0)
        
        remaining_time = deadline - time.time()
        if tb > remaining_time:
            tb = max(1.0, remaining_time - 1.0)
            
        subset = solve_instance(elems, tb)
        
        # Final verification
        if not is_sum_free(subset):
            # If it failed, try a very simple greedy
            S = []
            S_set = set()
            for x in sorted(elems, key=abs, reverse=True):
                conflict = False
                for a in S:
                    if (x - a) in S_set or (a + x) in S_set:
                        conflict = True
                        break
                if (2 * x) in S_set:
                    conflict = True
                if not conflict:
                    S.append(x)
                    S_set.add(x)
            subset = S
            
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
