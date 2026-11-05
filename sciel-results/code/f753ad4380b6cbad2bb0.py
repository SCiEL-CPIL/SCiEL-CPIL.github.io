"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_max_independent_set(vertices, adj, time_limit=10.0):
        """
        Find a Maximum Independent Set using Bron-Kerbosch with pivot and coloring.
        vertices: list of vertex indices
        adj: list of sets, where adj[i] is the set of neighbors of vertex i
        Returns: list of vertex indices in the maximum independent set found
        """
        n = len(vertices)
        if n == 0:
            return []
        
        best_set = []
        best_size = 0
        start_ts = time.time()
        
        # Precompute non-neighbors for each vertex (candidates that can be added)
        # In Bron-Kerbosch for MIS, we maintain P (candidates), X (excluded)
        # P: vertices that can be added to the current path
        # X: vertices that cannot be added (already processed or conflict)
        
        # We use a recursive backtracking approach
        # To avoid deep recursion issues, we implement it iteratively or with a safe limit
        
        # Coloring heuristic for pivot selection
        # We want to pick a pivot u from P U X that maximizes the number of neighbors in P
        # This reduces the branching factor
        
        # For efficiency, we represent P and X as bitsets or sets
        # Given n <= 240, sets are fine
        
        def bron_kerbosch(P, X, path):
            nonlocal best_set, best_size, start_ts
            
            if time.time() - start_ts > time_limit:
                return
                
            if not P and not X:
                if len(path) > best_size:
                    best_size = len(path)
                    best_set = list(path)
                return
            
            # Pivot selection: choose u in P U X with max |P ∩ N(u)|
            # This minimizes the number of recursive calls
            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return
                
            # Find pivot with maximum neighbors in P
            max_neighbors = -1
            pivot = None
            for u in pivot_candidates:
                # Count neighbors of u in P
                count = 0
                for v in P:
                    if v in adj[u]:
                        count += 1
                if count > max_neighbors:
                    max_neighbors = count
                    pivot = u
                    
            # Branch on P \ N(pivot)
            # If P is empty, we should have returned above
            # Elements to branch on: P - N(pivot)
            branch_candidates = [v for v in P if v not in adj[pivot]]
            
            # Optimization: If branch_candidates is empty, no solution here
            if not branch_candidates:
                return
                
            # Order branch candidates by degree or some heuristic? 
            # Usually, we just iterate. Coloring can help order them.
            # Let's just iterate through them
            
            for v in branch_candidates:
                # Recursive call:
                # P' = P ∩ N(v)
                # X' = X ∩ N(v)
                # path' = path + [v]
                
                # Check time limit
                if time.time() - start_ts > time_limit:
                    return
                    
                P_new = [x for x in P if x in adj[v]]
                X_new = [x for x in X if x in adj[v]]
                
                bron_kerbosch(P_new, X_new, path + [v])
                
                P.remove(v)
                X.add(v)
                
                if time.time() - start_ts > time_limit:
                    return

        # Initial call
        # P = all vertices, X = empty
        P_init = list(range(n))
        X_init = set()
        
        bron_kerbosch(P_init, X_init, [])
        
        return best_set

    def solve_one(vals):
        """Find a large sum-free subset of vals using MIS on conflict graph."""
        n = len(vals)
        if n == 0:
            return []
        
        # Build conflict graph
        # Vertex i conflicts with vertex j if adding both to a set creates a sum-free violation?
        # No, the constraint is: no a, b, c in S such that a+b=c.
        # This is a 3-uniform hypergraph constraint, not a pairwise graph constraint.
        # MIS on a graph handles pairwise conflicts (a+b=c where a,b are fixed, c is fixed).
        # Wait, if we select a set S, it is sum-free if for ALL a,b in S, a+b is NOT in S.
        # This is NOT a pairwise conflict. It's a global constraint.
        # Example: {1, 2, 3}. 1+2=3. Conflict involves 3 vertices.
        # MIS on a graph where edge (i,j) means i+j is in the set? No, that depends on the whole set.
        
        # Therefore, standard MIS on a static graph does NOT directly apply to sum-free sets
        # because the conflict between two elements depends on whether a third element is present.
        
        # However, we can model this differently.
        # A set S is sum-free iff S ∩ (S+S) = empty.
        # This is equivalent to finding a large subset with no internal sums.
        
        # Alternative approach:
        # The current program uses greedy. The inspiration mentioned "greedy largest-magnitude".
        # The prompt asks for a "MATERIALLY DIFFERENT" strategy.
        # Since sum-free is a 3-uniform constraint, exact search like Bron-Kerbosch on a graph is not directly applicable unless we reformulate.
        
        # Let's reconsider the "different strategy".
        # Maybe use a different heuristic: **Simulated Annealing** or **Genetic Algorithm**?
        # Or **Local Search with 2-opt swaps** (which is in the current code but maybe poorly implemented)?
        # The current code has `try_swaps` but it's only called if... wait, `solve_one` does NOT call `try_swaps`.
        # It only calls `greedy_augment`.
        
        # So the current program is purely greedy.
        # A materially different strategy could be **Local Search / Hill Climbing** starting from multiple random initializations or the greedy result, using 1-move (add/remove/swap) and 2-move (swap two) operations.
        
        # Let's implement a robust Local Search with Simulated Annealing or just Hill Climbing with restarts.
        
        # Strategy:
        # 1. Generate a few diverse initial sets (Greedy Odd, Greedy Largest, Random).
        # 2. For each, run Local Search (Hill Climbing) to optimize size.
        # 3. Return the best one.
        
        def is_sum_free(S):
            S_set = set(S)
            for a in S:
                for b in S:
                    if a + b in S_set:
                        return False
            return True

        def can_add(S_set, x):
            """Check if adding x to S keeps it sum-free."""
            if 2 * x in S_set:
                return False
            for a in S_set:
                if a + x in S_set:
                    return False
            return True

        def can_remove(S_set, x):
            """Removing x always keeps it sum-free (if it was sum-free before)."""
            return True

        def local_search(S_init, all_elements, max_iters=500):
            """
            Hill Climbing with 1-move (add/remove) and 2-move (swap) operations.
            """
            S_set = set(S_init)
            S_list = list(S_init)
            current_size = len(S_set)
            
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                
                # Try Add
                for x in all_elements:
                    if x not in S_set:
                        if can_add(S_set, x):
                            S_set.add(x)
                            S_list.append(x)
                            current_size += 1
                            improved = True
                            break  # Restart search after improvement
                
                if improved:
                    continue
                
                # Try Remove (usually not helpful for maximizing size, but maybe needed to enable adds? 
                # No, removing an element never creates new conflicts, it only removes potential sums.
                # But we want to maximize size. So we only remove if it allows adding more later?
                # That's a 2-move: Remove x, Add y.
                
                # Try Swap: Remove x, Add y
                for i, x in enumerate(S_list):
                    for y in all_elements:
                        if y in S_set or y == x:
                            continue
                        
                        # Simulate removal of x
                        temp_set = S_set - {x}
                        
                        # Check if y can be added to temp_set
                        if can_add(temp_set, y):
                            # Accept swap
                            S_set = temp_set | {y}
                            S_list = list(S_set)
                            improved = True
                            break
                    if improved:
                        break
                            
            return S_list

        # Initial Heuristics
        candidates = []
        
        # 1. Greedy Odd
        odd = [x for x in vals if x % 2 != 0]
        candidates.append(odd)
        
        # 2. Greedy Largest Magnitude
        sorted_by_abs = sorted(vals, key=abs, reverse=True)
        S_greedy_abs = []
        S_set_greedy_abs = set()
        for x in sorted_by_abs:
            if can_add(S_set_greedy_abs, x):
                S_set_greedy_abs.add(x)
                S_greedy_abs.append(x)
        candidates.append(S_greedy_abs)
        
        # 3. Greedy Random (several times)
        for _ in range(3):
            random.shuffle(vals)
            S_greedy_rand = []
            S_set_greedy_rand = set()
            for x in vals:
                if can_add(S_set_greedy_rand, x):
                    S_set_greedy_rand.add(x)
                    S_greedy_rand.append(x)
            candidates.append(S_greedy_rand)
            
        # Run Local Search on each candidate
        best = []
        best_size = 0
        
        for cand in candidates:
            if time.time() - start_time > DEADLINE - 2:
                break
            result = local_search(cand, vals)
            if len(result) > best_size:
                best_size = len(result)
                best = result
                
        # Final verification
        if not is_sum_free(best):
            # Fallback to odd
            best = [x for x in vals if x % 2 != 0]
            
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - 5:
            # Not enough time, return odd numbers as fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        vals = instance(size, seed)
        result = solve_one(vals)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
