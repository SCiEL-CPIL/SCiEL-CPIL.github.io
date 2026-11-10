"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit
    n = 512
    
    # Precompute all 4-APs (hyperedges)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= n
    # We represent the set of integers as a bitset or boolean list.
    # For Branch and Bound, we need to check if a set of vertices is independent.
    
    # Strategy:
    # 1. Greedy initialization to get a good lower bound (incumbent).
    # 2. Branch and Bound on the remaining vertices to see if we can improve.
    #    However, standard B&B on 4-uniform hypergraphs is hard.
    #    We will use a "conflict-based" pruning.
    #    
    #    Let S be the current set of chosen vertices.
    #    Let R be the remaining vertices.
    #    A vertex v in R is "blocked" if adding it to S creates a 4-AP.
    #    Actually, we must ensure that NO 4-AP is fully contained in S ∪ {v}.
    #    
    #    Let's refine the B&B approach:
    #    We process vertices in a specific order.
    #    State: (index, current_set, current_size)
    #    If we include vertex i, we must remove all vertices j such that
    #    {i, j} is part of a 4-AP that is "almost complete" in current_set.
    #    This is complex.
    
    # Alternative Structural Change:
    # Use a "Coloring" based heuristic or a different graph formulation?
    # The prompt suggests: "Maximum Weight Independent Set on a bipartite graph derived from a 2-coloring... edges represent 4-AP conflicts".
    # This is theoretically flawed because 4-APs are 4-uniform, not 2-uniform (edges).
    # However, we can model "conflicts" as pairs. If two numbers are in the set, they might force others out.
    
    # Let's stick to a robust, fast Branch-and-Bound for Max Independent Set on the 4-AP hypergraph.
    # Since n=512 is small, we can represent the set as a Python integer bitset for speed.
    
    # Precompute for each vertex v, the list of "conflict tuples" involving v.
    # Actually, it's better to precompute for each pair (u, v), the set of other pairs (w, x) such that
    # {u, v, w, x} is a 4-AP.
    # If u and v are in the set, then w and x CANNOT both be in the set.
    # This reduces the problem to checking pairwise conflicts conditional on the first two.
    
    # Let's try a different structural approach:
    # Iterated Local Search with a "Greedy Coloring" upper bound.
    # Or, use the "Maximum Independent Set" solver logic directly on the 4-AP hypergraph.
    
    # Given the 90s limit, a full B&B might not finish.
    # Let's implement a fast Greedy + Local Search with a smarter move set:
    # "2-Swap" moves: Remove two elements, add two different elements.
    # This requires efficient conflict checking.
    
    # Precomputation:
    # For each x, list of (y, z, w) such that (x, y, z, w) is a 4-AP.
    # This is too big.
    
    # Let's use the "Bipartite" inspiration but correctly.
    # Split [1, n] into two sets A and B (e.g., parity or modulo).
    # This doesn't directly help with 4-APs.
    
    # Let's implement a specialized Branch and Bound that works on the "forbidden pairs" structure.
    # If we pick x, we restrict the future choices.
    
    # Let's try a different strategy: **Randomized Greedy with Backtracking (Limited Depth)**
    # 1. Order elements randomly.
    # 2. Pick an element.
    # 3. Remove all elements that form a 4-AP with the picked element AND any other already picked element?
    #    No, that's not right.
    
    # Let's go with **Simulated Annealing with a better move set**:
    # Moves:
    # 1. Add one element (if valid).
    # 2. Remove one element.
    # 3. Swap one element with another (remove x, add y).
    # 4. "Kick": Remove k elements, then greedily add back k+1 elements.
    
    # The current program does Greedy + Simple Kick.
    # The improvement should be in the **Conflict Checking Speed** and **Move Quality**.
    
    # Speedup: Precompute for each x, the list of "potential 4-APs" involving x.
    # For a 4-AP (a, b, c, d), if we add x, we check if x completes any 4-AP with 3 existing elements.
    
    # Let's precompute `ap_conflicts[x]` = list of tuples (y, z, w) such that (x, y, z, w) is a 4-AP.
    # When checking if x can be added to set S:
    #   For each (y, z, w) in ap_conflicts[x]:
    #       if y in S and z in S and w in S: return False
    # This is O(deg(x)) per check.
    # deg(x) is roughly O(n) in worst case? No, number of 4-APs involving x is O(n).
    # Total precomputation: O(n^2).
    
    ap_conflicts = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Add to conflict lists for each member
            # If we add 'a', we need to check if b, c, e are present.
            ap_conflicts[a].append((b, c, e))
            # If we add 'b', we need to check if a, c, e are present.
            ap_conflicts[b].append((a, c, e))
            # If we add 'c', we need to check if a, b, e are present.
            ap_conflicts[c].append((a, b, e))
            # If we add 'e', we need to check if a, b, c are present.
            ap_conflicts[e].append((a, b, c))
            
    # Note: The order in the tuple doesn't matter for the check `all in S`.
    
    def can_add(x, in_set):
        """Check if x can be added to in_set without creating a 4-AP."""
        for y, z, w in ap_conflicts[x]:
            if in_set[y] and in_set[z] and in_set[w]:
                return False
        return True

    def is_valid_set(in_set):
        """Verify the entire set is 4-AP free."""
        # This is O(n * deg) if we iterate through all x in set and check conflicts?
        # No, if we just check if any 4-AP is fully in S.
        # We can iterate through all 4-APs generated above.
        # But we didn't store the list of all 4-APs separately, just the conflicts.
        # Let's store all 4-APs for final verification.
        pass # We'll use a simpler check for the final result.

    # Store all 4-APs for final verification
    all_4aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            all_4aps.append((a, a+d, a+2*d, a+3*d))

    def check_validity(in_set):
        for a, b, c, d in all_4aps:
            if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                return False
        return True

    # 1. Greedy Initialization
    # Use a random order to allow different starting points, or a heuristic order.
    # Heuristic: Elements with fewer conflicts first?
    # Let's try multiple random restarts with Greedy.
    
    best_set = []
    best_size = 0
    random.seed(42)
    
    # Precompute a static order for greedy? No, random is better for diversity.
    # But we want to maximize size.
    
    # Let's run a few greedy passes with different orderings.
    # Ordering 1: Random
    # Ordering 2: Base-3 digit sum (known to be good for 3-APs, maybe 4-APs)
    
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s

    # Try base-3 ordering first
    order = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    in_set = [False] * (n + 1)
    current_set = []
    
    for x in order:
        if can_add(x, in_set):
            in_set[x] = True
            current_set.append(x)
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
        
    # 2. Local Search (Simulated Annealing / Hill Climbing)
    # We will use a "Kick and Recover" strategy which is effectively a restart.
    # But we can do better with "2-Swap" if we can make it fast.
    # With precomputed conflicts, `can_add` is fast.
    
    current_set = list(best_set)
    in_set = [False] * (n + 1)
    for x in current_set:
        in_set[x] = True
        
    # Identify "missing" elements
    missing = [x for x in range(1, n + 1) if not in_set[x]]
    
    # Hill Climbing: Try to add one element.
    # If we can add any, add the one that allows the most future additions?
    # For now, just add the first one found in random order.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        random.shuffle(missing)
        for x in missing:
            if time.time() > deadline - 10:
                break
            if can_add(x, in_set):
                in_set[x] = True
                current_set.append(x)
                missing.remove(x)
                best_size = len(current_set)
                best_set = list(current_set)
                improved = True
                # Restart the loop to check for more additions
                break
        
    # Now, try "Swaps" to escape local optima.
    # A swap: Remove x from current_set, Add y from missing.
    # This is valid if y can be added to (current_set - {x}).
    # We need to check if y conflicts with current_set - {x}.
    
    # To make this faster, we can precompute for each y, the conflicts with current_set.
    # But current_set changes.
    
    # Let's do a limited number of random swaps.
    for _ in range(5000): # Fixed number of swap attempts
        if time.time() > deadline - 5:
            break
            
        if not current_set or not missing:
            break
            
        # Pick a random element to remove
        idx = random.randint(0, len(current_set) - 1)
        x = current_set[idx]
        
        # Pick a random element to add
        y_idx = random.randint(0, len(missing) - 1)
        y = missing[y_idx]
        
        # Temporarily remove x
        in_set[x] = False
        current_set.pop(idx)
        
        # Check if y can be added
        if can_add(y, in_set):
            # Add y
            in_set[y] = True
            current_set.append(y)
            missing.pop(y_idx)
            missing.append(x)
            
            # Try to fill up again
            improved = True
            while improved:
                improved = False
                random.shuffle(missing)
                for z in missing:
                    if time.time() > deadline - 5:
                        break
                    if can_add(z, in_set):
                        in_set[z] = True
                        current_set.append(z)
                        missing.remove(z)
                        improved = True
                        break
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
        else:
            # Revert
            current_set.append(x)
            in_set[x] = True
            # y remains in missing (it was already there, we just checked it)
            # Wait, we didn't remove y from missing, so it's still there.
            
    # Final Verification
    final_in_set = [False] * (n + 1)
    for x in best_set:
        final_in_set[x] = True
        
    if not check_validity(final_in_set):
        # Fallback to greedy base-3
        fallback_in_set = [False] * (n + 1)
        fallback_set = []
        for x in order:
            if can_add(x, fallback_in_set):
                fallback_in_set[x] = True
                fallback_set.append(x)
        return sorted(fallback_set)
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
