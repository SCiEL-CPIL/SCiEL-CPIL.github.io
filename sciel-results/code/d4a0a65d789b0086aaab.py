import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
    
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # We need to identify which triples conflict with which.
    # Two triples conflict if they are part of the same K(2,2,2) transversal set.
    # Actually, the condition is: "no three disjoint pairs have all eight transversal triples present".
    # This means we cannot have *all* 8 triples of any K(2,2,2) configuration.
    # This is NOT a pairwise conflict graph in the simple sense of "edge means cannot coexist".
    # Wait. If I have triples $t_1, ..., t_8$ forming a K(2,2,2), I cannot have all of them.
    # I can have $t_1, ..., t_7$.
    # So, the constraint is: For every K(2,2,2) configuration $C$ (set of 8 triples), $|S \cap C| \le 7$.
    # This is a "7-out-of-8" constraint, not a "0-out-of-2" (pairwise) constraint.
    # Therefore, a simple Maximum Independent Set on a pairwise conflict graph is INCORRECT.
    # The "conflict graph" approach in the inspiration might be flawed if it treats it as pairwise.
    
    # Let's re-read the task: "no three disjoint pairs have all eight transversal triples present".
    # This is a hypergraph independence problem.
    # Vertices: Triples.
    # Hyperedges: Sets of 8 triples forming a K(2,2,2).
    # We want a subset of vertices that does not contain any hyperedge.
    
    # The "Branch and Bound on Conflict Graph" strategy in the prompt assumes a pairwise conflict.
    # Does a pairwise conflict exist?
    # If two triples $t_1, t_2$ are in a K(2,2,2), can they coexist? Yes, as long as not all 8 are present.
    # So, $t_1$ and $t_2$ do NOT have a hard pairwise conflict.
    
    # HOWEVER, if we look at the structure:
    # Maybe the problem can be modeled differently?
    # No, the constraint is clearly on the set of 8.
    
    # So, the "Maximum Independent Set in a conflict graph" strategy is **incorrect** for this problem definition unless the "conflict graph" is defined such that edges represent something else, or the problem is actually pairwise (which it isn't).
    
    # Let's check the inspiration code again.
    # Inspiration 1 code:
    # "Precompute all K(2,2,2) configurations... config_count[ci] = number of triples from config ci in solution... A config is 'full' if config_count[ci] == 8"
    # This inspiration code correctly models the 8-triple constraint using counts.
    # It does NOT use a pairwise conflict graph for the B&B. It uses greedy + local search with these counts.
    
    # The DIRECTIVE says: "Formulate the problem as a maximum independent set in a conflict graph... then solve it using a branch-and-bound".
    # This directive might be a generic suggestion that doesn't fit the specific constraint structure (hyperedges vs edges).
    # If I follow the directive strictly, I will produce an incorrect answer (allowing sets of 8).
    # I must prioritize **correctness**.
    
    # Strategy:
    # 1. Use the correct constraint model: Hyperedges of size 8.
    # 2. Use a Branch and Bound that handles hyperedges.
    #    State: `selected` (set of triples), `counts` (count of selected triples in each K(2,2,2)).
    #    Constraint: `counts[i] < 8` for all i.
    #    This is hard to branch on efficiently because adding a triple increments many counts.
    
    # Alternative:
    # The problem is small (560 triples).
    # We can use a **Greedy + Local Search** approach which is proven to work well for such packing problems (like the current program and inspiration 1).
    # To satisfy the "structural change" and "B&B" directive, I can implement a **B&B on the hypergraph** but with strong pruning.
    # Or, I can reformulate:
    # If we select 7 triples from a K(2,2,2), the 8th is "blocked".
    # This looks like a maximum packing problem.
    
    # Let's look at the "Conflict Graph" idea again.
    # Is it possible that the optimal solution never contains 2 triples from the same K(2,2,2)?
    # No, that would be very restrictive.
    
    # Let's stick to the correct model:
    # We want to maximize $|S|$ such that for all $K \in \mathcal{K}$, $|S \cap K| \le 7$.
    
    # Algorithm:
    # 1. Precompute all K(2,2,2) configurations (hyperedges).
    # 2. Map triple -> list of config indices.
    # 3. Use a B&B search:
    #    - Order triples by some heuristic (e.g., degree in hypergraph).
    #    - Recursively decide to include or exclude the next triple.
    #    - Maintain `config_counts`.
    #    - Pruning:
    #       - If `current_size + remaining_triples <= best_size`, prune.
    #       - If adding a triple makes any `config_count` reach 8, it's invalid.
    #       - **Stronger Pruning**: If there is a config $C$ with `count[C] == 7`, then none of the other 7 triples in $C$ can be added. We can mark them as "blocked" in the remaining search space.
    
    # This is a valid B&B.
    
    # Implementation details:
    # - `configs`: List of lists of triple indices.
    # - `triple_to_configs`: List of lists of config indices.
    # - `counts`: Array of length `num_configs`.
    # - `selected`: List of triple indices.
    # - `available`: Bitmask of triples not yet decided.
    
    # To make it fast:
    # - Use bitmasks for `available`.
    # - The "blocked" logic: If a config has count 7, the other 7 triples are blocked.
    #   This is dynamic.
    
    # Given the 120s limit, a pure B&B might be too slow.
    # I will combine:
    # 1. Fast Greedy (multiple restarts) to get a good lower bound.
    # 2. Local Search (1-swaps) to improve.
    # 3. A limited B&B that tries to beat the best found, with a strict time limit.
    
    # Precomputation:
    k222_configs = []
    seen_configs = set()
    
    # Generate all K(2,2,2) configurations
    # Iterate over 6 points
    for six in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six
        # 15 pairings
        pairs_list = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        for pairs in pairs_list:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            t_indices = tuple(sorted(triple_index[t] for t in transversal))
            if t_indices not in seen_configs:
                seen_configs.add(t_indices)
                k222_configs.append(list(t_indices))
    
    num_configs = len(k222_configs)
    
    # Map triple -> configs
    triple_to_configs = [[] for _ in range(num_triples)]
    for ci, cfg in enumerate(k222_configs):
        for ti in cfg:
            triple_to_configs[ti].append(ci)
    
    # Degrees (number of configs a triple is in)
    degrees = [len(t) for t in triple_to_configs]
    
    # Sort triples by degree (descending) for B&B ordering
    # Higher degree triples are more constrained, better to decide earlier?
    # Or lower degree? Usually, in MIS, high degree is better for branching.
    # Here, high degree means it blocks more configs.
    order = sorted(range(num_triples), key=lambda x: degrees[x], reverse=True)
    
    best_size = 0
    best_mask = 0
    best_list = []
    
    # 1. Greedy Phase
    # Use random shuffles and degree-based ordering
    
    def greedy_run(seed):
        rng = random.Random(seed)
        rng.shuffle(order)
        counts = [0] * num_configs
        selected = []
        selected_mask = 0
        for ti in order:
            # Check if adding ti is safe
            safe = True
            for ci in triple_to_configs[ti]:
                if counts[ci] == 7:
                    safe = False
                    break
            if safe:
                selected.append(ti)
                selected_mask |= (1 << ti)
                for ci in triple_to_configs[ti]:
                    counts[ci] += 1
        return selected, selected_mask
    
    # Run multiple greedy runs
    for i in range(100):
        if time.time() > deadline - 30:
            break
        sel, mask = greedy_run(i)
        if len(sel) > best_size:
            best_size = len(sel)
            best_mask = mask
            best_list = sel
    
    # 2. Local Search Phase (1-swaps)
    # Try to improve best_list
    
    def local_search(initial_list, max_steps=500):
        if time.time() > deadline - 10:
            return initial_list
            
        counts = [0] * num_configs
        selected_set = set(initial_list)
        for ti in initial_list:
            for ci in triple_to_configs[ti]:
                counts[ci] += 1
        
        current_list = list(initial_list)
        current_size = len(current_list)
        
        # Precompute unselected
        unselected = [ti for ti in range(num_triples) if ti not in selected_set]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try removing one and adding another?
            # Or just try adding? (Greedy already did that).
            # Swap: Remove r, Add a.
            # We need to check if removing r allows adding a.
            
            # Randomly pick a candidate 'a' from unselected
            if not unselected:
                break
            a = random.choice(unselected)
            
            # Check if a is currently blocked
            blocking_configs = []
            for ci in triple_to_configs[a]:
                if counts[ci] == 7:
                    blocking_configs.append(ci)
            
            if not blocking_configs:
                # Can add directly? (Shouldn't happen if greedy was exhaustive, but maybe after removals)
                # Add a
                selected_set.add(a)
                current_list.append(a)
                current_size += 1
                for ci in triple_to_configs[a]:
                    counts[ci] += 1
                unselected.remove(a)
                improved = True
                continue
            
            # If blocked, try to remove a triple from one of the blocking configs
            ci = random.choice(blocking_configs)
            # Find a triple r in config ci that is selected
            candidates_to_remove = []
            for ti in k222_configs[ci]:
                if ti in selected_set:
                    candidates_to_remove.append(ti)
            
            if not candidates_to_remove:
                continue
                
            r = random.choice(candidates_to_remove)
            
            # Simulate removing r
            # Decrement counts for r
            for c in triple_to_configs[r]:
                counts[c] -= 1
            
            # Check if a can now be added
            safe_to_add = True
            for c in triple_to_configs[a]:
                if counts[c] == 7:
                    safe_to_add = False
                    break
            
            if safe_to_add:
                # Perform swap
                selected_set.remove(r)
                current_list.remove(r)
                selected_set.add(a)
                current_list.append(a)
                for c in triple_to_configs[a]:
                    counts[c] += 1
                # r is now unselected
                unselected.append(r)
                unselected.remove(a)
                improved = True
            else:
                # Rollback
                for c in triple_to_configs[r]:
                    counts[c] += 1
            
            if not improved and step % 100 == 0:
                # Try a different strategy or break if stuck?
                pass
        
        return current_list
    
    # Run local search on best found
    if time.time() < deadline - 20:
        improved_list = local_search(best_list, max_steps=2000)
        if len(improved_list) > best_size:
            best_size = len(improved_list)
            best_list = improved_list
            best_mask = 0
            for ti in best_list:
                best_mask |= (1 << ti)
        
        # Run local search on a few other good solutions
        for i in range(5, 20):
            if time.time() > deadline - 10:
                break
            sel, _ = greedy_run(i)
            if len(sel) > best_size - 5: # Only improve if close
                improved_list = local_search(sel, max_steps=500)
                if len(improved_list) > best_size:
                    best_size = len(improved_list)
                    best_list = improved_list
                    best_mask = 0
                    for ti in best_list:
                        best_mask |= (1 << ti)

    # 3. B&B Phase (Optional, if time remains)
    # Given the complexity, we might skip full B&B if greedy+LS is strong.
    # But let's try a very limited B&B to see if it can jump.
    
    # B&B uses the order sorted by degree.
    # We pass the current best_size as the lower bound.
    
    bb_nodes = 0
    
    def branch_and_bound(rem_mask, curr_mask, count, counts, index):
        nonlocal best_size, best_mask, bb_nodes, deadline
        
        if time.time() > deadline - 2:
            return
            
        bb_nodes += 1
        if bb_nodes % 1000 == 0 and time.time() > deadline - 2:
            return

        # Pruning 1: Upper bound
        # Max possible is count + popcount(rem_mask)
        if count + rem_mask.bit_count() <= best_size:
            return
            
        # Pruning 2: If no more vertices
        if rem_mask == 0:
            if count > best_size:
                best_size = count
                best_mask = curr_mask
            return
            
        # Pick next vertex
        # Find lowest set bit in rem_mask
        # But we want to follow the `order`?
        # The `rem_mask` contains all undecided vertices.
        # We should pick the next vertex in `order` that is in `rem_mask`.
        
        # To do this efficiently, we can just iterate `order` from `index`?
        # No, `rem_mask` changes.
        # Let's just pick the lowest index in `rem_mask` that is in `order`?
        # Actually, `order` is a permutation.
        # We can maintain a pointer? No, because we skip vertices.
        
        # Simple approach: Find the first vertex in `order` that is set in `rem_mask`.
        # This is O(N) per node. Too slow.
        
        # Alternative: Use `rem_mask` to pick a vertex.
        # Heuristic: Pick the vertex with highest degree in the remaining graph?
        # Hard to compute.
        
        # Let's pick the lowest set bit.
        # v = (rem_mask & -rem_mask).bit_length() - 1
        
        v = (rem_mask & -rem_mask).bit_length() - 1
        v_bit = 1 << v
        
        # Branch 1: Include v
        # Check if v is safe
        safe = True
        for ci in triple_to_configs[v]:
            if counts[ci] == 7:
                safe = False
                break
        
        if safe:
            # Update counts
            for ci in triple_to_configs[v]:
                counts[ci] += 1
            # Recurse
            branch_and_bound(rem_mask & ~v_bit, curr_mask | v_bit, count + 1, counts, index + 1)
            # Rollback
            for ci in triple_to_configs[v]:
                counts[ci] -= 1
        
        # Branch 2: Exclude v
        branch_and_bound(rem_mask & ~v_bit, curr_mask, count, counts, index + 1)
        
    # Initialize B&B
    # Start with all vertices available
    initial_rem_mask = (1 << num_triples) - 1
    initial_counts = [0] * num_configs
    
    # To make B&B faster, we can start with a greedy solution as the "current" state?
    # No, B&B explores from scratch.
    # But we can use the best greedy solution to set a high `best_size` for pruning.
    
    if time.time() < deadline - 5:
        # Run B&B
        # To avoid deep recursion, we limit the depth or nodes.
        # Python recursion limit is ~1000. 560 depth is fine.
        try:
            branch_and_bound(initial_rem_mask, 0, 0, initial_counts, 0)
        except RecursionError:
            pass
        except Exception:
            pass

    # Convert best_mask back to list
    result_list = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result_list.append(list(all_triples[i]))
            
    return result_list

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
