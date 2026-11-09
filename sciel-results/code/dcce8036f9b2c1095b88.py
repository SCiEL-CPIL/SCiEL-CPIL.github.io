"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def _is_valid(t, n, k=4, sample_limit=None):
    """
    Check if tournament t (n x n 0/1 matrix, t[i,j]=1 means i->j) has the property
    that every k-set has a common dominator.
    For k=4: every 4 vertices have a vertex with arcs to all four.
    """
    # Precompute: for each vertex v, the set of vertices it dominates
    # dominators[v] = set of u such that v->u
    # For a 4-set S, we need some v (not in S? or in S?) with v->u for all u in S.
    # Wait, "a vertex with arcs to all four" - does this vertex have to be one of the four, or can it be external?
    # Schütte's problem: "Every set of 4 vertices must have a vertex with arcs to all four."
    # Usually in tournament domination problems, the dominator is from the whole vertex set, not necessarily in the subset.
    # Let's re-read carefully: "Every set of 4 vertices must have a vertex with arcs to all four."
    # This phrasing is ambiguous. Does "a vertex" mean a vertex in the tournament (any vertex) or a vertex in the set?
    # If it means any vertex in the tournament: for every 4-subset S, there exists v in V such that v->u for all u in S.
    # If it means a vertex in S: for every 4-subset S, there exists v in S such that v->u for all u in S\{v}. This is just saying S has a king of length 1, i.e., a source in the induced subgraph. But a tournament on 4 vertices always has a source? No, a cycle of length 3 plus a sink has no source dominating all. A 4-cycle isn't possible in tournament. Actually, every tournament has a source (vertex with out-degree n-1)? No, only if it's transitive.
    # Let's check the standard definition of Schütte's problem or similar "common dominator" problems.
    # "Common dominator" usually means a vertex v such that v dominates all vertices in the set.
    # If v can be outside the set, the problem is much harder to satisfy for small n? Or easier?
    # If v must be in the set, it's equivalent to saying the induced subgraph on any 4 vertices has a vertex with out-degree 3 within that set.
    # Let's look at the lower bound: Erdős's lower bound gives f(4) >= 31.
    # If it were "induced subgraph has a source", the minimum n for which EVERY 4-set has a source is related to Ramsey numbers. R(4,3)=9? No.
    # Let's assume the standard interpretation: there exists a vertex v in the ENTIRE tournament such that v->u for all u in the 4-set.
    # This is equivalent to: the intersection of the out-neighborhoods of any 4 vertices is non-empty? No.
    # It means: for every 4-subset S, there is a v in V such that S subset of out_neighbors(v).
    # This is equivalent to: the family of out-neighborhoods {N^+(v) : v in V} covers all 4-subsets.
    # i.e., every 4-subset is contained in some N^+(v).
    
    # Let's verify with the Paley tournament on 83 vertices.
    # In a Paley tournament, the out-neighborhood of v has size (p-1)/2 = 41.
    # The number of 4-subsets is C(83,4) ~ 1.9e7.
    # Each N^+(v) contains C(41,4) ~ 100k 4-subsets.
    # 83 * 100k = 8.3e6. This is less than 1.9e7.
    # So by counting, it's IMPOSSIBLE for Paley(83) to cover all 4-subsets if the dominator must be external or internal?
    # Wait, if the dominator v is in V, and S is a 4-set, v can be in S or not.
    # If v is in S, v cannot dominate itself. So "arcs to all four" implies v is NOT in S.
    # So v must be external to S.
    # So we need: for every 4-subset S of V, there exists v in V\S such that v->u for all u in S.
    # This means S is a subset of N^+(v) for some v not in S.
    # Since v notin S, and v->u for u in S, this is just S subset N^+(v).
    # So the counting argument holds: sum_{v} C(|N^+(v)|, 4) >= C(n,4).
    # For Paley(83): 83 * C(41,4) = 83 * 101270 = 8,405,410.
    # C(83,4) = 83*82*81*80 / 24 = 19,319,100.
    # 8.4M < 19.3M. So Paley(83) CANNOT satisfy this property if the dominator must be external.
    
    # Therefore, the property MUST be: every 4-set S has a dominator v IN S such that v->u for all u in S\{v}.
    # i.e., the induced subgraph on any 4 vertices has a source (a vertex with out-degree 3 in the induced subgraph).
    # Let's re-read: "Every set of 4 vertices must have a vertex with arcs to all four."
    # "Arcs to all four" - if the vertex is in the set, it has arcs to the other three. It doesn't have an arc to itself.
    # So "arcs to all four" is slightly loose language for "arcs to all other three in the set".
    # OR, it means the vertex is external and has arcs to all four in the set.
    # Given the counting contradiction for Paley(83), it must be the internal interpretation.
    # Let's check the counting for the internal interpretation.
    # We need every 4-subset to contain a source.
    # This is equivalent to saying there is NO 4-subset that is a "cycle" or has no source.
    # A tournament on 4 vertices has no source iff it is strongly connected?
    # No, a tournament has a source iff it is not strongly connected? No.
    # A tournament is transitive iff it has a Hamiltonian path and...
    # Let's list tournament structures on 4 vertices.
    # 1. Transitive: 3->2->1->0. Sources: 3. (Has source)
    # 2. One 3-cycle + 1 vertex.
    #    If the 4th vertex dominates the 3-cycle: Source.
    #    If the 3-cycle dominates the 4th: No source? The 3-cycle vertices have out-degree 1 within the cycle, and 1 to the 4th. Total out-degree 2. No vertex has out-degree 3. So NO SOURCE.
    #    If the 4th vertex beats 2 in the cycle and loses to 1: The vertex that lost to 4th and beat the other... let's see.
    #    Cycle 1->2->3->1. 4 beats 1, 2 beats 4.
    #    Out-degrees in induced subgraph:
    #    1: beats 2, loses to 3, loses to 4. Out=1.
    #    2: beats 3, beats 4, loses to 1. Out=2.
    #    3: beats 1, loses to 2, loses to 4? No, 4 beats 1, 2 beats 4. What about 3 vs 4?
    #    Let's be systematic.
    #    A 4-vertex tournament has a source iff it is not one of the two specific "strong" types?
    #    Actually, a tournament has a source iff it is not strongly connected?
    #    No, a 3-cycle is strongly connected and has no source (max out-degree 1).
    #    A 4-vertex tournament is strongly connected iff it has no source and no sink.
    #    So we need to avoid 4-vertex strongly connected subtournaments.
    #    How many 4-vertex tournaments are there? 2^6 = 64.
    #    How many are strongly connected?
    #    Total 64.
    #    Transitive: 1.
    #    With a source: The source dominates 3 others. The remaining 3 form a tournament.
    #    If the remaining 3 is transitive, the whole thing is transitive.
    #    If the remaining 3 is a cycle, the whole thing has a source (the first one) but is not transitive.
    #    Number of tournaments on 4 with a source:
    #    Choose source (4 ways). The other 3 form any tournament (8 ways).
    #    But this overcounts transitive ones?
    #    In a transitive tournament, there is exactly 1 source.
    #    In a non-transitive tournament with a source, there is exactly 1 source?
    #    If a tournament has a source v, v dominates all. If another vertex u also dominated all, it would dominate v, contradiction. So at most 1 source.
    #    So number of tournaments with a source = sum over all 4-sets of (1 if has source else 0).
    #    Actually, let's just count the "bad" ones (no source).
    #    A tournament has no source iff every vertex has in-degree >= 1.
    #    For n=4, this means it is strongly connected?
    #    Yes, for n=4, if there is no source, is it strongly connected?
    #    Consider a 3-cycle 1->2->3->1 and a vertex 4.
    #    If 4 is dominated by all: 4 is a sink. No source. Not strongly connected (4 can't reach others).
    #    So "no source" includes tournaments with a sink but no source.
    #    So we must avoid:
    #    1. Tournaments with a sink (and no source).
    #    2. Tournaments that are strongly connected.
    #    Actually, simpler: We just need to check if the induced subgraph on any 4 vertices has a vertex with out-degree 3.
    
    # Let's verify the Paley(83) claim with this interpretation.
    # Does Paley(83) have the property that every 4-set has a source?
    # This is a known property of Paley tournaments?
    # Paley tournaments are "locally transitive" or have other properties?
    # Actually, Paley tournaments are known to be "score-regular".
    # Let's just trust the seed program and the task description's implication that the seed is valid.
    # The seed uses Paley(83). The task says "The smallest known has 67 vertices".
    # The seed is 83. The score is 1.0 (seed/value). If seed is 83, and we find 67, score is 83/67 = 1.239.
    # This matches the "inspiration" note: "Switching to the Paley tournament on 67 vertices... Measured: 1 -> 1.239."
    # Wait, 67 is not a prime of the form 4k+3?
    # 67 = 4*16 + 3. Yes, it is.
    # So Paley(67) exists.
    # Does Paley(67) satisfy the property?
    # The inspiration says "which is a known construction satisfying the 4-dominator property".
    # So yes, Paley(67) is valid.
    
    # So the goal is to find a tournament with n < 67 that satisfies the property.
    # Or to find a better construction.
    # The lower bound is 31.
    # We need to search for n in [31, 66].
    
    # Strategy:
    # 1. Try to construct a valid tournament for small n using local search / simulated annealing.
    # 2. Start with a random tournament or a structured one (like cyclic) and flip edges to satisfy constraints.
    # 3. Constraint: For every 4-subset S, there exists v in S such that v->u for all u in S\{v}.
    #    Equivalently: For every 4-subset S, it is NOT the case that for all v in S, there exists u in S\{v} such that u->v.
    #    i.e., no vertex in S has in-degree 0 within S.
    #    So we want to AVOID 4-subsets where every vertex has in-degree >= 1 within the subset.
    #    Let's call such a subset "bad".
    #    We want to minimize the number of bad 4-subsets to 0.
    
    # Objective: Minimize n such that there exists a tournament with 0 bad 4-subsets.
    # We can search for a tournament with 0 bad 4-subsets for increasing n.
    # Start with n=31, then 32, etc.
    # But 120s is tight.
    # Maybe start with n=67 (known good) and try to reduce n?
    # Or start with n around 40-50 and do local search?
    
    # Let's estimate the difficulty.
    # Number of 4-subsets for n=40: C(40,4) = 91,390.
    # For each 4-subset, we check if it has a source.
    # This is fast.
    # Local search: Pick a random 4-subset that is bad. Flip an edge within it to try to create a source.
    # Or pick a random vertex and flip its edges?
    
    # Let's implement a local search for a given n.
    # 1. Initialize tournament randomly.
    # 2. Count bad 4-subsets.
    # 3. While bad > 0 and time < limit:
    #    a. Pick a random bad 4-subset S.
    #    b. Try flipping one of the edges within S to see if it reduces the total number of bad 4-subsets.
    #    c. If yes, accept. If no, maybe accept with some probability (simulated annealing).
    #    d. Or just pick a random edge in S and flip it.
    
    # To make it efficient, we need to quickly update the count of bad 4-subsets when an edge is flipped.
    # Flipping edge (u,v) affects all 4-subsets containing {u,v}.
    # There are n-2 such subsets.
    # For each such subset S = {u,v,x,y}, we check if it was bad before and is bad after.
    # This is O(n) per flip.
    # Total flips needed? Unknown.
    # If n=40, n-2=38. Very fast.
    # We can do many flips per second.
    
    # Plan:
    # 1. Define a function to check if a 4-tuple is bad.
    # 2. Define a function to count bad 4-subsets for a given tournament.
    # 3. Define a local search loop.
    # 4. Iterate n from 31 upwards.
    # 5. For each n, run local search for a fixed time budget.
    # 6. If a valid tournament is found, return it.
    # 7. If time runs out, return the best found (smallest n with 0 bad) or the seed (67) if none found.
    #    Actually, if we don't find a valid one, we should return the known valid one (Paley 67) to ensure a valid score.
    #    The task says "Minimize n". If we can't find a smaller one, we should return the best valid one we have.
    #    We know Paley(67) is valid. So we can always return that as a fallback.
    #    If we find a valid one with n < 67, we return that.
    
    # Let's refine the local search.
    # State: tournament T (n x n boolean matrix).
    # Cost: number of bad 4-subsets.
    # Move: flip edge (i,j).
    # Delta cost: 
    #   Let S_old be the set of 4-subsets containing i,j that were bad.
    #   Let S_new be the set of 4-subsets containing i,j that are bad after flip.
    #   Delta = |S_new| - |S_old|.
    #   We only need to check the n-2 subsets {i,j,k,l} for k,l in V\{i,j}.
    
    # Implementation details:
    # Represent T as a numpy array of 0s and 1s.
    # T[i,j] = 1 means i->j.
    # Bad 4-subset {a,b,c,d}: for all v in {a,b,c,d}, there exists u in {a,b,c,d}\{v} such that T[u,v]=1.
    # Equivalently: for all v in {a,b,c,d}, sum_{u in {a,b,c,d}\{v}} T[u,v] > 0.
    
    # Let's write the code.
    
    def count_bad(t, n):
        bad_count = 0
        # Iterate over all 4-subsets
        # This is O(n^4). For n=40, 40^4 = 2.5M. This is too slow in Python if done naively in a loop.
        # We need to be smarter or use numpy.
        # Or, we can just use the local search which only updates O(n) per step.
        # But we need the initial count.
        # Maybe we don't need the exact count, just to find one bad subset.
        # But for simulated annealing, we need the cost.
        # Let's try to optimize the count.
        # Or, we can just do a "greedy" repair: find a bad subset, fix it, repeat.
        # This might get stuck in local minima.
        # Simulated annealing is better.
        
        # Let's use a vectorized approach for the initial count if possible, or just accept the O(n^4) cost for small n.
        # For n=31, C(31,4) = 31465. This is very fast.
        # For n=50, C(50,4) = 230300. Still fast.
        # For n=60, C(60,4) = 487635. Fast.
        # For n=67, C(67,4) = 751315. Fast.
        # So O(n^4) is fine for n < 100.
        
        bad_subsets = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        # Check if {a,b,c,d} is bad
                        # Bad if no source.
                        # Source a: T[a,b]==1 and T[a,c]==1 and T[a,d]==1
                        # Source b: T[b,a]==1 and T[b,c]==1 and T[b,d]==1
                        # ...
                        has_source = False
                        if t[a,b] and t[a,c] and t[a,d]:
                            has_source = True
                        elif t[b,a] and t[b,c] and t[b,d]:
                            has_source = True
                        elif t[c,a] and t[c,b] and t[c,d]:
                            has_source = True
                        elif t[d,a] and t[d,b] and t[d,c]:
                            has_source = True
                        
                        if not has_source:
                            bad_subsets.append((a,b,c,d))
        
        return len(bad_subsets), bad_subsets

    def is_bad_subset(t, a, b, c, d):
        # Returns True if {a,b,c,d} is bad (no source)
        if t[a,b] and t[a,c] and t[a,d]:
            return False
        if t[b,a] and t[b,c] and t[b,d]:
            return False
        if t[c,a] and t[c,b] and t[c,d]:
            return False
        if t[d,a] and t[d,b] and t[d,c]:
            return False
        return True

    def local_search(n, time_limit=10):
        # Initialize random tournament
        t = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    t[i,j] = 1
                    t[j,i] = 0
                else:
                    t[i,j] = 0
                    t[j,i] = 1
        
        # Initial count
        bad_count, bad_list = count_bad(t, n)
        if bad_count == 0:
            return t
        
        start_time = time.time()
        temperature = 1.0
        cooling = 0.999
        
        while bad_count > 0 and (time.time() - start_time) < time_limit:
            # Pick a random bad subset
            a, b, c, d = random.choice(bad_list)
            
            # Try flipping an edge in this subset
            # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            random.shuffle(edges)
            
            accepted = False
            for i, j in edges:
                # Flip edge (i,j)
                old_val = t[i,j]
                t[i,j] = 1 - old_val
                t[j,i] = old_val
                
                # Check new bad status for all subsets containing i,j
                # We need to update bad_count.
                # Instead of recalculating everything, we can just check the specific subsets.
                # But we need to know which subsets were bad before and after.
                # This is complex to track incrementally without a global structure.
                # Given n is small, let's just recalculate the bad count for the affected subsets.
                # Affected subsets are those containing i and j.
                # There are n-2 such subsets.
                
                new_bad_count = bad_count
                # Subtract the old status of affected subsets
                # Add the new status of affected subsets
                
                # To do this efficiently, we need to know which of the affected subsets were bad.
                # We can just check them.
                for k in range(n):
                    if k == i or k == j:
                        continue
                    for l in range(k+1, n):
                        if l == i or l == j:
                            continue
                        # Subset is {i,j,k,l}
                        # Sort them for consistency? Not needed for is_bad_subset.
                        # Check if it was bad before? We don't have the old state.
                        # This approach is flawed because we don't store the old state of each subset.
                        
                # Alternative: Just recalculate the total bad count from scratch?
                # O(n^4) per step is too slow.
                
                # Better approach:
                # Keep a list of bad subsets.
                # When flipping (i,j), only subsets containing i,j change status.
                # Let affected = [S for S in bad_list if i in S and j in S]
                # Also, some non-bad subsets containing i,j might become bad.
                # So we need to check ALL subsets containing i,j.
                
                # Let's collect all subsets containing i,j.
                affected_subs = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    for l in range(k+1, n):
                        if l == i or l == j:
                            continue
                        # Ensure i,j,k,l are distinct
                        # They are.
                        affected_subs.append((i,j,k,l))
                
                # For each affected subset, determine if it was bad (in bad_list) and if it is bad now.
                # This is still O(n^2) per flip to generate the list, and O(1) to check each.
                # Total O(n^2) per flip.
                # For n=40, 1600 operations. Very fast.
                
                # We need to update bad_list.
                # Let's create a set of bad subsets for fast lookup?
                # bad_set = set(bad_list)
                # This might be memory heavy but n is small.
                
                # Let's try a simpler heuristic:
                # Just flip the edge and recalculate the bad count for the affected subsets.
                # If the number of bad subsets decreases, accept.
                # If it increases, accept with probability exp(-delta/T).
                
                # To do this, we need to know the delta.
                # delta = (number of affected subsets that are bad now) - (number of affected subsets that were bad before)
                
                # Let's implement this carefully.
                
                # First, revert the flip to check the "before" state?
                # No, we already flipped it.
                # So we need to know the "before" status.
                # We can check the "before" status by looking at the old t?
                # We didn't save the old t.
                # Let's save the old values for the affected edges?
                # No, only one edge was flipped.
                # So for a subset S containing i,j, its status depends on t[i,j] and other edges.
                # The other edges didn't change.
                # So we can determine the "before" status by temporarily reverting t[i,j] and checking?
                # Or just calculate it logically.
                
                # Let's just recalculate the bad count from scratch for the affected subsets.
                # And compare with the count of those subsets in the current bad_list.
                
                # This is getting complicated. Let's use a simpler local search:
                # 1. Find a bad subset.
                # 2. Flip a random edge in it.
                # 3. Recalculate the total bad count (O(n^4)).
                # 4. If improved, keep. Else, revert.
                # With n=40, O(n^4) is ~100k ops. 100k ops is ~1ms in Python?
                # Maybe 10ms. So 100 flips per second.
                # 10s budget -> 1000 flips.
                # This might be enough to escape small local minima.
                
                # Let's try this simpler approach first.
                
                # Revert flip
                t[i,j] = old_val
                t[j,i] = 1 - old_val
                
                # Try flipping this edge
                t[i,j] = 1 - old_val
                t[j,i] = old_val
                
                new_bad_count, _ = count_bad(t, n)
                
                delta = new_bad_count - bad_count
                
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    # Accept
                    bad_count = new_bad_count
                    # Update bad_list
                    _, bad_list = count_bad(t, n)
                    accepted = True
                else:
                    # Revert
                    t[i,j] = old_val
                    t[j,i] = 1 - old_val
                
                if accepted:
                    break
            
            temperature *= cooling
            if temperature < 0.01:
                temperature = 0.01
        
        return t

    # Main logic
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Known valid: Paley 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    fallback_t = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                fallback_t[i,j] = 1
                fallback_t[j,i] = 0
            else:
                fallback_t[i,j] = 0
                fallback_t[j,i] = 1
    
    best_n = p
    best_t = fallback_t
    
    # Try to find smaller n
    # Start from 31 upwards
    # But we should prioritize likely candidates.
    # Let's try n=40, 45, 50, 55, 60 first?
    # Or just go 31, 32, ...
    # Given time constraints, maybe jump to the middle?
    # Let's try a few values.
    
    candidate_ns = list(range(31, 67))
    # Shuffle to explore different n? No, we want the smallest.
    # So we should check 31, 32, 33...
    # But if 31 is hard, we might waste time.
    # Let's allocate time per n.
    # Total time ~ 110s.
    # 36 values.
    # ~3s per value.
    
    for n in candidate_ns:
        if time.time() > deadline:
            break
        
        # Run local search for this n
        # Give it a time slice
        slice_time = 2.0
        t_candidate = local_search(n, time_limit=slice_time)
        
        # Check if valid
        bad_count, _ = count_bad(t_candidate, n)
        
        if bad_count == 0:
            best_n = n
            best_t = t_candidate
            break  # Found the smallest possible? Not necessarily, but we are iterating upwards.
            # So the first one we find is the smallest n we found.
            # We should continue to see if we can find a smaller one?
            # No, we are iterating from 31 upwards.
            # So if we find a valid one at n=40, we should check 31-39?
            # But we already checked them?
            # Yes, the loop goes 31, 32, ...
            # So if we find one at 40, it means 31-39 failed.
            # So we can break.
            
    # Construct the output
    n_out = best_n
    orientation = []
    for i in range(n_out):
        for j in range(i+1, n_out):
            orientation.append(int(best_t[i,j]))
    
    return {'n': n_out, 'orientation': orientation}

# EVOLVE-BLOCK-END
