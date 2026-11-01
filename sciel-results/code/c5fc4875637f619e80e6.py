import time
import itertools
import numpy as np
import random

# EVOLVE-BLOCK-START

def _precompute_graph(q, n, radius, deadline):
    """
    Precompute the adjacency structure for the covering problem.
    For radius 1, a vertex x covers itself and its neighbors at dist 1.
    We represent the 'ball' of each vertex as a bitset (or list) for fast coverage checking.
    """
    total = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # For q=3, n=8, total=6561.
    # We need to know, for each vertex x, which vertices it covers.
    # cover_mask[x] is a set/list of vertices covered by x.
    # Also, for local search, we need to quickly check if a vertex y is covered by the current set S.
    # This is equivalent to checking if any x in S is within distance <= radius of y.
    
    # Let's precompute neighbors for each vertex.
    # neighbors[x] = list of y such that d(x,y) <= radius
    # Since radius=1, neighbors[x] includes x and all words at dist 1.
    
    neighbors = [None] * total
    for x in range(total):
        ball = [x]
        for i in range(n):
            old_digit = (x // powers[i]) % q
            for new_digit in range(q):
                if new_digit != old_digit:
                    y = x + (new_digit - old_digit) * powers[i]
                    ball.append(y)
        neighbors[x] = ball
        
        if time.time() > deadline - 10:
            # Fallback if too slow (unlikely for this size)
            return None

    return neighbors

def _int_to_digits(x, q, n, powers):
    digits = []
    for i in range(n):
        digits.append((x // powers[i]) % q)
    return digits

def _greedy_init(neighbors, total, deadline):
    """
    Greedy initialization: Repeatedly pick the vertex that covers the most currently uncovered vertices.
    To break ties, we can prefer vertices that are 'central' or just random.
    """
    covered = np.zeros(total, dtype=bool)
    # uncovered_count[x] = number of uncovered vertices in neighbors[x]
    # Initially, all are uncovered, so uncovered_count[x] = len(neighbors[x])
    # Note: All balls have the same size for radius 1 in a regular graph?
    # Yes, for Hamming graph, degree is n*(q-1), so ball size is 1 + n*(q-1).
    # So initial counts are all equal. We need a tie-breaker.
    # We can just pick randomly or by index. But to get a better start, 
    # we can do a few passes or just pick the first max.
    
    # Let's use a simple greedy with a twist: 
    # Instead of just max uncovered, we can look at the "efficiency" but here all are same size.
    # So the first choice is arbitrary. Let's pick a random start or 0.
    # Actually, the structure of the graph matters.
    
    # Let's maintain a count of how many selected vertices cover each vertex? No, that's for removal.
    # For selection, we need: for each candidate x, how many *new* vertices does it cover?
    
    # Efficient update:
    # When we pick x, we mark all y in neighbors[x] as covered.
    # For every other candidate z, if it shares any newly covered vertex with x, its "new coverage" count decreases.
    # This is expensive if done naively.
    
    # Alternative: Just use the standard greedy:
    # 1. Pick x with max new coverage.
    # 2. Mark its neighbors covered.
    # 3. For all z, recompute new coverage? No, just decrement if overlap.
    
    # Let's precompute an inverse index: inv[y] = list of x such that y in neighbors[x].
    # This is the same as neighbors since the graph is undirected and symmetric for radius 1.
    # Wait, if y is in neighbors[x], then x is in neighbors[y]. Yes.
    
    # So, when we pick x:
    # For each y in neighbors[x]:
    #   If y was not covered:
    #     covered[y] = True
    #     For each z in neighbors[y] (which are the candidates that cover y):
    #       If z is not selected yet:
    #         new_coverage_count[z] -= 1
    #
    # This is O(|ball| * |ball|) per selection. |ball| = 17. 17*17=289.
    # 400 selections * 289 ops = ~115k ops. Very fast.
    
    new_cov = np.array([len(neighbors[i]) for i in range(total)], dtype=np.int32)
    selected = []
    
    # To avoid O(N) argmax every time, we could use a heap, but N=6561 is small.
    # However, updating the heap is hard. N is small enough for O(N) scan.
    
    while True:
        if time.time() > deadline - 5:
            break
            
        # Find max new_cov
        # If max is 0, we are done? Not necessarily, if some vertices are uncovered but no candidate covers them?
        # No, if a vertex is uncovered, any of its neighbors is a candidate with new_cov >= 1 (itself).
        # So if max new_cov is 0, all are covered.
        
        max_val = np.max(new_cov)
        if max_val == 0:
            break
            
        # Find indices with max_val
        # To add some diversity/optimization, we can pick randomly among the maxes?
        # Or just the first one.
        candidates = np.where(new_cov == max_val)[0]
        x = candidates[0] # Or random choice
        
        selected.append(x)
        
        # Update coverage
        for y in neighbors[x]:
            if not covered[y]:
                covered[y] = True
                # Decrement new_cov for all z that cover y
                # z in neighbors[y]
                for z in neighbors[y]:
                    # z might be x (already selected, we don't care about its count for future picks, 
                    # but we can set it to 0 to ignore)
                    # Actually, we only care about unselected candidates.
                    # If z is in selected, we don't pick it again.
                    # We can just decrement. If it becomes negative, it's fine, we only pick max > 0.
                    # But we should not decrement if z is already selected? 
                    # It doesn't matter, we won't pick it again.
                    new_cov[z] -= 1
                    # Ensure it doesn't go below 0? 
                    # new_cov[z] counts uncovered neighbors. It can't go below 0 if logic is correct.
                    # But if z=x, and we just picked x, we are decrementing x's count.
                    # We should probably set new_cov[selected] = 0 to be safe.
                    
        new_cov[x] = 0 # Ensure we don't pick x again
        
        if np.all(covered):
            break
            
    return selected

def _is_covered(selected, neighbors, total):
    """Check if all vertices are covered by selected."""
    covered = np.zeros(total, dtype=bool)
    for x in selected:
        covered[neighbors[x]] = True
    return np.all(covered)

def _local_search(selected, neighbors, total, deadline):
    """
    Iterated Local Search to minimize |selected| while maintaining coverage.
    1. Try to remove any vertex (if coverage remains).
    2. If stuck, try swaps (remove one, add one) or perturbations.
    """
    selected_set = set(selected)
    # coverage_count[y] = number of selected vertices that cover y
    coverage_count = np.zeros(total, dtype=np.int32)
    for x in selected:
        coverage_count[neighbors[x]] += 1
    
    # Check initial validity
    if np.any(coverage_count == 0):
        # Invalid start, shouldn't happen with greedy
        pass
    
    best_size = len(selected)
    best_set = set(selected)
    
    while time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Removal
        # Iterate through selected vertices. If removing x leaves all coverage_count[y] >= 1, remove it.
        # To be efficient, we can check if min(coverage_count[neighbors[x]]) > 1.
        # If so, x is redundant.
        
        # We need to be careful with order. Removing one might make another redundant.
        # So we loop until no more removals.
        while True:
            removed_any = False
            # Shuffle to avoid bias?
            current_list = list(selected_set)
            random.shuffle(current_list)
            
            for x in current_list:
                if time.time() > deadline - 5:
                    return list(best_set)
                
                # Check if x is redundant
                # x is redundant if for all y in neighbors[x], coverage_count[y] > 1
                # i.e., min(coverage_count[neighbors[x]]) >= 2
                
                # Optimize: precompute min?
                # neighbors[x] is a list of ints.
                min_cov = np.min(coverage_count[neighbors[x]])
                if min_cov >= 2:
                    # Remove x
                    selected_set.remove(x)
                    coverage_count[neighbors[x]] -= 1
                    removed_any = True
                    improved = True
                    best_size = len(selected_set)
                    best_set = set(selected_set)
            
            if not removed_any:
                break
        
        # Phase 2: Swaps
        # If no removals, try to swap.
        # Find a vertex x in selected that is "critical" (has some y with coverage 1).
        # Find a vertex z not in selected that covers many of those critical y's.
        
        # This is more complex.
        # Heuristic: 
        # 1. Identify "weak" points: y where coverage_count[y] == 1.
        # 2. For each weak point y, the only selected vertex covering it is x_y.
        # 3. We want to replace x_y with some z that covers y and potentially other weak points.
        
        weak_points = np.where(coverage_count == 1)[0]
        
        if len(weak_points) == 0:
            # No weak points, but no removals? 
            # This implies every vertex has a "private" neighbor?
            # If coverage is all >= 2, removal would have worked.
            # So if no removals, there must be weak points.
            pass
            
        # Group weak points by their unique coverer
        # unique_coverer[y] = the only x in selected s.t. y in neighbors[x]
        # We can compute this by iterating selected and checking neighbors.
        
        # Simpler Swap Strategy:
        # Randomly pick a selected vertex x.
        # Try to find a z not in selected such that:
        # 1. z covers all weak points that x uniquely covers.
        # 2. Ideally, z covers more than x.
        
        # Let's do a simpler perturbation if stuck:
        # Remove a random x, add a random z that covers the most newly uncovered vertices.
        
        if not improved:
            # Perturb
            # Pick a random x from selected
            if not selected_set:
                break
            x = random.choice(list(selected_set))
            
            # Remove x
            selected_set.remove(x)
            coverage_count[neighbors[x]] -= 1
            
            # Find uncovered vertices (coverage_count == 0)
            uncovered = np.where(coverage_count == 0)[0]
            
            if len(uncovered) == 0:
                # We removed x but nothing became uncovered?
                # Then x was redundant, which should have been caught in Phase 1.
                # Unless we missed it due to order?
                # Anyway, just keep the smaller set.
                best_size = len(selected_set)
                best_set = set(selected_set)
                continue
                
            # Find best z to add back
            # Score(z) = number of uncovered vertices covered by z
            # We can estimate this by checking neighbors of z against uncovered set.
            
            # To speed up, we can just pick a random z from the neighbors of the first uncovered vertex?
            # Or sample a few z's.
            
            # Let's try to find a z that covers the most uncovered.
            # Since uncovered might be large, we can't check all z in O(N).
            # But we can check z's that are likely to cover uncovered.
            # Uncovered vertices are neighbors of x (mostly).
            # So good candidates for z are neighbors of uncovered vertices.
            
            candidate_zs = set()
            for y in uncovered:
                # Add neighbors of y to candidates
                candidate_zs.update(neighbors[y])
            
            # Filter out selected
            candidate_zs -= selected_set
            
            if not candidate_zs:
                # Add x back
                selected_set.add(x)
                coverage_count[neighbors[x]] += 1
                continue
                
            # Score candidates
            # To avoid O(|candidates| * |uncovered|), we can just pick a few random ones?
            # Or iterate and calculate score.
            # |uncovered| is small (<= 17). |candidates| might be large.
            
            best_z = None
            best_score = -1
            
            # Sample if too many candidates
            cand_list = list(candidate_zs)
            if len(cand_list) > 100:
                cand_list = random.sample(cand_list, 100)
            
            for z in cand_list:
                # Count how many uncovered y are in neighbors[z]
                # neighbors[z] is a list.
                # We can use numpy intersection?
                # Or just loop uncovered (small).
                score = 0
                for y in uncovered:
                    if y in neighbors[z]: # This is O(17) if neighbors[z] is a set? No, list.
                        # Better: precompute neighbors as sets? 
                        # Or use a boolean mask for uncovered?
                        pass
                
                # Let's use a boolean mask for uncovered for speed
                # Actually, let's just do a simple check.
                # Since neighbors are small lists, we can make a set of neighbors[z] once?
                # No, let's just count.
                
                # Optimization: 
                # We want to maximize coverage of 'uncovered'.
                # Let's convert uncovered to a set for O(1) lookup?
                # No, 'uncovered' is a numpy array.
                
                # Let's just compute score for top few candidates.
                # For now, simple loop.
                # score = sum(1 for y in uncovered if y in neighbors[z])
                # This is slow if done for many z.
                
                # Let's just pick the first z that covers all uncovered?
                # Or the one with max overlap.
                
                # Let's try a different approach:
                # Add the z that covers the most uncovered.
                # We can approximate by picking z from the intersection of neighbors of uncovered?
                
                # Let's just pick a random z from candidates and see if it fixes it.
                # If not, we might end up with a larger set.
                # This is a "random restart" style move.
                
                # Let's calculate score properly for the sampled candidates.
                # To speed up, we can check if z covers the *first* uncovered vertex.
                # If not, skip.
                
                first_unc = uncovered[0]
                if first_unc not in neighbors[z]:
                    continue
                    
                # Now check full score
                score = 0
                for y in uncovered:
                    if y in neighbors[z]:
                        score += 1
                        if score == len(uncovered):
                            break
                            
                if score > best_score:
                    best_score = score
                    best_z = z
                    
            if best_z is not None:
                selected_set.add(best_z)
                coverage_count[neighbors[best_z]] += 1
                # Update best if size decreased? 
                # Size is same as before removal.
                # But we might have improved structure for future removals.
                # So we continue.
            else:
                # Revert
                selected_set.add(x)
                coverage_count[neighbors[x]] += 1

    return list(best_set)

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    total = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute neighbors
    neighbors = _precompute_graph(q, n, radius, deadline)
    if neighbors is None:
        # Fallback
        return []
        
    # Greedy Init
    initial_selected = _greedy_init(neighbors, total, deadline)
    
    # Local Search
    final_selected = _local_search(initial_selected, neighbors, total, deadline)
    
    # Convert to digits
    codewords = []
    for idx in final_selected:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

# EVOLVE-BLOCK-END
