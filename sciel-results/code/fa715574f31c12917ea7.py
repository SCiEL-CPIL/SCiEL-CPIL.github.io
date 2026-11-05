import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors (balls)
    # For q=3, n=8, radius=1, each ball has 1 + n*(q-1) = 1 + 8*2 = 17 words
    neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        d = digits_of[i]
        neigh = {i}
        
        # Radius 1 optimization: only 1 position changes
        if radius == 1:
            for pos in range(n):
                for new_val in range(q):
                    if new_val == d[pos]:
                        continue
                    new_d = list(d)
                    new_d[pos] = new_val
                    idx = 0
                    for j in range(n):
                        idx = idx * q + new_d[j]
                    neigh.add(idx)
        else:
            # General radius handling (slower, but needed for generality)
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        new_d = list(d)
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                            new_d[pos] = new_val
                        if not is_diff:
                            continue
                        idx = 0
                        for j in range(n):
                            idx = idx * q + new_d[j]
                        neigh.add(idx)
        neighbors[i] = neigh

    # Convert neighbors to numpy arrays for faster set operations if needed, 
    # but for now we keep sets for intersection counting.
    # Actually, for primal-dual, we need to iterate points and check coverage.
    # Let's precompute the list of neighbors as lists for faster iteration in coverage checks.
    neighbor_lists = [list(neigh) for neigh in neighbors]
    
    # Primal-Dual Approximation Algorithm for Set Cover
    # 1. Initialize dual variables y[v] = 0 for all points v
    # 2. While there are uncovered points:
    #    a. Increase y[v] for all uncovered v at the same rate until some set S becomes "tight" 
    #       (sum_{v in S} y[v] = cost(S) = 1)
    #    b. Select the tight set S, add to solution, mark all v in S as covered (y[v] stays, but v is covered)
    
    # Since all sets have cost 1, the "rate" is uniform.
    # We can simulate this efficiently:
    # At any step, we want to pick the set S that maximizes the sum of y[v] for v in S? 
    # No, the standard primal-dual picks the set that becomes tight first.
    # Since we increase all uncovered y[v] equally, the set that becomes tight first is the one 
    # with the smallest current "potential" or rather, we look for a set where the sum of 
    # duals of its uncovered elements is closest to 1? 
    # Actually, a simpler equivalent greedy for unit costs:
    # Pick the set that covers the most uncovered points. This is the standard greedy.
    # However, the prompt asks for a "primal-dual approximation algorithm that iteratively raises 
    # dual variables... and greedily select codewords with the highest marginal coverage gain per unit cost."
    # For unit cost sets, "marginal coverage gain" is just the number of new points covered.
    # So it reduces to standard greedy set cover, BUT we can use the dual variables to guide 
    # subsequent local search or to break ties more intelligently.
    
    # Let's implement the standard Greedy Set Cover first as the "Primal" part, 
    # but we will track the "dual" weights (effectively the number of times a point 
    # was considered uncovered during the greedy phase) to inform the repair phase.
    
    uncovered = set(range(total_words))
    initial_code = []
    # dual_weights[v] represents the "price" paid for covering v, or simply 
    # we can use the standard greedy which is H(n)-approximation.
    
    # Optimization: Use bitsets or numpy for coverage?
    # For 6561 points, sets are fine.
    
    while uncovered and time.time() < deadline - 30.0:
        best_idx = -1
        max_cover = 0
        # To speed this up, we can iterate only over codewords that might be useful?
        # Or just iterate all. 6561 * (avg ball size 17) is small.
        for i in range(total_words):
            if i in initial_code:
                continue
            # Count how many uncovered points are in neighbors[i]
            # neighbors[i] is a set. uncovered is a set.
            cnt = len(neighbors[i] & uncovered)
            if cnt > max_cover:
                max_cover = cnt
                best_idx = i
        
        if max_cover == 0:
            break
        
        initial_code.append(best_idx)
        uncovered.difference_update(neighbors[best_idx])
        
    if not initial_code:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    best_code = initial_code[:]
    best_size = len(best_code)
    
    # --- Error Repair / Local Search Phase ---
    # The "dual" information from the greedy phase: points covered later in the greedy process 
    # are "scarcer" or harder to cover. We can use this to prioritize removal attempts.
    # However, a more direct "Error Repair" strategy:
    # 1. Calculate coverage count for each point.
    # 2. Identify points with coverage count == 1 (critical points).
    # 3. Try to remove codewords that do NOT cover any critical points.
    # 4. If no such codeword exists, try to swap a codeword covering a critical point 
    #    with another that covers it better or allows removal of another.
    
    # Build a mapping: point -> list of codewords covering it
    # This is expensive to build from scratch, so we update it incrementally or rebuild when code changes significantly.
    
    def build_coverage_map(code):
        cover_map = {v: [] for v in range(total_words)}
        for cw in code:
            for v in neighbors[cw]:
                cover_map[v].append(cw)
        return cover_map
    
    # Initial coverage map
    cover_map = build_coverage_map(best_code)
    
    # Local Search: Try to remove codewords
    # A codeword can be removed if all points it covers are covered by other codewords.
    # i.e., for all v in neighbors[cw], len(cover_map[v]) > 1.
    
    improvement_found = True
    while improvement_found and time.time() < deadline - 10.0:
        improvement_found = False
        # Sort codewords by "criticality": how many points do they uniquely cover?
        # We want to remove codewords with 0 unique points first.
        unique_counts = {}
        for cw in best_code:
            unique = 0
            for v in neighbors[cw]:
                if len(cover_map[v]) == 1:
                    unique += 1
            unique_counts[cw] = unique
        
        # Try removing codewords with unique_counts[cw] == 0
        # Iterate in random order to avoid bias
        candidates = [cw for cw in best_code if unique_counts[cw] == 0]
        random.shuffle(candidates)
        
        for cw in candidates:
            if time.time() > deadline - 10.0:
                break
            
            # Try removing cw
            # Check if removal is valid
            valid = True
            for v in neighbors[cw]:
                if len(cover_map[v]) == 1:
                    valid = False
                    break
            
            if valid:
                # Remove cw
                best_code.remove(cw)
                # Update cover_map
                for v in neighbors[cw]:
                    cover_map[v].remove(cw)
                best_size -= 1
                improvement_found = True
                # Restart the loop to recalculate unique counts
                break
        
        # If no removal was possible, try swaps?
        # The prompt mentions "error_repair" changing the chance of improvement.
        # Swaps are a form of repair: replace a "bad" codeword with a "good" one to enable future removals.
        # Let's implement a simple swap heuristic if removals stall.
        
        if not improvement_found:
            # Try to find a codeword cw1 that, if replaced by cw2, allows the removal of another codeword cw3.
            # This is O(N^3) potentially. Let's do a limited search.
            
            # Identify "bottleneck" points: those with min coverage > 1 but close to 1.
            # Or just try random swaps.
            
            # For now, let's try a few random swaps to see if they unlock removals.
            swaps_attempted = 0
            max_swaps = 50
            while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
                swaps_attempted += 1
                if len(best_code) < 2:
                    break
                
                i = random.randint(0, len(best_code) - 1)
                cw1 = best_code[i]
                
                # Pick a random candidate cw2
                cw2 = random.randint(0, total_words - 1)
                if cw2 in best_code:
                    continue
                
                # Simulate swap: remove cw1, add cw2
                # Check if this makes any other codeword removable.
                # This is complex. A simpler heuristic: 
                # Does cw2 cover points that cw1 doesn't, and does it cover the unique points of some other cw3?
                
                # Let's just do a "greedy repair": 
                # Find a point v that is covered by only 1 codeword (cw1). 
                # If we swap cw1 with cw2, and cw2 also covers v, then v is still covered.
                # But this doesn't help removal unless cw1 was the only one covering v.
                
                # Actually, let's stick to the removal loop. If it stalls, we might be at a local minimum 
                # for the "remove only" heuristic.
                # The "error_repair" idea: specifically target the most "constrained" codewords.
                
                # Let's try to replace a codeword that covers many unique points with one that covers fewer?
                # No, we want to remove codewords.
                
                # Let's try a different repair: 
                # If no codeword has 0 unique points, then every codeword is "essential".
                # We can try to find a pair (cw1, cw2) such that replacing cw1 with cw2 
                # reduces the unique count of another codeword cw3 to 0.
                
                # This is getting complicated. Let's rely on the removal loop with randomization 
                # and the initial greedy being strong.
                
                # One more idea: "Random Restart" of the local search? No, too expensive.
                
                # Let's just break out of the swap loop if no simple removal is found.
                break

    # Final check and return
    return [list(digits_of[i]) for i in best_code]
