import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # Precompute neighbor lists for fast lookup
    # For 4-APs: for each element x, list of 4-AP indices containing x
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            four_ap_neighbors[x].append(idx)
            
    # For 3-APs: for each element x, list of 3-AP indices containing x
    three_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        for x in (a, b, c):
            three_ap_neighbors[x].append(idx)
    
    # Helper: check if a set S is 4-AP free
    def is_4ap_free(S):
        for idx, (a, b, c, d) in enumerate(four_ap_tuples):
            if a in S and b in S and c in S and d in S:
                return False
        return True
        
    # Helper: count 3-APs in a set S
    def count_3aps(S):
        count = 0
        for a, b, c in three_ap_tuples:
            if a in S and b in S and c in S:
                count += 1
        return count
        
    # Helper: check if adding x to S creates a 4-AP
    def creates_4ap(S, x):
        for idx in four_ap_neighbors[x]:
            a, b, c, d = four_ap_tuples[idx]
            if a in S and b in S and c in S and d in S:
                return True
        return False
        
    # Helper: check if removing x from S breaks any 4-AP? No, removing never creates a 4-AP.
    # But we need to ensure S \ {x} is still 4-AP free. Since S is 4-AP free, S\{x} is also 4-AP free.
    
    # Strategy: Multi-start Greedy Construction with Incremental Delta
    
    # Greedy Construction:
    # Start with empty set.
    # At each step, add the element x that maximizes the number of new 3-APs formed,
    # subject to not creating any 4-AP.
    # Tie-breaker: random or lowest index.
    
    def greedy_construct():
        S = set()
        current_score = 0
        
        # Precompute for each x, the list of 3-APs it participates in.
        # When adding x, the new 3-APs are those where the other two elements are already in S.
        
        while True:
            best_x = None
            best_gain = 0
            best_new_score = current_score
            
            # Try adding each candidate x not in S
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                break
                
            # For efficiency, we can precompute potential gains, but N=256 is small.
            # Let's iterate through candidates.
            for x in candidates:
                # Check if adding x creates a 4-AP
                if creates_4ap(S, x):
                    continue
                    
                # Calculate gain in 3-APs
                gain = 0
                for idx in three_ap_neighbors[x]:
                    a, b, c = three_ap_tuples[idx]
                    # The other two elements must be in S
                    others = [y for y in (a, b, c) if y != x]
                    if others[0] in S and others[1] in S:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_new_score = current_score + gain
                elif gain == best_gain and gain > 0:
                    # Tie-breaker: pick the one that allows more future growth? 
                    # For now, just pick the first one or random.
                    # Let's stick to the first one found with max gain.
                    pass
                    
            if best_x is None:
                # No more elements can be added without creating a 4-AP
                break
                
            S.add(best_x)
            current_score = best_new_score
            
        return S, current_score
    
    # Local Search with Swap:
    # Move: Swap x in S with y not in S.
    # Condition: S' = S \ {x} U {y} is 4-AP free.
    # Objective: Maximize 3-APs.
    
    def local_search(S, current_score):
        # Precompute 3-AP counts for incremental updates?
        # Delta_3ap = (3-APs gained by adding y) - (3-APs lost by removing x) + (interaction)
        # Actually, it's easier to just recalculate if we are doing many swaps, 
        # but we can be smarter.
        
        # Let's use a simple hill-climbing swap local search.
        # We will iterate through pairs (x in S, y not in S) and try the swap.
        # If the swap improves the score and maintains 4-AP freeness, accept it.
        
        improved = True
        while improved:
            improved = False
            S_list = list(S)
            not_in_S = [x for x in range(1, N + 1) if x not in S]
            
            # Shuffle for variety
            random.shuffle(S_list)
            random.shuffle(not_in_S)
            
            for x in S_list:
                if time.time() > deadline - 5:
                    return S, current_score
                for y in not_in_S:
                    if time.time() > deadline - 5:
                        return S, current_score
                    
                    # Check if swap is valid (no 4-AP)
                    # S' = S - {x} + {y}
                    # Since S is 4-AP free, S-{x} is 4-AP free.
                    # We only need to check if adding y to S-{x} creates a 4-AP.
                    temp_S = set(S)
                    temp_S.discard(x)
                    temp_S.add(y)
                    
                    if creates_4ap(temp_S, y):
                        continue
                        
                    # Calculate new score
                    # We can calculate the delta:
                    # Lost: 3-APs containing x and two other elements in S\{x}
                    # Gained: 3-APs containing y and two other elements in S\{x}
                    
                    lost = 0
                    for idx in three_ap_neighbors[x]:
                        a, b, c = three_ap_tuples[idx]
                        others = [z for z in (a, b, c) if z != x]
                        if others[0] in temp_S and others[1] in temp_S:
                            lost += 1
                            
                    gained = 0
                    for idx in three_ap_neighbors[y]:
                        a, b, c = three_ap_tuples[idx]
                        others = [z for z in (a, b, c) if z != y]
                        if others[0] in temp_S and others[1] in temp_S:
                            gained += 1
                            
                    delta = gained - lost
                    if delta > 0:
                        current_score += delta
                        S = temp_S
                        improved = True
                        # Restart the loop? Or continue? 
                        # Continuing is fine, but the lists S_list and not_in_S are stale.
                        # For simplicity and speed, break out and restart.
                        break
                if improved:
                    break
                    
        return S, current_score
    
    best_set = set()
    best_score = -1
    
    # Multi-start
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Add some randomness to the greedy construction
        # We can randomize the tie-breaking or the order of candidates.
        # Let's just run the greedy construction. It's deterministic.
        # To make it multi-start, we can randomize the initial choice or use different heuristics.
        # For now, let's just run it once and then do local search.
        # Actually, the greedy is deterministic. Let's add a random seed to the local search or 
        # perturb the greedy.
        
        # Let's try a slightly different greedy: 
        # At each step, pick the best x from the top 3 candidates?
        
        S, score = greedy_construct()
        
        # Local search
        S, score = local_search(S, score)
        
        if score > best_score:
            best_score = score
            best_set = S
            
        # If we have time, do another round of local search or a different start
        if time.time() < deadline - 20:
            # Try a random restart with a small random set?
            # Or just run local search again on the same set? It might not improve.
            # Let's try a different greedy: prioritize elements with high degree in 3-APs?
            pass
            
    # Final polish: one more local search pass
    if time.time() < deadline - 5:
        best_set, best_score = local_search(best_set, best_score)
        
    return sorted(best_set)
