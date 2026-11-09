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
    
    # Precompute 4-AP neighbors: for each x, list of (idx, role) where role is 0,1,2,3
    # This helps in checking if adding x creates a 4-AP
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        four_ap_neighbors[a].append((idx, 0))
        four_ap_neighbors[b].append((idx, 1))
        four_ap_neighbors[c].append((idx, 2))
        four_ap_neighbors[d].append((idx, 3))
        
    # Precompute 3-AP partners for delta calculation
    # For a 3-AP (a, b, c), the pairs are (a,b), (a,c), (b,c)
    # If we add x, we gain a 3-AP if (x, y, z) is a 3-AP and y,z in S.
    # We can precompute for each x, the list of (y, z) such that (x, y, z) is a 3-AP.
    # Roles: x can be start, mid, end.
    # If x is start: (x, x+d, x+2d) -> need (x+d, x+2d) in S
    # If x is mid: (x-d, x, x+d) -> need (x-d, x+d) in S
    # If x is end: (x-2d, x-d, x) -> need (x-2d, x-d) in S
    
    # Let's build a map: x -> list of (y, z) where {x,y,z} form a 3-AP and y!=z
    # Actually, simpler: for each 3-AP (a,b,c):
    #   a's partners: (b,c)
    #   b's partners: (a,c)
    #   c's partners: (a,b)
    
    three_ap_partners = {i: [] for i in range(1, N + 1)}
    for a, b, c in three_ap_tuples:
        three_ap_partners[a].append((b, c))
        three_ap_partners[b].append((a, c))
        three_ap_partners[c].append((a, b))
        
    # Helper: count 3-APs in a set (for initial verification)
    def count_3aps(S):
        count = 0
        for a, b, c in three_ap_tuples:
            if a in S and b in S and c in S:
                count += 1
        return count
        
    # Helper: check if adding x to S creates a 4-AP
    def creates_4ap(S, x):
        for idx, role in four_ap_neighbors[x]:
            a, b, c, d = four_ap_tuples[idx]
            # Check if the other 3 are in S
            # Optimization: check only the ones that are in S
            # Since S is a set, 'in' is O(1)
            # We need to be careful not to check x itself (but x is not in S yet)
            # So we just check the other 3 elements of the tuple
            # The tuple is (a,b,c,d). x is one of them.
            # We need to check if the other 3 are in S.
            
            # Instead of extracting, we can just check all 4 and ensure x is the only one "new"
            # But x is not in S, so if a in S and b in S and c in S and d in S is false, it's fine.
            # Wait, if x is in the tuple, and we are adding x, we check if the OTHERS are in S.
            # If we just check `a in S and b in S and c in S and d in S`, it will be False because x is not in S.
            # So we need to check the specific others.
            
            # Let's identify the others
            others = []
            if role == 0: others = [b, c, d]
            elif role == 1: others = [a, c, d]
            elif role == 2: others = [a, b, d]
            elif role == 3: others = [a, b, c]
            
            if all(o in S for o in others):
                return True
        return False

    # Initial Seed: Greedy construction
    # Start with empty set, add elements if they don't create a 4-AP.
    # To maximize 3-APs, we might want a denser start.
    # The standard greedy (add if valid) usually yields a decent set.
    # Let's try to start with a random permutation and add greedily.
    
    selected = set()
    # Try multiple random permutations for the seed and keep the best
    best_seed = set()
    best_seed_score = 0
    
    for _ in range(10):
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        S = set()
        for x in perm:
            if not creates_4ap(S, x):
                S.add(x)
        score = count_3aps(S)
        if score > best_seed_score:
            best_seed_score = score
            best_seed = S
            
    selected = best_seed
    current_score = best_seed_score
    best_set = set(selected)
    best_score = current_score
    
    # Local Search: Greedy Hill Climbing
    # Moves:
    # 1. Swap: Remove x in S, Add y not in S.
    #    Constraint: S' = S - {x} + {y} must be 4-AP free.
    #    Note: Removing x cannot create a 4-AP. Adding y might.
    #    So we only need to check if adding y to (S - {x}) creates a 4-AP.
    #    Delta 3-APs: 
    #       Lost: 3-APs containing x that are broken.
    #       Gained: 3-APs containing y that are formed.
    #       Note: 3-APs containing both x and y? 
    #       If a 3-AP has x and y, and we swap x for y... wait, y is not in S.
    #       So no 3-AP in S contains y.
    #       When we remove x, we lose all 3-APs in S that contain x.
    #       When we add y, we gain all 3-APs in (S - {x}) that contain y.
    #       Are there 3-APs containing both x and y? 
    #       If (x, y, z) is a 3-AP, y is not in S, so it wasn't counted.
    #       If we add y, we count it if z is in S.
    #       So Delta = (3-APs formed with y in S\{x}) - (3-APs broken by removing x from S).
        
    # Precompute for each x:
    #   lost_3aps[x] = list of (y, z) such that (x, y, z) is a 3-AP and y,z in S.
    #   This is dynamic. We need to calculate it on the fly or maintain counts.
    
    # To speed up, we can precompute the "potential" partners.
    # For a given S, the number of 3-APs lost by removing x is the number of pairs (y,z) in three_ap_partners[x] such that y in S and z in S.
    # Similarly, gained by adding y is the number of pairs (u,v) in three_ap_partners[y] such that u in (S\{x}) and v in (S\{x}).
    
    # Since N=256, the number of 3-APs is ~16k.
    # The number of partners for any x is small (O(N)).
    # Calculating delta for a swap is O(degree(x) + degree(y)).
    # This is very fast.
    
    # We will iterate through all possible swaps in a random order and apply the first improving one.
    # If no improving swap is found, we can try random restarts or accept neutral moves (plateau).
    # Given the time budget, we should focus on finding improving moves.
    
    # To avoid getting stuck, we can randomize the order of swaps.
    # We can also try "Add" moves if the set is not maximal (but greedy seed is usually maximal or close).
    # Let's stick to Swaps for now, as they preserve cardinality (which is good for 3-APs).
    
    # Optimization: 
    # Instead of checking all x in S and y not in S (which is ~100*150 = 15000 checks),
    # we can be smarter? 
    # 15000 checks * O(10) operations = 150k operations per "pass".
    # We can do many passes in 50s.
    
    # Let's implement the swap logic.
    
    no_improvement_count = 0
    max_no_improvement = 5 # After 5 full passes without improvement, break
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        improved = False
        no_improvement_count += 1
        
        # Generate list of elements in S and not in S
        S_list = list(selected)
        not_S_list = [x for x in range(1, N + 1) if x not in selected]
        
        # Shuffle to avoid bias
        random.shuffle(S_list)
        random.shuffle(not_S_list)
        
        # To speed up, we can limit the number of swaps checked per pass if time is tight?
        # No, let's try to do a full pass.
        
        # We need to be careful: if we apply a swap, S changes.
        # So we should check swaps based on the current S.
        # If we apply a swap, we should restart the check or continue?
        # Standard first-improvement: apply the first valid improving swap found, then restart the scan.
        # This is safer but slower.
        # Alternatively, apply it and continue, but the "lost/gained" calcs for subsequent swaps
        # based on the old S would be wrong.
        # So: Find one swap, apply it, restart the loop.
        
        found_swap = False
        best_delta = 0
        best_x = -1
        best_y = -1
        
        # To find the BEST swap (steepest ascent) might be better than first improvement?
        # Steepest ascent is more expensive but makes bigger jumps.
        # Let's try First Improvement for speed, but maybe look at a few candidates.
        
        # Actually, let's just iterate and take the first valid improving swap.
        for x in S_list:
            # Calculate lost 3-APs if we remove x
            # lost = count of (u, v) in three_ap_partners[x] where u in S and v in S
            # Note: u and v are distinct from x.
            lost = 0
            for u, v in three_ap_partners[x]:
                if u in selected and v in selected:
                    lost += 1
            
            # If lost is 0, removing x doesn't hurt 3-APs.
            # Now we want to find y not in S such that:
            # 1. Adding y to S\{x} doesn't create a 4-AP.
            # 2. Gained 3-APs > lost.
            
            # To speed up, we can check y's.
            # But we need to check 4-AP constraint for each y.
            # creates_4ap(S\{x}, y)
            
            # Let's iterate y
            for y in not_S_list:
                # Check 4-AP
                # We need to check if adding y to (selected - {x}) creates a 4-AP.
                # This is equivalent to: are there any 4-APs containing y where the other 3 are in (selected - {x})?
                # Which means: other 3 are in selected, AND none of them is x.
                
                # Let's precompute the "blockers" for y.
                # A 4-AP (a,b,c,d) containing y blocks y if the other 3 are in S.
                # If one of the other 3 is x, then removing x unblocks it.
                
                # So, for a given x and y:
                # y is valid if for every 4-AP containing y, the other 3 are NOT all in S\{x}.
                # i.e., for every 4-AP containing y, at least one of the other 3 is NOT in S\{x}.
                # "Not in S\{x}" means "Not in S" OR "Is x".
                # Since the other 3 are distinct from y, and we are adding y, the others must be in S to form the AP.
                # So the condition for y being INVALID is:
                # There exists a 4-AP containing y such that the other 3 are ALL in S, AND NONE of them is x.
                
                # So y is VALID if:
                # For all 4-APs containing y, either:
                # a) The other 3 are not all in S.
                # b) The other 3 are all in S, but x is one of them.
                
                # Let's check this.
                valid = True
                for idx, role in four_ap_neighbors[y]:
                    a, b, c, d = four_ap_tuples[idx]
                    others = []
                    if role == 0: others = [b, c, d]
                    elif role == 1: others = [a, c, d]
                    elif role == 2: others = [a, b, d]
                    elif role == 3: others = [a, b, c]
                    
                    # Check if all others are in S
                    all_in_S = True
                    x_in_others = False
                    for o in others:
                        if o in selected:
                            if o == x:
                                x_in_others = True
                        else:
                            all_in_S = False
                            break
                    
                    if all_in_S:
                        # The other 3 are in S.
                        # If x is NOT in others, then removing x doesn't help, so y still creates a 4-AP.
                        if not x_in_others:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Calculate gained 3-APs
                # Gained = count of (u, v) in three_ap_partners[y] where u in S\{x} and v in S\{x}
                gained = 0
                for u, v in three_ap_partners[y]:
                    # u, v must be in S and not x
                    if u in selected and v in selected:
                        if u != x and v != x:
                            gained += 1
                
                delta = gained - lost
                if delta > 0:
                    # Apply swap
                    selected.remove(x)
                    selected.add(y)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(selected)
                    
                    improved = True
                    found_swap = True
                    break # Restart the scan for x
            
            if found_swap:
                break
        
        if not improved:
            # No improving swap found in this pass
            # We can try a random restart or accept a neutral move?
            # Let's try a few random neutral moves to escape local optima.
            # Or just break if we've done enough passes.
            
            # Let's try one random swap (even if neutral or slightly worse) to escape?
            # Risky. Let's stick to strict improvement for now.
            # If we can't improve, we are likely at a local optimum.
            # We can try to change the seed?
            # Given the time, let's just break.
            pass

    return sorted(best_set)
