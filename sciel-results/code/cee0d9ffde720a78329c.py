import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
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
    
    # Map element indices (1..N) to 0..N-1 for bit operations or list indexing
    # We will use a list of booleans for the assignment, or a set.
    # For WalkSAT, we need to evaluate clause satisfaction.
    
    # Clause representation for Max-SAT
    # Hard clauses: 4-APs. A 4-AP (a, b, c, d) is forbidden if all are True.
    # Clause: NOT (a AND b AND c AND d) is equivalent to (NOT a OR NOT b OR NOT c OR NOT d).
    # In Max-SAT terms, we want to minimize the number of violated hard clauses (should be 0)
    # and maximize the number of satisfied soft clauses.
    
    # Soft clauses: 3-APs. A 3-AP (a, b, c) is rewarded if all are True.
    # Clause: (a AND b AND c).
    # In Max-SAT, a clause is satisfied if at least one literal is true.
    # But here we want the clause to be satisfied only if ALL are true?
    # Standard Max-SAT clauses are ORs of literals.
    # To reward (a AND b AND c), we can't directly use a single clause.
    # However, we can use a weighted objective function directly.
    
    # Objective:
    # Score = (Number of satisfied 3-APs) - M * (Number of violated 4-APs)
    # where M is a large penalty.
    
    # We can use a local search that directly optimizes this objective.
    # State: A boolean array or set of selected elements.
    
    # Precompute for each element x:
    # - list of 4-AP indices containing x
    # - list of 3-AP indices containing x
    
    x_to_4aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            x_to_4aps[x].append(idx)
            
    x_to_3aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        for x in (a, b, c):
            x_to_3aps[x].append(idx)
            
    # WalkSAT-style Local Search
    # We maintain:
    # - assignment: set of True variables
    # - val_4ap[idx]: number of True literals in the 4-AP (violated if 4)
    # - val_3ap[idx]: number of True literals in the 3-AP (satisfied if 3)
    
    # Initialize assignment
    # A good initial assignment is important.
    # Let's try a random dense set, or a greedy set.
    # Greedy: Start with all True, remove elements to break 4-APs.
    
    def greedy_init():
        selected = set(range(1, N + 1))
        # Randomize order of 4-APs to break ties
        ap_indices = list(range(len(four_ap_tuples)))
        random.shuffle(ap_indices)
        
        for idx in ap_indices:
            a, b, c, d = four_ap_tuples[idx]
            if a in selected and b in selected and c in selected and d in selected:
                # Remove one randomly
                to_remove = random.choice([a, b, c, d])
                selected.remove(to_remove)
        return selected
    
    # Calculate current score
    def calc_score(selected):
        score_3 = 0
        for a, b, c in three_ap_tuples:
            if a in selected and b in selected and c in selected:
                score_3 += 1
        score_4 = 0
        for a, b, c, d in four_ap_tuples:
            if a in selected and b in selected and c in selected and d in selected:
                score_4 += 1
        return score_3, score_4
        
    # Let's use a more efficient state representation for local search.
    # assignment: list of 0/1 of length N+1 (index 1..N)
    # val_4: list of counts for each 4-AP
    # val_3: list of counts for each 3-AP
    
    # Initialize
    initial_set = greedy_init()
    assignment = [0] * (N + 1)
    for x in initial_set:
        assignment[x] = 1
        
    val_4 = [0] * len(four_ap_tuples)
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        val_4[idx] = assignment[a] + assignment[b] + assignment[c] + assignment[d]
        
    val_3 = [0] * len(three_ap_tuples)
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        val_3[idx] = assignment[a] + assignment[b] + assignment[c]
        
    # Current Objective
    # We want to maximize score_3 and minimize score_4.
    # Let's use a weighted sum: Total = sum(val_3 == 3) - 1000 * sum(val_4 == 4)
    # Actually, just track counts.
    
    curr_3_count = sum(1 for v in val_3 if v == 3)
    curr_4_count = sum(1 for v in val_4 if v == 4)
    
    def get_obj(c3, c4):
        return c3 - 1000 * c4
        
    best_obj = get_obj(curr_3_count, curr_4_count)
    best_assignment = assignment[:]
    
    # WalkSAT Parameters
    # Flip a variable (add/remove element).
    # When flipping x:
    # - For each 4-AP containing x: val_4 changes by +/- 1
    # - For each 3-AP containing x: val_3 changes by +/- 1
    
    # To make it "WalkSAT", we pick a random violated hard clause (if any) and flip a literal in it.
    # If no violated hard clauses, we pick a random clause (hard or soft?) and flip a literal to satisfy it?
    # Since we want to maximize soft clauses, we can also pick a random unsatisfied soft clause and flip a literal to satisfy it.
    
    # Hybrid approach:
    # 1. If there are violated 4-APs, pick one randomly.
    # 2. Pick a random element in that 4-AP.
    # 3. Flip it.
    # 4. If no violated 4-APs, pick a random unsatisfied 3-AP.
    # 5. Pick a random element in that 3-AP that is currently 0 (to make it 1).
    # 6. Flip it.
    # 7. Also allow random flips to escape local minima?
    
    # Let's implement a simpler local search:
    # Try flipping each element, calculate delta in objective.
    # If delta > 0, accept.
    # If stuck, do a random walk.
    
    # For speed, we only calculate delta for flips.
    
    def flip(x):
        nonlocal curr_3_count, curr_4_count
        # Update val_4
        for idx in x_to_4aps[x]:
            old_val = val_4[idx]
            new_val = old_val + (1 if assignment[x] == 0 else -1)
            val_4[idx] = new_val
            if old_val == 4: curr_4_count -= 1
            if new_val == 4: curr_4_count += 1
            
        # Update val_3
        for idx in x_to_3aps[x]:
            old_val = val_3[idx]
            new_val = old_val + (1 if assignment[x] == 0 else -1)
            val_3[idx] = new_val
            if old_val == 3: curr_3_count -= 1
            if new_val == 3: curr_3_count += 1
            
        assignment[x] = 1 - assignment[x]
        
    # Local Search Loop
    # We will use a "best-first" or "hill-climbing" approach with random restarts if stuck.
    # But N=256, so checking all 256 flips is fast enough (256 * avg_degree).
    # Avg degree of 3-APs is ~50. So 256 * 50 = 12800 ops per step.
    # We can do many steps.
    
    max_steps = 10000
    steps = 0
    no_improve_count = 0
    
    while steps < max_steps and time.time() < deadline:
        improved = False
        
        # Try all possible flips
        # To avoid bias, shuffle the order
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        best_delta = 0
        best_x = None
        best_new_c3 = curr_3_count
        best_new_c4 = curr_4_count
        
        for x in candidates:
            # Simulate flip
            delta_4 = 0
            delta_3 = 0
            
            # Calculate change in counts
            # This is expensive if done naively.
            # Let's just flip, calc, unflip?
            # Or calculate delta analytically.
            
            # Analytical Delta:
            # For 4-APs:
            # If assignment[x] == 1 (removing x):
            #   For each 4-AP containing x:
            #     if val_4[idx] == 4: it becomes 3. curr_4_count -= 1
            #     else: no change in count (unless it was 3 and becomes 2, etc, but we only care about 4s)
            #     Wait, if it was 4, it becomes 3. Count of 4s decreases by 1.
            #     If it was < 4, it becomes < 3. Count of 4s unchanged.
            # If assignment[x] == 0 (adding x):
            #   For each 4-AP containing x:
            #     if val_4[idx] == 3: it becomes 4. curr_4_count += 1
            #     else: no change.
            
            # For 3-APs:
            # If assignment[x] == 1 (removing x):
            #   For each 3-AP containing x:
            #     if val_3[idx] == 3: it becomes 2. curr_3_count -= 1
            #     else: no change.
            # If assignment[x] == 0 (adding x):
            #   For each 3-AP containing x:
            #     if val_3[idx] == 2: it becomes 3. curr_3_count += 1
            #     else: no change.
            
            if assignment[x] == 1:
                # Remove x
                for idx in x_to_4aps[x]:
                    if val_4[idx] == 4:
                        delta_4 -= 1
                for idx in x_to_3aps[x]:
                    if val_3[idx] == 3:
                        delta_3 -= 1
            else:
                # Add x
                for idx in x_to_4aps[x]:
                    if val_4[idx] == 3:
                        delta_4 += 1
                for idx in x_to_3aps[x]:
                    if val_3[idx] == 2:
                        delta_3 += 1
            
            # Objective change:
            # New Obj = (curr_3_count + delta_3) - 1000 * (curr_4_count + delta_4)
            # Delta Obj = delta_3 - 1000 * delta_4
            
            delta_obj = delta_3 - 1000 * delta_4
            
            if delta_obj > best_delta:
                best_delta = delta_obj
                best_x = x
                best_new_c3 = curr_3_count + delta_3
                best_new_c4 = curr_4_count + delta_4
                
        if best_x is not None and best_delta > 0:
            # Perform the flip
            flip(best_x)
            curr_3_count = best_new_c3
            curr_4_count = best_new_c4
            
            if curr_3_count - 1000 * curr_4_count > best_obj:
                best_obj = curr_3_count - 1000 * curr_4_count
                best_assignment = assignment[:]
                
            no_improve_count = 0
        else:
            # No improving move found.
            # Do a random walk to escape local minimum.
            # Pick a random x and flip it.
            if no_improve_count > 50:
                # Restart or aggressive random walk
                # Let's just pick a random element and flip
                x = random.choice(range(1, N + 1))
                flip(x)
                # Recalculate counts? No, flip updates them.
                # But we need to update curr_3_count and curr_4_count.
                # My flip function updates val_4 and val_3, but not the counts directly?
                # Ah, I defined flip to update counts. Let me check.
                # Yes, flip updates curr_3_count and curr_4_count.
                
                no_improve_count += 1
            else:
                # Stuck. Break and try a new restart?
                # Or just break.
                break
                
        steps += 1
        
    # Convert best_assignment to sorted list
    result = [i for i in range(1, N + 1) if best_assignment[i] == 1]
    
    return sorted(result)
