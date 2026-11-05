import random

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    # Run multiple trials to escape local minima or find better random tie-breaks
    for trial in range(50):
        S = set()
        sums = set()
        
        while True:
            # Identify U: elements that can be added
            U = []
            for x in range(1, N + 1):
                if x in S:
                    continue
                # Quick check: 2x
                if 2 * x in sums:
                    continue
                # Check x + a for a in S
                conflict = False
                for a in S:
                    if x + a in sums:
                        conflict = True
                        break
                if not conflict:
                    U.append(x)
            
            if not U:
                break
            
            # Find y in U that blocks the most others in U
            best_y = None
            max_blocked = -1
            candidates = []
            
            # Optimization: Pre-compute S list for faster iteration
            S_list = list(S)
            
            for y in U:
                # Simulate sums added by y
                new_sums_y = {2 * y}
                for a in S_list:
                    new_sums_y.add(y + a)
                
                # Count how many x in U (x != y) are blocked by adding these sums
                blocked_count = 0
                for x in U:
                    if x == y:
                        continue
                    # Check if x conflicts with existing sums OR new_sums_y
                    # 1. Check against existing sums (should be false since x in U)
                    # 2. Check against new_sums_y
                    # x is blocked if {2x} U {x+a for a in S U {y}} intersects sums U new_sums_y
                    # Since x in U, {2x} U {x+a for a in S} does NOT intersect sums.
                    # So we only need to check if the NEW sums for x intersect (sums U new_sums_y)
                    # The new sums for x are {2x} U {x+a for a in S} U {x+y}
                    # We know {2x} U {x+a for a in S} does not intersect sums.
                    # So we just need to check if any of these intersect new_sums_y?
                    # Wait. If x is added, its sums are $S_x = \{2x\} \cup \{x+a \mid a \in S\} \cup \{x+y\}$.
                    # The total sum set would be $\Sigma \cup \text{new\_sums\_y} \cup S_x$.
                    # Collision if $S_x \cap (\Sigma \cup \text{new\_sums\_y}) \neq \emptyset$.
                    # We know $S_x \setminus \{x+y\}$ does not intersect $\Sigma$.
                    # So we check:
                    # 1. Does $\{x+y\}$ intersect $\Sigma$? (If yes, blocked)
                    # 2. Does $\{x+y\}$ intersect $\text{new\_sums\_y}$? (If yes, blocked)
                    # 3. Does $S_x \setminus \{x+y\}$ intersect $\text{new\_sums\_y}$? (If yes, blocked)
                    
                    blocked = False
                    if (x + y) in sums:
                        blocked = True
                    elif (x + y) in new_sums_y:
                        blocked = True
                    else:
                        # Check if any other sum of x is in new_sums_y
                        # new_sums_y = {2y} U {y+a for a in S}
                        # S_x\{x+y} = {2x} U {x+a for a in S}
                        if 2 * x in new_sums_y:
                            blocked = True
                        else:
                            for a in S_list:
                                if (x + a) in new_sums_y:
                                    blocked = True
                                    break
                    if blocked:
                        blocked_count += 1
                
                if blocked_count > max_blocked:
                    max_blocked = blocked_count
                    best_y = y
                    candidates = [y]
                elif blocked_count == max_blocked:
                    candidates.append(y)
            
            # Pick one
            chosen_y = random.choice(candidates)
            
            # Update
            S.add(chosen_y)
            sums.add(2 * chosen_y)
            for a in S:
                if a != chosen_y:
                    sums.add(chosen_y + a)
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
