import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s limit

    # 1. Precompute 4-AP constraints (Hard Clauses)
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # We want to forbid any such set from being fully included in S.
    # This is equivalent to: NOT (x_a AND x_{a+d} AND x_{a+2d} AND x_{a+3d})
    # In MaxSAT/Local Search, we can treat this as a penalty if violated.
    
    # 2. Precompute 3-AP rewards (Soft Clauses)
    # A 3-AP is (a, a+d, a+2d).
    # We want to maximize the count of these.
    # Reward: x_a AND x_{a+d} AND x_{a+2d}
    
    # Represent variables as a boolean vector of length n (index 0..n-1 corresponds to 1..n)
    
    # Precompute indices for 4-APs
    ap4_indices = []
    for a in range(1, n + 1):
        for d in range(1, n // 3 + 1):
            if a + 3 * d <= n:
                # Store 0-based indices
                ap4_indices.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    ap4_arr = np.array(ap4_indices, dtype=np.int32)
    
    # Precompute indices for 3-APs
    ap3_indices = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            ap3_indices.append((a - 1, a + d - 1, a + 2 * d - 1))
    ap3_arr = np.array(ap3_indices, dtype=np.int32)
    
    # We will use a vectorized approach to calculate the objective function.
    # Objective = -HARD_WEIGHT * (number of violated 4-APs) + (number of satisfied 3-APs)
    # We want to maximize this.
    # If any 4-AP is violated, the result is invalid. 
    # However, for the search to work, we need a landscape. 
    # Standard WalkSAT on MaxSAT:
    # Minimize unsatisfied hard clauses. If 0 hard clauses unsatisfied, maximize satisfied soft clauses.
    
    # Let's define the score function using NumPy.
    # x is a boolean array of shape (n,)
    
    def evaluate(x):
        # x is boolean numpy array
        # Check 4-APs: violation if all 4 are True
        if len(ap4_arr) > 0:
            # ap4_arr is shape (M, 4)
            # x[ap4_arr] gives shape (M, 4)
            # all(axis=1) gives shape (M,)
            violated_4 = np.all(x[ap4_arr], axis=1)
            num_violated_4 = np.sum(violated_4)
        else:
            num_violated_4 = 0
            
        # Check 3-APs: satisfied if all 3 are True
        if len(ap3_arr) > 0:
            satisfied_3 = np.all(x[ap3_arr], axis=1)
            num_satisfied_3 = np.sum(satisfied_3)
        else:
            num_satisfied_3 = 0
            
        return num_violated_4, num_satisfied_3

    def get_score(x):
        v4, s3 = evaluate(x)
        # We prioritize removing 4-APs. 
        # Score = -10000 * v4 + s3
        # If v4 > 0, score is very negative.
        return -10000 * v4 + s3

    # Initial state: Random or Greedy?
    # Greedy is better for initialization to start valid or near-valid.
    
    def greedy_init():
        x = np.zeros(n, dtype=bool)
        # Try to add elements in random order, checking if it creates a 4-AP.
        # Checking if adding i creates a 4-AP is expensive if done naively.
        # However, n=256 is small. We can check all 4-APs containing i.
        # Precompute which 4-APs contain which index for faster check?
        # Or just use the global check? 
        # Global check is O(M4) per insertion. M4 ~ 1000. 256 * 1000 = 256k ops. Fast enough.
        
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            x[i] = True
            # Check if any 4-AP is now violated
            # Only need to check 4-APs that include i.
            # But for simplicity in this prototype, let's just check all? 
            # No, let's be smarter.
            # Actually, let's just do a full check and revert if violated.
            v4, _ = evaluate(x)
            if v4 > 0:
                x[i] = False
        return x

    # WalkSAT Logic
    # 1. Start with x
    # 2. While time remains:
    #    a. If no 4-APs violated, we are in a valid state. Try to improve 3-AP count.
    #    b. If 4-APs violated, pick a random violated 4-AP, and flip one of its variables to try to satisfy it.
    #       (Weighted random selection based on impact on other clauses? Or just random flip in the clause)
    #    c. Occasionally do a random flip to escape local minima.
    
    best_score = -10**9
    best_x = None
    current_x = greedy_init()
    current_v4, current_s3 = evaluate(current_x)
    current_score = get_score(current_x)
    
    if current_score > best_score:
        best_score = current_score
        best_x = current_x.copy()
        
    # Parameters
    # p_flip_clause: probability to flip a variable in a violated clause (if any)
    # p_random: probability to flip a random variable
    # p_stick: probability to stick with current heuristic
    
    # We will use a simple WalkSAT variant:
    # If unsatisfied hard clauses exist:
    #   Pick random unsatisfied hard clause.
    #   With prob p, flip a random variable in that clause.
    #   With prob 1-p, flip the variable in that clause that maximizes the number of satisfied soft clauses (or minimizes violated hard clauses).
    # If no unsatisfied hard clauses:
    #   Pick random variable, flip it if it improves soft score (or with some probability).
    
    # To make it faster, let's precompute the "impact" of flipping a variable.
    # This is complex because it depends on current state.
    
    # Simplified WalkSAT for this specific structure:
    
    # Precompute for each variable i, the list of 4-AP indices containing i.
    # And list of 3-AP indices containing i.
    # This allows faster delta calculation?
    # Actually, the vectorized evaluation is quite fast for n=256.
    # M4 ~ 1000, M3 ~ 8000.
    # evaluate() takes ~1ms?
    # We can do ~1000 evaluations/sec.
    # 58s * 1000 = 58,000 steps.
    
    # Let's optimize the inner loop.
    # Instead of full vectorized eval every step, we can do incremental updates?
    # Or just rely on NumPy speed.
    
    # Let's try to speed up evaluate by using bit operations or precomputed lookups?
    # For now, let's stick with NumPy and see if it's fast enough.
    
    # Optimization: 
    # ap4_arr and ap3_arr are fixed.
    # x[ap4_arr] is a gather operation.
    
    # Let's add a restart mechanism.
    # If stuck (no improvement in X steps), restart with a new greedy init or random perturbation.
    
    steps_since_improvement = 0
    max_steps_no_improve = 2000
    
    while time.time() - start_time < time_limit:
        v4, s3 = evaluate(current_x)
        score = get_score(current_x)
        
        if score > best_score:
            best_score = score
            best_x = current_x.copy()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            
        if steps_since_improvement > max_steps_no_improve:
            # Restart
            current_x = greedy_init()
            # Perturb
            for _ in range(10):
                idx = random.randint(0, n - 1)
                current_x[idx] = not current_x[idx]
            steps_since_improvement = 0
            continue
            
        if v4 > 0:
            # Try to fix a violated 4-AP
            # Find indices of violated 4-APs
            violated_indices = np.where(np.all(current_x[ap4_arr], axis=1))[0]
            if len(violated_indices) == 0:
                # Should not happen if v4 > 0, but safety
                continue
            target_clause_idx = violated_indices[random.randint(0, len(violated_indices) - 1)]
            clause_vars = ap4_arr[target_clause_idx]
            
            # Choose which variable to flip
            # Option 1: Random
            # Option 2: Variable that "breaks" the least other 4-APs or "saves" the most 3-APs
            # Let's try a weighted choice: 
            # For each var in clause, calculate "cost" of flipping it.
            # Cost = (change in violated 4-APs) * 10000 - (change in satisfied 3-APs)
            # We want to minimize cost (or maximize improvement).
            
            best_var = -1
            best_cost = 10**9
            
            for var in clause_vars:
                # Flip var
                current_x[var] = not current_x[var]
                new_v4, new_s3 = evaluate(current_x)
                new_score = get_score(current_x)
                # Unflip
                current_x[var] = not current_x[var]
                
                # We want to maximize new_score.
                if new_score > best_score_current: # This is not right, we want local improvement
                    pass
                
                # Let's just calculate delta
                # delta = new_score - current_score
                # If delta > 0, it's good.
                # But we are in a violated state, so current_score is very low.
                # We just want to reduce v4.
                
                # Heuristic: Flip the variable that is in the fewest other violated 4-APs.
                # Or just random.
                
                # Let's use a simple heuristic: 
                # Count how many other violated 4-APs contain this variable.
                # If we flip it, we satisfy this one, but might create new ones or keep others violated.
                
                # For speed, let's just pick random variable in the clause 50% of the time,
                # and the one with lowest degree in the "violated clause graph" 50% of the time.
                
                # Actually, let's just do random flip in the clause for now, it's simpler.
                pass
            
            # Let's implement the "evaluate all 4 options" approach since clause size is 4.
            best_var_to_flip = clause_vars[0]
            best_new_score = -10**9
            
            for var in clause_vars:
                old_val = current_x[var]
                current_x[var] = not old_val
                nv4, ns3 = evaluate(current_x)
                nscore = get_score(current_x)
                current_x[var] = old_val
                
                if nscore > best_new_score:
                    best_new_score = nscore
                    best_var_to_flip = var
                    
            # With some probability, do a random flip instead of greedy
            if random.random() < 0.3:
                best_var_to_flip = clause_vars[random.randint(0, 3)]
                
            current_x[best_var_to_flip] = not current_x[best_var_to_flip]
            
        else:
            # No 4-APs violated. Maximize 3-APs.
            # Try to find a flip that increases 3-AP count.
            # Scan a subset of variables? Or random?
            # Let's try a random variable and see if flipping it improves the score.
            
            # To be more effective, let's try a few random variables and pick the best improvement.
            best_var_to_flip = -1
            best_delta = 0
            current_score_val = get_score(current_x)
            
            for _ in range(5): # Try 5 random candidates
                var = random.randint(0, n - 1)
                old_val = current_x[var]
                current_x[var] = not old_val
                nv4, ns3 = evaluate(current_x)
                nscore = get_score(current_x)
                current_x[var] = old_val
                
                delta = nscore - current_score_val
                if delta > best_delta:
                    best_delta = delta
                    best_var_to_flip = var
                    
            if best_var_to_flip != -1:
                current_x[best_var_to_flip] = not current_x[best_var_to_flip]
            else:
                # No immediate improvement. Do a random flip to escape local optimum.
                var = random.randint(0, n - 1)
                current_x[var] = not current_x[var]
                
    # Final check
    if best_x is None:
        return [1]
        
    # Convert boolean array to list of indices (1-based)
    result_indices = np.where(best_x)[0] + 1
    return list(result_indices)

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
