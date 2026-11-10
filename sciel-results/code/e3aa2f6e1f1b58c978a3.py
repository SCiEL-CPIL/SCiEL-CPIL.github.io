"""Reciprocal-weight 3-AP-free set on [1,256] using Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 256
DEADLINE_MARGIN = 5.0  # seconds before the 60s limit (or 90s if evaluator allows, but contract says 60s for construct)
# The prompt says "Keep that claim conditional... evaluator KILLS after 90s... contract says 60s".
# I will use 55s to be safe for the 60s contract, but optimize heavily.

def _precompute_conflicts():
    """
    Precompute for each number x, the list of numbers y such that {x, y} forms a pair 
    that cannot coexist in a 3-AP-free set? No, that's not right.
    A set S is 3-AP free if there are no a,b,c in S such that a+c=2b.
    This is a constraint on triples.
    However, for local search, we need to quickly check if adding x to S creates a 3-AP.
    This happens if there exists a,b in S such that:
    1. a, x, b is AP -> x = (a+b)/2 -> 2x = a+b
    2. a, b, x is AP -> x = a + 2(b-a) = 2b - a (if b is middle, a is start, x is end? No)
       Let's stick to: x creates AP if:
       - x is middle: a, x, c -> c = 2x - a. If c in S, conflict.
       - x is end: a, b, x -> b = (a+x)/2. If b in S, conflict.
       - x is start: x, a, b -> a = (x+b)/2. If a in S, conflict.
       
    Actually, simpler: 
    If we add x to S, a 3-AP is formed iff there exist a, b in S such that:
    {a, b, x} contains a 3-AP.
    The possible APs involving x are:
    1. x, y, z with x < y < z. x is first. y = (x+z)/2.
    2. y, x, z with y < x < z. x is middle. y+z = 2x.
    3. y, z, x with y < z < x. x is last. y+x = 2z.
    
    To check if x can be added to S:
    For all a in S:
       - If 2*x - a is in S (and != x), then (a, x, 2x-a) is AP.
       - If (x + a) is even, let b = (x+a)/2. If b in S (and b != x, b != a), then (a, b, x) or (x, b, a) is AP.
       
    We can precompute for each x, the list of "forbidden partners" relative to other numbers?
    Better: Represent S as a bitset.
    Check(x, S_bitset):
       # Case 1: x is middle. a, x, c. c = 2x - a.
       # We need to know if there is a in S such that c = 2x-a is in S.
       # This is equivalent to: (S & (S shifted)) logic? 
       # Simpler loop over S is O(|S|). |S| is at most ~100. 256*100 = 25600 ops per check.
       # If we do 10000 checks, that's 256M ops. Might be slow in Python.
       
    Let's optimize the check using precomputed lists.
    For each x, precompute:
    - lefts[x]: list of a < x such that (a+x)/2 is integer. i.e., a+x even.
      Actually, if we add x, and a is in S, we need to check if b=(a+x)/2 is in S.
      So for each a in S, if (a+x)%2==0, check if (a+x)//2 in S.
      
    Let's just use a set for S for O(1) lookups, but the loop over S is the bottleneck.
    To speed up, we can precompute for each x, the list of pairs (a, c) such that a+c=2x.
    If we add x, we need to ensure NO pair (a, c) from this list is fully contained in S.
    Wait, if we add x, the AP is (a, x, c). Both a and c must be in S.
    So for each x, let Pairs[x] be a list of (a, c) such that a+c=2x, a != x, c != x, a < c.
    When adding x, we check if any (a, c) in Pairs[x] has both a and c in S.
    
    Also, x can be an endpoint.
    If x is endpoint, the AP is (a, b, x) or (x, a, b).
    (a, b, x) => b = (a+x)/2. So a and b must be in S.
    Let's precompute for each x, the list of pairs (a, b) such that a+x=2b (x is end) OR x+a=2b (x is start).
    Actually, it's symmetric.
    If we add x, conflicts arise if:
    1. There exists a, c in S such that a+c=2x. (x is middle)
    2. There exists a, b in S such that a+x=2b. (x is end, b is middle)
    3. There exists a, b in S such that x+a=2b. (x is start, b is middle)
    
    Note that 2 and 3 are the same condition: exists b in S such that 2b - x in S.
    So we need to check:
    - For each b in S: is (2b - x) in S? (And 2b-x != x, 2b-x != b? If 2b-x=b, then b=x, impossible since x not in S. If 2b-x=x, then b=x, impossible.)
      So simply: if (2b - x) in S, conflict.
    - For each a in S: is (2x - a) in S? (And 2x-a != x? If 2x-a=x, a=x, impossible.)
      So simply: if (2x - a) in S, conflict.
      
    So, check(x, S_set):
       # 1. x is middle
       for a in S_set:
           c = 2*x - a
           if c in S_set:
               return False
       # 2. x is end/start
       for b in S_set:
           a = 2*b - x
           if a in S_set:
               return False
       return True
       
    This is O(|S|) with set lookups. |S| ~ 100. 200 lookups per check.
    Python set lookups are fast (~50ns). 200*50ns = 10us per check.
    100,000 checks = 1 second. Feasible.
    """
    pass

def _is_valid_addition(x, S_set):
    """Check if x can be added to S_set without creating a 3-AP."""
    # x is middle: a, x, c -> c = 2x - a
    for a in S_set:
        c = 2 * x - a
        if c in S_set:
            return False
    # x is end/start: a, b, x -> b = (a+x)/2 => a = 2b - x
    for b in S_set:
        a = 2 * b - x
        if a in S_set:
            return False
    return True

def _construct_greedy():
    """Initial greedy solution."""
    S = []
    S_set = set()
    # Try to add small numbers first for high reciprocal weight
    for x in range(1, N + 1):
        if _is_valid_addition(x, S_set):
            S.append(x)
            S_set.add(x)
    return S

def _reciprocal_sum(S):
    return sum(1.0 / x for x in S)

def _simulated_annealing(S_init, deadline):
    """
    Simulated Annealing on the set of 3-AP-free subsets.
    Moves:
    1. Add a random valid element.
    2. Remove a random element.
    3. Swap: Remove one, Add another (if valid).
    Objective: Maximize sum(1/x).
    """
    S = list(S_init)
    S_set = set(S)
    current_sum = _reciprocal_sum(S)
    best_S = list(S)
    best_sum = current_sum
    
    # Temperature schedule
    T_start = 0.1
    T_end = 0.001
    # Estimate number of steps possible. 
    # Let's do 5000 steps per restart? 
    # 5000 * 10us = 0.05s. We can do many restarts.
    
    steps_per_run = 2000
    T = T_start
    
    start_time = time.time()
    step = 0
    
    while time.time() < deadline:
        # Adjust temperature
        # Linear cooling
        T = T_start - (T_start - T_end) * (step / steps_per_run)
        if T < T_end:
            T = T_end
        
        # Pick a move
        move_type = random.random()
        
        # Candidate move
        new_S = None
        new_sum = None
        delta = 0
        
        if move_type < 0.4:
            # Try to add a random element
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            if candidates:
                x = random.choice(candidates)
                if _is_valid_addition(x, S_set):
                    new_S = S + [x]
                    new_sum = current_sum + 1.0 / x
                    delta = 1.0 / x
                else:
                    # Invalid move, skip or treat as huge penalty? 
                    # In SA, we usually only consider valid moves or penalize invalid ones.
                    # Here, the constraint is hard. We can only move to valid states.
                    # So if invalid, we don't move.
                    pass
        elif move_type < 0.8:
            # Try to remove a random element
            if S:
                idx = random.randint(0, len(S) - 1)
                x = S[idx]
                new_S = [y for i, y in enumerate(S) if i != idx]
                new_sum = current_sum - 1.0 / x
                delta = -1.0 / x
        else:
            # Try to swap: remove one, add one
            if S:
                idx = random.randint(0, len(S) - 1)
                x_rem = S[idx]
                S_temp = [y for i, y in enumerate(S) if i != idx]
                S_temp_set = set(S_temp)
                
                # Try to add a random valid element to S_temp
                candidates = [x for x in range(1, N + 1) if x not in S_temp_set]
                if candidates:
                    # Pick a few candidates to try? Or just one?
                    # To keep it fast, pick one random candidate.
                    x_add = random.choice(candidates)
                    if _is_valid_addition(x_add, S_temp_set):
                        new_S = S_temp + [x_add]
                        new_sum = current_sum - 1.0 / x_rem + 1.0 / x_add
                        delta = 1.0 / x_add - 1.0 / x_rem
                    else:
                        pass

        if new_S is not None:
            # Acceptance probability
            if delta > 0:
                # Always accept improvements
                S = new_S
                S_set = set(S)
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_S = list(S)
            else:
                # Accept worse moves with probability exp(delta / T)
                # delta is negative
                prob = math.exp(delta / T)
                if random.random() < prob:
                    S = new_S
                    S_set = set(S)
                    current_sum = new_sum
        
        step += 1
        
        # Restart if stuck or temperature too low?
        # For now, just let it run until deadline, maybe with a simple restart logic
        if step % 5000 == 0:
             # Check if we are improving
             pass

    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + (60.0 - DEADLINE_MARGIN)
    
    # Initial greedy
    S_greedy = _construct_greedy()
    best_S = list(S_greedy)
    best_sum = _reciprocal_sum(best_S)
    
    # Run Simulated Annealing with multiple restarts
    # We can run SA starting from greedy, and also from random valid sets
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Determine initial state for this restart
        if i == 0:
            S_init = list(S_greedy)
        else:
            # Random valid set? 
            # Just start from greedy and let SA explore, or randomize the greedy order?
            # Let's try randomizing the insertion order of the greedy set to get different local optima
            S_rand = list(S_greedy)
            random.shuffle(S_rand)
            S_init = S_rand
            
        S_res = _simulated_annealing(S_init, deadline)
        s_res = _reciprocal_sum(S_res)
        
        if s_res > best_sum:
            best_sum = s_res
            best_S = S_res
    
    # Final local search polish? 
    # The SA should have done most of the work.
    # A quick greedy addition pass on the best found set
    if time.time() < deadline:
        S_polish = list(best_S)
        S_set = set(S_polish)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x not in S_set and _is_valid_addition(x, S_set):
                    S_polish.append(x)
                    S_set.add(x)
                    improved = True
        s_polish = _reciprocal_sum(S_polish)
        if s_polish > best_sum:
            best_sum = s_polish
            best_S = S_polish

    if not best_S:
        best_S = [1]
        
    return best_S

# EVOLVE-BLOCK-END
