import time
import random
import sys
import numpy as np

_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0

# Global cache for priority sets
_PRIORITY_CACHE = {}

def _int_to_tuple(idx, n):
    v = []
    x = idx
    for _ in range(n):
        v.append(x % 3)
        x //= 3
    return tuple(v)

def _tuple_to_int(v):
    idx = 0
    for i, d in enumerate(reversed(v)):
        idx += d * (3 ** i)
    return idx

def _reflection_index(v_idx, a_idx, n):
    """
    Compute the index of w where w_i = (2*v_i - a_i) mod 3.
    This is the third point in the arithmetic progression (a, v, w) in F_3^n.
    """
    w_idx = 0
    power = 1
    vi = v_idx
    ai = a_idx
    for _ in range(n):
        vd = vi % 3
        ad = ai % 3
        wd = (2 * vd - ad) % 3
        w_idx += wd * power
        vi //= 3
        ai //= 3
        power *= 3
    return w_idx

def _greedy_cap_set_fast(n, seed, time_limit):
    """
    Fast greedy cap set using integer indices.
    """
    start = time.time()
    total = 3 ** n
    
    # Precompute digits for all vectors to speed up feature extraction
    # For small n, this is fine. For large n, we might need to be careful.
    # n is usually small in these tasks (e.g., 5-10).
    all_digits = []
    for i in range(total):
        all_digits.append(_int_to_tuple(i, n))
        
    # Simple heuristic ordering: prefer vectors with more 0s? 
    # Actually, random is often robust. Let's use a mixed strategy.
    # We'll sort by number of zeros descending, then random shuffle within groups.
    # This tends to find "sparse" vectors first which are less likely to conflict?
    # Or maybe dense? Let's stick to random for speed and simplicity in the seed phase,
    # as local search will fix it.
    
    rng = random.Random(seed)
    order = list(range(total))
    rng.shuffle(order)
    
    S = set()
    forbidden = set()
    
    for idx in order:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        S.add(idx)
        v = all_digits[idx]
        
        # Update forbidden set
        # For each a in S, the reflection w = 2v - a is forbidden.
        # Also, for each a in S, the reflection of a with respect to v is w.
        # Wait, if we add v, we must ensure no a in S such that 2v-a is in S.
        # If we maintain 'forbidden' as the set of vectors that CANNOT be added,
        # then when we add v, we add all w such that (a, v, w) is a line for any a in S.
        # w = 2v - a.
        for a_idx in S:
            if a_idx == idx:
                continue
            w_idx = _reflection_index(idx, a_idx, n)
            forbidden.add(w_idx)
            
    return S

def _local_search_cap_set(n, initial_set, seed, time_limit):
    """
    Local search (Simulated Annealing) to improve the cap set.
    State: Set of indices S.
    Move: Remove one element, add another (or just add another if space allows? No, size is fixed or we maximize size).
    Actually, we want to maximize size. 
    Standard approach: 
    1. Try to add random vectors.
    2. If stuck, remove a "bad" vector and try to add others.
    
    Better approach for Cap Sets:
    Maintain S. 
    1. Generate candidate vector v not in S.
    2. Check if S U {v} is a cap set.
       - This requires checking if v forms a line with any pair in S.
       - Or, maintain the set of "forbidden" vectors for the current S.
       - If v is not in forbidden(S), then S U {v} is a cap set.
    
    So, we can maintain `forbidden` dynamically.
    When we add v:
      forbidden_new = forbidden_old U { 2v - a for a in S }
    When we remove v:
      This is hard to invert easily.
    
    Alternative: 
    Use a fixed size K and try to find a cap set of size K.
    Or, use a "greedy + repair" strategy.
    
    Let's try a simpler Local Search:
    Start with S from greedy.
    Maintain `forbidden` set.
    Loop:
      1. Pick a random vector v from the "available" pool (not in S, not in forbidden).
         If we find one, add it to S, update forbidden, and continue.
      2. If no available vector is found (or time is running out), we are stuck.
         To escape, remove a random element u from S.
         Recompute forbidden from scratch? O(|S|^2). 
         Or, keep a count of how many times each vector is forbidden.
    
    Let's implement the count-based forbidden set.
    forbidden_count[w] = number of pairs (a, v) in S such that w = 2v - a.
    If forbidden_count[w] == 0, w is available.
    
    When adding v:
      For each a in S:
        w = 2v - a
        forbidden_count[w] += 1
      S.add(v)
      
    When removing v:
      For each a in S (excluding v):
        w = 2v - a
        forbidden_count[w] -= 1
      S.remove(v)
      
    This allows O(|S|) updates for add/remove.
    
    Strategy:
    1. Greedy seed.
    2. While time remains:
       a. Try to add a random available vector.
          - Maintain a list of available vectors? Updating it dynamically is O(N).
          - Instead, pick random v from [0, N). Check if v in S? No. Check if forbidden_count[v] == 0?
          - If yes, add it.
       b. If we fail to find an available vector in M random tries, we are likely stuck or full.
          - If stuck, remove a random element from S to create space.
          - This "shake" allows us to explore different configurations.
    
    """
    start = time.time()
    total = 3 ** n
    
    # Initialize forbidden_count
    forbidden_count = np.zeros(total, dtype=np.int32)
    S = set(initial_set)
    
    # Rebuild forbidden_count for the initial set
    for v in S:
        for a in S:
            if v == a:
                continue
            w = _reflection_index(v, a, n)
            forbidden_count[w] += 1
            
    # Note: The reflection of v with respect to a is w.
    # The line is (a, v, w).
    # If we add v, we forbid w for all a in S.
    # Also, if we add v, we forbid a' = 2v - w? No, the condition is symmetric.
    # A set is a cap set if it contains no 3 points in a line.
    # A line is {x, y, z} with x+y+z = 0 (in F_3).
    # If we have S, and we add v, we must ensure that for all a in S,
    # the third point w = -a - v (which is 2v - a in F_3? No. 2v-a is (2v-a) mod 3.
    # In F_3, -1 = 2. So -a - v = 2a + 2v = 2(a+v).
    # Wait. The line is x, y, z. x+y+z = 0.
    # If we have a and v, the third point is z = -a - v.
    # In integer arithmetic mod 3: z = (3 - (a+v)%3)%3?
    # Let's check the reflection formula: w = 2v - a.
    # If a=0, v=1, w=2. 0+1+2 = 3 = 0. Correct.
    # If a=1, v=1, w=1. 1+1+1 = 3 = 0. Correct.
    # If a=2, v=1, w=0. 2+1+0 = 3 = 0. Correct.
    # So w = (2v - a) % 3 is correct for the third point.
    
    rng = random.Random(seed)
    
    # List of vectors in S for quick removal
    # We just iterate over S for updates.
    
    while time.time() - start < time_limit:
        # Try to add a vector
        added = False
        # Try a few random candidates
        for _ in range(100):
            v = rng.randint(0, total - 1)
            if v in S:
                continue
            if forbidden_count[v] == 0:
                # Add v
                S.add(v)
                for a in S:
                    if a == v:
                        continue
                    w = _reflection_index(v, a, n)
                    forbidden_count[w] += 1
                added = True
                break
        
        if added:
            continue
            
        # If we couldn't add, try to remove and reshuffle
        # Remove a random element
        if len(S) > 0:
            victim = rng.choice(list(S))
            S.remove(victim)
            for a in S:
                w = _reflection_index(victim, a, n)
                forbidden_count[w] -= 1
            # Optional: try to add something else immediately?
            # The loop will handle it in the next iteration.
            
    return S

def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        # Perform search for this n
        # Time budget management
        time_limit_greedy = 30.0
        time_limit_ls = 200.0
        
        start_search = time.time()
        
        best_set = set()
        best_size = 0
        
        # 1. Greedy Seed
        # Use multiple seeds for greedy to get diverse starting points
        greedy_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
        
        for seed in greedy_seeds:
            if time.time() - start_search > time_limit_greedy:
                break
            S_greedy = _greedy_cap_set_fast(n, seed, time_limit_greedy / len(greedy_seeds))
            if len(S_greedy) > best_size:
                best_size = len(S_greedy)
                best_set = S_greedy
        
        # 2. Local Search
        # Run local search on the best greedy set, and maybe a few others
        ls_seeds = [1, 2, 3, 4, 5]
        
        for seed in ls_seeds:
            if time.time() - start_search > time_limit_greedy + time_limit_ls:
                break
            remaining = time_limit_greedy + time_limit_ls - (time.time() - start_search)
            if remaining < 10.0:
                break
            
            # We can run LS on the best set found so far
            # Or on a fresh greedy set? 
            # Let's run LS on the best set found in greedy phase.
            # To allow for different trajectories, we might want to run LS on different seeds' greedy outputs.
            # But LS is expensive. Let's run it on the best greedy set.
            
            S_ls = _local_search_cap_set(n, best_set, seed, remaining)
            
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_set = S_ls
            
            # If we have time, run LS on a second best greedy set?
            # For now, stick to improving the best one.
            
        # Convert best_set (indices) to set of tuples
        best_tuples = set()
        for idx in best_set:
            best_tuples.add(_int_to_tuple(idx, n))
        
        _PRIORITY_CACHE[n] = best_tuples
    
    return 1.0 if vector in _PRIORITY_CACHE[n] else 0.0
