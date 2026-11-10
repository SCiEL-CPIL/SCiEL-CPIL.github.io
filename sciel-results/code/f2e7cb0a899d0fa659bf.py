import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    N = 1000
    MAX_SUM = 2 * N
    
    # Data structures
    # sum_counts[s] = number of pairs (a,b) in S such that a+b=s and a<=b
    sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    # in_set[x] = 1 if x is in S, 0 otherwise
    in_set = np.zeros(N + 1, dtype=np.int8)
    # active_elements: list of elements currently in S
    active_elements = []
    
    # Helper to check if adding x is valid (doesn't make any sum exceed 2)
    # Returns True if valid, and updates nothing.
    def is_valid_add(x):
        if in_set[x]:
            return False
        # Check 2x
        if sum_counts[2 * x] >= 2:
            return False
        # Check x + y for all y in S
        # Using numpy for speed if possible, but loop is safer for small N
        # Actually, let's stick to the efficient loop or vectorized check
        # For a single x, checking against active_elements is O(|S|)
        # We can vectorize: sums = x + np.array(active_elements)
        # But creating array each time is overhead.
        # Let's use the standard loop first, it's fast enough for ~700 elements.
        for y in active_elements:
            if sum_counts[x + y] >= 2:
                return False
        return True

    def add_to_set(x):
        in_set[x] = 1
        active_elements.append(x)
        sum_counts[2 * x] += 1
        for y in active_elements:
            if y == x:
                continue
            sum_counts[x + y] += 1

    def remove_from_set(x):
        in_set[x] = 0
        if x in active_elements:
            active_elements.remove(x)
        sum_counts[2 * x] -= 1
        for y in active_elements:
            if y == x:
                continue
            sum_counts[x + y] -= 1

    def get_score(x):
        """
        Score for adding x: 
        We want to pick x that uses up the 'scarcest' resources or balances the load.
        Actually, a common heuristic for packing is to pick the item that fits in the 
        'tightest' spots, or conversely, to preserve flexibility, pick the one that 
        uses the most 'abundant' resources?
        
        Let's think about the "Pressure" heuristic from the prompt's inspiration.
        If we pick x, we increment counts for 2x and x+y.
        If count becomes 2, that sum is "full".
        We want to minimize the number of sums that become full? 
        Or maximize the number of sums that remain < 2?
        
        Alternative: Pick x that minimizes the maximum count increase.
        Or: Pick x such that the resulting state has the most "slack".
        
        Let's try: Score = - (sum of counts of affected sums after addition).
        We want to minimize the sum of counts, meaning we prefer to use sums that are already low.
        This tends to spread out the representations.
        """
        if not is_valid_add(x):
            return -np.inf
        
        # Affected sums: 2x, and x+y for y in S
        # We want to minimize the "load" we add to.
        # Load = sum( (sum_counts[s] + 1) for s in affected )
        # Minimizing this means we prefer sums that are currently 0 or 1.
        
        score = sum_counts[2 * x] + 1
        for y in active_elements:
            score += (sum_counts[x + y] + 1)
            
        return -score

    # Phase 1: Greedy Construction with "Pressure" Heuristic
    # We will run multiple restarts with different random seeds for tie-breaking.
    
    best_size = 0
    best_state = None
    best_sum_counts = None
    best_in_set = None
    best_active = None
    
    num_restarts = 100 # Increased restarts due to better scoring
    time_per_restart = 1.0
    
    for r in range(num_restarts):
        if time.time() > deadline - 20: # Leave 20s for final polish
            break
            
        # Reset state
        sum_counts[:] = 0
        in_set[:] = 0
        active_elements = []
        
        # Candidate pool
        candidates = list(range(1, N + 1))
        random.shuffle(candidates) # Shuffle to break ties or randomize start? 
        # Actually, for the heuristic to work well, we should evaluate all valid candidates
        # at each step and pick the best one. 
        # But O(N * |S| * N) might be too slow.
        # N=1000, |S|~700. 
        # Evaluating score for all 1000 candidates: 1000 * 700 = 700k ops per step.
        # 700 steps * 700k = 490M ops. Too slow in Python.
        
        # Optimization: 
        # We can't evaluate all candidates every step.
        # We can maintain a priority queue? 
        # Or we can sample a subset of candidates?
        
        # Let's try a "Beam Search" or "Top-K" approach.
        # At each step, evaluate a random subset of K valid candidates and pick the best.
        # Or, simpler: Evaluate all candidates, but optimize the score calculation.
        
        # Can we optimize is_valid_add and get_score?
        # is_valid_add is O(|S|). get_score is O(|S|).
        # Total O(N * |S|) per step.
        
        # Let's try to be smarter.
        # Instead of picking the global best, pick a local best from a random sample.
        # Sample size K=50.
        
        K = 50
        
        while True:
            if time.time() > deadline - 20:
                break
                
            # Get valid candidates
            # This is still O(N * |S|) if we check all.
            # We can just iterate and collect valid ones?
            # Or just pick K random candidates and check them.
            
            valid_candidates = []
            # Randomly sample indices to check
            # If we just check all, it's slow.
            # Let's check a random subset first. If we find few valid, check more.
            
            # Actually, let's just iterate through all candidates but use a faster check?
            # No, let's stick to the random sampling for speed, or just iterate all if N is small enough?
            # 1000 candidates * 700 elements = 700k checks.
            # In Python, 700k simple checks is ~0.05s.
            # 700 steps * 0.05s = 35s per restart.
            # 100 restarts * 35s = 3500s. Too slow.
            
            # We need a faster greedy.
            # Standard greedy (smallest valid) is O(N * |S|) total if we just scan 1..N once?
            # No, we scan 1..N, check validity (O(|S|)), add if valid.
            # Total O(N * |S|) for the whole construction.
            # 1000 * 700 = 700k ops. ~0.05s.
            # This is very fast.
            
            # The problem with "smallest valid" is it might paint itself into a corner.
            # The "Pressure" heuristic requires evaluating ALL candidates to pick the best.
            # That makes it O(N^2 * |S|) which is too slow.
            
            # Compromise: 
            # 1. Find ALL valid candidates. (O(N * |S|))
            # 2. If none, stop.
            # 3. If few (< K), evaluate all and pick best.
            # 4. If many, pick a random subset of size K, evaluate them, pick best.
            
            valid_candidates = []
            for x in range(1, N + 1):
                if is_valid_add(x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if len(valid_candidates) <= K:
                best_x = max(valid_candidates, key=get_score)
            else:
                # Sample K candidates
                sample = random.sample(valid_candidates, K)
                best_x = max(sample, key=get_score)
            
            add_to_set(best_x)
            
        current_size = len(active_elements)
        if current_size > best_size:
            best_size = current_size
            best_state = active_elements[:]
            best_sum_counts = sum_counts.copy()
            best_in_set = in_set.copy()
            best_active = active_elements[:]
            
    # Phase 2: Local Search (Swap Moves)
    # Start from the best greedy result found.
    if best_state is None:
        # Fallback: simple greedy
        sum_counts[:] = 0
        in_set[:] = 0
        active_elements = []
        for x in range(1, N + 1):
            if is_valid_add(x):
                add_to_set(x)
        best_state = active_elements[:]
        best_sum_counts = sum_counts.copy()
        best_in_set = in_set.copy()
        best_active = active_elements[:]
        best_size = len(active_elements)
        
    # Load best state into working variables
    sum_counts = best_sum_counts
    in_set = best_in_set
    active_elements = best_active
    
    # Local Search: Hill Climbing with Kicks
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Greedy Fill (try to add any valid number)
        added_any = False
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            if is_valid_add(x):
                add_to_set(x)
                added_any = True
                improved = True
        
        if not added_any:
            # 2. Try a Kick (Swap to enable growth)
            kick_success = False
            # Try multiple random kicks
            for _ in range(200): # Limit kicks per stagnation phase
                if time.time() > deadline - 5:
                    break
                    
                if not active_elements:
                    break
                    
                u = random.choice(active_elements)
                
                # Remove u
                remove_from_set(u)
                
                # Try to add a new element v
                # We want v such that adding v is valid.
                # Then we try to add w.
                
                # Strategy: Pick v randomly from non-in_set.
                v = random.randint(1, N)
                if in_set[v]:
                    # Restore u and try again
                    add_to_set(u)
                    continue
                    
                if is_valid_add(v):
                    add_to_set(v)
                    
                    # Now try to add w
                    w_found = False
                    # To speed up, we can just scan for any valid w
                    for w in range(1, N + 1):
                        if in_set[w]:
                            continue
                        if is_valid_add(w):
                            add_to_set(w)
                            w_found = True
                            break
                            
                    if w_found:
                        # We have increased size by 1 (removed u, added v, added w)
                        improved = True
                        kick_success = True
                        break
                    else:
                        # Revert v
                        remove_from_set(v)
                        # Restore u
                        add_to_set(u)
                else:
                    # v was not valid, so we didn't add it. 
                    # But we removed u! We must restore u.
                    add_to_set(u)
                    
            if not kick_success:
                # Stuck
                break
                
    # Final Greedy Fill to be safe
    for x in range(1, N + 1):
        if in_set[x]:
            continue
        if is_valid_add(x):
            add_to_set(x)
            
    return active_elements

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    s = set(res)
    counts = defaultdict(int)
    valid = True
    for i, a in enumerate(res):
        for b in res[i:]:
            sum_val = a + b
            counts[sum_val] += 1
    for k, v in counts.items():
        if v > 2:
            valid = False
            break
    print(f"Valid: {valid}")
