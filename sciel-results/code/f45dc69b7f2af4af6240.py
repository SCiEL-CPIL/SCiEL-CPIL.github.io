import time
import random
import sys

START_TIME = time.time()
DEADLINE = 85.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def validate(A):
    """
    Validate the set A using bitsets.
    Returns True if valid, False otherwise.
    """
    if not A:
        return True
    if len(set(A)) != len(A):
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 256:
            return False
    
    n = len(A)
    # S[k] is a bitset where bit s is 1 if sum s is achievable with cardinality k
    S = [0] * (n + 1)
    S[0] = 1  # Sum 0 is achievable with cardinality 0
    
    for i in range(n):
        x = A[i]
        # Update in reverse to avoid using x multiple times in the same step
        for j in range(i, -1, -1):
            if S[j]:
                S[j+1] |= (S[j] << x)
        
    # Check for collisions between different cardinalities
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def check_validity_bitset(S, n):
    """
    Check if the current bitset state S (length n+1) has no cross-cardinality collisions.
    """
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def greedy_build_bitset(time_budget=10.0):
    """
    Greedy construction using bitsets.
    Returns a valid set.
    """
    start = time.time()
    chosen = []
    # S[k] bitset for current chosen set
    # Max size 60, max sum 60*256 = 15360. Bitset size needs to be large enough.
    # Python ints handle arbitrary size, so we don't need to pre-allocate, just shift.
    
    # We maintain S for the current 'chosen' set.
    # But 'chosen' changes as we add.
    # Let's just build S incrementally.
    
    # To make it robust, we'll try multiple random orders.
    best_set = []
    
    for attempt in range(50):
        if (time.time() - start) > time_budget or is_done():
            break
        
        chosen = []
        S = [0] * 61
        S[0] = 1
        used = set()
        
        # Heuristic: Try smaller numbers first? Or random?
        # Smaller numbers create more dense sum sets, which might collide more?
        # Actually, large numbers are "safer" because their sums are distinct and far apart.
        # Let's try a mix.
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Sort by some heuristic? Maybe just random is fine with bitset speed.
        # Let's try sorting descending to keep sums sparse initially?
        # Or ascending to pack tight?
        # Let's try random order first, then if we fail, try sorted.
        
        for x in candidates:
            if len(chosen) >= 60:
                break
            if (time.time() - start) > time_budget or is_done():
                break
            
            k = len(chosen)
            
            # Tentatively add x
            # New sums at k+1: S[k+1] | (S[k] << x)
            new_S_k_plus_1 = S[k+1] | (S[k] << x)
            
            # Check if this new set of sums collides with any other S[j]
            valid = True
            for j in range(k + 1):
                if new_S_k_plus_1 & S[j]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also need to ensure that the NEW sums don't collide with each other?
            # No, S[k+1] already contains previous sums. The new sums are added to it.
            # The collision is between the *set* of sums at cardinality k+1 and sets at other cardinalities.
            # Wait, adding x creates new sums. These new sums are part of S[k+1].
            # We must check if ANY sum in the new S[k+1] collides with ANY sum in S[j] for j != k+1.
            # Since S[k+1] was valid before (no collision with S[j]), we only need to check if the NEWLY ADDED sums collide.
            # Newly added sums are in (S[k] << x).
            # So we check if (S[k] << x) & S[j] != 0 for any j != k+1.
            # Note: S[k+1] might have changed, but the old part was already checked.
            # So checking (S[k] << x) & S[j] is sufficient for j < k+1.
            # What about j > k+1? S[j] for j > k+1 is currently 0 (since we build up to k).
            # So we only need to check j in 0..k.
            
            # Re-evaluate:
            # Current state: S[0..k] are valid. S[k+1..60] are 0.
            # We add x. New S[k+1] = S[k+1] | (S[k] << x).
            # We need new S[k+1] to be disjoint from S[0..k].
            # Since old S[k+1] was disjoint from S[0..k], we just need (S[k] << x) to be disjoint from S[0..k].
            
            # Let's re-verify the loop above.
            # My previous loop: `if new_S_k_plus_1 & S[j]`.
            # This checks if ANY sum in the new set collides.
            # This is correct.
            
            if valid:
                # Update S
                S[k+1] = new_S_k_plus_1
                chosen.append(x)
                used.add(x)
        
        if len(chosen) > len(best_set):
            best_set = chosen
            
    return best_set

def backtrack_search(time_budget=50.0):
    """
    Backtracking search to maximize set size.
    Uses bitsets for fast collision checking.
    """
    start = time.time()
    best_set = []
    
    # We try to find a set of size N.
    # Start with a greedy solution as a lower bound.
    initial = greedy_build_bitset(5.0)
    best_set = initial
    current_size = len(best_set)
    
    # Try to extend or find larger sets
    # We can use a DFS that tries to add elements.
    # To avoid getting stuck, we can use a "repair" strategy or just pure backtracking with good heuristics.
    
    # Let's define a recursive function
    sys.setrecursionlimit(10000)
    
    def dfs(chosen, S, k, candidates):
        nonlocal best_set, current_size
        
        if is_done() or (time.time() - start) > time_budget:
            return
        
        if len(chosen) > current_size:
            best_set = list(chosen)
            current_size = len(chosen)
            
        if len(chosen) >= 60:
            return
            
        # Heuristic: Order candidates
        # We want to try candidates that are likely to be valid.
        # Large numbers are safer.
        # Let's sort candidates descending?
        # But we need to try many.
        
        # To prevent exponential blowup, we limit the branching factor or use randomization.
        # Let's try a limited lookahead.
        
        # Filter candidates that are immediately valid
        valid_candidates = []
        for x in candidates:
            if x in chosen:
                continue
            
            # Check if x can be added
            new_sums = S[k] << x
            # Check collision with S[0..k]
            collides = False
            for j in range(k + 1):
                if new_sums & S[j]:
                    collides = True
                    break
            
            if not collides:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return
            
        # Shuffle valid candidates to add randomness
        random.shuffle(valid_candidates)
        
        # Limit branching to avoid explosion if time is tight
        # If we have many valid candidates, pick a subset to explore deeply?
        # Or just explore all?
        # With bitsets, checking is fast. But recursion depth is the issue.
        
        # Let's try all valid candidates, but prune if time is running out.
        # We'll iterate through them.
        
        for x in valid_candidates:
            if is_done() or (time.time() - start) > time_budget:
                return
                
            # Add x
            new_S = S[:]  # Copy
            new_S[k+1] = new_S[k+1] | (new_S[k] << x)
            
            # Recurse
            dfs(chosen + [x], new_S, k + 1, candidates)
            
            # If we found a better solution, we might want to restart or continue?
            # Continuing is fine, we keep the best.
            
            # Optimization: If we just found a solution of size 60, we can stop?
            # No, we want to be sure. But 60 is the max.
            if current_size == 60:
                return

    # Start DFS from empty set? Or from greedy solution?
    # Starting from empty is more general but slower.
    # Starting from greedy might miss other branches.
    # Let's start from empty but use the greedy result to set the initial best.
    
    # Candidates are 1..256
    candidates = list(range(1, 257))
    
    # To make DFS efficient, we should order candidates globally?
    # The `valid_candidates` list is built at each step.
    
    # Let's run the DFS
    dfs([], [0]*61, 0, candidates)
    
    return best_set

def construct():
    # 1. Greedy phase to get a decent baseline
    t0 = time.time()
    greedy_res = greedy_build_bitset(10.0)
    
    best = greedy_res
    
    # 2. Backtracking phase to try to improve
    # If greedy found < 60, try to improve with backtracking
    if len(best) < 60 and not is_done():
        t1 = time.time()
        time_left = DEADLINE - t1
        if time_left > 5:
            bt_res = backtrack_search(time_budget=time_left - 5)
            if len(bt_res) > len(best):
                best = bt_res
                
    # Final validation
    if validate(best):
        return best
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
