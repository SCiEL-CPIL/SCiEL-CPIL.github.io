import time
import random
from itertools import combinations_with_replacement

def _is_valid_set(S):
    """Check if all unordered triple sums (with replacement) are unique."""
    if not S:
        return True
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = S[i] + S[j] + S[k]
                if s in sums:
                    return False
                sums.add(s)
    return True

def _add_candidate(S, x):
    """Try to add x to S. Returns True if successful."""
    if x in S:
        return False
    new_S = S + [x]
    return _is_valid_set(new_S)

def _remove_element(S, idx):
    """Remove element at index idx from S."""
    return S[:idx] + S[idx+1:]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = []
    best_size = 0
    
    # Initial phase: Greedy build to get a decent starting point
    # This is fast and provides a baseline
    S_greedy = []
    for x in range(1, 4097):
        if time.time() > deadline - 5.0:
            break
        if _add_candidate(S_greedy, x):
            S_greedy.append(x)
            if len(S_greedy) > best_size:
                best_size = len(S_greedy)
                best_set = S_greedy[:]
        
        # If we've reached a reasonable size, stop greedy and switch to hill climbing
        if len(S_greedy) >= 60:
            break

    # Hill Climbing Phase
    # Strategy: Start with a random valid subset of the greedy set or a fresh random set
    # Then try to improve by swapping or inserting.
    
    # Let's try multiple random restarts with hill climbing
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize current set
        # Option 1: Use a random subset of the best found so far
        # Option 2: Generate a new random valid set
        if best_set and len(best_set) > 20:
            # Take a random subset of the best set to allow rearrangement
            k = max(10, len(best_set) - 5)
            curr_set = rng.sample(best_set, k)
        else:
            # Generate a small random valid set
            curr_set = []
            for _ in range(20):
                if time.time() > deadline - 10.0:
                    break
                x = rng.randint(1, 4096)
                if _add_candidate(curr_set, x):
                    curr_set.append(x)
        
        curr_size = len(curr_set)
        stuck_count = 0
        max_stuck = 100
        
        while time.time() < deadline - 5.0 and stuck_count < max_stuck:
            # Try to improve
            improved = False
            
            # 1. Try to insert a new random element
            if rng.random() < 0.5:
                # Pick a candidate not in curr_set
                # To be efficient, pick from a pool of "likely" candidates
                # or just random
                x = rng.randint(1, 4096)
                if x not in curr_set:
                    if _add_candidate(curr_set, x):
                        curr_set.append(x)
                        curr_size += 1
                        improved = True
                        stuck_count = 0
                        if curr_size > best_size:
                            best_size = curr_size
                            best_set = curr_set[:]
            else:
                # 2. Try to swap an existing element with a new one to enable growth
                # Pick an index to remove
                idx = rng.randint(0, curr_size - 1)
                old_val = curr_set[idx]
                new_S = _remove_element(curr_set, idx)
                
                # Try to find a replacement that allows the set to be valid and potentially larger
                # Actually, a simpler swap: replace old_val with a new random value, check validity
                # If valid, we have same size. But we want to grow.
                # So, swap is only useful if it "unblocks" an insertion.
                # Let's try: Remove old_val, then try to insert TWO new random values?
                # Or just swap and then try to insert.
                
                # Let's try a direct swap first to see if it helps future insertions
                # But since we track global best, let's focus on insertion attempts after swaps.
                
                # Alternative: Just try to insert into the set with one element removed
                # This is complex. Let's stick to:
                # A) Insert random
                # B) Swap random element with another random element (to shuffle)
                # C) Insert random into the set after a swap
                
                # Let's try swapping an element with a random new one
                new_val = rng.randint(1, 4096)
                if new_val not in new_S:
                    test_set = new_S + [new_val]
                    if _is_valid_set(test_set):
                        curr_set = test_set
                        # Size is same, but structure changed. Reset stuck count slightly?
                        # No, only reset on size increase.
                        # But this might allow future insertions.
                        # Let's count this as a "move" but not an improvement.
                        # We need to be careful not to loop forever.
                        # Let's say if we do 50 swaps without an insertion, we stop.
                        improved = True # Mark as move made to avoid immediate re-check of same state? 
                        # Actually, we should track if size increased.
                        # Let's distinguish "size_improved" from "move_made"
                        
            if not improved:
                stuck_count += 1
            else:
                # If we made a move (swap or insert), we might still be stuck in size
                # But we reset stuck_count only if size increased?
                # No, if we swap, we haven't increased size. 
                # Let's reset stuck_count on any successful move (insert or valid swap)
                # to allow exploration.
                pass 

            # Optimization: If we haven't increased size in a while, force a larger jump?
            # For now, simple hill climbing.

    return best_set

# Quick sanity check for the logic
# The _is_valid_set function is O(N^3). 
# For N=80, N^3 = 512,000 operations. 
# In Python, this might be slow if called millions of times.
# Let's optimize _is_valid_set or the insertion check.

def _is_valid_set_fast(S):
    """Faster validity check using sets."""
    sums = set()
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            sj = S[j]
            for k in range(j, n):
                sk = S[k]
                s_val = si + sj + sk
                if s_val in sums:
                    return False
                sums.add(s_val)
    return True

def _add_candidate_fast(S, x):
    """Check if adding x to S keeps it valid."""
    if x in S:
        return False
    
    # We need to check if any triple sum involving x conflicts with existing sums
    # or if any new triple sum involving x conflicts with another new triple sum involving x.
    
    # Existing sums in S
    # New sums involving x:
    # 1. (x, x, x)
    # 2. (x, x, y) for y in S
    # 3. (x, y, y) for y in S
    # 4. (x, y, z) for y < z in S
    
    new_sums = set()
    
    # 1. (x, x, x)
    s1 = 3 * x
    new_sums.add(s1)
    
    # 2. (x, x, y)
    for y in S:
        new_sums.add(2 * x + y)
        
    # 3. (x, y, y)
    for y in S:
        new_sums.add(x + 2 * y)
        
    # 4. (x, y, z)
    n = len(S)
    for i in range(n):
        yi = S[i]
        for j in range(i + 1, n):
            zj = S[j]
            new_sums.add(x + yi + zj)
            
    # Check for conflicts within new_sums
    # If len(new_sums) < number of generated triples, there's a conflict
    # Number of triples involving x:
    # 1 (xxx) + N (xxy) + N (xyy) + N*(N-1)/2 (xyz)
    # Total = 1 + 2N + N(N-1)/2
    
    expected_count = 1 + 2 * n + n * (n - 1) // 2
    if len(new_sums) != expected_count:
        return False
        
    # Check for conflicts with existing sums in S
    # We need the set of all existing sums in S.
    # Computing this from scratch every time is O(N^3).
    # It's better to maintain the set of sums.
    
    # For simplicity in this hill-climbing approach, let's just compute all sums of S
    # and check intersection.
    existing_sums = set()
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            sj = S[j]
            for k in range(j, n):
                existing_sums.add(si + sj + sk)
                
    if existing_sums & new_sums:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    # Phase 1: Greedy
    S_greedy = []
    for x in range(1, 4097):
        if time.time() > deadline - 10.0:
            break
        if _add_candidate_fast(S_greedy, x):
            S_greedy.append(x)
            if len(S_greedy) > best_size:
                best_size = len(S_greedy)
                best_set = S_greedy[:]
        if len(S_greedy) >= 70:
            break

    # Phase 2: Hill Climbing
    rng = random.Random(123)
    
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        if best_set and len(best_set) > 10:
            # Start from a random subset of best_set
            k = max(5, len(best_set) - rng.randint(0, 5))
            curr_set = rng.sample(best_set, k)
        else:
            curr_set = []
            
        curr_size = len(curr_set)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 2.0 and no_improve_count < max_no_improve:
            improved_size = False
            made_move = False
            
            # Strategy: Try to insert a random number
            # Or try to swap a number to enable insertion
            
            if rng.random() < 0.3:
                # Try Swap
                if curr_size > 0:
                    idx = rng.randint(0, curr_size - 1)
                    old_val = curr_set[idx]
                    new_val = rng.randint(1, 4096)
                    if new_val not in curr_set:
                        test_set = curr_set[:idx] + [new_val] + curr_set[idx+1:]
                        # Check validity
                        if _is_valid_set_fast(test_set):
                            curr_set = test_set
                            made_move = True
            else:
                # Try Insert
                x = rng.randint(1, 4096)
                if x not in curr_set:
                    if _add_candidate_fast(curr_set, x):
                        curr_set.append(x)
                        curr_size += 1
                        improved_size = True
                        made_move = True
                        if curr_size > best_size:
                            best_size = curr_size
                            best_set = curr_set[:]
                            
            if improved_size:
                no_improve_count = 0
            elif made_move:
                # Made a swap, didn't increase size. Count as progress?
                # Let's allow a few swaps before giving up.
                no_improve_count += 1
            else:
                no_improve_count += 2 # Penalize failed moves more

    return best_set
