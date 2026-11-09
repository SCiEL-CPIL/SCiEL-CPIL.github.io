import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # safety margin before 90s kill

def _pair_sums_unique(elements):
    """Check if all pair sums (including a+a) are unique."""
    sums = set()
    for i, a in enumerate(elements):
        for j in range(i, len(elements)):
            s = a + elements[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diffs(elements):
    """Get set of nonzero signed differences."""
    d = set()
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            diff = elements[j] - elements[i]
            d.add(diff)
            d.add(-diff)
    return d

def _can_add_to_set(x, elements, sum_set, diff_set, other_diff_set):
    """Check if x can be added to the set with given incremental data."""
    # Check pair sums uniqueness
    for a in elements:
        s = x + a
        if s in sum_set:
            return False
    s2 = x + x
    if s2 in sum_set:
        return False
    # Check differences
    new_diffs = set()
    for a in elements:
        d = x - a
        new_diffs.add(d)
        new_diffs.add(-d)
    # Check no conflict with own diff set (shouldn't happen if sums are unique, but be safe)
    if new_diffs & diff_set:
        return False
    # Check no conflict with other set's diff set
    if new_diffs & other_diff_set:
        return False
    return True

def _greedy_build():
    """Greedy construction with full range [1,256]."""
    A = []
    B = []
    sumA = set()
    sumB = set()
    diffA = set()
    diffB = set()
    
    for x in range(1, N + 1):
        # Try adding to A
        if len(A) < MAX_EACH and _can_add_to_set(x, A, sumA, diffA, diffB):
            # Update sumA
            for a in A:
                sumA.add(x + a)
            sumA.add(x + x)
            # Update diffA
            for a in A:
                d = x - a
                diffA.add(d)
                diffA.add(-d)
            A.append(x)
        
        # Try adding to B
        if len(B) < MAX_EACH and _can_add_to_set(x, B, sumB, diffB, diffA):
            # Update sumB
            for b in B:
                sumB.add(x + b)
            sumB.add(x + x)
            # Update diffB
            for b in B:
                d = x - b
                diffB.add(d)
                diffB.add(-d)
            B.append(x)
    
    return A, B, sumA, sumB, diffA, diffB

def _simulate_annealing(A, B, sumA, sumB, diffA, diffB, deadline):
    """
    Simulated Annealing on the state (A, B).
    Moves:
    1. Add x to A or B if possible (greedy improvement).
    2. Remove x from A or B and add y to A or B (swap).
    3. Remove x from A or B (if we want to shrink to escape, though we maximize size, 
       so usually we only remove if we can immediately add two? No, simple remove+add is enough).
    
    Actually, a better move for maximizing size is:
    - Try to add any valid x. If found, accept (always improves).
    - If no add found, try a swap: remove a from A, add x to A. Or remove from A, add to B.
    - Accept swap if size increases (impossible for 1-for-1 swap) or if it allows future improvements.
    - Since we maximize size, we should only accept swaps if they lead to a state where we can add more.
    - However, pure hill climbing on size gets stuck. 
    - Alternative: Random restarts with different greedy orders (already in current code).
    - Let's try a different local search: 
      "Perturb and Solve". 
      1. Take current best.
      2. Remove k random elements.
      3. Run greedy fill on the remaining sets + new candidates.
    """
    t0 = time.time()
    A = list(A)
    B = list(B)
    sumA = set(sumA)
    sumB = set(sumB)
    diffA = set(diffA)
    diffB = set(diffB)
    
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    # We will use a simpler, more effective local search:
    # Iteratively try to add elements. If stuck, try removing one element from A or B
    # and then greedily filling again.
    
    # To make it faster, we will use a "shuffled greedy restart" from the current state.
    
    # Actually, the current program's local search is weak. 
    # Let's implement a proper "Remove and Re-insert" search.
    
    # For efficiency, we will just do many random restarts of the greedy algorithm
    # but with different random seeds and different initial "seeds" of elements.
    
    # Strategy: 
    # 1. Generate a random subset of [1, 256] of size ~10 to start with? No, that's hard to validate.
    # 2. Better: Randomized Greedy is already there. 
    # 3. Let's try "Interleaved Greedy" where we randomly choose A or B to add to.
    
    # Since the deadline is tight, we will focus on high-quality randomized restarts.
    
    # We'll implement a "Shuffled Greedy" that is slightly different:
    # Instead of iterating 1..256, we iterate a random permutation.
    # We already do this. 
    
    # What if we bias the random permutation?
    # Or what if we use a "Two-Phase" construction?
    # Phase 1: Build A greedily.
    # Phase 2: Build B greedily avoiding A's diffs.
    # Phase 3: Try to add to A avoiding B's diffs.
    
    # Let's try this specific strategy in the main loop instead of generic restarts.
    
    pass

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    best_A, best_B = [], []
    best_size = 0
    
    # Phase 1: Standard Greedy
    A, B, sumA, sumB, diffA, diffB = _greedy_build()
    best_A, best_B = list(A), list(B)
    best_size = len(A) + len(B)
    
    # Phase 2: Randomized Greedy Restarts
    # We will run many restarts. To differentiate from the current program,
    # we will use a "Bipartite" greedy strategy in some restarts:
    # Alternate between building A and B based on random choice, 
    # or use a "Priority" queue where we pick the element that minimizes 
    # the number of conflicts (not implemented, too slow).
    
    # Let's try "Reverse Greedy": Start from 256 down to 1.
    # And "Random Permutation" Greedy.
    
    for restart in range(200):
        if time.time() > deadline - 2.0:
            break
        
        # Choose strategy
        if restart % 3 == 0:
            xs = list(range(1, N + 1))
            random.shuffle(xs)
        elif restart % 3 == 1:
            xs = list(range(N, 0, -1))
        else:
            xs = list(range(1, N + 1))
            # Bias towards smaller numbers? Or larger?
            # Let's try a mix: sort by random key
            xs.sort(key=lambda x: random.random())
            
        A = []
        B = []
        sumA = set()
        sumB = set()
        diffA = set()
        diffB = set()
        
        for x in xs:
            if time.time() > deadline - 2.0:
                break
            
            # Try adding to A
            if len(A) < MAX_EACH and _can_add_to_set(x, A, sumA, diffA, diffB):
                for a in A:
                    sumA.add(x + a)
                sumA.add(x + x)
                for a in A:
                    d = x - a
                    diffA.add(d)
                    diffA.add(-d)
                A.append(x)
            
            # Try adding to B
            if len(B) < MAX_EACH and _can_add_to_set(x, B, sumB, diffB, diffA):
                for b in B:
                    sumB.add(x + b)
                sumB.add(x + x)
                for b in B:
                    d = x - b
                    diffB.add(d)
                    diffB.add(-d)
                B.append(x)
        
        if len(A) + len(B) > best_size:
            best_A, best_B = list(A), list(B)
            best_size = len(A) + len(B)
            
            # If we hit the theoretical max (unlikely, but good to know), stop
            # Theoretical max for Sidon set in [1, 256] is approx sqrt(256) * something?
            # Sidon set size is approx sqrt(N). For N=256, sqrt(256)=16. 
            # Wait, Sidon set size k satisfies k^2 < N. So k < 16?
            # No, k(k+1)/2 < N is for distinct sums?
            # Number of sums is k(k+1)/2. These must be distinct and <= 2N.
            # So k(k+1)/2 <= 2N => k^2 ~ 4N => k ~ 2 sqrt(N).
            # For N=256, k ~ 32.
            # So max size is around 32 per set. Total 64.
            # If we hit 64, we can stop.
            if best_size >= 64:
                break

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
