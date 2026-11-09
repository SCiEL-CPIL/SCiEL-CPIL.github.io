"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 80  # 80s deadline to leave safety margin
    
    def check_sidon(arr):
        """Check if arr is a Sidon set (all pair sums including a+a are unique)."""
        if len(arr) == 0:
            return True
        arr_set = set(arr)
        # Use a set to track sums
        seen_sums = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                s = arr[i] + arr[j]
                if s in seen_sums:
                    return False
                seen_sums.add(s)
        return True
    
    def get_diffs(arr):
        """Get the set of nonzero absolute differences (or signed? Let's use signed to be safe, but actually the problem says 'nonzero difference sets' which usually means {b-a | a,b in set, a!=b}. Let's use signed differences to be precise, or absolute? 
        'disjoint difference sets' usually implies the set of values d = b-a. If A has {1,3} and B has {2,4}, diffs of A are {2,-2}, diffs of B are {2,-2}. They intersect.
        Let's check the seed code:
        differences={b-a for a,b in combinations(candidate,2)}
        So it's signed differences.
        """
        if len(arr) < 2:
            return set()
        return {b - a for a, b in combinations(arr, 2)}
    
    def is_valid(a, b):
        """Check if A and B are valid: Sidon and disjoint nonzero diffs."""
        if not check_sidon(a):
            return False
        if not check_sidon(b):
            return False
        if len(a) == 0 or len(b) == 0:
            return True
        diff_a = get_diffs(a)
        diff_b = get_diffs(b)
        if diff_a & diff_b:
            return False
        return True

    # Greedy construction with multiple random restarts and orders
    best_a = []
    best_b = []
    best_score = 0
    
    # Precompute potential candidates
    candidates = list(range(1, n + 1))
    
    def greedy_build(order, side_priority):
        """
        Build A and B greedily following the given order of candidates.
        side_priority: 0 means try A first, 1 means try B first.
        """
        A = []
        B = []
        # To speed up, we can maintain incremental sums and diffs, but for simplicity in restarts,
        # we might just check at the end or use a fast incremental check.
        # Given n=256, checking Sidon is O(k^2). k up to 64. 64^2 = 4096 ops. Very fast.
        # Diffs O(k^2).
        
        # We need to decide which side to add to.
        # Strategy: Iterate through candidates in 'order'. For each candidate x:
        # Try adding to A. If valid, add.
        # Else try adding to B. If valid, add.
        # Else skip.
        
        # To allow for better packing, we can randomize the order of candidates.
        
        for x in order:
            # Try A
            A_new = A + [x]
            if check_sidon(A_new):
                # Check diff disjointness
                if B:
                    diff_a = get_diffs(A_new)
                    diff_b = get_diffs(B)
                    if not (diff_a & diff_b):
                        A = A_new
                        continue
            # Try B
            B_new = B + [x]
            if check_sidon(B_new):
                # Check diff disjointness
                if A:
                    diff_a = get_diffs(A)
                    diff_b = get_diffs(B_new)
                    if not (diff_a & diff_b):
                        B = B_new
                        continue
            # If neither, skip x
        
        return A, B

    # Run multiple random restarts
    num_restarts = 2000  # Adjust based on time
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    restart_count = 0
    
    while time.time() < deadline and restart_count < num_restarts:
        restart_count += 1
        
        # Random order of candidates
        order = candidates[:]
        random.shuffle(order)
        
        # Sometimes try sorted, sometimes reverse sorted
        if restart_count % 3 == 0:
            order = sorted(candidates)
        elif restart_count % 3 == 1:
            order = sorted(candidates, reverse=True)
        
        A, B = greedy_build(order, 0)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_a = A
            best_b = B
            
        # If we hit the theoretical max or close to it, we can stop early?
        # Theoretical max for Sidon in [1,n] is approx sqrt(n).
        # For n=256, sqrt(256)=16. But wait, Sidon sets can be larger.
        # Actually, max size of Sidon set in [1,n] is ~ sqrt(n).
        # However, the constraint is disjoint diffs.
        # Let's just let it run.
        
        # Local search: try to improve by swapping or adding
        # Simple local search: try to add any remaining candidate to A or B
        # Or try to swap an element in A with one outside to free up space?
        # For now, let's stick to random restarts as they are effective for this type of constraint.
        
        # Optimization: If best_score is high, we might break early if time is tight?
        # No, keep searching for better.
        
        # A more sophisticated local search:
        # Try to find an element x not in A or B that can be added to A (if valid) or B (if valid).
        # This is essentially what greedy does, but greedy processes in a fixed order.
        # We can do a second pass:
        # Check all candidates not in A or B.
        # If any can be added to A or B, add it.
        # Repeat until no more can be added.
        
        # Let's implement a "fill" pass after the greedy build
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in candidates:
                if x in A or x in B:
                    continue
                # Try A
                A_new = A + [x]
                if check_sidon(A_new):
                    if B:
                        diff_a = get_diffs(A_new)
                        diff_b = get_diffs(B)
                        if not (diff_a & diff_b):
                            A = A_new
                            changed = True
                            break
                # Try B
                B_new = B + [x]
                if check_sidon(B_new):
                    if A:
                        diff_a = get_diffs(A)
                        diff_b = get_diffs(B_new)
                        if not (diff_a & diff_b):
                            B = B_new
                            changed = True
                            break
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_a = A
            best_b = B

    # Ensure non-empty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
