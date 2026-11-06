"""Improved program: greedy + local search with incremental validity checks."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Returns a list of distinct positive integers <= 192 (max 48) such that
    no selected integer divides the sum of any 3 distinct OTHER selected integers.
    Maximizes cardinality.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    def is_valid_addition(S, x):
        """
        Check if adding x to a valid set S keeps it valid.
        S is assumed to be a valid set. x is not in S.
        Only checks new constraints introduced by x:
        1. For each a in S: does x + b + c ≡ 0 (mod a) for some distinct b,c in S\{a}?
        2. For x itself: does any triple in S sum to a multiple of x?
        """
        S_set = set(S)
        # Check 1: x as part of triple for each existing element a
        for a in S:
            others = [v for v in S if v != a]
            # Need to check if there exist distinct b, c in others such that (x + b + c) % a == 0
            # i.e., (b + c) % a == (-x) % a
            target = (-x) % a
            # Count pairs (b, c) in others with (b + c) % a == target
            # Use residue counting for efficiency
            residue_count = {}
            for v in others:
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            
            # For each b in others, check if there exists c in others (c != b) with (b + c) % a == target
            # i.e., c % a == (target - b) % a
            found = False
            for b in others:
                needed = (target - b) % a
                cnt = residue_count.get(needed, 0)
                # Subtract 1 if b itself has the needed residue
                if b % a == needed:
                    cnt -= 1
                if cnt > 0:
                    found = True
                    break
            if found:
                return False
        
        # Check 2: x as the tested element - any triple in S sums to multiple of x
        if len(S) >= 3:
            for triple in combinations(S, 3):
                if sum(triple) % x == 0:
                    return False
        
        return True
    
    def is_valid_full(S):
        """Full validation of set S. Used sparingly for verification."""
        S_list = list(S)
        n = len(S_list)
        for a in S_list:
            others = [x for x in S_list if x != a]
            m = len(others)
            if m < 3:
                continue
            # Check if any triple in others sums to multiple of a
            residue_count = {}
            for v in others:
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            
            for i in range(m):
                b = others[i]
                for j in range(i + 1, m):
                    c = others[j]
                    target = (-b - c) % a
                    cnt = residue_count.get(target, 0)
                    if b % a == target:
                        cnt -= 1
                    if c % a == target:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    def greedy_construct(order):
        """Greedy construction: add numbers in given order if valid."""
        S = []
        for x in order:
            if x in S:
                continue
            if is_valid_addition(S, x):
                S.append(x)
            if len(S) >= MAX_K:
                break
        return S
    
    def local_search(S, remaining_time):
        """Local search: try to improve by adding or swapping elements."""
        S_set = set(S)
        all_vals = set(range(1, N + 1))
        
        while time.time() - start_time < remaining_time:
            improved = False
            
            # Try to add any number not in S
            for x in range(N, 0, -1):
                if x in S_set:
                    continue
                if is_valid_addition(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add one (only if it increases size, which it won't,
            # so this is for diversification)
            # Actually, for maximizing cardinality, we want to find configurations where
            # we can add more. Let's try: remove one element, then try to add two.
            
            # Try 1-swap: remove x, try to add y (same size, but different composition)
            # This can enable further additions
            swapped = False
            for i, x in enumerate(list(S)):
                if time.time() - start_time > remaining_time:
                    break
                S_without = [v for j, v in enumerate(S) if j != i]
                S_without_set = set(S_without)
                for y in range(N, 0, -1):
                    if y in S_without_set or y == x:
                        continue
                    if is_valid_addition(S_without, y):
                        # Check if this swap enables further additions
                        S_candidate = S_without + [y]
                        S_candidate_set = set(S_candidate)
                        # Try to add more
                        extra_added = 0
                        for z in range(N, 0, -1):
                            if z in S_candidate_set:
                                continue
                            if is_valid_addition(S_candidate, z):
                                S_candidate.append(z)
                                S_candidate_set.add(z)
                                extra_added += 1
                                if len(S_candidate) >= MAX_K:
                                    break
                        if extra_added > 0 or (len(S_candidate) == len(S) and y > x):
                            # Prefer if it adds more, or same size but larger value (for diversity)
                            if extra_added > 0:
                                S = S_candidate
                                S_set = set(S)
                                improved = True
                                swapped = True
                                break
                if swapped:
                    break
            
            if improved:
                continue
            
            # Try 2-remove, 2-add moves
            found_two_swap = False
            S_list = list(S)
            n = len(S_list)
            for i in range(n):
                if time.time() - start_time > remaining_time:
                    break
                for j in range(i + 1, n):
                    if time.time() - start_time > remaining_time:
                        break
                    S_without = [v for k, v in enumerate(S_list) if k != i and k != j]
                    S_without_set = set(S_without)
                    removed = {S_list[i], S_list[j]}
                    # Try to add two new elements
                    added = []
                    for y in range(N, 0, -1):
                        if y in S_without_set or y in removed:
                            continue
                        if is_valid_addition(S_without + added, y):
                            added.append(y)
                            S_without_set.add(y)
                            if len(added) == 2:
                                break
                    if len(added) == 2:
                        S_candidate = S_without + added
                        S_candidate_set = set(S_candidate)
                        # Try to add even more
                        for z in range(N, 0, -1):
                            if z in S_candidate_set:
                                continue
                            if is_valid_addition(S_candidate, z):
                                S_candidate.append(z)
                                S_candidate_set.add(z)
                                if len(S_candidate) >= MAX_K:
                                    break
                        if len(S_candidate) > len(S):
                            S = S_candidate
                            S_set = set(S)
                            improved = True
                            found_two_swap = True
                            break
                if found_two_swap:
                    break
            
            if not improved:
                # No improvement found, try randomized restart
                break
        
        return S
    
    # Phase 1: Greedy construction with multiple orders
    best_S = []
    best_len = 0
    
    # Descending order
    S1 = greedy_construct(range(N, 0, -1))
    if len(S1) > best_len:
        best_S = S1
        best_len = len(S1)
    
    # Ascending order
    if time.time() - start_time < DEADLINE:
        S2 = greedy_construct(range(1, N + 1))
        if len(S2) > best_len:
            best_S = S2
            best_len = len(S2)
    
    # Random orders
    random.seed(42)
    for trial in range(200):
        if time.time() - start_time > DEADLINE - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S3 = greedy_construct(order)
        if len(S3) > best_len:
            best_S = S3
            best_len = len(S3)
    
    # Phase 2: Local search on best solution
    if time.time() - start_time < DEADLINE:
        best_S = local_search(best_S, DEADLINE - 5)
    
    # Phase 3: More random restarts with local search
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S4 = greedy_construct(order)
        S4 = local_search(S4, DEADLINE - 10)
        if len(S4) > best_len:
            best_S = S4
            best_len = len(S4)
    
    # Final verification
    if not is_valid_full(best_S):
        # Fallback: rebuild from scratch with simple greedy
        best_S = greedy_construct(range(N, 0, -1))
    
    return best_S

# EVOLVE-BLOCK-END
