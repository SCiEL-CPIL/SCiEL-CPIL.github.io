import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 192
    MAX_SIZE = 48
    
    def is_valid_set(S):
        """Check if S is valid: no element divides sum of any 3 distinct others."""
        S_list = list(S)
        n = len(S_list)
        if n < 4:
            return True
        for idx_a, a in enumerate(S_list):
            others = [S_list[i] for i in range(n) if i != idx_a]
            m = len(others)
            if m < 3:
                continue
            residue_set = {}
            for x in others:
                r = x % a
                residue_set[r] = residue_set.get(r, 0) + 1
            for i in range(m):
                b = others[i]
                for j in range(i + 1, m):
                    c = others[j]
                    need = (-b - c) % a
                    cnt = residue_set.get(need, 0)
                    if b % a == need:
                        cnt -= 1
                    if c % a == need:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to valid set S. Optimized version."""
        S_list = list(S)
        n = len(S_list)
        
        # Check a = x: no triple in S sums to multiple of x
        if n >= 3:
            residue_count = {}
            for v in S_list:
                r = v % x
                residue_count[r] = residue_count.get(r, 0) + 1
            for i in range(n):
                b = S_list[i]
                rb = b % x
                for j in range(i + 1, n):
                    c = S_list[j]
                    rc = c % x
                    need = (-rb - rc) % x
                    cnt = residue_count.get(need, 0)
                    if rb == need:
                        cnt -= 1
                    if rc == need:
                        cnt -= 1
                    if cnt > 0:
                        return False
        
        # Check for each a in S: no pair (b,c) in S\{a} with (b+c+x) % a == 0
        for a in S_list:
            ra = x % a
            others = [v for v in S_list if v != a]
            m = len(others)
            if m < 2:
                continue
            # For each pair (b,c) in others, check if (b+c+x) % a == 0
            # i.e., (b%a + c%a + ra) % a == 0
            residues = {}
            for v in others:
                r = v % a
                residues[r] = residues.get(r, 0) + 1
            for i in range(m):
                b = others[i]
                rb = b % a
                for j in range(i + 1, m):
                    c = others[j]
                    rc = c % a
                    if (rb + rc + ra) % a == 0:
                        return False
        return True
    
    def greedy_build(order, max_size=MAX_SIZE, time_limit=None):
        """Build a set greedily in the given order."""
        if time_limit is None:
            time_limit = deadline
        selected = []
        sel_set = set()
        for x in order:
            if time.time() > time_limit:
                break
            if len(selected) >= max_size:
                break
            if can_add(sel_set, x):
                selected.append(x)
                sel_set.add(x)
        return selected
    
    def local_search(current_set, time_limit=None):
        """Try to improve the set via additions and swaps."""
        if time_limit is None:
            time_limit = deadline
        current_set = set(current_set)
        full_range = set(range(1, N + 1))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            complement = full_range - current_set
            
            # Try single additions
            for y in sorted(complement):
                if time.time() > time_limit:
                    break
                if len(current_set) >= MAX_SIZE:
                    break
                if can_add(current_set, y):
                    current_set.add(y)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try remove-1-add-1 swaps
            comp_list = list(complement)
            random.shuffle(comp_list)
            
            found_swap = False
            for x in list(current_set):
                if time.time() > time_limit:
                    break
                S_minus_x = current_set - {x}
                
                for y in comp_list:
                    if y in S_minus_x:
                        continue
                    new_set = S_minus_x | {y}
                    if can_add(S_minus_x, y):
                        # Check if we can extend further
                        comp2 = full_range - new_set
                        extended = False
                        for y2 in sorted(comp2):
                            if time.time() > time_limit:
                                break
                            if len(new_set) >= MAX_SIZE:
                                break
                            if can_add(new_set, y2):
                                new_set.add(y2)
                                extended = True
                                break
                        if len(new_set) >= len(current_set):
                            current_set = new_set
                            found_swap = True
                            break
                if found_swap:
                    break
            
            # Try remove-1-add-2 moves
            if not found_swap:
                comp_list = list(complement)
                random.shuffle(comp_list)
                
                for x in list(current_set):
                    if time.time() > time_limit:
                        break
                    S_minus_x = current_set - {x}
                    
                    found = False
                    for y1 in comp_list[:40]:
                        if time.time() > time_limit:
                            break
                        if not can_add(S_minus_x, y1):
                            continue
                        S_temp = S_minus_x | {y1}
                        for y2 in comp_list[:40]:
                            if y1 == y2:
                                continue
                            if can_add(S_temp, y2):
                                new_set = S_temp | {y2}
                                # Try to extend further
                                comp2 = full_range - new_set
                                for y3 in sorted(comp2):
                                    if time.time() > time_limit:
                                        break
                                    if len(new_set) >= MAX_SIZE:
                                        break
                                    if can_add(new_set, y3):
                                        new_set.add(y3)
                                        break
                                if len(new_set) > len(current_set):
                                    current_set = new_set
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
            
            improved = found_swap or found
        
        return current_set
    
    def final_polish(current_set, time_limit=None):
        """Final pass: try all single swaps and small perturbations."""
        if time_limit is None:
            time_limit = deadline
        current_set = set(current_set)
        full_range = set(range(1, N + 1))
        
        # Try each single swap
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            complement = full_range - current_set
            
            for x in list(current_set):
                if time.time() > time_limit:
                    break
                S_minus_x = current_set - {x}
                
                for y in sorted(complement):
                    if time.time() > time_limit:
                        break
                    new_set = S_minus_x | {y}
                    if can_add(S_minus_x, y):
                        # Try to extend
                        comp2 = full_range - new_set
                        for y2 in sorted(comp2):
                            if time.time() > time_limit:
                                break
                            if len(new_set) >= MAX_SIZE:
                                break
                            if can_add(new_set, y2):
                                new_set.add(y2)
                                improved = True
                                break
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            break
                if improved:
                    break
        
        return current_set
    
    # Strategy: Start with large numbers (they're less restrictive)
    best_set = []
    best_len = 0
    
    # Try multiple initial orderings
    random.seed(42)
    
    # Ordering 1: Descending (large numbers first)
    s = greedy_build(list(range(N, 0, -1)))
    s = local_search(set(s))
    s = final_polish(s)
    if len(s) > best_len:
        best_len = len(s)
        best_set = list(s)
    
    # Ordering 2: Random shuffles
    for trial in range(15):
        if time.time() > deadline - 8:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_build(order)
        s = local_search(set(s))
        s = final_polish(s)
        if len(s) > best_len:
            best_len = len(s)
            best_set = list(s)
    
    # Ordering 3: Focus on upper range
    upper = list(range(N, N // 2, -1))  # 192 down to 97
    lower = list(range(N // 2, 0, -1))  # 96 down to 1
    s = greedy_build(upper + lower)
    s = local_search(set(s))
    s = final_polish(s)
    if len(s) > best_len:
        best_len = len(s)
        best_set = list(s)
    
    # Ordering 4: Numbers with few small divisors (primes first)
    def score_num(x):
        if x == 1:
            return -1000
        # Count divisors
        div_count = 0
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                div_count += 2
        return -div_count
    
    order4 = sorted(range(1, N + 1), key=score_num, reverse=True)
    s = greedy_build(order4)
    s = local_search(set(s))
    s = final_polish(s)
    if len(s) > best_len:
        best_len = len(s)
        best_set = list(s)
    
    # Ordering 5: Mix of large primes and composite numbers
    primes = [x for x in range(1, N + 1) if x > 1 and all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    composites = [x for x in range(1, N + 1) if x not in primes and x > 1]
    order5 = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    s = greedy_build(order5)
    s = local_search(set(s))
    s = final_polish(s)
    if len(s) > best_len:
        best_len = len(s)
        best_set = list(s)
    
    # Final verification
    if not is_valid_set(set(best_set)):
        # Fallback
        best_set = greedy_build(list(range(N, 0, -1)))
    
    return best_set[:MAX_SIZE]
