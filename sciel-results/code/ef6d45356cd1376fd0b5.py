import time
import random
from collections import defaultdict

def is_valid_set(S):
    """Check if S is valid: no element divides sum of any 3 distinct others."""
    S_list = list(S)
    n = len(S_list)
    if n < 4:
        return True
    for a in S_list:
        others = [x for x in S_list if x != a]
        m = len(others)
        if m < 3:
            continue
        residue_count = defaultdict(int)
        for x in others:
            residue_count[x % a] += 1
        for i in range(m):
            b = others[i]
            for j in range(i + 1, m):
                c = others[j]
                need = (-b - c) % a
                cnt = residue_count.get(need, 0)
                avail = cnt
                if b % a == need:
                    avail -= 1
                if c % a == need:
                    avail -= 1
                if avail > 0:
                    return False
    return True

def can_add(S, x):
    """Check if x can be added to valid set S."""
    S_list = list(S)
    n = len(S_list)
    # Check a = x: no triple in S sums to multiple of x
    if n >= 3:
        residue_count = defaultdict(int)
        for v in S_list:
            residue_count[v % x] += 1
        m = n
        for i in range(m):
            b = S_list[i]
            for j in range(i + 1, m):
                c = S_list[j]
                need = (-b - c) % x
                cnt = residue_count.get(need, 0)
                avail = cnt
                if b % x == need:
                    avail -= 1
                if c % x == need:
                    avail -= 1
                if avail > 0:
                    return False
    # Check for each a in S: no pair (b,c) in S\{a} with (b+c+x) % a == 0
    for a in S_list:
        others = [v for v in S_list if v != a]
        m = len(others)
        if m < 2:
            continue
        for i in range(m):
            b = others[i]
            for j in range(i + 1, m):
                c = others[j]
                if (b + c + x) % a == 0:
                    return False
    return True

def greedy_build(order, max_size=48):
    """Build a set greedily in the given order."""
    selected = []
    sel_set = set()
    for x in order:
        if len(selected) >= max_size:
            break
        if can_add(sel_set, x):
            selected.append(x)
            sel_set.add(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # Safety margin
    
    best_set = []
    best_len = 0
    
    # Multiple initial strategies
    strategies = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        list(range(128, 0, -1)),
        list(range(192, 64, -1)) + list(range(64, 0, -1)),
    ]
    
    random.seed(42)
    for _ in range(40):
        order = list(range(1, 193))
        random.shuffle(order)
        strategies.append(order)
    
    for order in strategies:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s
    
    # Local search to improve
    current_set = set(best_set)
    full_range = set(range(1, 193))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        complement = full_range - current_set
        
        # Try single additions (most impactful)
        for y in sorted(complement):
            if time.time() > deadline:
                break
            if len(current_set) >= 48:
                break
            if can_add(current_set, y):
                current_set.add(y)
                best_set = list(current_set)
                best_len = len(current_set)
                improved = True
                break
        
        if improved:
            continue
        
        # Try remove-1-add-2 moves (same size, may unblock further additions)
        comp_list = list(complement)
        random.shuffle(comp_list)
        
        for x in list(current_set):
            if time.time() > deadline:
                break
            S_minus_x = current_set - {x}
            
            found = False
            for i in range(min(50, len(comp_list))):
                y1 = comp_list[i]
                if not can_add(S_minus_x, y1):
                    continue
                S_temp = S_minus_x | {y1}
                for j in range(min(50, len(comp_list))):
                    if i == j:
                        continue
                    y2 = comp_list[j]
                    if can_add(S_temp, y2):
                        new_set = S_temp | {y2}
                        # Try to extend further
                        comp2 = full_range - new_set
                        for y3 in sorted(comp2):
                            if time.time() > deadline:
                                break
                            if len(new_set) >= 48:
                                break
                            if can_add(new_set, y3):
                                new_set.add(y3)
                                break
                        if len(new_set) > len(best_set):
                            current_set = new_set
                            best_set = list(current_set)
                            best_len = len(current_set)
                            improved = True
                            found = True
                            break
                if found:
                    break
            if found:
                break
    
    # Final verification
    if not is_valid_set(set(best_set)):
        # Fallback: rebuild from scratch with a simple strategy
        best_set = greedy_build(list(range(192, 0, -1)))
    
    return best_set
