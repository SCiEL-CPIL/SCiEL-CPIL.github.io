"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute gcd for all pairs in [1, 128]
    max_n = 128
    gcd_table = {}
    for i in range(1, max_n + 1):
        for j in range(i + 1, max_n + 1):
            gcd_table[(i, j)] = gcd(i, j)
    
    def get_gcd(a, b):
        if a > b:
            a, b = b, a
        return gcd_table.get((a, b), 1)
    
    # Start with all elements
    current_set = set(range(1, max_n + 1))
    
    def find_conflicts(s):
        """Find all triples (a, b, c) in s where gcd(a,b) == gcd(a,c) == gcd(b,c)"""
        conflicts = []
        elements = list(s)
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = elements[i], elements[j], elements[k]
                    g_ab = get_gcd(a, b)
                    g_ac = get_gcd(a, c)
                    g_bc = get_gcd(b, c)
                    if g_ab == g_ac == g_bc:
                        conflicts.append((a, b, c))
        return conflicts
    
    def is_valid(s):
        """Check if set s has no equal-gcd triples"""
        elements = list(s)
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = elements[i], elements[j], elements[k]
                    g_ab = get_gcd(a, b)
                    g_ac = get_gcd(a, c)
                    g_bc = get_gcd(b, c)
                    if g_ab == g_ac == g_bc:
                        return False
        return True
    
    # Greedy removal: repeatedly remove the element that resolves the most conflicts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        conflicts = find_conflicts(current_set)
        if not conflicts:
            break
        
        # Count how many conflicts each element is involved in
        conflict_count = {}
        for a, b, c in conflicts:
            for x in (a, b, c):
                conflict_count[x] = conflict_count.get(x, 0) + 1
        
        # Find the element with the most conflicts
        if conflict_count:
            max_conflicts = max(conflict_count.values())
            candidates = [x for x, cnt in conflict_count.items() if cnt == max_conflicts]
            # Remove one of the candidates (choose the smallest for determinism)
            to_remove = min(candidates)
            current_set.discard(to_remove)
            improved = True
    
    # Try to add back removed elements if possible
    removed = set(range(1, max_n + 1)) - current_set
    for x in sorted(removed, reverse=True):
        if time.time() > deadline:
            break
        trial_set = current_set | {x}
        if is_valid(trial_set):
            current_set = trial_set
    
    # Ensure nonempty
    if not current_set:
        return {1}
    
    return current_set
# EVOLVE-BLOCK-END
