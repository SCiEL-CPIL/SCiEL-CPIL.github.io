import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs and Conflicts
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # conflicts[v] is a set of frozensets {u, w} such that {v, u, w} is a bad triple
    # A triple {a,b,c} is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    conflicts = [set() for _ in range(N + 1)]
    
    # Also precompute the set of all bad triples for potential use
    bad_triples = []
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))
            bad_triples.append((a, b, c))
    
    # Precompute degrees (number of pairs that conflict with v)
    degrees = [len(conflicts[v]) for v in range(N + 1)]
    
    # Precompute conflict counts: for each pair {u,v}, how many w have {u,v} in conflicts[w]
    # This helps quickly check if adding v1,v2 would create conflicts
    pair_conflict_count = {}
    for w in range(1, N + 1):
        for p in conflicts[w]:
            pair_conflict_count[p] = pair_conflict_count.get(p, 0) + 1
    
    best_size = 0
    best_set = set()
    
    # Helper function to check if a set is valid
    def is_valid(S):
        S_list = list(S)
        for a, b, c in combinations(S_list, 3):
            if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                return False
        return True

    # Improved Greedy + Local Search
    # We will use multiple restarts with different random seeds/orders
    
    num_restarts = 80
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Strategy: Start with a random subset of "easy" numbers (low degree)
        # Or start with a greedy construction.
        
        # Let's try a greedy construction first, then local search.
        # Order: Randomize, but bias towards low degree?
        # Or just random shuffle?
        
        candidates = numbers[:]
        rng.shuffle(candidates)
        
        # Greedy Add
        S = set()
        active_pairs = set()
        
        def can_add(v, S, active_pairs):
            for p in conflicts[v]:
                if p in active_pairs:
                    return False
            return True

        def add(v):
            S.add(v)
            # Update active_pairs
            for u in S:
                if u != v:
                    p = frozenset((u, v))
                    active_pairs.add(p)

        def remove(v):
            # Update active_pairs
            for u in S:
                if u != v:
                    p = frozenset((u, v))
                    if p in active_pairs:
                        active_pairs.remove(p)
            S.remove(v)

        # Phase 1: Greedy Initialization
        # We can try different orders.
        # Let's try sorting by degree ascending, then shuffling ties?
        # Or just random.
        
        # Let's use a "weighted" random order based on degree?
        # Lower degree is better.
        # Let's just shuffle for now, maybe try sorted by degree in some restarts.
        
        if restart % 3 == 0:
            candidates = sorted(numbers, key=lambda x: (degrees[x], rng.random()))
        elif restart % 3 == 1:
            rng.shuffle(candidates)
        else:
            # Try a mixed strategy: half sorted, half random
            half = numbers[:len(numbers)//2]
            other_half = numbers[len(numbers)//2:]
            half.sort(key=lambda x: degrees[x])
            rng.shuffle(other_half)
            candidates = half + other_half

        for v in candidates:
            if can_add(v, S, active_pairs):
                add(v)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = set(S)
        
        # Phase 2: Local Search
        # 1. Add any valid element
        # 2. Try 1-2 swap (remove 1, add 2)
        # 3. Try 2-2 swap (remove 2, add 2) for structural improvement
        # 4. Try 2-3 swap (remove 2, add 3) for larger gains
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try to add
            remaining = set(numbers) - S
            for v in remaining:
                if can_add(v, S, active_pairs):
                    add(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-2 swap
            found_swap = False
            for u in list(S):
                if time.time() > deadline - 2:
                    break
                
                remove(u)
                
                valid_v = []
                for v in remaining:
                    if can_add(v, S, active_pairs):
                        valid_v.append(v)
                
                # Check pairs in valid_v
                if len(valid_v) >= 2:
                    # Try to add two that don't conflict with each other
                    added = False
                    for i in range(len(valid_v)):
                        for j in range(i + 1, len(valid_v)):
                            v1 = valid_v[i]
                            v2 = valid_v[j]
                            p12 = frozenset((v1, v2))
                            
                            # Check if p12 conflicts with any w in S
                            conflict = False
                            for w in S:
                                if p12 in conflicts[w]:
                                    conflict = True
                                    break
                            
                            if not conflict:
                                # Add v1 and v2
                                add(v1)
                                add(v2)
                                found_swap = True
                                improved = True
                                added = True
                                break
                        if added:
                            break
                    
                    if added:
                        break
                
                if not found_swap:
                    # Restore u
                    add(u)
            
            if improved:
                continue
            
            # Try 2-2 swap: remove u1, u2, add v1, v2 (size same, but better structure)
            # This can help escape local optima by changing the structure
            found_22_swap = False
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 2:
                    break
                u1 = S_list[i]
                for j in range(i + 1, len(S_list)):
                    u2 = S_list[j]
                    
                    remove(u1)
                    remove(u2)
                    
                    valid_v = []
                    for v in remaining:
                        if can_add(v, S, active_pairs):
                            valid_v.append(v)
                    
                    if len(valid_v) >= 2:
                        added = False
                        for a in range(len(valid_v)):
                            for b in range(a + 1, len(valid_v)):
                                v1 = valid_v[a]
                                v2 = valid_v[b]
                                p12 = frozenset((v1, v2))
                                
                                conflict = False
                                for w in S:
                                    if p12 in conflicts[w]:
                                        conflict = True
                                        break
                                
                                if not conflict:
                                    add(v1)
                                    add(v2)
                                    found_22_swap = True
                                    added = True
                                    break
                            if added:
                                break
                        
                        if not added:
                            # Restore u1, u2
                            add(u1)
                            add(u2)
                    
                    if found_22_swap:
                        improved = True
                        break
                    
                    # Restore u1, u2 if not added
                    if not added:
                        add(u1)
                        add(u2)
                
                if found_22_swap:
                    break
            
            if improved:
                continue
            
            # Try 2-3 swap: remove u1, u2, add v1, v2, v3 (size +1)
            found_23_swap = False
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 2:
                    break
                u1 = S_list[i]
                for j in range(i + 1, len(S_list)):
                    u2 = S_list[j]
                    
                    remove(u1)
                    remove(u2)
                    
                    valid_v = []
                    for v in remaining:
                        if can_add(v, S, active_pairs):
                            valid_v.append(v)
                    
                    if len(valid_v) >= 3:
                        added = False
                        # Try to find 3 that can be added together
                        for a in range(len(valid_v)):
                            for b in range(a + 1, len(valid_v)):
                                for c in range(b + 1, len(valid_v)):
                                    v1, v2, v3 = valid_v[a], valid_v[b], valid_v[c]
                                    
                                    # Check all pairs among v1,v2,v3 against S
                                    ok = True
                                    for pair in [frozenset((v1,v2)), frozenset((v1,v3)), frozenset((v2,v3))]:
                                        for w in S:
                                            if pair in conflicts[w]:
                                                ok = False
                                                break
                                        if not ok:
                                            break
                                    
                                    if ok:
                                        add(v1)
                                        add(v2)
                                        add(v3)
                                        found_23_swap = True
                                        added = True
                                        break
                                if added:
                                    break
                            if added:
                                break
                        
                        if not added:
                            add(u1)
                            add(u2)
                    
                    if found_23_swap:
                        improved = True
                        break
                    
                    if not added:
                        add(u1)
                        add(u2)
                
                if found_23_swap:
                    break
            
            if improved:
                continue
            
            # If stuck, break and do new restart
            break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    return list(best_set)
