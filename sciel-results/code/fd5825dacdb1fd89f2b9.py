"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, n+1))
    
    # For fast conflict checking, precompute for each pair (x,y) whether they form a bad triple with some z
    # Actually, for local search we need: given a set S, check if adding x creates a bad triple.
    # A bad triple involving x and two existing elements a,b in S means lcm(x,a)==lcm(x,b)==lcm(a,b).
    
    # Precompute: for each element x, and each pair (a,b) where a<b, does {x,a,b} form a bad triple?
    # Better: for each element x, precompute the set of pairs (a,b) such that {x,a,b} is a bad triple.
    # Then when checking if x can be added to S, we check if any such pair (a,b) is fully in S.
    
    # For each x, store list of (a,b) pairs that with x form a bad triple
    x_conflict_pairs = {x: [] for x in range(1, n+1)}
    for (a, b, c) in bad_triples:
        x_conflict_pairs[a].append((b, c))
        x_conflict_pairs[b].append((a, c))
        x_conflict_pairs[c].append((a, b))
    
    # Count how many bad triples each element is in
    elem_triple_count = [0]*(n+1)
    for (a, b, c) in bad_triples:
        elem_triple_count[a] += 1
        elem_triple_count[b] += 1
        elem_triple_count[c] += 1
    
    def is_valid_set(S):
        """Check if set S has no bad triple."""
        S_set = set(S)
        for (a, b, c) in bad_triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True
    
    def count_bad_triples_in(S):
        """Count bad triples fully contained in S."""
        S_set = set(S)
        count = 0
        for (a, b, c) in bad_triples:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
    
    def can_add(S_set, x):
        """Check if adding x to S_set creates any bad triple."""
        for (a, b) in x_conflict_pairs[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    def try_add(S_list, S_set, x):
        """Try to add x to the set. Returns True if successful."""
        if x in S_set:
            return False
        if can_add(S_set, x):
            S_list.append(x)
            S_set.add(x)
            return True
        return False
    
    def try_remove(S_list, S_set, x):
        """Remove x from the set."""
        if x in S_set:
            S_set.discard(x)
            S_list.remove(x)
            return True
        return False
    
    # Strategy: Start with a good seed, then do local search with 1-swap and 2-swap moves.
    
    # Seed: use elements from [65, 128] (large numbers tend to have fewer LCM conflicts)
    # Plus some well-chosen smaller elements
    
    # First, try greedy construction from the large range
    random.seed(42)
    
    best_set = None
    best_size = 0
    
    # Multiple random restarts with different seeds
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Deterministic seed: large range
            candidates = list(range(65, 129))
        elif restart == 1:
            # Large range + some smaller
            candidates = list(range(65, 129)) + list(range(33, 65))
        elif restart == 2:
            # Random order
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
        elif restart == 3:
            # Sorted by descending triple count (least constrained first)
            candidates = list(range(1, n+1))
            candidates.sort(key=lambda x: -elem_triple_count[x])
        elif restart == 4:
            # Sorted by ascending triple count
            candidates = list(range(1, n+1))
            candidates.sort(key=lambda x: elem_triple_count[x])
        elif restart < 10:
            # Random subsets of large range
            large_range = list(range(65, 129))
            random.shuffle(large_range)
            candidates = large_range + [x for x in range(1, 65) if random.random() < 0.3]
        else:
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
        
        # Greedy construction
        S_list = []
        S_set = set()
        for x in candidates:
            if time.time() > deadline:
                break
            if try_add(S_list, S_set, x):
                pass
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)
        
        # Local search: 1-swap moves (replace one element with another)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_set = set(best_set)
            
            # Try 1-swap: remove x from S, add y not in S
            for x in list(best_set):
                if time.time() > deadline:
                    break
                # Try removing x
                new_S_set = S_set - {x}
                for y in range(1, n+1):
                    if y in S_set:
                        continue
                    if can_add(new_S_set, y):
                        # This swap is valid
                        new_set = [z for z in best_set if z != x] + [y]
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            S_set = set(new_set)
                            improved = True
                            break
                if improved:
                    break
            
            # Try 2-swap (kick): remove x1, x2 from S, add y1, y2 not in S
            if not improved and time.time() < deadline:
                for i in range(len(best_set)):
                    if time.time() > deadline:
                        break
                    x1 = best_set[i]
                    for j in range(i+1, len(best_set)):
                        if time.time() > deadline:
                            break
                        x2 = best_set[j]
                        new_S_set = S_set - {x1, x2}
                        
                        # Find two elements to add
                        added = []
                        for y in range(1, n+1):
                            if y in S_set or y in added:
                                continue
                            if can_add(new_S_set | set(added), y):
                                added.append(y)
                                if len(added) == 2:
                                    break
                        
                        if len(added) == 2:
                            new_set = [z for z in best_set if z != x1 and z != x2] + added
                            if len(new_set) > best_size:
                                best_set = new_set
                                best_size = len(new_set)
                                S_set = set(new_set)
                                improved = True
                                break
                    if improved:
                        break
        
        # Also try adding elements greedily to the current best
        if time.time() < deadline:
            S_set = set(best_set)
            for x in range(1, n+1):
                if time.time() > deadline:
                    break
                if x not in S_set and can_add(S_set, x):
                    best_set.append(x)
                    S_set.add(x)
                    best_size = len(best_set)
    
    # Final: ensure we have at least one element
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
