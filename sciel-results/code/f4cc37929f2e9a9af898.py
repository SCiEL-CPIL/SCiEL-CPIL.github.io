"""Strong algorithm for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    N = 128
    universe = list(range(1, N + 1))
    
    # Build list of all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                d_ac = gcd(a, c)
                if d_ac != d_ab:
                    continue
                d_bc = gcd(b, c)
                if d_bc == d_ab:
                    bad_triples.append((a, b, c))
    
    print(f"Total bad triples: {len(bad_triples)}", flush=True)
    
    # Precompute partner_pairs: for each x, list of (y,z) such that {x,y,z} is a bad triple
    partner_pairs = defaultdict(list)
    elem_triples = defaultdict(list)  # for each x, list of triple indices
    for idx, (a, b, c) in enumerate(bad_triples):
        partner_pairs[a].append((b, c))
        partner_pairs[b].append((a, c))
        partner_pairs[c].append((a, b))
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    # Triple count per element
    triple_count = defaultdict(int)
    for (a, b, c) in bad_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    def greedy_construct(order):
        S = set()
        S.add(order[0])
        for x in order[1:]:
            valid = True
            for (y, z) in partner_pairs[x]:
                if y in S and z in S:
                    valid = False
                    break
            if valid:
                S.add(x)
        return S
    
    def is_valid_set(S):
        for (a, b, c) in bad_triples:
            if a in S and b in S and c in S:
                return False
        return True
    
    def local_search_with_counts(S):
        S = set(S)
        n_triples = len(bad_triples)
        triple_counts = [0] * n_triples
        
        for idx, (a, b, c) in enumerate(bad_triples):
            cnt = (1 if a in S else 0) + (1 if b in S else 0) + (1 if c in S else 0)
            triple_counts[idx] = cnt
        
        not_in_S = set(universe) - S
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in list(S):
                for idx in elem_triples[x]:
                    triple_counts[idx] -= 1
                
                for y in not_in_S:
                    valid = True
                    for idx in elem_triples[y]:
                        if triple_counts[idx] >= 2:
                            valid = False
                            break
                    if valid:
                        for idx in elem_triples[y]:
                            triple_counts[idx] += 1
                        S = (S - {x}) | {y}
                        not_in_S = set(universe) - S
                        improved = True
                        break
                
                if not improved:
                    for idx in elem_triples[x]:
                        triple_counts[idx] += 1
                else:
                    break
            
            if not improved:
                break
        
        return S
    
    # Multiple greedy orderings
    best_set = set()
    
    order1 = sorted(universe, key=lambda x: (triple_count[x], x))
    best_set = greedy_construct(order1)
    
    order2 = sorted(universe, key=lambda x: (-triple_count[x], x))
    s2 = greedy_construct(order2)
    if len(s2) > len(best_set):
        best_set = s2
    
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > deadline - 15:
            break
        order = universe[:]
        rng.shuffle(order)
        s = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
    
    print(f"After greedy: {len(best_set)}", flush=True)
    
    # Local search
    if time.time() < deadline - 10:
        S_ls = local_search_with_counts(best_set)
        if len(S_ls) > len(best_set):
            best_set = S_ls
            print(f"After local search: {len(best_set)}", flush=True)
    
    # Random restarts with local search
    rng2 = random.Random(123)
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        order = universe[:]
        rng2.shuffle(order)
        S = greedy_construct(order)
        S = local_search_with_counts(S)
        if len(S) > len(best_set):
            best_set = S
            print(f"Restart {restart}: {len(best_set)}", flush=True)
    
    # Final: try adding any remaining elements
    for x in universe:
        if time.time() > deadline - 5:
            break
        if x not in best_set:
            valid = True
            for (y, z) in partner_pairs[x]:
                if y in best_set and z in best_set:
                    valid = False
                    break
            if valid:
                best_set.add(x)
    
    # Verify
    assert is_valid_set(best_set), "Invalid set returned!"
    assert len(best_set) > 0, "Empty set!"
    
    print(f"Final size: {len(best_set)}", flush=True)
    return list(best_set)

# EVOLVE-BLOCK-END
