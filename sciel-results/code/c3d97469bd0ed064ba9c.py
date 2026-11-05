import time
import random
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    N = 256
    # Precompute all 3-term AP triples (a, b, c) with a < b < c and b-a == c-b
    # i.e., a + c == 2b, all in [1, 256]
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > b and c <= N:
                triples.append((a, b, c))
    
    # Greedy initialization: add elements in order of decreasing 1/x
    # (i.e., smallest x first), skipping if it would create a 3-AP with two already selected
    selected = []
    in_set = [False] * (N + 1)
    # To quickly check: for a candidate x, is there a pair (a, b) in selected with 2b - a == x?
    # Equivalently: does x form a 3-AP with two existing elements?
    # For x to be in a 3-AP with two existing elements (u, v) where u < x < v or v < x < u or x < u < v:
    #   Case 1: x is the middle: need a, c in S with a + c = 2x, a < x < c
    #   Case 2: x is the end: need a, b in S with x + b = 2a (x is smaller end) -> a = (x+b)/2, b < a < ... wait
    #   Let's just check: for each pair (u, v) in S with u < v, does 2v - u == x? (x is the larger end)
    #   or 2u - v == x? (x is the smaller end, but x < v so 2u-v < v, and x could be < u)
    #   Actually, the triple (a, b, c) with a<b<c: x could be a, b, or c.
    #   If x = a: need b, c in S with a + c = 2b => x + c = 2b => c = 2b - x, with b > x, c > b
    #   If x = b: need a, c in S with a + c = 2x, a < x < c
    #   If x = c: need a, b in S with a + c = 2b => a = 2b - c = 2b - x, with a < b < x
    
    # Simpler: for candidate x, check all triples involving x
    # Precompute for each x, the list of triples it participates in
    x_to_triples = [[] for _ in range(N + 1)]
    for (a, b, c) in triples:
        x_to_triples[a].append((b, c))
        x_to_triples[b].append((a, c))
        x_to_triples[c].append((a, b))
    
    def can_add(x):
        # Check if adding x would create a 3-AP with two existing elements
        for (u, v) in x_to_triples[x]:
            if in_set[u] and in_set[v]:
                return False
        return True
    
    for x in range(1, N + 1):  # increasing x = decreasing 1/x
        if can_add(x):
            selected.append(x)
            in_set[x] = True
    
    def reciprocal_sum(s):
        return sum(Fraction(1, x) for x in s)
    
    best_set = set(selected)
    best_val = reciprocal_sum(selected)
    
    # Local search: try to improve by swaps (remove one, add one or more)
    # and by single additions after removals
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try single element swaps: remove r, add a
        for r in list(best_set):
            for a in range(1, N + 1):
                if a in best_set or a == r:
                    continue
                # Try: best_set - {r} + {a}
                candidate = best_set - {r}
                candidate.add(a)
                # Validate: check all triples
                valid = True
                for (ta, tb, tc) in triples:
                    if ta in candidate and tb in candidate and tc in candidate:
                        valid = False
                        break
                if not valid:
                    continue
                val = reciprocal_sum(candidate)
                if val > best_val:
                    best_set = candidate
                    best_val = val
                    improved = True
        
        # Try single element additions (if we can add without removing)
        if time.time() < deadline - 5.0:
            for a in range(1, N + 1):
                if a in best_set:
                    continue
                candidate = best_set | {a}
                valid = True
                for (ta, tb, tc) in triples:
                    if ta in candidate and tb in candidate and tc in candidate:
                        valid = False
                        break
                if not valid:
                    continue
                val = reciprocal_sum(candidate)
                if val > best_val:
                    best_set = candidate
                    best_val = val
                    improved = True
        
        # Try two-element improvements: remove r1, r2 and add a1 (net -1 element)
        if time.time() < deadline - 5.0:
            r_list = list(best_set)
            for i in range(len(r_list)):
                for j in range(i + 1, len(r_list)):
                    r1, r2 = r_list[i], r_list[j]
                    for a in range(1, N + 1):
                        if a in best_set or a == r1 or a == r2:
                            continue
                        candidate = best_set - {r1, r2}
                        candidate.add(a)
                        valid = True
                        for (ta, tb, tc) in triples:
                            if ta in candidate and tb in candidate and tc in candidate:
                                valid = False
                                break
                        if not valid:
                            continue
                        val = reciprocal_sum(candidate)
                        if val > best_val:
                            best_set = candidate
                            best_val = val
                            improved = True
                            break
    
    # Final: try to add any remaining elements
    for a in range(1, N + 1):
        if a in best_set:
            continue
        if time.time() > deadline - 2.0:
            break
        candidate = best_set | {a}
        valid = True
        for (ta, tb, tc) in triples:
            if ta in candidate and tb in candidate and tc in candidate:
                valid = False
                break
        if valid:
            val = reciprocal_sum(candidate)
            if val > best_val:
                best_set = candidate
                best_val = val
    
    return sorted(best_set)
