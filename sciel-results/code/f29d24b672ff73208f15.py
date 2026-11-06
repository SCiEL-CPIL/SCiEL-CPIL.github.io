import time
import random
from itertools import combinations, product

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def all_triples():
        return list(combinations(range(16), 3))
    
    def check_valid(triples_set, triples_list):
        """Check if adding a new triple to triples_set would create a K(2,2,2)."""
        # For the new triple {a,b,c}, check all ways it could be a transversal triple
        # A transversal triple for pairs (p1,p2),(p3,p4),(p5,p6) is {x,y,z} with x in pair1, y in pair2, z in pair3
        # So for triple {a,b,c}, we need to check: are there disjoint pairs containing a, b, c respectively
        # such that all 8 transversal triples are present?
        
        a, b, c = triples_list[-1] if triples_list else (0,0,0)
        # Actually, let me restructure: check if the full set is valid
        pass
    
    def is_k222_free(triples_set):
        """Check if the hypergraph is K(2,2,2)-free."""
        # For every three disjoint pairs, check that not all 8 transversal triples are present
        # This is expensive, so we use it only for final verification
        for p1 in combinations(range(16), 2):
            for p2 in combinations(range(16), 2):
                if set(p1) & set(p2):
                    continue
                for p3 in combinations(range(16), 2):
                    if set(p1) & set(p3) or set(p2) & set(p3):
                        continue
                    # Check all 8 transversal triples
                    all_present = True
                    for x, y, z in product(p1, p2, p3):
                        t = tuple(sorted((x, y, z)))
                        if t not in triples_set:
                            all_present = False
                            break
                    if all_present:
                        return False
        return True
    
    def greedy_add(triples_set, order):
        """Try to add triples in the given order, maintaining K(2,2,2)-freeness."""
        for t in order:
            new_set = triples_set | {t}
            # Quick check: does adding t create a K(2,2,2)?
            # t = (a,b,c). For t to be a transversal triple of three disjoint pairs,
            # a, b, c must be in different pairs.
            # The pairs containing a, b, c must be disjoint from each other.
            # For pair containing a: {a, a'} where a' != a,b,c
            # For pair containing b: {b, b'} where b' != a,b,c
            # For pair containing c: {c, c'} where c' != a,b,c
            # And a', b', c' must be distinct and not in {a,b,c}
            a, b, c = t
            others = [x for x in range(16) if x not in (a, b, c)]
            for a_prime in others:
                for b_prime in others:
                    if b_prime == a_prime:
                        continue
                    for c_prime in others:
                        if c_prime == a_prime or c_prime == b_prime:
                            continue
                        # Check all 8 transversal triples for pairs ({a,a'}, {b,b'}, {c,c'})
                        all_present = True
                        for x, y, z in product((a, a_prime), (b, b_prime), (c, c_prime)):
                            tt = tuple(sorted((x, y, z)))
                            if tt not in new_set:
                                all_present = False
                                break
                        if all_present:
                            break
                    if all_present:
                        break
                if all_present:
                    break
            if not all_present:
                triples_set.add(t)
        return triples_set
    
    def greedy_add_fast(triples_set, order):
        """Faster greedy: only check the specific K(2,2,2) that would be completed."""
        for t in order:
            a, b, c = t
            others = [x for x in range(16) if x not in (a, b, c)]
            conflict = False
            for a_prime in others:
                for b_prime in others:
                    if b_prime == a_prime:
                        continue
                    for c_prime in others:
                        if c_prime == a_prime or c_prime == b_prime:
                            continue
                        # Check all 8 transversal triples
                        all_present = True
                        for x in (a, a_prime):
                            for y in (b, b_prime):
                                for z in (c, c_prime):
                                    tt = tuple(sorted((x, y, z)))
                                    if tt not in triples_set and tt != t:
                                        all_present = False
                                        break
                                if not all_present:
                                    break
                            if not all_present:
                                break
                        if all_present:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if not conflict:
                triples_set.add(t)
        return triples_set
    
    best_size = 0
    best_set = set()
    
    all_tr = list(combinations(range(16), 3))
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        order = all_tr[:]
        random.shuffle(order)
        
        triples_set = set()
        for t in order:
            a, b, c = t
            others = [x for x in range(16) if x not in (a, b, c)]
            conflict = False
            for a_prime in others:
                for b_prime in others:
                    if b_prime == a_prime:
                        continue
                    for c_prime in others:
                        if c_prime == a_prime or c_prime == b_prime:
                            continue
                        all_present = True
                        for x in (a, a_prime):
                            for y in (b, b_prime):
                                for z in (c, c_prime):
                                    tt = tuple(sorted((x, y, z)))
                                    if tt not in triples_set and tt != t:
                                        all_present = False
                                        break
                                if not all_present:
                                    break
                            if not all_present:
                                break
                        if all_present:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if not conflict:
                triples_set.add(t)
        
        if len(triples_set) > best_size:
            best_size = len(triples_set)
            best_set = triples_set.copy()
    
    return [list(t) for t in best_set]
