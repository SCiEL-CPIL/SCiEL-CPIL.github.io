import random
from itertools import combinations
from math import gcd
import time

def compute_bad_pairs():
    """
    Precompute bad_pairs[x] = set of (u, v) tuples (u < v) 
    such that {x, u, v} is a bad triple (gcds are equal).
    """
    n = 128
    bad_pairs = [set() for _ in range(n + 1)]
    
    # Iterate over all triples
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, n + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    # Triple (a, b, c) is bad
                    # For x=a, pair is (b, c)
                    bad_pairs[a].add((b, c))
                    # For x=b, pair is (a, c) -> sort to (a, c)
                    bad_pairs[b].add((a, c))
                    # For x=c, pair is (a, b) -> sort to (a, b)
                    bad_pairs[c].add((a, b))
    
    return bad_pairs

BAD_PAIRS = compute_bad_pairs()

def can_add(x, current_set):
    """
    Check if x can be added to current_set without forming a bad triple.
    Uses precomputed BAD_PAIRS[x].
    """
    # If current_set is empty or size 1, no triple can be formed
    if len(current_set) < 2:
        return True
        
    # Iterate over pairs in current_set
    # To speed up, we convert to list and iterate
    # Since we need to check if ANY pair is in BAD_PAIRS[x], we can break early
    pairs = combinations(current_set, 2)
    for u, v in pairs:
        # Normalize pair to (min, max) to match precomputed format
        if u > v:
            u, v = v, u
        if (u, v) in BAD_PAIRS[x]:
            return False
    return True

def greedy_build(order):
    """
    Build a set by iterating through 'order' and adding if possible.
    """
    s = set()
    for x in order:
        if can_add(x, s):
            s.add(x)
    return s

def local_search(s):
    """
    Improve set s using add and 1-for-2 swap moves.
    """
    improved = True
    max_iter = 100 # Safety limit for inner loops to prevent infinite loops in bad cases
    while improved:
        improved = False
        iter_count = 0
        while iter_count < max_iter and not improved:
            iter_count += 1
            # 1. Try to add any single element
            candidates = [x for x in range(1, 129) if x not in s]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, s):
                    s.add(x)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try 1-for-2 swaps
            # Remove u, add v and w
            list_s = list(s)
            random.shuffle(list_s)
            
            for u in list_s:
                s_without_u = s - {u}
                
                # Find all elements that can be added to s_without_u
                addable = []
                for x in range(1, 129):
                    if x not in s_without_u:
                        if can_add(x, s_without_u):
                            addable.append(x)
                
                if len(addable) >= 2:
                    # We can add at least 2 elements
                    # Pick the first 2 (randomized by shuffle? No, addable is sorted by range)
                    # Let's shuffle addable to diversify
                    random.shuffle(addable)
                    v, w = addable[0], addable[1]
                    
                    # Construct new set
                    new_s = s_without_u | {v, w}
                    
                    # Verify validity (should be true by construction, but good to be safe)
                    # Actually, can_add ensures v and w are individually compatible with s_without_u
                    # But do v and w conflict with each other in the context of s_without_u?
                    # can_add(v, s_without_u) checks pairs in s_without_u.
                    # It does NOT check if (v, w) is a bad pair with some element in s_without_u?
                    # Wait.
                    # If we add v, we check pairs in s_without_u.
                    # If we then add w, we check pairs in (s_without_u U {v}).
                    # The pair (v, w) is checked when adding w?
                    # No, can_add(w, s_without_u U {v}) will check pairs in s_without_u U {v}.
                    # The pairs are (x, y) where x, y in s_without_u U {v}.
                    # One such pair is (v, y) for y in s_without_u.
                    # One such pair is (x, v) for x in s_without_u.
                    # Does it check (v, w)? No, w is the element being added.
                    # It checks if (v, w) is a bad pair with ANY element in the current set?
                    # No. The condition is: no triple {a, b, c} in the final set is bad.
                    # When adding w to S_temp, we check if there exists a pair (u, v) in S_temp
                    # such that {w, u, v} is bad.
                    # Here S_temp = s_without_u U {v}.
                    # So we check pairs in s_without_u U {v}.
                    # This includes pairs from s_without_u, and pairs involving v.
                    # It does NOT check the pair (v, w) itself against a third element, 
                    # because w is the new element.
                    # The triple would be {w, v, z} where z is in s_without_u.
                    # This is checked by: is (v, z) in BAD_PAIRS[w]?
                    # Yes.
                    
                    # So the construction is valid.
                    if len(new_s) > len(s):
                        s = new_s
                        improved = True
                        break
                        
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second margin
    
    best_set = set()
    
    # Initial greedy with sorted order
    initial_order = list(range(1, 129))
    s = greedy_build(initial_order)
    s = local_search(s)
    if len(s) > len(best_set):
        best_set = s.copy()
        
    # Random restarts
    while time.time() < deadline:
        random_order = list(range(1, 129))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s)
        
        if len(s) > len(best_set):
            best_set = s.copy()
            
    return list(best_set)
