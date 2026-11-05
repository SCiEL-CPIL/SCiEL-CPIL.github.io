import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256

    def is_valid_addition(x, S):
        """
        Check if adding x to S keeps it valid.
        S is a list of elements.
        For every a<b<=c in the new set, a must NOT divide b+c.
        New violations must involve x.
        1. x is a: exists b,c in S with x<b<=c and x|(b+c)
        2. x is b: exists a,c in S with a<x<=c and a|(x+c)
        3. x is c: exists a,b in S with a<b<=x and a|(b+x)
        """
        if not S:
            return True

        # Optimization: if x is larger than all elements in S, it can only be 'c'.
        # If x is smaller than all elements, it can only be 'a'.
        # But checking all cases is fast enough for n=256.
        
        # Case 1: x is 'a'
        # We need b, c in S such that x < b <= c and (b+c) % x == 0
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b'
        # We need a, c in S such that a < x <= c and (x+c) % a == 0
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c'
        # We need a, b in S such that a < b <= x and (b+x) % a == 0
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
                    
        return True

    def is_valid_set(S):
        """Check if S is valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def greedy_construct(order):
        """Construct a set by adding elements in the given order if valid."""
        S = []
        for x in order:
            if is_valid_addition(x, S):
                S.append(x)
        return S

    def local_search_improve(S):
        """Try to add any missing elements to S."""
        S_set = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if x not in S_set:
                    if is_valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        improved = True
        return S

    # Conflict graph model:
    # A set S is valid iff for every a<b<=c in S, a does NOT divide b+c.
    # Equivalently, S is an independent set in a hypergraph of "bad triples".
    # We approximate with a pairwise conflict graph:
    #   For each a in [1,n], and each b,c with a<b<=c and a|(b+c),
    #   any two of {b,c} cannot both be in S *if a is also in S*.
    #   But since a must be in S for the violation, we can't just precompute
    #   pairwise edges independent of S.
    #
    # Instead, use a MIS-style greedy:
    #   - Start with all elements as candidates.
    #   - Repeatedly pick an element x to add (or keep), and remove elements
    #     that would conflict with x given the current set.
    #
    # Practical approach: greedy MIS heuristic with random restarts.
    # At each step, among remaining candidates, pick the one with fewest
    # conflicts (minimum degree in the "conflict with current set" sense),
    # add it, and remove all candidates that conflict with it.
    #
    # To make this efficient and diverse:
    #   1. Precompute for each a, the set of (b,c) pairs that conflict.
    #   2. Use a fast validity check.
    #   3. Greedy: start with empty set, repeatedly add the element that
    #      causes the fewest future conflicts (or use random ordering).
    #   4. Local search: try to add any missing element.
    #   5. Removal-based: start with full set, remove elements to resolve
    #      conflicts (like greedy set cover / hitting set).

    # Precompute conflict structure for fast checks
    # conflict[a] = set of (b, c) pairs where a < b <= c and a | (b+c)
    # This is large, so we'll compute on-the-fly with memoization.

    def conflicts_with(x, S):
        """Return set of elements in S that conflict with x (i.e., adding x to S would violate validity)."""
        result = set()
        S_list = list(S)
        # x as 'a': exists b,c in S with x < b <= c and x | (b+c)
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for c in S_list[i:]:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    result.add(b)
                    result.add(c)
        # x as 'b': exists a,c in S with a < x <= c and a | (x+c)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    result.add(a)
                    result.add(c)
        # x as 'c': exists a,b in S with a < b <= x and a | (b+x)
        for a in S_list:
            if a >= x:
                continue
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    result.add(a)
                    result.add(b)
        return result

    def mis_greedy(order, S_init=None):
        """
        MIS-style greedy: process elements in order, add if valid.
        This is equivalent to greedy_construct but we keep it for clarity.
        """
        S = list(S_init) if S_init else []
        S_set = set(S)
        for x in order:
            if x in S_set:
                continue
            if is_valid_addition(x, S):
                S.append(x)
                S_set.add(x)
        return S

    def remove_to_fix(S):
        """
        Start with S, remove elements to make it valid.
        Greedy: find a violating triple, remove the element with the most
        conflicts (or the smallest, to keep larger elements).
        """
        S_set = set(S)
        S_list = sorted(S_set)
        changed = True
        while changed:
            changed = False
            # Find a violation
            m = len(S_list)
            violation = None
            for i in range(m):
                a = S_list[i]
                for j in range(i+1, m):
                    b = S_list[j]
                    for k in range(j, m):
                        c = S_list[k]
                        if (b + c) % a == 0:
                            violation = (a, b, c)
                            break
                    if violation:
                        break
                if violation:
                    break
            if not violation:
                break
            a, b, c = violation
            # Remove the element that appears in the most violations
            # Simple heuristic: remove the smallest (a), since small numbers
            # conflict with many pairs.
            S_set.discard(a)
            S_list = sorted(S_set)
            changed = True
        return list(S_set)

    best_set = []
    best_size = 0

    # Initial strategies
    # Descending order
    S1 = greedy_construct(range(n, 0, -1))
    S1 = local_search_improve(S1)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1[:]

    # Ascending order
    S2 = greedy_construct(range(1, n + 1))
    S2 = local_search_improve(S2)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2[:]

    # Remove-to-fix: start with all elements, remove to fix
    S3 = remove_to_fix(list(range(1, n + 1)))
    S3 = local_search_improve(S3)
    if len(S3) > best_size:
        best_size = len(S3)
        best_set = S3[:]

    # Random restarts with MIS-style greedy (random order)
    random.seed(42)
    for attempt in range(800):
        if time.time() > deadline - 5:
            break
        
        # Shuffle the order
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S_try = greedy_construct(order)
        S_try = local_search_improve(S_try)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]

    # Random restarts with remove-to-fix (random removal order)
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Randomly permute and remove
        S_try = list(range(1, n + 1))
        random.shuffle(S_try)
        S_set = set(S_try)
        S_list = sorted(S_set)
        changed = True
        while changed:
            changed = False
            m = len(S_list)
            violation = None
            for i in range(m):
                a = S_list[i]
                for j in range(i+1, m):
                    b = S_list[j]
                    for k in range(j, m):
                        c = S_list[k]
                        if (b + c) % a == 0:
                            violation = (a, b, c)
                            break
                    if violation:
                        break
                if violation:
                    break
            if not violation:
                break
            # Remove a random element from the violating triple
            a, b, c = violation
            candidates = [a, b, c]
            random.shuffle(candidates)
            for cand in candidates:
                if cand in S_set:
                    S_set.discard(cand)
                    S_list = sorted(S_set)
                    changed = True
                    break
        S_try = list(S_set)
        S_try = local_search_improve(S_try)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]

    # Final local search on best_set
    if time.time() < deadline - 5:
        best_set = local_search_improve(best_set)

    return sorted(best_set)
