import time
import random

def construct():
    deadline = time.time() + 85
    
    n = 128
    best_size = 0
    best_set = None
    
    def check_valid(S):
        """Check if set S has no bad triple. Returns True if valid."""
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i+1, m):
                b = elems[j]
                for k in range(j+1, m):
                    c = elems[k]
                    if (a+b) in S and (a+c) in S and (b+c) in S:
                        return False
        return True
    
    def find_one_bad_triple(S):
        """Find one bad triple in S, or return None."""
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i+1, m):
                b = elems[j]
                for k in range(j+1, m):
                    c = elems[k]
                    if (a+b) in S and (a+c) in S and (b+c) in S:
                        return (a, b, c)
        return None
    
    def greedy_remove(S, deadline):
        """Iteratively remove elements that are in bad triples."""
        S = set(S)
        while time.time() < deadline:
            bad = find_one_bad_triple(S)
            if bad is None:
                break
            a, b, c = bad
            # Try removing each of the 6 values and pick the one that breaks most triples
            candidates = [a, b, c, a+b, a+c, b+c]
            candidates = [x for x in candidates if x in S]
            if not candidates:
                break
            # Just remove the first one for speed, or try a few
            best_remove = None
            best_count = -1
            for x in candidates:
                S_test = S - {x}
                # Quick check: count how many bad triples remain (sample)
                cnt = 0
                for _ in range(50):
                    if find_one_bad_triple(S_test) is not None:
                        cnt += 1
                        break
                if cnt < best_count or best_remove is None:
                    best_count = cnt
                    best_remove = x
            S.discard(best_remove)
        return S
    
    # Strategy 1: Start with all, greedily remove
    S1 = set(range(1, n + 1))
    S1 = greedy_remove(S1, deadline)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1
    
    # Strategy 2: Multiple random restarts with different initial subsets
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start with a random subset of ~60-70% of elements
        all_elems = list(range(1, n + 1))
        random.shuffle(all_elems)
        size = n - random.randint(30, 50)
        S = set(all_elems[:size])
        
        # Greedy remove
        S = greedy_remove(S, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 3: Local improvement - try to add back removed elements
    if best_set is not None:
        S = set(best_set)
        removed = set(range(1, n + 1)) - S
        removed_list = sorted(removed)
        
        for x in removed_list:
            if time.time() > deadline - 2:
                break
            S_test = S | {x}
            if check_valid(S_test):
                S = S_test
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Final verification
    if best_set is not None:
        if not check_valid(best_set):
            # Fallback: use a known valid small set
            best_set = set(range(1, 40))
            while not check_valid(best_set):
                best_set.pop()
    
    return list(sorted(best_set)) if best_set else []
