import time
import random

def construct():
    deadline = time.time() + 85
    N = 128
    
    # Precompute: for each value x in 1..128, which pairs (a,b) with a<b in 1..128
    # have a+b == x? We need this to check if adding x creates a forbidden triple.
    # Actually, let's think differently.
    # A forbidden triple is {a,b,c} (distinct) where a+b, a+c, b+c are all in S.
    # When we add x to S, the only new forbidden triples are those containing x.
    # So {x, a, b} is forbidden iff a+b in S AND a+x in S∪{x} AND b+x in S∪{x}.
    # Since a,b >= 1, a+x > x and b+x > x, so a+x in S∪{x} means a+x in S (not equal to x).
    # So: {x,a,b} forbidden iff a+b in S AND a+x in S AND b+x in S.
    
    # Strategy: Greedy construction with lookahead, then local search.
    
    def can_add(S_set, S_list, x):
        """Check if x can be added to S (given by S_set and S_list)."""
        if x in S_set:
            return False
        # For each pair (a,b) in S_list, check if a+b, a+x, b+x are all in S
        # (a+x and b+x are > x >= 1, so they can't equal x)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if (a + b) in S_set and (a + x) in S_set and (b + x) in S_set:
                    return False
        return True
    
    # Greedy construction with multiple random orderings
    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if can_add(S_set, S, x):
                S.append(x)
                S_set.add(x)
        return S
    
    # Local search: try to add elements
    def local_add(S, S_set):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_set = set(S)
            for x in range(1, N+1):
                if time.time() > deadline:
                    break
                if x not in S_set and can_add(S_set, S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
        return S, S_set
    
    # Local search: try swaps (remove one, add one)
    def local_swap(S, S_set):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_set = set(S)
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                a_remove = S[i]
                S_minus = S_set - {a_remove}
                S_minus_list = [x for x in S if x != a_remove]
                for x in range(1, N+1):
                    if time.time() > deadline:
                        break
                    if x in S_set:
                        continue
                    if can_add(S_minus, S_minus_list, x):
                        # Verify full validity
                        new_S = S_minus_list + [x]
                        new_S_set = set(new_S)
                        valid = True
                        for ii in range(len(new_S)):
                            for jj in range(ii+1, len(new_S)):
                                for kk in range(jj+1, len(new_S)):
                                    a, b, c = new_S[ii], new_S[jj], new_S[kk]
                                    if (a+b) in new_S_set and (a+c) in new_S_set and (b+c) in new_S_set:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            S = new_S
                            S_set = new_S_set
                            improved = True
                            break
                if time.time() > deadline:
                    break
        return S, S_set
    
    # Start with a good baseline: upper half
    best_S = list(range(65, 129))
    best_size = len(best_S)
    
    random.seed(123)
    
    # Try multiple greedy constructions with different orderings
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            S = list(best_S)
        else:
            # Random ordering with bias
            order = list(range(1, N+1))
            random.shuffle(order)
            S = greedy_construct(order)
        
        S_set = set(S)
        S, S_set = local_add(S, S_set)
        S, S_set = local_swap(S, S_set)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Final verification
    best_S.sort()
    S_set = set(best_S)
    for i in range(len(best_S)):
        for j in range(i+1, len(best_S)):
            for k in range(j+1, len(best_S)):
                a, b, c = best_S[i], best_S[j], best_S[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return list(range(65, 129))
    
    return best_S
