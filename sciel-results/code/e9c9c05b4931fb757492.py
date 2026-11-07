import random
import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s (or 90s)
    N = 128
    
    def is_valid(S):
        """Check no three distinct a,b,c in S have a+b, a+c, b+c all in S."""
        Sset = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (a+b) in Sset and (a+c) in Sset and (b+c) in Sset:
                        return False
        return True
    
    def check_add(S, x):
        """Check if adding x to S keeps it valid. S is a list, x not in S."""
        Sset = set(S)
        if x in Sset:
            return False
        # For every pair (a,b) in S, check if a+b, a+x, b+x would all be present
        # After adding x, the new triples involving x are (x, a, b) for a,b in S
        # We need: NOT (a+b in S AND a+x in Sset_after AND b+x in Sset_after)
        # Sset_after = Sset | {x}
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a, b = S[i], S[j]
                # Check if a+b is in S (it is, since a,b in S and a+b might be in S)
                # Actually a+b might or might not be in S
                if (a+b) in Sset:
                    # Then we need NOT (a+x in Sset_after) AND NOT (b+x in Sset_after)
                    # i.e., a+x not in Sset and a+x != x (always true since a>0)
                    # and b+x not in Sset and b+x != x
                    if (a+x) in Sset or (a+x) == x:
                        # a+x == x implies a==0, not possible
                        if (a+x) in Sset:
                            return False
                    if (b+x) in Sset or (b+x) == x:
                        if (b+x) in Sset:
                            return False
        # Also need to check triples where x is one of the three, but we already did (x,a,b)
        # What about triples where two are x? No, distinct members.
        return True
    
    # Better check_add: for each pair (a,b) in S, after adding x, the triple (a,b,x) is forbidden
    # iff a+b, a+x, b+x are all in S∪{x}
    def check_add_fast(S, Sset, x):
        if x in Sset:
            return False
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a, b = S[i], S[j]
                if (a+b) in Sset:
                    if (a+x) in Sset or (b+x) in Sset:
                        # Wait, we need ALL THREE: a+b, a+x, b+x
                        # a+b is in Sset (checked)
                        # a+x is in Sset OR a+x==x? No, a+x==x => a=0
                        # So a+x in Sset_after means a+x in Sset (since a+x != x for a>=1)
                        # Similarly b+x in Sset_after means b+x in Sset
                        if (a+x) in Sset and (b+x) in Sset:
                            return False
        return True
    
    # Start with a good initial solution
    best_set = list(range(65, 129))  # size 64
    best_size = len(best_set)
    
    # Try to improve with local search
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from a random perturbation of best or a fresh greedy
        if restart == 0:
            S = list(best_set)
        else:
            # Randomly pick a subset of [1..128] and greedily build
            candidates = list(range(1, 129))
            random.shuffle(candidates)
            S = []
            Sset = set()
            for x in candidates:
                if check_add_fast(S, Sset, x):
                    S.append(x)
                    Sset.add(x)
        
        Sset = set(S)
        current_size = len(S)
        
        # Local search: try adding elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            Sset = set(S)
            for x in range(1, 129):
                if time.time() > deadline:
                    break
                if x not in Sset:
                    if check_add_fast(S, Sset, x):
                        S.append(x)
                        Sset.add(x)
                        current_size = len(S)
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_set = list(S)
        
        # Local search: try swaps (remove one, add one)
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            Sset = set(S)
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                for x in range(1, 129):
                    if x in Sset:
                        continue
                    # Try removing S[i] and adding x
                    S_minus = Sset - {S[i]}
                    if check_add_fast(list(S_minus), S_minus, x):
                        # Check full validity after swap
                        new_S = list(S_minus) + [x]
                        new_Sset = set(new_S)
                        # Quick check
                        valid = True
                        for ii in range(len(new_S)):
                            for jj in range(ii+1, len(new_S)):
                                for kk in range(jj+1, len(new_S)):
                                    a, b, c = new_S[ii], new_S[jj], new_S[kk]
                                    if (a+b) in new_Sset and (a+c) in new_Sset and (b+c) in new_Sset:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            S = new_S
                            Sset = new_Sset
                            current_size = len(S)
                            improved = True
                            if current_size > best_size:
                                best_size = current_size
                                best_set = list(S)
                            break
                if time.time() > deadline:
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(S)
    
    # Final verification
    best_set = list(best_set)
    best_set.sort()
    # Verify
    Sset = set(best_set)
    for i in range(len(best_set)):
        for j in range(i+1, len(best_set)):
            for k in range(j+1, len(best_set)):
                a, b, c = best_set[i], best_set[j], best_set[k]
                if (a+b) in Sset and (a+c) in Sset and (b+c) in Sset:
                    # Invalid, fall back to safe solution
                    return list(range(65, 129))
    
    return best_set
