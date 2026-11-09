"""Combined program: fast greedy initialization + efficient local search with deadline guard."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s limit
    
    n = 256
    
    def can_add(S, x):
        """Check if adding x to S maintains AP-free property."""
        # Case 1: x is the middle term
        # Exists a in S such that 2*x - a is in S and 2*x - a != a
        for a in S:
            b = 2 * x - a
            if b != a and b in S:
                return False
        # Case 2: x is an end term
        # Exists b in S such that 2*b - x is in S and 2*b - x != b and 2*b - x != x
        for b in S:
            a = 2 * b - x
            if a != b and a != x and a in S:
                return False
        return True

    # Greedy initialization: add smallest elements first
    S = set()
    for x in range(1, n + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_val = sum(1.0 / x for x in S)
    
    # Local Search
    # We will try to improve by:
    # 1. Adding any valid excluded element (should be none if greedy was optimal, but check just in case)
    # 2. Single swaps: remove u, add v (v < u)
    # 3. Double swaps: remove u1, u2, add v1, v2
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(S)
        excluded = sorted(set(range(1, n + 1)) - S)
        
        # Phase 1: Try adding excluded elements (in case greedy missed something or previous swaps opened space)
        for v in excluded:
            if time.time() > deadline:
                break
            if can_add(S, v):
                S.add(v)
                best_S = set(S)
                best_val += 1.0 / v
                improved = True
                # Restart search as S changed
                break
        
        if time.time() > deadline or not improved:
            # If no simple add, try swaps
            if time.time() > deadline:
                break
            
            # Phase 2: Single Swaps
            # We want to maximize sum(1/x). So we want to replace large u with small v.
            # Sort S descending, Excluded ascending.
            S_desc = sorted(S, reverse=True)
            E_asc = sorted(excluded)
            
            best_swap_gain = -1e-9
            best_swap_u = None
            best_swap_v = None
            
            for u in S_desc:
                if time.time() > deadline:
                    break
                # Potential gain is 1/v - 1/u. Since v < u, gain is positive.
                # We only care about v < u.
                for v in E_asc:
                    if v >= u:
                        break # No more potential gains for this u (v would be larger or equal, gain <= 0)
                    # Check if swapping u for v is valid
                    S_temp = S - {u}
                    if can_add(S_temp, v):
                        gain = 1.0 / v - 1.0 / u
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_u = u
                            best_swap_v = v
            
            if best_swap_u is not None:
                S.remove(best_swap_u)
                S.add(best_swap_v)
                best_S = set(S)
                best_val += best_swap_gain
                improved = True
                continue # Restart search
        
        # If no single swap improved, try double swaps (if time)
        if not improved and time.time() < deadline - 5.0:
            # Try removing 2 large elements and adding 2 small ones
            S_desc = sorted(S, reverse=True)
            E_asc = sorted(excluded)
            
            # Limit search space to top 10 largest in S and top 10 smallest in E
            # This is O(100 * 100) = 10,000 checks, which is fast.
            candidates_S = S_desc[:15]
            candidates_E = E_asc[:15]
            
            best_double_gain = -1e-9
            best_double_u1, best_double_u2 = None, None
            best_double_v1, best_double_v2 = None, None
            
            for i, u1 in enumerate(candidates_S):
                for u2 in candidates_S[i+1:]:
                    S_temp = S - {u1} - {u2}
                    base_val = best_val - 1.0/u1 - 1.0/u2
                    
                    for j, v1 in enumerate(candidates_E):
                        if v1 == u1 or v1 == u2:
                            continue
                        if not can_add(S_temp, v1):
                            continue
                        S_temp2 = S_temp | {v1}
                        
                        for v2 in candidates_E[j+1:]:
                            if v2 == u1 or v2 == u2 or v2 == v1:
                                continue
                            if can_add(S_temp2, v2):
                                gain = (1.0/v1 + 1.0/v2) - (1.0/u1 + 1.0/u2)
                                if gain > best_double_gain:
                                    best_double_gain = gain
                                    best_double_u1, best_double_u2 = u1, u2
                                    best_double_v1, best_double_v2 = v1, v2
            
            if best_double_u1 is not None:
                S.remove(best_double_u1)
                S.remove(best_double_u2)
                S.add(best_double_v1)
                S.add(best_double_v2)
                best_S = set(S)
                best_val += best_double_gain
                improved = True
                continue
        
        if not improved:
            # Try random perturbations if stuck? 
            # Or just break.
            break

    return list(best_S)
