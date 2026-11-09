import time
import random

N = 256

def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to S while maintaining validity.
    S is a set of integers.
    x is an integer not in S.
    """
    if not S:
        return True
        
    # Case 1: x is 'a' (smallest)
    # a=x, b in S, c in S, x < b <= c
    # Check if x | (b+c)
    
    # Optimization for small x: use residues
    if x < 128: # Threshold for optimization
        # We need b, c in S such that b > x, c >= b and (b+c) % x == 0
        # Let's collect residues of elements in S that are > x
        # Actually, we need to be careful about b <= c.
        # If we just check existence of pair with sum 0 mod x, we might pick b > c.
        # But since the set is unordered, if there exist distinct u, v in S > x such that (u+v)%x==0, 
        # we can assign b=min(u,v), c=max(u,v). Then b<=c.
        # If u==v, we need 2u % x == 0.
        
        # So:
        # 1. Check if exists u in S, u > x, such that 2*u % x == 0.
        # 2. Check if exists distinct u, v in S, u,v > x, such that (u+v) % x == 0.
        
        # Efficient check for 2:
        # Count residues of elements in S that are > x.
        counts = {}
        for val in S:
            if val > x:
                r = val % x
                counts[r] = counts.get(r, 0) + 1
        
        # Check 1
        for val in S:
            if val > x and (2 * val) % x == 0:
                return False
        
        # Check 2
        # We need r1 + r2 = 0 mod x.
        # r2 = (-r1) % x.
        for r, cnt in counts.items():
            r_neg = (-r) % x
            if r == r_neg:
                # Need two distinct elements with residue r
                if cnt >= 2:
                    return False
            else:
                # Need one element with residue r and one with residue r_neg
                if r_neg in counts and counts[r_neg] >= 1:
                    return False
    else:
        # x is large, just iterate
        # b, c in S, x < b <= c
        S_list = [v for v in S if v > x]
        for i in range(len(S_list)):
            b = S_list[i]
            for j in range(i, len(S_list)):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False

    # Case 2: x is 'b' (middle)
    # a in S, b=x, c in S, a < x <= c
    # Check if a | (x+c)
    for a in S:
        if a >= x:
            continue
        # Need c in S such that c >= x and (x+c) % a == 0
        # (x+c) % a == 0 => c % a == (-x) % a
        target_res = (-x) % a
        for c in S:
            if c >= x and c % a == target_res:
                return False

    # Case 3: x is 'c' (largest)
    # a in S, b in S, c=x, a < b <= x
    # Check if a | (b+x)
    for a in S:
        if a >= x:
            continue
        # Need b in S such that a < b <= x and (b+x) % a == 0
        # (b+x) % a == 0 => b % a == (-x) % a
        target_res = (-x) % a
        for b in S:
            if a < b <= x and b % a == target_res:
                return False

    return True

def _greedy_descending():
    n = N
    S = set()
    for x in range(n, 0, -1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_ascending():
    n = N
    S = set()
    for x in range(1, n + 1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_random(seed):
    random.seed(seed)
    n = N
    S = set()
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(x, S):
            S.add(x)
    return S

def _local_search_add(S, deadline):
    """Try to add elements to S. Returns (new_S, added_count)."""
    added = 0
    changed = True
    # Prioritize adding smaller numbers? Or larger?
    # Smaller numbers are more constrained. If we can add them, it's good.
    # But larger numbers are easier to add.
    # Let's just iterate 1 to N.
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in S:
                continue
            if _can_add(x, S):
                S.add(x)
                added += 1
                changed = True
    return S, added

def _local_search_swap_1_for_1(S, deadline):
    """Try to remove one element and add one other."""
    S_list = list(S)
    for y in S_list:
        if time.time() > deadline:
            break
        S_temp = S - {y}
        for x in range(1, N + 1):
            if x == y or x in S:
                continue
            if _can_add(x, S_temp):
                S_new = S_temp | {x}
                # Optional: try to add more after this swap
                # But that's expensive. Just return the swap.
                return S_new, True
    return S, False

def _local_search_swap_1_for_2(S, deadline):
    """Try to remove one element and add two others."""
    S_list = list(S)
    # To avoid O(N^3) explosion, we can limit the search for x, z
    # or just be efficient.
    for y in S_list:
        if time.time() > deadline:
            break
        S_temp = S - {y}
        added = 0
        new_elements = []
        # Try to add two elements
        # We can iterate x, then z
        # To speed up, we can try to add the "easiest" elements first?
        # Just iterate 1..N
        for x in range(1, N + 1):
            if x == y or x in S:
                continue
            if _can_add(x, S_temp):
                S_temp.add(x)
                new_elements.append(x)
                added += 1
                if added == 2:
                    break
        if added == 2:
            return S_temp, True
    return S, False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate initial candidates
    candidates = []
    candidates.append(_greedy_descending())
    candidates.append(_greedy_ascending())
    
    # Random restarts
    for i in range(5):
        candidates.append(_greedy_random(i))
        
    # Interval seeds
    # Try starting with [N/2+1, N] and repairing? 
    # _greedy_descending already does this effectively.
    # Maybe try a set of large primes? No, keep it simple.
    
    # Pick the best valid candidate
    best_set = set()
    best_size = 0
    for S in candidates:
        if _validate(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
    
    if not best_set:
        best_set = {N}
        best_size = 1
        
    # Local Search
    # Loop until no improvement or deadline
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add
        new_set, added = _local_search_add(best_set, deadline)
        if added > 0:
            best_set = new_set
            best_size = len(best_set)
            improved = True
            continue
        
        # 2. Try 1-for-1 swap
        new_set, sw = _local_search_swap_1_for_1(best_set, deadline)
        if sw:
            best_set = new_set
            best_size = len(best_set)
            improved = True
            continue
            
        # 3. Try 1-for-2 swap
        # This is expensive. Only do it if we have plenty of time left.
        if time.time() < deadline - 10.0:
            new_set, sw = _local_search_swap_1_for_2(best_set, deadline)
            if sw:
                best_set = new_set
                best_size = len(best_set)
                improved = True
                continue
        
        if not improved:
            # Stuck in local minimum.
            # Try a random perturbation?
            # Remove 2 random elements and re-add greedily?
            if time.time() < deadline - 20.0:
                # Random restart from current set
                S_perturbed = set(best_set)
                # Remove 2 random elements
                if len(S_perturbed) > 2:
                    items = list(S_perturbed)
                    random.shuffle(items)
                    S_perturbed.remove(items[0])
                    S_perturbed.remove(items[1])
                    # Greedy add
                    new_set, _ = _local_search_add(S_perturbed, deadline)
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(best_set)
                        improved = True
                        continue
            break
            
    return sorted(best_set)

if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
