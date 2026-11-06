import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 192
    
    # Precompute for each d in 1..n, and for each triple sum modulo d,
    # whether that residue is "forbidden" for a set S.
    # Key insight: for a given d and set S (excluding d itself), d violates
    # iff there exist 3 distinct elements in S whose sum ≡ 0 (mod d).
    
    # Precompute all triple sums and their residues
    # triple_sums[d][r] = list of (a,b,c) indices where (a+b+c) % d == r
    # But we need to check against the actual set S, not all numbers.
    
    # Better approach: for each d, precompute for each pair (i,j) the residue
    # that a third element would need to make the sum ≡ 0 mod d.
    
    def can_add_fast(S, x, S_list):
        """Check if x can be added to S. S is a set, S_list is list(S)."""
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S
        if K >= 3:
            for i in range(K):
                si = S_list[i]
                for j in range(i + 1, K):
                    sj = S_list[j]
                    rem = (-si - sj) % x
                    # Need a k > j with S_list[k] % x == rem
                    for k in range(j + 1, K):
                        if S_list[k] % x == rem:
                            return False
        
        # Check 2: For each y in S, y divides sum of any triple in S U {x} that includes x
        for y in S_list:
            # Need: exists u, v in S, u != v, u != y, v != y, such that (x + u + v) % y == 0
            # i.e., u + v ≡ -x (mod y)
            target = (-x) % y
            # Find pairs (u,v) in S_list (excluding y) with u + v ≡ target (mod y)
            others = [z for z in S_list if z != y]
            L = len(others)
            if L < 2:
                continue
            # Use residue counting
            counts = {}
            for val in others:
                r = val % y
                counts[r] = counts.get(r, 0) + 1
            
            found = False
            for u_idx in range(L):
                u = others[u_idx]
                ru = u % y
                needed = (target - ru) % y
                cnt = counts.get(needed, 0)
                if needed == ru:
                    if cnt >= 2:
                        found = True
                        break
                else:
                    if cnt >= 1:
                        # Need to ensure the element with residue 'needed' is not u itself
                        # Since needed != ru, any element with that residue works
                        found = True
                        break
            
            if found:
                return False
        
        return True

    def is_valid_fast(S):
        """Check if S is a valid set."""
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for i, d in enumerate(S_list):
            others = [x for j, x in enumerate(S_list) if j != i]
            L = len(others)
            if L < 3:
                continue
            
            # Check if d divides sum of any 3 distinct elements in others
            counts = {}
            for val in others:
                r = val % d
                counts[r] = counts.get(r, 0) + 1
            
            found = False
            for u_idx in range(L):
                u = others[u_idx]
                ru = u % d
                for v_idx in range(u_idx + 1, L):
                    v = others[v_idx]
                    rv = v % d
                    target_r = (- (ru + rv)) % d
                    
                    cnt = counts.get(target_r, 0)
                    if cnt == 0:
                        continue
                    
                    if target_r != ru and target_r != rv:
                        found = True
                        break
                    elif target_r == ru and target_r != rv:
                        if cnt >= 2:
                            found = True
                            break
                    elif target_r != ru and target_r == rv:
                        if cnt >= 2:
                            found = True
                            break
                    elif target_r == ru and target_r == rv:
                        if cnt >= 3:
                            found = True
                            break
                
                if found:
                    break
            
            if found:
                return False
        
        return True

    def greedy_construct(order):
        S = set()
        S_list = []
        for x in order:
            if time.time() - start_time > time_limit * 0.85:
                break
            if can_add_fast(S, x, S_list):
                S.add(x)
                S_list.append(x)
        return S

    def local_search(S, max_iters=2000):
        S = set(S)
        S_list = list(S)
        best_S = set(S)
        best_size = len(S)
        
        not_in_S = [x for x in range(1, n + 1) if x not in S]
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit * 0.9:
                break
            
            # Try adding an element
            if not_in_S:
                random.shuffle(not_in_S)
                added = False
                for x in not_in_S:
                    if can_add_fast(S, x, S_list):
                        S.add(x)
                        S_list.append(x)
                        not_in_S.remove(x)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
            
            # Try 1-swap: remove one, add one
            if not_in_S and len(S) > 0:
                S_list_copy = list(S_list)
                random.shuffle(S_list_copy)
                for r in S_list_copy[:5]:
                    new_S = S - {r}
                    new_S_list = [x for x in S_list if x != r]
                    candidates = [x for x in not_in_S if x != r]
                    if not candidates:
                        continue
                    random.shuffle(candidates)
                    for x in candidates[:10]:
                        if can_add_fast(new_S, x, new_S_list):
                            new_S.add(x)
                            if is_valid_fast(new_S):
                                S = new_S
                                S_list = list(S)
                                not_in_S = [y for y in range(1, n + 1) if y not in S]
                                if len(S) > best_size:
                                    best_S = set(S)
                                    best_size = len(S)
                                break
                    if len(S) > best_size:
                        break
            
            # Try 2-swap: remove two, add two
            if len(not_in_S) >= 2 and len(S) >= 2:
                S_list_copy = list(S_list)
                random.shuffle(S_list_copy)
                for i in range(min(3, len(S_list_copy))):
                    r1 = S_list_copy[i]
                    for j in range(i + 1, min(len(S_list_copy), i + 5)):
                        r2 = S_list_copy[j]
                        new_S = S - {r1, r2}
                        new_S_list = [x for x in S_list if x != r1 and x != r2]
                        candidates = [x for x in not_in_S if x not in (r1, r2)]
                        if len(candidates) < 2:
                            continue
                        random.shuffle(candidates)
                        for x1 in candidates[:8]:
                            if can_add_fast(new_S, x1, new_S_list):
                                new_S2 = new_S | {x1}
                                new_S_list2 = new_S_list + [x1]
                                remaining_cands = [x for x in candidates if x != x1]
                                for x2 in remaining_cands[:8]:
                                    if can_add_fast(new_S2, x2, new_S_list2):
                                        new_S3 = new_S2 | {x2}
                                        if is_valid_fast(new_S3):
                                            S = new_S3
                                            S_list = list(S)
                                            not_in_S = [y for y in range(1, n + 1) if y not in S]
                                            if len(S) > best_size:
                                                best_S = set(S)
                                                best_size = len(S)
                                            break
                                if len(S) > best_size:
                                    break
                        if len(S) > best_size:
                            break
                    if len(S) > best_size:
                        break
        
        return best_S

    # Phase 1: Multiple greedy seeds
    best_S = set()
    best_size = 0
    
    # Greedy from high to low
    S1 = greedy_construct(range(n, 0, -1))
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Greedy from low to high
    S2 = greedy_construct(range(1, n + 1))
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # Randomized greedy
    for _ in range(5):
        if time.time() - start_time > time_limit * 0.3:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = greedy_construct(order)
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(S_rand)
    
    # Phase 2: Local search from best seed
    best_S = local_search(best_S)
    
    # Phase 3: More local search iterations with restarts
    while time.time() - start_time < time_limit * 0.85:
        # Try local search from current best
        improved_S = local_search(best_S, max_iters=1000)
        if len(improved_S) > len(best_S):
            best_S = improved_S
        
        # Occasionally try a fresh random seed
        if random.random() < 0.1:
            order = list(range(1, n + 1))
            random.shuffle(order)
            S_new = greedy_construct(order)
            S_new = local_search(S_new, max_iters=500)
            if len(S_new) > len(best_S):
                best_S = S_new
    
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
