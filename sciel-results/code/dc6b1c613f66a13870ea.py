"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def build_conflict_structure(S):
        """Build for each element, the set of pairs (i,j) such that S[i]+S[j]=S[k]"""
        S_set = set(S)
        n = len(S)
        # For each element k, find all pairs (i,j) where S[i]+S[j]=S[k]
        # conflict[k] = set of (i,j) pairs that would conflict if k is included
        # Actually, we need: if we include elements i and j, then we cannot include k where S[i]+S[j]=S[k]
        # And if we include i and k, we cannot include j where S[i]+S[j]=S[k]
        # Let's build: for each element, the set of other elements that form a sum triple with it and some third element
        
        # Simpler: build a 3-uniform hypergraph of conflicts
        # hyperedges: set of (i,j,k) where S[i]+S[j]=S[k]
        # For greedy: when considering adding element x, check if there exist a,b already in set such that a+b=x or a+x=b or x+b=a
        
        # For local search, let's build: for each pair (i,j), the set of k where S[i]+S[j]=S[k]
        pair_to_k = {}
        for i in range(n):
            for j in range(i, n):  # a=b allowed, so i can equal j
                s = S[i] + S[j]
                if s in S_set:
                    # Find index of s
                    # We need to be careful: s might equal S[i] or S[j]
                    if s not in pair_to_k:
                        pair_to_k[s] = []
                    pair_to_k[s].append((i, j))
        
        # For each element k, find all pairs (i,j) such that S[i]+S[j]=S[k]
        # i.e., if k is in the set, then for any (i,j) in pair_to_k[S[k]], we can't have both i and j
        element_conflicts = [set() for _ in range(n)]
        for k_val, pairs in pair_to_k.items():
            # Find index of k_val
            k_idx = S.index(k_val) if k_val in S else -1
            if k_idx == -1:
                continue
            for (i, j) in pairs:
                # If we include k_idx, we can't include both i and j
                # If we include i and j, we can't include k_idx
                # For the purpose of "can I add element x to current set S_cur":
                # x conflicts with current set if there exist a,b in S_cur such that a+b=x or a+x=b or x+b=a
                # i.e., x = a+b for some a,b in S_cur, or a = x+b for some b in S_cur (i.e., x = a-b), or b = x+a
                # More simply: for each a in S_cur, check if a+x in S_cur or x-a in S_cur... 
                # Let's just check: for a in S_cur, is a+x in S_cur? Is x-a in S_cur? Is -x+a in S_cur? No...
                # a + b = c means: given x, check if there exist a,b in S_cur such that:
                #   a + b = x  (x is the sum)
                #   a + x = b  => b - a = x  (x is the difference, b = a+x)
                #   x + a = b  => b - a = x  (same as above)
                # So: x conflicts if (exists a,b in S_cur: a+b=x) or (exists a in S_cur: a+x in S_cur) or (exists a in S_cur: x-a in S_cur and x-a != 0... wait)
                # Actually a+b=c with a,b,c in set. If x is new:
                # Case 1: x = a+b for a,b in S_cur
                # Case 2: a = x+b => a-b=x for a,b in S_cur, meaning there's a pair in S_cur with difference x
                # Case 3: b = x+a => same as case 2
                # So x can be added iff: no a,b in S_cur with a+b=x, AND no a,b in S_cur with |a-b|=x (and a != b if x=0, but x != 0)
                # Wait, a=b is allowed, so a-a=0, but x != 0 so that's fine.
                # Actually if a=b, then a+b=2a, and |a-b|=0. Since x != 0, the difference condition with a=b gives 0 != x, so it's fine.
                pass
        
        # Let's use a different approach for the greedy: just check directly
        return S_set, pair_to_k
    
    def greedy_sum_free(S, order='conflict_degree'):
        """Greedy: process elements in given order, include if no conflict with already included"""
        S_set = set(S)
        n = len(S)
        
        # Compute conflict degree for each element: number of triples it participates in
        # For element x, it's in a triple if:
        #   x = a+b for some a,b in S
        #   a = x+b for some b in S (i.e., a-b=x)
        #   b = x+a for some a in S (same)
        # Count: for each x, count pairs (a,b) in S such that a+b=x, plus count pairs (a,b) in S such that a-b=x or b-a=x
        
        conflict_degree = [0] * n
        S_list = list(S)
        for i in range(n):
            x = S_list[i]
            cnt = 0
            for j in range(n):
                a = S_list[j]
                # Case 1: a + b = x => b = x - a
                b = x - a
                if b in S_set and b != 0:
                    cnt += 1
                # Case 2: a + x = b => b = a + x
                b2 = a + x
                if b2 in S_set and b2 != 0:
                    cnt += 1
                # Case 3: x + a = b => same as case 2
            conflict_degree[i] = cnt
        
        # Sort by conflict degree (ascending) - include low-conflict elements first
        indices = list(range(n))
        indices.sort(key=lambda i: conflict_degree[i])
        
        included = []
        included_set = set()
        for i in indices:
            x = S_list[i]
            can_add = True
            for a in included_set:
                # Check a + b = x => b = x - a
                b = x - a
                if b in included_set:
                    can_add = False
                    break
                # Check a + x = b => b = a + x
                b2 = a + x
                if b2 in included_set:
                    can_add = False
                    break
            if can_add:
                included.append(x)
                included_set.add(x)
        
        return included
    
    def local_search(S, initial, max_time=30):
        """Try to improve the initial sum-free subset by adding elements or swapping"""
        S_set = set(S)
        current = set(initial)
        best = set(initial)
        best_size = len(initial)
        
        # Elements not in current
        candidates = [x for x in S if x not in current]
        
        def can_add(x, current_set):
            for a in current_set:
                b = x - a
                if b in current_set:
                    return False
                b2 = a + x
                if b2 in current_set:
                    return False
            return True
        
        # Try adding elements
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            candidates = [x for x in S if x not in current]
            # Sort candidates by how many conflicts they have with current
            def conflict_count(x):
                cnt = 0
                for a in current:
                    if (x - a) in current or (a + x) in current:
                        cnt += 1
                return cnt
            candidates.sort(key=conflict_count)
            for x in candidates:
                if can_add(x, current):
                    current.add(x)
                    improved = True
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                    break  # Restart scanning after each addition
        
        return list(best)
    
    def solve_instance(S):
        n = len(S)
        # Start with greedy
        initial = greedy_sum_free(S)
        # Try local search
        result = local_search(S, initial)
        return result
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = solve_instance(S)
        # Verify sum-free
        assert is_sum_free(subset), f"Not sum-free for size={size}"
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
