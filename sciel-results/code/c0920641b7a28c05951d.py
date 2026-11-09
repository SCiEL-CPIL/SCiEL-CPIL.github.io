import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    N = 512
    target_size = 48

    def is_pair_valid(S, x):
        """Check if x creates any pair-mean violation with S."""
        S_set = set(S)
        for a in S:
            b = 2 * x - a
            if b in S_set and b != a:
                return False
        for y in S:
            a = 2 * y - x
            if a in S_set and a != y:
                return False
        return True

    def is_valid_global(S):
        """Full validity check using bitset DP for each element."""
        m = len(S)
        if m < 3:
            return True

        S_sorted = sorted(S)
        max_k = m - 1
        max_sum = 512 * 48

        for idx, a in enumerate(S_sorted):
            dp = [0] * (max_k + 1)
            dp[0] = 1

            count = 0
            for v in S_sorted:
                if v == a and count == 0:
                    count = 1
                    continue

                for k in range(min(count, max_k - 1), -1, -1):
                    if dp[k]:
                        dp[k + 1] |= dp[k] << v
                count += 1

            for k in range(2, max_k + 1):
                target_sum = a * k
                if target_sum > max_sum:
                    break
                if (dp[k] >> target_sum) & 1:
                    return False

        return True

    def greedy_build(order, use_pair_filter=True):
        current_set = []
        current_set_set = set()

        for c in order:
            if len(current_set) >= target_size:
                break
            if c in current_set_set:
                continue

            if use_pair_filter and not is_pair_valid(current_set, c):
                continue

            test_set = current_set + [c]
            if is_valid_global(test_set):
                current_set.append(c)
                current_set_set.add(c)

        return current_set

    # Build conflict graph
    def build_conflict_graph():
        """Build adjacency list for conflict graph."""
        # vertices 1..512
        # edge (u,v) if u and v cannot coexist
        # For simplicity, we'll use a heuristic:
        # Two numbers conflict if one is the mean of the other and some third number
        # But this is complex. Instead, we'll use a simpler approach:
        # We'll build the graph incrementally during the search.
        
        # For now, return an empty graph and we'll use the greedy approach
        # with the existing validity checks.
        return {}

    # Greedy independent set with degree-based ordering
    def greedy_independent_set_with_degree():
        """
        Greedy independent set:
        1. Start with all vertices 1..512
        2. Repeatedly pick the vertex with minimum degree (conflicts)
        3. Add it to the independent set
        4. Remove it and its neighbors
        5. Repeat until no vertices remain or we reach target size
        """
        # We'll use a heuristic: order by some property and greedily add
        # vertices that don't conflict with already selected ones.
        
        # First, let's compute a "conflict count" for each vertex
        # A vertex v conflicts with u if there exists w such that v = (u+w)/2
        # or u = (v+w)/2, etc.
        
        # For efficiency, we'll use a simpler ordering:
        # Order by value, but with some randomization to explore different solutions
        
        rng = random.Random(42)
        
        # Try multiple orderings
        best_set = []
        best_len = 0
        
        # Ordering 1: ascending
        order1 = list(range(1, N + 1))
        s1 = greedy_build(order1)
        if len(s1) > best_len:
            best_len = len(s1)
            best_set = s1[:]
        
        # Ordering 2: descending
        order2 = list(range(N, 0, -1))
        s2 = greedy_build(order2)
        if len(s2) > best_len:
            best_len = len(s2)
            best_set = s2[:]
        
        # Ordering 3: odd numbers first
        order3 = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        s3 = greedy_build(order3)
        if len(s3) > best_len:
            best_len = len(s3)
            best_set = s3[:]
        
        # Ordering 4: even numbers first
        order4 = list(range(2, N + 1, 2)) + list(range(1, N + 1, 2))
        s4 = greedy_build(order4)
        if len(s4) > best_len:
            best_len = len(s4)
            best_set = s4[:]
        
        # Randomized restarts
        for i in range(50):
            if time.time() > deadline:
                break
            p = list(range(1, N + 1))
            rng.shuffle(p)
            s = greedy_build(p)
            if len(s) > best_len:
                best_len = len(s)
                best_set = s[:]
                if best_len == target_size:
                    return best_set
        
        # Local search to improve
        if best_len < target_size:
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                current_set = best_set[:]
                current_set_set = set(current_set)
                
                candidates = [x for x in range(1, N + 1) if x not in current_set_set]
                if not candidates:
                    break
                
                for x in rng.sample(candidates, min(len(candidates), 20)):
                    if not is_pair_valid(current_set, x):
                        continue
                    
                    test_set = current_set + [x]
                    if is_valid_global(test_set):
                        current_set = test_set
                        current_set_set.add(x)
                        
                        remaining = [y for y in range(1, N + 1) if y not in current_set_set]
                        rng.shuffle(remaining)
                        for y in remaining:
                            if len(current_set) >= target_size:
                                break
                            if not is_pair_valid(current_set, y):
                                continue
                            test_set2 = current_set + [y]
                            if is_valid_global(test_set2):
                                current_set.append(y)
                                current_set_set.add(y)
                        
                        if len(current_set) > best_len:
                            best_len = len(current_set)
                            best_set = current_set[:]
                            if best_len == target_size:
                                return best_set
                        break
                
                if current_set and candidates:
                    a = rng.choice(current_set)
                    x = rng.choice(candidates)
                    if a != x:
                        new_set = [v for v in current_set if v != a]
                        new_set.append(x)
                        new_set_set = set(new_set)
                        
                        if is_valid_global(new_set):
                            remaining = [y for y in range(1, N + 1) if y not in new_set_set]
                            rng.shuffle(remaining)
                            for y in remaining:
                                if len(new_set) >= target_size:
                                    break
                                if not is_pair_valid(new_set, y):
                                    continue
                                test_set3 = new_set + [y]
                                if is_valid_global(test_set3):
                                    new_set.append(y)
                                    new_set_set.add(y)
                            
                            if len(new_set) > best_len:
                                best_len = len(new_set)
                                best_set = new_set[:]
                                if best_len == target_size:
                                    return best_set
        
        return best_set

    return greedy_independent_set_with_degree()
