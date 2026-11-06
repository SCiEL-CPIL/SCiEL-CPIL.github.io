import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 180
    
    # 1. Generate all forbidden triples (a, b, c) distinct in [1, N] with 1/a = 1/b + 1/c
    # Using a = bc/(b+c) => (b-a)(c-a) = a^2
    triples = []
    seen = set()
    
    for a in range(1, N + 1):
        a_sq = a * a
        for k in range(1, int(a_sq**0.5) + 1):
            if a_sq % k == 0:
                m = a_sq // k
                b1 = a + k
                c1 = a + m
                if b1 <= N and c1 <= N and b1 != c1 and b1 != a and c1 != a:
                    t = tuple(sorted([a, b1, c1]))
                    if t not in seen:
                        seen.add(t)
                        triples.append(t)
                
                # Also check if k and m are different (they always are unless k=m)
                # If k != m, we already have the pair (k,m) and (m,k) gives same triple
                # But we should also check the case where b = a + m, c = a + k
                # This is the same triple {a, a+k, a+m}, so no new triple
    
    # 2. Precompute conflict pairs for each element
    conflict_pairs = defaultdict(list)
    for t in triples:
        a, b, c = t
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    def count_violations(S_set):
        count = 0
        for t in triples:
            if all(x in S_set for x in t):
                count += 1
        return count
    
    def can_add(x, S_set):
        for (y, z) in conflict_pairs[x]:
            if y in S_set and z in S_set:
                return False
        return True
    
    def greedy_remove_violations(S_set):
        """Remove elements to eliminate all violations"""
        S = set(S_set)
        violated = [t for t in triples if all(x in S for x in t)]
        while violated:
            elem_count = defaultdict(int)
            for t in violated:
                for x in t:
                    if x in S:
                        elem_count[x] += 1
            if not elem_count:
                break
            best_elem = max(elem_count, key=elem_count.get)
            S.discard(best_elem)
            violated = [t for t in triples if all(x in S for x in t)]
        return S
    
    def greedy_add_all(S_set):
        """Add all elements that can be added without creating violations"""
        S = set(S_set)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    changed = True
        return S
    
    # 3. Initial construction: start with all, remove violations
    S = set(range(1, N + 1))
    S = greedy_remove_violations(S)
    S = greedy_add_all(S)
    
    best_set = set(S)
    best_size = len(S)
    
    # 4. Multiple restarts with simulated annealing
    random.seed(12345)
    all_elements = list(range(1, N + 1))
    
    num_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 15 and num_restarts < max_restarts:
        num_restarts += 1
        
        # Initialize current set
        if num_restarts == 1:
            current_set = set(best_set)
        else:
            # Perturb from best_set
            current_set = set(best_set)
            # Remove 3-7 random elements
            num_remove = random.randint(3, 7)
            to_remove = random.sample(list(current_set), min(num_remove, len(current_set)))
            for x in to_remove:
                current_set.discard(x)
            # Add some random elements
            not_in_set = [x for x in all_elements if x not in current_set]
            if not_in_set:
                num_add = random.randint(1, 4)
                to_add = random.sample(not_in_set, min(num_add, len(not_in_set)))
                for x in to_add:
                    current_set.add(x)
            # Fix any violations
            current_set = greedy_remove_violations(current_set)
            current_set = greedy_add_all(current_set)
        
        current_violations = count_violations(current_set)
        
        # Simulated annealing
        temp = 3.0
        cooling = 0.997
        min_temp = 0.05
        
        for sa_iter in range(3000):
            if time.time() > deadline - 15:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.35:
                # Try to add a random element
                not_in_set = [x for x in all_elements if x not in current_set]
                if not not_in_set:
                    continue
                x = random.choice(not_in_set)
                
                # Calculate change in violations
                new_violations = 0
                for (y, z) in conflict_pairs[x]:
                    if y in current_set and z in current_set:
                        new_violations += 1
                
                if new_violations == 0:
                    current_set.add(x)
                    current_violations = 0
                else:
                    delta = new_violations
                    # Accept with probability
                    prob = min(1.0, (temp / 3.0) / (1.0 + delta))
                    if random.random() < prob:
                        current_set.add(x)
                        current_violations += delta
                        
            elif move_type < 0.70:
                # Try to remove a random element
                if not current_set:
                    continue
                x = random.choice(list(current_set))
                
                # Calculate how many violations are resolved
                resolved = 0
                for (y, z) in conflict_pairs[x]:
                    if y in current_set and z in current_set:
                        resolved += 1
                
                current_set.discard(x)
                current_violations = max(0, current_violations - resolved)
                
                # After removal, try to greedily add back as many as possible
                for y in all_elements:
                    if y not in current_set and can_add(y, current_set):
                        current_set.add(y)
                        
            else:
                # Try a swap: remove x, add y
                if len(current_set) < 1:
                    continue
                x = random.choice(list(current_set))
                not_in_set = [z for z in all_elements if z not in current_set]
                if not not_in_set:
                    continue
                y = random.choice(not_in_set)
                
                # Calculate delta for the swap
                # Removing x resolves some violations, adding y creates some
                resolved_by_x = 0
                for (u, v) in conflict_pairs[x]:
                    if u in current_set and v in current_set and u != y and v != y:
                        resolved_by_x += 1
                    elif u in current_set and v == y:
                        # This violation would be resolved by removing x, but y is being added
                        # Actually, if we remove x and add y, the triple {x, u, y} is no longer
                        # a problem since x is gone. But {u, v, y} might be a new violation.
                        # Let's just compute directly.
                        pass
                
                # Direct computation
                old_set = set(current_set)
                current_set.discard(x)
                current_set.add(y)
                
                new_violations = count_violations(current_set)
                delta = new_violations - current_violations
                
                if delta <= 0:
                    current_violations = new_violations
                else:
                    prob = min(1.0, (temp / 3.0) / (1.0 + delta))
                    if random.random() < prob:
                        current_violations = new_violations
                    else:
                        current_set = old_set
                
            temp *= cooling
            if temp < min_temp:
                temp = min_temp
        
        # After SA, polish
        current_set = greedy_remove_violations(current_set)
        current_set = greedy_add_all(current_set)
        
        if len(current_set) > best_size and count_violations(current_set) == 0:
            best_set = set(current_set)
            best_size = len(current_set)
    
    # 5. Final polish: try removing each element individually and see if we can do better
    if time.time() < deadline - 5:
        for x in list(best_set):
            if time.time() > deadline - 5:
                break
            test_set = set(best_set)
            test_set.discard(x)
            test_set = greedy_remove_violations(test_set)
            test_set = greedy_add_all(test_set)
            if len(test_set) > len(best_set) and count_violations(test_set) == 0:
                best_set = test_set
    
    # Final validation and cleanup
    best_set = greedy_remove_violations(best_set)
    best_set = greedy_add_all(best_set)
    
    return list(best_set)
