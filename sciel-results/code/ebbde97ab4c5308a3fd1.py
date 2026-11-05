import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)
    
    N = 512
    weights = [1.0 / (N - a) for a in range(N)]
    
    # Precompute conflict graph: conflicts[a] = set of b != a with gcd(a,b) > 1
    conflicts = [set() for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            if math.gcd(i, j) > 1:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Candidates sorted by weight descending (a close to 511 first)
    candidates = list(range(N-1, 0, -1))
    
    def get_score(s):
        return sum(weights[a] for a in s)
    
    def is_coprime_to_all(a, chosen_set):
        for b in chosen_set:
            if math.gcd(a, b) > 1:
                return False
        return True
    
    def greedy_init(order):
        chosen = []
        chosen_set = set()
        for a in order:
            if is_coprime_to_all(a, chosen_set):
                chosen.append(a)
                chosen_set.add(a)
        return chosen
    
    # Multiple initialization strategies
    random.seed(42)
    
    # Strategy 1: Largest to smallest
    init1 = greedy_init(candidates)
    
    # Strategy 2: Randomized (several tries)
    best_random_init = []
    best_random_score = 0
    for trial in range(20):
        order = list(candidates)
        random.shuffle(order)
        init = greedy_init(order)
        s = get_score(init)
        if s > best_random_score:
            best_random_score = s
            best_random_init = init
    
    # Strategy 3: Perturbed greedy
    order3 = list(candidates)
    for i in range(len(order3)-1, 0, -1):
        if random.random() < 0.15:
            j = random.randint(0, i)
            order3[i], order3[j] = order3[j], order3[i]
    init3 = greedy_init(order3)
    
    # Strategy 4: Weighted random (bias toward high-weight items)
    best_weighted_init = []
    best_weighted_score = 0
    for trial in range(10):
        order = list(candidates)
        # Bias shuffle: items with higher weight more likely to stay in front
        for i in range(len(order) - 1, 0, -1):
            r = random.random()
            if r < 0.3:  # 30% chance to swap with a random earlier position
                j = random.randint(0, i)
                order[i], order[j] = order[j], order[i]
        init = greedy_init(order)
        s = get_score(init)
        if s > best_weighted_score:
            best_weighted_score = s
            best_weighted_init = init
    
    candidates_list = [init1, best_random_init, init3, best_weighted_init]
    best_init = max(candidates_list, key=get_score)
    best_score = get_score(best_init)
    best_chosen = list(best_init)
    
    # Branch and bound with tighter upper bound
    def branch_and_bound():
        nonlocal best_score, best_chosen
        if time.time() > deadline:
            return
        
        order = candidates  # already sorted by weight desc
        
        def upper_bound(idx, chosen_set, current_score):
            if time.time() > deadline:
                return 0.0
            ub = current_score
            # Tighter bound: for each remaining candidate coprime to chosen,
            # also check it's coprime to other remaining candidates we might add
            # Simple bound: sum of all remaining coprime-to-chosen weights
            for a in order[idx:]:
                if a in chosen_set:
                    continue
                if is_coprime_to_all(a, chosen_set):
                    ub += weights[a]
            return ub
        
        # DFS with pruning - use a smarter ordering
        stack = [(0, frozenset(), 0.0)]
        nodes_visited = 0
        
        while stack:
            if time.time() > deadline - 10:  # leave time for local search
                break
            idx, chosen_set, current_score = stack.pop()
            nodes_visited += 1
            
            if current_score > best_score:
                best_score = current_score
                best_chosen = list(chosen_set)
            
            if idx >= len(order):
                continue
            
            # Upper bound check
            ub = upper_bound(idx, chosen_set, current_score)
            if ub <= best_score + 1e-12:
                continue
            
            a = order[idx]
            
            # Option 1: skip a
            stack.append((idx + 1, chosen_set, current_score))
            
            # Option 2: include a if coprime to all chosen
            if is_coprime_to_all(a, chosen_set):
                stack.append((idx + 1, chosen_set | {a}, current_score + weights[a]))
        
        return nodes_visited
    
    # Run branch and bound
    nodes = branch_and_bound()
    
    # Local search polish on best solution found
    if time.time() < deadline - 5:
        current = list(best_chosen)
        current_set = set(current)
        current_score = best_score
        
        # 1-swap local search (remove one, add best replacement)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                temp_set = current_set - {a}
                
                best_b = -1
                best_gain = 0
                for b in range(N-1, 0, -1):
                    if b in temp_set:
                        continue
                    if is_coprime_to_all(b, temp_set):
                        gain = weights[b] - weights[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                
                if best_b != -1 and best_gain > 1e-12:
                    current = [x for x in current if x != a]
                    current.append(best_b)
                    current_set.remove(a)
                    current_set.add(best_b)
                    current_score += best_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(current)
                    break
            
            if not improved:
                break
        
        # 2-opt: swap two elements simultaneously
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            n_curr = len(current)
            for i in range(n_curr):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n_curr):
                    a, b = current[i], current[j]
                    temp_set = current_set - {a, b}
                    
                    # Find best pair (x, y) to replace (a, b)
                    best_pair = None
                    best_pair_gain = 0
                    for x in range(N-1, 0, -1):
                        if x in temp_set:
                            continue
                        if not is_coprime_to_all(x, temp_set):
                            continue
                        for y in range(x-1, 0, -1):
                            if y in temp_set or y == x:
                                continue
                            if not is_coprime_to_all(y, temp_set | {x}):
                                continue
                            gain = weights[x] + weights[y] - weights[a] - weights[b]
                            if gain > best_pair_gain:
                                best_pair_gain = gain
                                best_pair = (x, y)
                    
                    if best_pair and best_pair_gain > 1e-12:
                        x, y = best_pair
                        current = [v for v in current if v != a and v != b]
                        current.extend([x, y])
                        current_set.discard(a)
                        current_set.discard(b)
                        current_set.add(x)
                        current_set.add(y)
                        current_score += best_pair_gain
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_chosen = list(current)
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        # Try adding elements
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for b in range(N-1, 0, -1):
                if time.time() > deadline - 5:
                    break
                if b in current_set:
                    continue
                if is_coprime_to_all(b, current_set):
                    current.append(b)
                    current_set.add(b)
                    current_score += weights[b]
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(current)
                    break
            
            if not improved:
                break
        
        # Final 1-swap pass after additions
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                temp_set = current_set - {a}
                
                best_b = -1
                best_gain = 0
                for b in range(N-1, 0, -1):
                    if b in temp_set:
                        continue
                    if is_coprime_to_all(b, temp_set):
                        gain = weights[b] - weights[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                
                if best_b != -1 and best_gain > 1e-12:
                    current = [x for x in current if x != a]
                    current.append(best_b)
                    current_set.remove(a)
                    current_set.add(best_b)
                    current_score += best_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(current)
                    break
            
            if not improved:
                break
    
    return best_chosen
