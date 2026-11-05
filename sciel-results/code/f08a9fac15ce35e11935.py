import time
import math
import random

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
    
    def greedy_init(order):
        chosen = []
        chosen_set = set()
        for a in order:
            if all(math.gcd(a, b) == 1 for b in chosen):
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
    for trial in range(10):
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
    
    candidates_list = [init1, best_random_init, init3]
    best_init = max(candidates_list, key=get_score)
    best_score = get_score(best_init)
    best_chosen = list(best_init)
    
    # Branch and bound with fractional knapsack upper bound
    def branch_and_bound():
        nonlocal best_score, best_chosen
        if time.time() > deadline:
            return
        
        # Order candidates by weight descending
        order = candidates  # already sorted by weight desc
        
        # Precompute for each position: which candidates are still available
        # State: (idx, chosen_set, current_score)
        # Use iterative DFS with explicit stack to avoid recursion depth issues
        
        # For upper bound: fractional knapsack on remaining candidates
        # that are coprime to all chosen, sorted by weight
        def upper_bound(idx, chosen_set, current_score):
            if time.time() > deadline:
                return 0.0
            # Greedy fractional knapsack: add fractions of remaining items
            # sorted by weight, skipping those conflicting with chosen
            ub = current_score
            remaining = [a for a in order[idx:] if a not in chosen_set]
            for a in remaining:
                if any(math.gcd(a, b) > 1 for b in chosen_set):
                    continue
                ub += weights[a]
            return ub
        
        # DFS with pruning
        stack = [(0, set(), 0.0)]
        nodes_visited = 0
        
        while stack:
            if time.time() > deadline:
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
            if all(math.gcd(a, b) == 1 for b in chosen_set):
                stack.append((idx + 1, chosen_set | {a}, current_score + weights[a]))
        
        return nodes_visited
    
    # Run branch and bound
    nodes = branch_and_bound()
    
    # Local search polish on best solution found
    if time.time() < deadline - 5:
        current = list(best_chosen)
        current_set = set(current)
        current_score = best_score
        
        # 1-swap local search
        for _ in range(50):  # limited iterations
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                a = current[i]
                temp = [x for x in current if x != a]
                temp_set = current_set - {a}
                
                best_b = -1
                best_gain = 0
                for b in range(N-1, a, -1):
                    if b in temp_set:
                        continue
                    if all(math.gcd(b, c) == 1 for c in temp):
                        gain = weights[b] - weights[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                
                if best_b != -1 and best_gain > 1e-12:
                    current = temp
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
        
        # Try adding elements
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for b in range(N-1, 0, -1):
                if time.time() > deadline:
                    break
                if b in current_set:
                    continue
                if all(math.gcd(b, c) == 1 for c in current):
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
    
    return best_chosen
