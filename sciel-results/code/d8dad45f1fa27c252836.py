import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # But the function signature is general. We assume the evaluator passes the specific task params.
    # If params are different, we might need a general solver, but the task says "Football pool problem for 7 matches".
    # The contract says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we should handle the specific case q=3, n=7, radius=1 efficiently.
    # If other params are passed, we can fall back to the seed or a generic greedy.
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_3_7_1()
    else:
        # Fallback to seed logic for other cases to ensure validity
        # Though the task is specific, safety first.
        if q == 3:
            # Try to adapt the tetracode logic if n is multiple of 4?
            # For now, return a valid but potentially suboptimal code
            # Simple greedy for general case
            return _generic_greedy(q, n, radius)
            
def _solve_3_7_1():
    N = 3**7
    # Generate all words
    words = [list(t) for t in itertools.product(range(3), repeat=7)]
    
    # Precompute Hamming distances or balls
    # Ball of radius 1 for word i: {i} U {words that differ in 1 pos}
    # To speed up, we can map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    balls = []
    for w in words:
        b = set()
        b.add(tuple(w))
        for i in range(7):
            for val in range(3):
                if w[i] != val:
                    new_w = w.copy()
                    new_w[i] = val
                    b.add(tuple(new_w))
        balls.append(b)
        
    # Convert balls to numpy arrays for fast coverage?
    # Or just use sets? N=2187, balls size 15.
    # Set operations are fast in C.
    
    def greedy_cover():
        uncovered = set(range(N))
        code = []
        # Precompute list of balls as lists for faster iteration?
        # balls_list = [list(b) for b in balls]
        
        while uncovered:
            best_idx = -1
            max_cover = -1
            # Optimization: Only consider candidates that cover at least one uncovered node
            # This is hard to filter quickly without precomputation.
            # Just scan all.
            for i in range(N):
                # Count how many of ball[i] are in uncovered
                # Using set intersection
                cover_count = len(balls[i] & uncovered)
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_idx = i
            code.append(best_idx)
            uncovered -= balls[best_idx]
        return code

    def reduce_cover(code):
        code_set = set(code)
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                # Check if all nodes in ball[c] are covered by code_set \ {c}
                # A node u is covered by code_set \ {c} if there exists v in code_set \ {c} such that u in ball[v]
                # Equivalent: Is ball[c] subset of Union(ball[v] for v in code_set \ {c})?
                # Faster: Check if any node in ball[c] is ONLY covered by c.
                # If no node is exclusively covered by c, we can remove c.
                
                # For each u in ball[c]:
                #   count how many v in code_set cover u.
                #   if count == 1 for all u in ball[c], then c is redundant?
                #   No, if for ALL u in ball[c], there is some OTHER v covering u, then c is redundant.
                #   i.e., ball[c] is subset of (Union of balls of others).
                
                others = code_set - {c}
                union_others = set()
                for v in others:
                    union_others |= balls[v]
                
                if balls[c].issubset(union_others):
                    code_set.remove(c)
                    changed = True
                    break # Restart loop to be safe
        return list(code_set)

    def local_search(code):
        code_set = set(code)
        current_size = len(code_set)
        best_size = current_size
        best_code = list(code_set)
        
        # Try swaps
        # To avoid O(N^2) full scan, we can try random swaps
        # Or structured swaps.
        
        for _ in range(5000): # Limit iterations
            if not code_set: break
            c = random.choice(list(code_set))
            # Find a candidate w not in code_set
            # Ideally one that covers nodes currently only covered by c?
            # Or just random.
            # To be efficient, pick a random word from the whole space
            w = random.randint(0, N-1)
            if w in code_set:
                continue
                
            new_set = (code_set - {c}) | {w}
            
            # Check if new_set is a valid cover
            # Union of balls in new_set must be all N
            # This is expensive to check fully.
            # Optimization: 
            # Nodes that were covered by c might now be uncovered.
            # Nodes that are covered by w are safe.
            # We only need to check nodes in ball[c] that are NOT in ball[w].
            # If those nodes are covered by (code_set - {c}), then it's valid.
            
            critical_nodes = balls[c] - balls[w]
            
            # Check if critical_nodes are covered by code_set - {c}
            valid = True
            for u in critical_nodes:
                # u is covered by v in code_set - {c} if u in balls[v]
                # We can check if any v in code_set - {c} has u in balls[v]
                # Since balls are small, we can precompute reverse balls?
                # reverse_balls[u] = list of v such that u in balls[v]
                pass
                
            # Let's precompute reverse_balls (who covers u)
            # reverse_balls[u] is the set of codewords that cover u.
            
            if valid:
                # Try to reduce new_set
                reduced = reduce_cover(list(new_set))
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_code = reduced
                    code_set = set(reduced)
                    current_size = len(code_set)
                    
        return best_code

    # Precompute reverse balls for speed
    reverse_balls = [set() for _ in range(N)]
    for i in range(N):
        for u in balls[i]:
            reverse_balls[u].add(i)

    def is_valid_cover(code_set):
        # Check if union of balls is N
        covered = set()
        for c in code_set:
            covered |= balls[c]
        return len(covered) == N

    def check_swap_valid(c, w, code_set):
        # c is in code_set, w is not
        # New set: (code_set - {c}) U {w}
        # Nodes that lose coverage: ball[c] - ball[w]
        # These must be covered by code_set - {c}
        critical = balls[c] - balls[w]
        for u in critical:
            # u must be in ball[v] for some v in code_set - {c}
            # i.e., u in balls[v]
            # Check reverse_balls[u]
            # If reverse_balls[u] has any element in code_set - {c}, it's covered.
            # Note: c is in reverse_balls[u] because u in ball[c].
            # We need another one.
            if len(reverse_balls[u] & (code_set - {c})) == 0:
                return False
        return True

    # Run Greedy
    code = greedy_cover()
    code = reduce_cover(code)
    
    # Run Local Search
    # Use the optimized check_swap_valid
    code_set = set(code)
    best_code = list(code_set)
    best_size = len(code_set)
    
    start_time = time.time()
    timeout = 110 # Leave 10s for safety
    
    while time.time() - start_time < timeout:
        # Try random swaps
        if not code_set: break
        c = random.choice(list(code_set))
        w = random.randint(0, N-1)
        if w in code_set:
            continue
            
        if check_swap_valid(c, w, code_set):
            new_set = (code_set - {c}) | {w}
            # Reduce
            reduced = reduce_cover(list(new_set))
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_code = reduced
                code_set = set(reduced)
            elif len(reduced) == best_size:
                # Accept to explore?
                # Let's just accept if it allows further reduction potential
                # For now, only accept if strictly better to ensure convergence
                pass
                
    # Convert best_code indices back to words
    return [words[i] for i in best_code]

def _generic_greedy(q, n, radius):
    # Fallback
    N = q**n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    # Simple greedy
    uncovered = set(range(N))
    code = []
    # Precompute balls
    balls = []
    for w in words:
        b = set()
        # Generate all neighbors within radius
        # This is expensive for large n, but n is small in fallback
        for dist in range(radius + 1):
            for combo in itertools.combinations(range(n), dist):
                for changes in itertools.product(range(1, q), repeat=dist):
                    new_w = w.copy()
                    for idx, ch in zip(combo, changes):
                        new_w[idx] = (new_w[idx] + ch) % q
                    b.add(tuple(new_w))
        balls.append(b)
        
    while uncovered:
        best_idx = -1
        max_cover = -1
        for i in range(N):
            cover_count = len(balls[i] & uncovered)
            if cover_count > max_cover:
                max_cover = cover_count
                best_idx = i
        code.append(best_idx)
        uncovered -= balls[best_idx]
        
    # Reduction
    code_set = set(code)
    changed = True
    while changed:
        changed = False
        for c in list(code_set):
            others = code_set - {c}
            union_others = set()
            for v in others:
                union_others |= balls[v]
            if balls[c].issubset(union_others):
                code_set.remove(c)
                changed = True
                break
    return [words[i] for i in code_set]
