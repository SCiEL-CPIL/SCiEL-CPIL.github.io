import time
import random

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    Uses Simulated Annealing to optimize the set size by minimizing forbidden triples.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 180
    all_numbers = set(range(1, N + 1))
    
    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  a = bc / (b+c)
    triples = []
    triple_index = {} # index: (a,b,c) sorted -> list of indices in triples
    # We need a fast way to count active triples for a given set S.
    # Let's precompute for each pair (u, v), the set of w such that (u, v, w) is a triple.
    # Actually, for SA, we need to evaluate the "cost" of a set.
    # Cost = number of forbidden triples fully contained in S.
    
    # Precompute conflicts:
    # For each element x, store the set of pairs (y, z) such that {x, y, z} is a forbidden triple.
    # When we add x to S, the new triples formed are those where y, z are already in S.
    # When we remove x from S, the triples removed are those where y, z are in S.
    
    conflicts = {i: set() for i in range(1, N + 1)}
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            denom = b + c
            num = b * c
            if num % denom == 0:
                a = num // denom
                if 1 <= a <= N and a != b and a != c:
                    # Store sorted tuple to avoid duplicates
                    triple = tuple(sorted((a, b, c)))
                    # Add to conflicts
                    for x in triple:
                        others = [y for y in triple if y != x]
                        conflicts[x].add(tuple(sorted(others)))
    
    # Initial solution: Greedy construction to get a valid start
    def greedy_construct():
        S = set()
        # Sort by degree ascending to pick "safe" nodes first
        sorted_nodes = sorted(range(1, N + 1), key=lambda x: len(conflicts[x]))
        
        for v in sorted_nodes:
            is_safe = True
            for u, w in conflicts[v]:
                if u in S and w in S:
                    is_safe = False
                    break
            if is_safe:
                S.add(v)
        return S

    current_set = greedy_construct()
    current_size = len(current_set)
    
    # Calculate initial cost (should be 0 if greedy is valid)
    def count_active_triples(S):
        count = 0
        # This is O(|S| * avg_degree) which is fast enough for N=180
        # But we can maintain it incrementally.
        # Let's just use it for verification or if we do full re-evals.
        # For SA, we need incremental updates.
        return count # Placeholder, we will use incremental logic in SA

    # Simulated Annealing
    # State: S (set of kept numbers)
    # Objective: Maximize |S| subject to no forbidden triples.
    # Alternatively: Minimize "violations" while maximizing size.
    # Let's use a penalty method:
    # Energy E = -|S| + Lambda * (number of forbidden triples in S)
    # We want E to be low. If Lambda is large enough, the solution will have 0 triples.
    # Then we maximize |S|.
    
    # Better approach:
    # Keep S valid always? No, SA usually allows invalid states.
    # Let's define:
    # Score = |S| - M * (number of forbidden triples in S)
    # We want to maximize Score.
    # If M is large, it forces 0 triples.
    
    M = 1000  # Large penalty
    
    def get_score(S, active_triples_count):
        return len(S) - M * active_triples_count

    # Initial active triples count
    def count_triples_in_set(S):
        count = 0
        # Iterate through all precomputed triples and check if all 3 are in S
        # This is O(T) where T is number of triples.
        # T is small for N=180.
        for t in triples_list:
            if t[0] in S and t[1] in S and t[2] in S:
                count += 1
        return count

    # We need the list of unique triples for counting
    unique_triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            denom = b + c
            num = b * c
            if num % denom == 0:
                a = num // denom
                if 1 <= a <= N and a != b and a != c:
                    unique_triples.add(tuple(sorted((a, b, c))))
    triples_list = list(unique_triples)
    
    current_triples_count = count_triples_in_set(current_set)
    current_score = get_score(current_set, current_triples_count)
    
    best_set = set(current_set)
    best_score = current_score
    
    # SA Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    steps_per_temp = 50
    
    random.seed(42)
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Move 1: Add a random element not in S
            # Move 2: Remove a random element from S
            # Move 3: Swap an element in S with an element not in S
            
            move_type = random.choice([1, 2, 3])
            new_set = set(current_set)
            
            if move_type == 1:
                # Add
                candidates = [x for x in range(1, N + 1) if x not in current_set]
                if not candidates: continue
                x = random.choice(candidates)
                new_set.add(x)
            elif move_type == 2:
                # Remove
                if not current_set: continue
                x = random.choice(list(current_set))
                new_set.remove(x)
            else:
                # Swap
                in_set = list(current_set)
                out_set = [x for x in range(1, N + 1) if x not in current_set]
                if not in_set or not out_set: continue
                x_in = random.choice(in_set)
                x_out = random.choice(out_set)
                new_set.remove(x_in)
                new_set.add(x_out)
            
            # Calculate Delta Score
            # Delta |S|:
            # Add: +1
            # Remove: -1
            # Swap: 0
            
            # Delta Triples:
            # We need to count how many triples are added/removed.
            # This requires checking which triples involve the changed elements.
            
            # Let's just recount for simplicity if N=180 is small enough?
            # T ~ 100-200. Counting is O(T).
            # 50 steps * 1000 temps * 200 ops = 1,000,000 ops. Very fast.
            
            new_triples_count = count_triples_in_set(new_set)
            
            new_score = get_score(new_set, new_triples_count)
            delta_score = new_score - current_score
            
            # Acceptance
            if delta_score >= 0:
                current_set = new_set
                current_triples_count = new_triples_count
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                # Metropolis criterion
                # P = exp(delta / T)
                # Since delta is negative, delta/T is negative.
                try:
                    prob = random.expovariate(1.0)
                    if prob < -delta_score / T:
                        current_set = new_set
                        current_triples_count = new_triples_count
                        current_score = new_score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                except OverflowError:
                    pass # delta is very negative, probability is ~0
            
            T *= cooling_rate
        
        # Optional: Re-seed or adjust if stuck?
        # For now, standard cooling.
        
    # Ensure we return a valid set (0 triples)
    # If best_set has triples, we need to clean it up.
    # Since M is large, best_set likely has 0 triples.
    # But to be safe, if best_set has triples, fall back to greedy or clean it.
    
    # Verify best_set
    if count_triples_in_set(best_set) > 0:
        # Clean up by removing elements from conflicting triples
        # Greedy removal
        S = set(best_set)
        while count_triples_in_set(S) > 0:
            # Find a triple
            removed = False
            for t in triples_list:
                if t[0] in S and t[1] in S and t[2] in S:
                    # Remove the element with highest degree in S? Or just first?
                    # Let's remove one element from this triple
                    # To maximize size, we should remove the element that is in the most other triples.
                    # But for now, just remove t[0] to fix validity quickly.
                    # Better: try removing each and pick the one that minimizes new triples?
                    # Simpler: Just remove t[0].
                    S.remove(t[0])
                    removed = True
                    break
            if not removed:
                break
        
        if len(S) > len(best_set): # Unlikely
             best_set = S
        else:
             # If cleaning reduced size significantly, maybe greedy was better?
             # Let's compare with initial greedy
             greedy_set = greedy_construct()
             if len(greedy_set) > len(S):
                 best_set = greedy_set
             else:
                 best_set = S
    
    # Final check
    if count_triples_in_set(best_set) > 0:
        # Last resort: return all numbers > 90? 
        # 1/91 > 1/180 + 1/180? 
        # 1/180+1/180 = 1/90. 
        # So 1/91 < 1/90. 
        # If a, b, c > 90, then 1/a < 1/90, 1/b+1/c > 1/180+1/180 = 1/90? No.
        # Max sum 1/b+1/c for b,c>90 is 1/91+1/92 < 1/90.
        # Min 1/a for a>90 is 1/180.
        # Is it possible 1/a = 1/b+1/c?
        # 1/a >= 1/b+1/c => a <= bc/(b+c).
        # If b,c > 90, bc/(b+c) < 180/2 = 90.
        # So a < 90. But a > 90. Contradiction.
        # So any subset of {91, ..., 180} is valid.
        # Size = 90.
        fallback = set(range(91, 181))
        if len(best_set) < len(fallback):
            best_set = fallback
            
    return list(best_set)

# Test run
if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    S = set(res)
    valid = True
    N = 180
    # Re-check triples
    for a in S:
        for b in S:
            if a == b: continue
            for c in S:
                if a == c or b == c: continue
                if 1.0/a == 1.0/b + 1.0/c:
                    valid = False
                    print(f"Found conflict: {a}, {b}, {c}")
    print("Valid:", valid)
