import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55  # safety margin before 60s
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict sets: conflict[i] is a set of j (0-indexed) such that (i+1)+(j+1) is a square
    # Include self-conflicts: if 2*(i+1) is a square, then i conflicts with itself
    conflict = [set() for _ in range(N)]
    for i in range(N):
        a = i + 1
        for s in squares:
            b = s - a
            if 1 <= b <= N:
                j = b - 1
                conflict[i].add(j)
    
    # Check self-conflicts: if i is in conflict[i], then vertex i+1 cannot be in any valid set
    # (because 2a would be a square)
    
    def is_valid_solution(S):
        """Check if S (list of 1-based values) is valid."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a <= b:
                    if (a + b) in squares:
                        return False
        return True
    
    def greedy_construct(order):
        """Greedy: add vertices in given order if no conflict with already added."""
        chosen = set()
        for idx in order:
            # Check if idx conflicts with any chosen
            ok = True
            for c in chosen:
                if c in conflict[idx]:
                    ok = False
                    break
            if ok:
                chosen.add(idx)
        return chosen
    
    def local_search(S, max_iters=5000):
        """Try to improve S by swapping or adding."""
        S = set(S)
        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            # Try adding each non-chosen vertex
            for v in range(N):
                if v not in S:
                    if not (conflict[v] & S):
                        S.add(v)
                        improved = True
                        break
            if improved:
                continue
            # Try swapping: remove one, add one or two
            for v in list(S):
                S.remove(v)
                added = 0
                for u in range(N):
                    if u not in S:
                        if not (conflict[u] & S):
                            S.add(u)
                            added += 1
                if added > 0:
                    improved = True
                else:
                    S.add(v)  # put back
            if (time.time() - start_time) > DEADLINE:
                break
        return S
    
    best_solution = None
    
    # Try multiple random orderings
    num_attempts = 200
    for attempt in range(num_attempts):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        
        random.seed(attempt * 7919 + 13)
        order = list(range(N))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = local_search(S)
        
        if best_solution is None or len(S) > len(best_solution):
            best_solution = S
    
    # Final local search on best
    if (time.time() - start_time) < DEADLINE - 2:
        best_solution = local_search(best_solution)
    
    # Convert to 1-based list
    result = sorted(v + 1 for v in best_solution)
    
    # Verify
    if not is_valid_solution(result):
        # Fallback: just return a safe small set
        result = [1]
    
    return result
