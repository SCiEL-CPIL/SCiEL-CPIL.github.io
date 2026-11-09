import time
import random
import math

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Precompute coordinates
    coords = [(i // N, i % N) for i in range(NUM_POINTS)]
    
    # Data structures for the set S
    # S is a list of point indices
    # dist_from is a list of sets, dist_from[i] contains squared distances from i to other points in S
    
    def can_add(S, dist_from, p):
        if not S:
            return True
        px, py = coords[p]
        d2s = set()
        for a in S:
            ax, ay = coords[a]
            dx = px - ax
            dy = py - ay
            d2 = dx*dx + dy*dy
            if d2 in d2s:
                return False
            d2s.add(d2)
            if d2 in dist_from[a]:
                return False
        return True

    def add_point(S, dist_from, p):
        S.append(p)
        px, py = coords[p]
        dist_from[p] = set()
        for a in S[:-1]:
            ax, ay = coords[a]
            dx = px - ax
            dy = py - ay
            d2 = dx*dx + dy*dy
            dist_from[a].add(d2)
            dist_from[p].add(d2)

    def remove_point(S, dist_from, p):
        idx = S.index(p)
        S.pop(idx)
        px, py = coords[p]
        for a in S:
            ax, ay = coords[a]
            dx = px - ax
            dy = py - ay
            d2 = dx*dx + dy*dy
            dist_from[a].discard(d2)
        dist_from[p] = set()

    # Initial Greedy Construction to get a good starting point
    def initial_greedy():
        S = []
        dist_from = [set() for _ in range(NUM_POINTS)]
        points = list(range(NUM_POINTS))
        random.shuffle(points)
        for p in points:
            if can_add(S, dist_from, p):
                add_point(S, dist_from, p)
        return S, dist_from

    S, dist_from = initial_greedy()
    best_S = S[:]
    best_dist = [s.copy() for s in dist_from]
    best_size = len(S)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We need to track the "state" of S and dist_from
    # We will operate on the current S and dist_from
    
    current_size = len(S)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move
        # 1. Add point (if possible)
        # 2. Remove point
        # 3. Swap point
        
        move_type = random.random()
        
        if move_type < 0.5 and current_size < NUM_POINTS:
            # Try to add
            p = random.randint(0, NUM_POINTS - 1)
            if p not in S:
                if can_add(S, dist_from, p):
                    add_point(S, dist_from, p)
                    current_size += 1
                    # Always accept improvement
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S[:]
                        best_dist = [s.copy() for s in dist_from]
                else:
                    # Reject, do nothing
                    pass
        elif move_type < 0.75 and current_size > 0:
            # Remove point
            q = random.choice(S)
            remove_point(S, dist_from, q)
            current_size -= 1
            # This is a "worse" move (size decreases)
            # Accept with probability 1 (since we want to explore) or based on T?
            # In SA, we accept worse moves with prob exp(-DeltaE/T).
            # Here DeltaE = -1 (cost increases).
            # Prob = exp(1/T). If T is high, prob is high.
            # But we want to keep the size high. 
            # Standard SA for maximization: 
            # If new_size < old_size, accept with prob exp((new_size - old_size)/T) = exp(-1/T).
            # If T is large, exp(-1/T) is close to 1.
            # If T is small, exp(-1/T) is close to 0.
            # So we should REJECT with prob 1 - exp(-1/T).
            # Or ACCEPT with prob exp(-1/T).
            
            # Let's implement the acceptance properly.
            # We already performed the move. Now decide if we keep it.
            # Actually, it's easier to decide BEFORE performing the move if it's a worsening move.
            # But for "Add", it's always improvement.
            # For "Remove", it's always worsening.
            # For "Swap", it's neutral (size same), so always accept if valid.
            
            # Let's restart the loop logic to be cleaner.
            # We will propose a move, calculate delta, and decide.
            
            # Revert for now, we will re-implement in the loop below.
            add_point(S, dist_from, q)
            current_size += 1
            
        else:
            # Swap
            if current_size > 0:
                q = random.choice(S)
                p = random.randint(0, NUM_POINTS - 1)
                while p == q or p in S:
                    p = random.randint(0, NUM_POINTS - 1)
                
                # Try swap: remove q, add p
                remove_point(S, dist_from, q)
                if can_add(S, dist_from, p):
                    add_point(S, dist_from, p)
                    # Size unchanged, accept
                else:
                    # Revert
                    add_point(S, dist_from, q)
            
        T *= cooling_rate

    # Convert best_S to tuples
    return [coords[i] for i in best_S]
