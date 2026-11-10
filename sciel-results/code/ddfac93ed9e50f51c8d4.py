import time
import random

N = 512

def construct():
    start_time = time.time()
    # Keep a safety margin. 90s limit, use 50s.
    deadline = start_time + 50.0 

    # Precompute all 4-term arithmetic progressions
    # aps[x] contains a list of tuples (a, b, c) such that {x, a, b, c} is a 4-AP
    # Note: The tuple elements are the OTHER three elements.
    aps = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    # a is the start, d is the difference
    # a, a+d, a+2d, a+3d <= N
    for a in range(1, N + 1):
        # Max d such that a + 3d <= N => d <= (N - a) / 3
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # The AP is (a, b, c, e)
            aps[a].append((b, c, e))
            aps[b].append((a, c, e))
            aps[c].append((a, b, e))
            aps[e].append((a, b, c))

    best_set = set()
    best_size = 0
    
    # We will run multiple restarts of the local search
    # Each restart starts with a greedy initialization (possibly randomized)
    # and then performs local search.
    
    while time.time() < deadline:
        # 1. Initialization: Greedy
        S = set()
        # conflict[x] is the number of 4-APs {x, u, v, w} such that {u, v, w} subset of S
        conflict = [0] * (N + 1)
        
        # Greedy fill
        # We can randomize the order to get different starting points
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if conflict[x] == 0:
                S.add(x)
                # Update conflict counts for partners
                for a, b, c in aps[x]:
                    # Check pairs in S
                    # If a, b in S, then c's conflict increases (because {x, a, b} are in S, 
                    # wait, conflict[c] counts APs {c, u, v, w} with u,v,w in S.
                    # The AP is {c, x, a, b}. If x, a, b are in S, then this AP contributes to conflict[c]?
                    # No. conflict[c] requires the OTHER 3 to be in S.
                    # The other 3 for c are x, a, b.
                    # So if x, a, b are in S, then conflict[c] should be incremented.
                    # But we are adding x. x is now in S.
                    # Before adding x, x was NOT in S.
                    # So the condition "x, a, b in S" was false (because x not in S).
                    # Now x is in S.
                    # So we need to check if a, b are in S.
                    # If a, b in S, then x, a, b are in S. So conflict[c] += 1.
                    
                    if a in S and b in S:
                        conflict[c] += 1
                    if a in S and c in S:
                        conflict[b] += 1
                    if b in S and c in S:
                        conflict[a] += 1

        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()

        # 2. Local Search
        # We try to add elements. If stuck, we remove one to try to break the deadlock.
        steps_since_improvement = 0
        max_steps = 10000 # Limit steps per restart to avoid wasting time on one bad path
        
        while steps_since_improvement < max_steps and time.time() < deadline:
            # Find all valid elements to add
            valids = []
            for x in range(1, N + 1):
                if x not in S and conflict[x] == 0:
                    valids.append(x)
            
            if valids:
                # Add a random valid element
                x = random.choice(valids)
                S.add(x)
                for a, b, c in aps[x]:
                    if a in S and b in S:
                        conflict[c] += 1
                    if a in S and c in S:
                        conflict[b] += 1
                    if b in S and c in S:
                        conflict[a] += 1
                
                # Improvement
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                steps_since_improvement = 0
            else:
                # Stuck: no element can be added
                if S:
                    # Remove a random element
                    x = random.choice(list(S))
                    S.remove(x)
                    for a, b, c in aps[x]:
                        if a in S and b in S:
                            conflict[c] -= 1
                        if a in S and c in S:
                            conflict[b] -= 1
                        if b in S and c in S:
                            conflict[a] -= 1
                    steps_since_improvement += 1
                else:
                    break

    return sorted(best_set)
