import time
import random

def construct():
    N = 256
    
    # Precompute APs
    ap3_containing = [[] for _ in range(N+1)]
    ap4_containing = [[] for _ in range(N+1)]
    
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            ap3_containing[a].append((b, c))
            ap3_containing[b].append((a, c))
            ap3_containing[c].append((a, b))
            
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            ap4_containing[a].append((b, c, e))
            ap4_containing[b].append((a, c, e))
            ap4_containing[c].append((a, b, e))
            ap4_containing[e].append((a, b, c))

    def count_3aps(S):
        count = 0
        # Iterate through all 3-APs
        # To be faster, we can iterate through elements in S and check?
        # No, precomputed list of all 3-APs is better.
        # There are ~20k 3-APs.
        for a in range(1, N+1):
             for b, c in ap3_containing[a]:
                 if b in S and c in S:
                     # Avoid double counting? 
                     # ap3_containing[a] has (b,c) where a is part of the AP.
                     # Each AP (a,b,c) is counted 3 times? 
                     # No, ap3_containing[a] contains pairs (b,c) such that (a,b,c) is an AP.
                     # If we sum for all a in S, we count each AP 3 times.
                     # So divide by 3.
                     count += 1
        return count // 3

    # Wait, the logic above is:
    # For each a in S, for each (b,c) in ap3_containing[a], if b,c in S, then (a,b,c) is a 3-AP in S.
    # Each 3-AP {x,y,z} will be counted when a=x (if y,z in S), a=y (if x,z in S), a=z (if x,y in S).
    # Since x,y,z are all in S, it is counted exactly 3 times.
    # So count // 3 is correct.
    
    # This is O(|S| * avg_degree). Much faster than iterating all 20k APs if |S| is small, but |S| is ~100.
    # 100 * 50 = 5000 ops. vs 20000 ops. Comparable.
    # But checking `b in S` is O(1) for set.
    
    def is_valid_addition(S, x):
        for others in ap4_containing[x]:
            if all(o in S for o in others):
                return False
        return True

    def get_delta_3aps(S, x):
        delta = 0
        for others in ap3_containing[x]:
            if all(o in S for o in others):
                delta += 1
        return delta

    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    best_set = set()
    best_score = -1
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        S = set()
        available = list(range(1, N+1))
        random.shuffle(available)
        
        # Adaptive randomness: 
        # Early in construction, be more random to explore structure.
        # Late, be greedy to fill gaps?
        # Actually, greedy (max delta) is usually good for 3-APs.
        # Let's use a "temperature" that cools down.
        # T = 1.0 at start, 0.1 at end.
        
        steps = 0
        max_steps = N
        while available and steps < max_steps:
            steps += 1
            valid_candidates = []
            
            # Find valid candidates
            # Optimization: We can break early if we find a candidate with high delta?
            # No, we need all to choose from.
            
            for x in available:
                if is_valid_addition(S, x):
                    delta = get_delta_3aps(S, x)
                    valid_candidates.append((x, delta))
            
            if not valid_candidates:
                break
            
            # Selection
            # Sort by delta
            valid_candidates.sort(key=lambda item: item[1], reverse=True)
            
            # Temperature based on progress
            progress = steps / max_steps
            T = 0.5 * (1 - progress) + 0.1
            
            # Softmax selection
            # Score = exp(delta / T)
            # But delta can be 0.
            
            # Simple heuristic: 
            # Pick from top 10% with prob 0.8, else random.
            k = max(1, len(valid_candidates) // 10)
            
            if random.random() < 0.8:
                # Pick from top k
                x_to_add = random.choice(valid_candidates[:k])[0]
            else:
                # Pick from all
                x_to_add = random.choice(valid_candidates)[0]
            
            S.add(x_to_add)
            available.remove(x_to_add)
            
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_set = S.copy()
            
    return sorted(best_set)
