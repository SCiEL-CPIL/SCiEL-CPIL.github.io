"""Improved program using local search to minimize chords."""
import random
import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Initial solution: 1..19
    best_set = set(range(1, 20))
    best_score = len(best_set)
    
    def get_cycle_lengths(S):
        S = sorted(S)
        lengths = set()
        lengths.add(N) # 40
        for s in S:
            lengths.add(1 + s)
            lengths.add(41 - s)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                lengths.add(2 + d)
                lengths.add(42 - d)
        return lengths
        
    def is_valid(S):
        lengths = get_cycle_lengths(S)
        return all(k in lengths for k in range(3, N + 1))

    if not is_valid(best_set):
        # Fallback
        return {'edges': [(i, (i+1)%N) for i in range(N)] + [(0, 2*j) for j in range(1, 20)], 'cycles': []}
        
    # Local search: try to remove elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for s in list(best_set):
            new_set = best_set - {s}
            if is_valid(new_set):
                best_set = new_set
                best_score = len(best_set)
                improved = True
                break
        
        # Try to swap elements
        for s in list(best_set):
            for new_s in range(1, N):
                if new_s not in best_set:
                    new_set = (best_set - {s}) | {new_s}
                    if is_valid(new_set):
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break

    # Construct edges
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
    for s in sorted(best_set):
        edges.append((0, s))
        
    # Construct cycles (verification and certificate)
    cycles = []
    lengths_map = {}
    S = sorted(best_set)
    for s in S:
        lengths_map.setdefault(1 + s, (0, s, 'single_short'))
        lengths_map.setdefault(41 - s, (0, s, 'single_long'))
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = S[j] - S[i]
            lengths_map.setdefault(2 + d, (S[i], S[j], 'pair_short'))
            lengths_map.setdefault(42 - d, (S[i], S[j], 'pair_long'))
    lengths_map[40] = (0, 39, 'outer')
    
    for k in range(3, N + 1):
        if k in lengths_map:
            info = lengths_map[k]
            if info[2] == 'outer':
                cycles.append(list(range(N)))
            elif info[2] == 'single_short':
                s = info[1]
                cycles.append(list(range(0, s + 1)))
            elif info[2] == 'single_long':
                s = info[1]
                cycles.append([0] + list(range(39, s - 1, -1)))
            elif info[2] == 'pair_short':
                a, b = info[0], info[1]
                cycles.append([0] + list(range(a, b + 1)))
            elif info[2] == 'pair_long':
                a, b = info[0], info[1]
                cycles.append([0] + list(range(39, a - 1, -1)) + list(range(b, a - 1, -1)))
        else:
            # Should not happen if valid
            cycles.append([])

    return {'edges': edges, 'cycles': cycles}
