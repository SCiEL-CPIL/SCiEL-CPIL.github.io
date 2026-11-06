"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    # 1. Identify all odd numbers. They are mutually compatible.
    odds = [i for i in range(1, N + 1) if i % 2 == 1]
    evens = [i for i in range(1, N + 1) if i % 2 == 0]
    
    # 2. Filter evens to find those compatible with ALL odds.
    safe_evens = []
    for e in evens:
        is_safe = True
        for o in odds:
            if (e * o) % (e + o) == 0:
                is_safe = False
                break
        if is_safe:
            safe_evens.append(e)
    
    # 3. Build conflict graph for safe_evens.
    # We want Max Independent Set in this subgraph.
    M = len(safe_evens)
    if M == 0:
        return odds
        
    # conflict_e[i][j] is True if safe_evens[i] and safe_evens[j] conflict
    conflict_e = [[False] * M for _ in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            e1 = safe_evens[i]
            e2 = safe_evens[j]
            if (e1 * e2) % (e1 + e2) == 0:
                conflict_e[i][j] = True
                conflict_e[j][i] = True
    
    # 4. Find Max Independent Set in conflict_e using Greedy + Local Search
    # Greedy initialization: Sort by degree, pick lowest degree first
    degrees = [sum(conflict_e[i]) for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x])
    
    selected_idx = []
    selected_set = set()
    
    for idx in order:
        if time.time() - start_time > time_limit:
            break
        compatible = True
        for sel in selected_idx:
            if conflict_e[idx][sel]:
                compatible = False
                break
        if compatible:
            selected_idx.append(idx)
            selected_set.add(idx)
            
    # Local Search: Try to add any remaining safe_even
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for idx in range(M):
            if idx not in selected_set:
                compatible = True
                for sel in selected_set:
                    if conflict_e[idx][sel]:
                        compatible = False
                        break
                if compatible:
                    selected_set.add(idx)
                    improved = True
                    break
    
    # Local Search: Swaps (optional, if time permits)
    # Try swapping out one selected idx and in another to potentially allow more additions
    # This is complex, so we stick to Add-only if time is tight.
    # Given M is likely small (<50), Add-only is fast.
    
    final_evens = [safe_evens[i] for i in selected_set]
    
    result = odds + final_evens
    return result
# EVOLVE-BLOCK-END
