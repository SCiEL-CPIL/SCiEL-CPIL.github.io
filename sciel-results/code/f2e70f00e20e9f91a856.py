import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def build_edge_index():
    idx = [[-1]*N for _ in range(N)]
    counter = 0
    for i in range(N):
        for j in range(i+1, N):
            idx[i][j] = counter
            idx[j][i] = counter
            counter += 1
    return idx

EDGE_IDX = build_edge_index()

def build_cycles():
    cycles = []
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def is_valid(colors):
    for cycle in CYCLES:
        if len(set(colors[e] for e in cycle)) > 3:
            return False
    return True

def can_take_color(colors, edge, new_color):
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        seen = set()
        for e in cycle:
            c = new_color if e == edge else colors[e]
            if c in seen:
                continue
            seen.add(c)
            if len(seen) > 3:
                return False
    return True

def local_search(colors, deadline):
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Kick: randomly recolor 5 edges to valid colors (might reduce count)
            for _ in range(5):
                e = random.randint(0, NUM_EDGES - 1)
                # Find a valid color for e
                # Try current color first, then randoms
                valid_found = False
                for _ in range(20):
                    nc = random.randint(0, 119)
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        valid_found = True
                        break
                if not valid_found:
                    # Revert to a safe color? Or just skip
                    pass
            
            used_colors = set(colors)
            # Reset no_improve
            no_improve_count = 0
            max_no_improve = random.randint(1000, 5000)
            continue

        # Try to improve
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            # Try to assign a new color
            # Prioritize unused colors
            all_colors = set(range(120))
            available = list(all_colors - used_colors)
            random.shuffle(available)
            
            # Also try some used colors just in case (re-coloring)
            # But adding new colors is the goal
            
            for nc in available:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                if len(used_colors) > best_score:
                    best_score = len(used_colors)
                    best_colors = list(colors)
                # Continue trying to add more in this pass
            else:
                no_improve_count += 1
        
        if not improved:
             no_improve_count += 100 # Penalize full pass without improvement

    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Try multiple random permutations for the base
    for perm_iter in range(50):
        if time.time() >= deadline:
            break
        
        perm = list(range(N))
        random.shuffle(perm)
        
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                e = EDGE_IDX[i][j]
                colors[e] = min(perm[i], perm[j])
        
        if not is_valid(colors):
            continue
            
        # Run local search with a portion of the time
        # We want to try many restarts
        # Estimate time per restart
        # If we have 50s left, and 50 restarts, 1s each?
        # Let's just run local search until deadline if it's the last one,
        # or for a fixed small time.
        
        # Better: run local search, if it finds something good, keep it.
        # To allow many restarts, we can limit local search time.
        
        # Strategy: 
        # 1. Quick local search (0.5s) to see potential.
        # 2. If potential > best_score, run longer local search on this start.
        
        # For simplicity, just run local search with the global deadline for the last few,
        # and short timeouts for the first few.
        
        if perm_iter < 40:
            ls_deadline = time.time() + 0.5
        else:
            ls_deadline = deadline
            
        improved_colors, score = local_search(colors, ls_deadline)
        
        if score > best_score:
            best_score = score
            best_colors = improved_colors
            
        # If we hit the theoretical max (or close), we can stop
        if best_score >= 25: # Arbitrary high target
            break

    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
