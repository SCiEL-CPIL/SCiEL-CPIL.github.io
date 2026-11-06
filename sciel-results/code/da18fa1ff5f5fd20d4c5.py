import time
import random
import numpy as np
from collections import deque

N = 12

def _compute_threshold(trans):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Uses integer bitmasks for speed.
    Returns -1 if not synchronizing.
    """
    # trans: list of two lists, trans[0], trans[1]
    # Convert to numpy arrays for faster mapping if needed, but for N=12, 
    # bit manipulation on Python ints is often faster than numpy overhead for small N
    # However, we can use a lookup table for subset transitions if we precompute?
    # No, subsets are dynamic. 
    # Let's stick to Python ints for bitsets.
    
    # Precompute transition of single states?
    # t0[s], t1[s]
    t0 = trans[0]
    t1 = trans[1]
    
    # Function to apply transition to a bitmask
    # For N=12, we can iterate bits or use a precomputed table for 0..2^12-1?
    # 2^12 = 4096. Precomputing table for a specific automaton is 4096 * 2 operations.
    # This is very fast.
    
    # Precompute next_state_table for this specific automaton
    # table[letter][mask] = next_mask
    # But we only have 2 letters.
    # Let's precompute for each letter a list of length 2^N
    # Actually, iterating 4096 entries is fine.
    
    # Optimization: Precompute the image of each singleton for each letter
    # img0[s] = 1 << t0[s]
    # img1[s] = 1 << t1[s]
    img0 = [1 << t0[s] for s in range(N)]
    img1 = [1 << t1[s] for s in range(N)]
    
    # To compute next_mask for a given mask:
    # next_mask = OR of img[letter][s] for all s in mask
    
    # We can precompute a table for all 2^N masks?
    # table0[mask] = next_mask for letter 0
    # table1[mask] = next_mask for letter 1
    # Size: 4096 * 2 ints. Memory is trivial. Time to build: 4096 * N ops ~ 50k ops.
    # This will make BFS extremely fast.
    
    size = 1 << N
    table0 = [0] * size
    table1 = [0] * size
    
    # Build tables
    # DP: table[mask] = table[mask & (mask-1)] | img[lsb_index]
    # lsb_index = (mask & -mask).bit_length() - 1
    
    for mask in range(1, size):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        rest = mask ^ lsb
        table0[mask] = table0[rest] | img0[idx]
        table1[mask] = table1[rest] | img1[idx]
        
    full_set = (1 << N) - 1
    
    visited = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current & (current - 1) == 0: # Singleton check
            return dist
            
        next0 = table0[current]
        next1 = table1[current]
        
        if next0 not in visited:
            visited[next0] = dist + 1
            queue.append(next0)
            
        if next1 not in visited:
            visited[next1] = dist + 1
            queue.append(next1)
            
    return -1

def _seed_population():
    """
    Generate diverse seeds.
    """
    population = []
    rng = random.Random(123)
    
    # 1. Classic Cerny
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    population.append([a, b])
    
    # 2. Random Permutations + Random Functions
    for _ in range(100):
        a = list(range(N))
        rng.shuffle(a)
        b = [rng.randint(0, N-1) for _ in range(N)]
        population.append([a, b])
        
    # 3. Structured "Pinwheel" like
    # Cycle for a, b maps i to i-1 with some defects
    for _ in range(100):
        a = list(range(N))
        rng.shuffle(a)
        b = [0] * N
        shift = rng.randint(1, N-1)
        for i in range(N):
            b[i] = (i - shift) % N
        # Defects
        num_defects = rng.randint(1, 3)
        for _ in range(num_defects):
            idx = rng.randint(0, N-1)
            b[idx] = rng.randint(0, N-1)
        population.append([a, b])
        
    return population

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin for 90s limit
    
    # Initialize Islands
    # 3 Islands, each with 20 individuals
    num_islands = 3
    pop_size = 20
    
    all_seeds = _seed_population()
    rng = random.Random(456)
    
    # Distribute seeds to islands
    islands = []
    for i in range(num_islands):
        island_pop = []
        # Pick random seeds for this island
        for _ in range(pop_size):
            trans = rng.choice(all_seeds)
            # Deep copy
            trans = [trans[0][:], trans[1][:]]
            score = _compute_threshold(trans)
            island_pop.append((score, trans))
        
        # Sort by score
        island_pop.sort(key=lambda x: x[0], reverse=True)
        islands.append(island_pop)
        
    # Track global best
    global_best_score = -1
    global_best_trans = None
    for island in islands:
        if island and island[0][0] > global_best_score:
            global_best_score = island[0][0]
            global_best_trans = [island[0][1][0][:], island[0][1][1][:]]
            
    # Evolution Loop
    while time.time() < deadline:
        # Select Island
        # Roulette wheel or just random? Random is fine for diversity.
        # Or select island with best recent performance? 
        # Let's use random island selection to maintain diversity.
        island_idx = rng.randint(0, num_islands - 1)
        island = islands[island_idx]
        
        # Select Parents
        # Tournament selection within island
        # Pick 2 random, take best
        p1_idx = rng.randint(0, len(island) - 1)
        p2_idx = rng.randint(0, len(island) - 1)
        
        if island[p1_idx][0] >= island[p2_idx][0]:
            parent1 = island[p1_idx][1]
            parent1_score = island[p1_idx][0]
        else:
            parent1 = island[p2_idx][1]
            parent1_score = island[p2_idx][0]
            
        if rng.random() < 0.5:
             # Crossover
             # Swap letter 0 or letter 1 between parents?
             # Or mix?
             # Structure preserving: Let's try swapping entire letter mappings.
             # Child gets letter 0 from P1 and letter 1 from P2, or vice versa.
             # Or better: Randomly pick for each state? No, that breaks structure.
             # Let's swap one letter.
             if rng.random() < 0.5:
                 child_trans = [parent1[0][:], parent1[1][:]]
                 # Find a second parent from the same island, distinct from parent1 if possible
                 # Just pick another random from island
                 second_parent = rng.choice(island)[1]
                 child_trans[1] = second_parent[1][:]
             else:
                 child_trans = [parent1[0][:], parent1[1][:]]
                 second_parent = rng.choice(island)[1]
                 child_trans[0] = second_parent[0][:]
        else:
             # Mutation
             child_trans = [parent1[0][:], parent1[1][:]]
             # Mutate 1-2 random entries
             num_mutations = rng.randint(1, 2)
             for _ in range(num_mutations):
                 letter = rng.randint(0, 1)
                 state = rng.randint(0, N - 1)
                 new_dest = rng.randint(0, N - 1)
                 child_trans[letter][state] = new_dest
                 
        # Evaluate Child
        child_score = _compute_threshold(child_trans)
        
        if child_score > -1:
            # Update Global Best
            if child_score > global_best_score:
                global_best_score = child_score
                global_best_trans = [child_trans[0][:], child_trans[1][:]]
                
            # Replace worst in island if child is better than worst
            if child_score > island[-1][0]:
                island[-1] = (child_score, child_trans)
                island.sort(key=lambda x: x[0], reverse=True)
            elif len(island) < pop_size:
                island.append((child_score, child_trans))
                island.sort(key=lambda x: x[0], reverse=True)
            else:
                # If not better than worst, maybe replace a random one to maintain diversity?
                # Or just discard. Let's discard to keep elite pressure.
                pass
                
        # Migration: Occasionally copy best from one island to another
        if rng.random() < 0.1:
            # Pick source and destination island
            src_idx = rng.randint(0, num_islands - 1)
            dst_idx = rng.randint(0, num_islands - 1)
            if src_idx != dst_idx and islands[src_idx]:
                best_in_src = islands[src_idx][0]
                # Replace worst in dst
                if best_in_src[0] > islands[dst_idx][-1][0]:
                    islands[dst_idx][-1] = best_in_src
                    islands[dst_idx].sort(key=lambda x: x[0], reverse=True)
                    
    # Return best found
    if global_best_trans is None:
        # Fallback to a known valid one if all failed (shouldn't happen with seeds)
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        for i in range(1, N):
            b[i] = i - 1
        return [a, b]
        
    return global_best_trans

# EVOLVE-BLOCK-END
