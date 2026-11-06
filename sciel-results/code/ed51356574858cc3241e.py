import time
import random
from collections import deque

N = 12

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    
    # Use a list for the queue and a dictionary for visited to allow O(1) lookup
    # and FIFO behavior.
    visited = {start: 0}
    queue = deque([start])
    
    # Precompute transitions for speed? 
    # For N=12, subsets are 2^12=4096. 
    # Applying a transition to a subset takes O(|subset|).
    # We can represent subsets as bitmasks for faster operations if needed, 
    # but frozenset is okay for N=12 given the time limit.
    
    while queue:
        subset = queue.popleft()
        dist = visited[subset]
        
        if len(subset) == 1:
            return dist
            
        # Apply a
        next_subset_a = frozenset(trans_a[i] for i in subset)
        if next_subset_a not in visited:
            visited[next_subset_a] = dist + 1
            queue.append(next_subset_a)
            
        # Apply b
        next_subset_b = frozenset(trans_b[i] for i in subset)
        if next_subset_b not in visited:
            visited[next_subset_b] = dist + 1
            queue.append(next_subset_b)
            
    return -1

def construct():
    deadline = time.time() + 85.0 # Safety margin for 90s limit
    
    # We will use a "Chain Construction" strategy.
    # A synchronizing automaton can be viewed as a sequence of subset reductions.
    # Start with S0 = {0..N-1}.
    # We want to find a sequence S0 -> S1 -> ... -> Sk = {t} such that |Si| < |Si-1| usually,
    # or at least the distance from Si to a singleton is increasing.
    
    # However, finding such a chain directly is hard.
    # Alternative: Local Search on the "Subset Graph" of the automaton.
    # Let's stick to a robust Hill Climbing but with a better initial guess and smarter moves.
    # The current program's score is 40.
    # Known best for N=12 is likely around 40-50.
    # Let's try to generate automata that are "close" to known optimal structures.
    
    # Strategy:
    # 1. Generate random complete deterministic automata.
    # 2. If synchronizing, compute length.
    # 3. Use a "Simulated Annealing" like approach or "Random Restarts" with Hill Climbing.
    
    # To be materially different: 
    # Use a "Bidirectional Search" heuristic for construction? No, construction is the hard part.
    # Let's use "Structured Mutation":
    # Instead of changing one transition, change a "block" of transitions that preserves synchronizing property locally?
    # Or, better: Start with a known good structure (e.g., cyclic shift + one sink) and perturb.
    
    # Let's implement a fast evaluation loop.
    
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_len = compute_reset_length(best_a, best_b)
    
    # If best_len is low, try to find a better seed.
    # Random search for a good seed
    seeds = 0
    while time.time() < deadline - 10 and seeds < 200:
        seeds += 1
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        l = compute_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a
            best_b = b
            
    # Now Hill Climb from the best seed found
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Define a mutation strategy:
    # 1. Pick a state i.
    # 2. Pick a letter (a or b).
    # 3. Change the destination of that transition to a random state.
    # 4. Check if it's still synchronizing and if length increased.
    # 5. If length decreased, revert with high probability (Annealing).
    # 6. If length same, keep with some probability.
    
    T = 5.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    steps = 0
    while time.time() < deadline - 2:
        steps += 1
        
        # Pick a random state and letter
        i = random.randint(0, N-1)
        if random.random() < 0.5:
            letter = 0 # a
            old_dest = current_a[i]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            current_a[i] = new_dest
            new_len = compute_reset_length(current_a, current_b)
            if new_len < 0:
                # Not synchronizing, revert
                current_a[i] = old_dest
                continue
            delta = new_len - current_len
            
            # Acceptance criterion
            if delta > 0:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # delta <= 0
                # Accept if random < exp(delta / T)
                if random.random() < (2.71828 ** (delta / T)):
                    current_len = new_len
                else:
                    # Revert
                    current_a[i] = old_dest
            
        else:
            letter = 1 # b
            old_dest = current_b[i]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            current_b[i] = new_dest
            new_len = compute_reset_length(current_a, current_b)
            if new_len < 0:
                current_b[i] = old_dest
                continue
            delta = new_len - current_len
            
            if delta > 0:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                if random.random() < (2.71828 ** (delta / T)):
                    current_len = new_len
                else:
                    current_b[i] = old_dest
                    
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Restart if stuck for a long time?
        # Simple check: if T is very low and no improvement for X steps, restart from best
        # We don't track steps since improvement, let's add that.
        
    return [best_a, best_b]
