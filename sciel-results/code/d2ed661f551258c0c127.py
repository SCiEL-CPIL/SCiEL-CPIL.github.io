import time
import random
import math
from collections import deque

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton
    with precomputed letter transition tables for bitmasks.
    Returns the length, or -1 if not synchronizing.
    """
    total_subsets = 1 << N
    trans_a_table = [0] * total_subsets
    trans_b_table = [0] * total_subsets
    
    for mask in range(1, total_subsets):
        lsb = mask & (-mask)
        s = lsb.bit_length() - 1
        rest = mask ^ lsb
        if rest:
            trans_a_table[mask] = trans_a_table[rest] | (1 << trans_a[s])
            trans_b_table[mask] = trans_b_table[rest] | (1 << trans_b[s])
        else:
            trans_a_table[mask] = 1 << trans_a[s]
            trans_b_table[mask] = 1 << trans_b[s]
    
    start_state = total_subsets - 1
    visited = bytearray(total_subsets)
    visited[start_state] = 1
    queue = deque([(start_state, 0)])
    
    while queue:
        state, dist = queue.popleft()
        if state & (state - 1) == 0: 
            return dist
        
        ns0 = trans_a_table[state]
        if not visited[ns0]:
            visited[ns0] = 1
            queue.append((ns0, dist + 1))
            
        ns1 = trans_b_table[state]
        if not visited[ns1]:
            visited[ns1] = 1
            queue.append((ns1, dist + 1))
            
    return -1

def construct():
    """
    Returns [trans_a, trans_b] for a 12-state synchronizing automaton maximizing
    the exact shortest reset word length.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best_threshold = -1
    best_a = None
    best_b = None
    
    def update_best(a, b):
        nonlocal best_threshold, best_a, best_b
        t = compute_threshold(a, b)
        if t > best_threshold:
            best_threshold = t
            best_a = a[:]
            best_b = b[:]
        return t
    
    # Family 1: Černý automaton (cycle + decrement)
    a_cz = [(i + 1) % N for i in range(N)]
    b_cz = [0 if i == 0 else i - 1 for i in range(N)]
    update_best(a_cz, b_cz)
    
    # Family 2: Cycle + single sink
    for sink in range(N):
        a_cy = [(i + 1) % N for i in range(N)]
        b_sink = [sink] * N
        update_best(a_cy, b_sink)
        if time.time() > deadline:
            break
    
    # Family 3: Cycle + "reset to k" pattern (b maps i to max(0, i-k))
    for k in range(1, N):
        a_cy = [(i + 1) % N for i in range(N)]
        b_dec = [max(0, i - k) for i in range(N)]
        update_best(a_cy, b_dec)
        if time.time() > deadline:
            break
    
    # Family 4: Cycle + "fold" pattern
    a_cy = [(i + 1) % N for i in range(N)]
    b_fold = [min(i, N - 1 - i) for i in range(N)]
    update_best(a_cy, b_fold)
    
    # Family 5: Systematic exploration of "cycle + structured b with exceptions"
    for offset in range(N):
        if time.time() > deadline:
            break
        a_cy = [(i + 1) % N for i in range(N)]
        b_base = [(i + offset) % N for i in range(N)]
        update_best(a_cy, b_base)
        # With one exception
        for exc_state in range(N):
            for exc_dest in range(N):
                if time.time() > deadline:
                    break
                b_mod = b_base[:]
                b_mod[exc_state] = exc_dest
                update_best(a_cy, b_mod)
    
    # Family 6: Perturbed cycles with local search
    for restart in range(15):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a random permutation as "cycle-like" a
        a_perm = list(range(N))
        random.shuffle(a_perm)
        b_rand = [random.randint(0, N - 1) for _ in range(N)]
        cur_t = update_best(a_perm, b_rand)
        
        # Local search on both a and b
        for ls_iter in range(20):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Try changing b
            for s in range(N):
                old = b_rand[s]
                for nd in range(N):
                    if nd == old:
                        continue
                    b_rand[s] = nd
                    nt = compute_threshold(a_perm, b_rand)
                    if nt > cur_t:
                        cur_t = nt
                        improved = True
                        if cur_t > best_threshold:
                            best_threshold = cur_t
                            best_a = a_perm[:]
                            best_b = b_rand[:]
                    else:
                        b_rand[s] = old
            # Try changing a
            for s in range(N):
                old = a_perm[s]
                for nd in range(N):
                    if nd == old:
                        continue
                    a_perm[s] = nd
                    nt = compute_threshold(a_perm, b_rand)
                    if nt > cur_t:
                        cur_t = nt
                        improved = True
                        if cur_t > best_threshold:
                            best_threshold = cur_t
                            best_a = a_perm[:]
                            best_b = b_rand[:]
                    else:
                        a_perm[s] = old
            if not improved:
                break
    
    # Family 7: Bidirectional cycle with asymmetric reset
    for hole_pos in range(N):
        if time.time() > deadline - 5.0:
            break
        a8 = [(i + 1) % N for i in range(N)]
        b8 = []
        for i in range(N):
            if i == 0:
                b8.append(0)
            elif i == hole_pos:
                b8.append(0)
            else:
                b8.append(i - 1)
        update_best(a8, b8)
    
    # Family 8: Two-letter structure where a is a long cycle and b has specific pattern
    for d in range(1, N):
        if time.time() > deadline - 5.0:
            break
        for exc_state in range(N):
            for exc_dest in range(N):
                if time.time() > deadline - 5.0:
                    break
                a10 = [(i + 1) % N for i in range(N)]
                b10 = []
                for i in range(N):
                    if i == exc_state:
                        b10.append(exc_dest)
                    else:
                        b10.append((i - d) % N)
                update_best(a10, b10)
    
    # If we have time, do simulated annealing from the best found
    if time.time() < deadline - 15.0 and best_a is not None:
        random.seed(42)
        current_a = best_a[:]
        current_b = best_b[:]
        current_score = best_threshold
        
        # Simulated annealing parameters
        initial_temp = 5.0
        cooling_rate = 0.999
        min_temp = 0.01
        max_steps = 30000
        
        temp = initial_temp
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if temp < min_temp:
                break
            
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            old_dest = current_a[state] if letter == 0 else current_b[state]
            
            # Try a random new destination
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            
            if letter == 0:
                current_a[state] = new_dest
            else:
                current_b[state] = new_dest
                
            new_score = compute_threshold(current_a, current_b)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_threshold:
                    best_threshold = new_score
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # Accept with probability exp(delta / temp)
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / temp)
                    if random.random() > prob:
                        # Reject: revert
                        if letter == 0:
                            current_a[state] = old_dest
                        else:
                            current_b[state] = old_dest
                        new_score = current_score
                # If delta >= 0, already accepted above
            
            temp *= cooling_rate
    
    # Final polish: try all single-state changes on the best solution
    if time.time() < deadline - 2.0 and best_a is not None:
        for letter_idx in range(2):
            for state in range(N):
                for new_dest in range(N):
                    if time.time() > deadline - 1.0:
                        break
                    if letter_idx == 0:
                        old = best_a[state]
                        if new_dest == old:
                            continue
                        best_a[state] = new_dest
                        nt = compute_threshold(best_a, best_b)
                        if nt > best_threshold:
                            best_threshold = nt
                        else:
                            best_a[state] = old
                    else:
                        old = best_b[state]
                        if new_dest == old:
                            continue
                        best_b[state] = new_dest
                        nt = compute_threshold(best_a, best_b)
                        if nt > best_threshold:
                            best_threshold = nt
                        else:
                            best_b[state] = old
    
    if best_a is None:
        # Fallback: should never happen
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0 if i == 0 else i - 1 for i in range(N)]
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
