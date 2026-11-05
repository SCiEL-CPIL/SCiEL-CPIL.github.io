"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 256
    
    # Weights: w[i] = 1/(i+1) for vertex i (representing number i+1)
    weights = [1.0 / (i + 1) for i in range(n)]
    
    # Precompute conflict bitsets
    # neighbors[i] is a bitmask of vertices j that conflict with i (share a 3-AP)
    neighbors = [0] * n
    
    # Generate all 3-APs: (a, b, c) with a < b < c, a+c=2b
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c > n:
                break
            ia, ib, ic = a - 1, b - 1, c - 1
            # Add edges (a,b), (b,c), (a,c)
            neighbors[ia] |= (1 << ib) | (1 << ic)
            neighbors[ib] |= (1 << ia) | (1 << ic)
            neighbors[ic] |= (1 << ia) | (1 << ib)
    
    # Precompute sorted order by weight (descending) for fast max-weight vertex selection
    sorted_vertices = sorted(range(n), key=lambda i: -weights[i])
    
    def get_max_weight_vertex(bitset):
        for idx in sorted_vertices:
            if bitset & (1 << idx):
                return idx
        return -1
    
    def sum_weights(bitset):
        total = 0.0
        b = bitset
        while b:
            lsb = b & -b
            idx = lsb.bit_length() - 1
            total += weights[idx]
            b ^= lsb
        return total
    
    # Greedy initialization: pick highest weight vertex, remove it and its neighbors
    def greedy_init():
        avail = (1 << n) - 1
        selected = []
        weight = 0.0
        while avail:
            v = get_max_weight_vertex(avail)
            if v == -1:
                break
            selected.append(v + 1)
            weight += weights[v]
            avail &= ~(neighbors[v] | (1 << v))
        return selected, weight
    
    best_set, best_weight = greedy_init()
    
    # Local search with simulated annealing
    # Moves:
    # 1. Add a vertex not in S that doesn't conflict with any vertex in S
    # 2. Remove a vertex from S
    # 3. Swap: remove a vertex from S, add a different vertex not in S
    # To make this efficient, we maintain a bitset for S and a bitset for "available"
    # (vertices not in S and not conflicting with S)
    
    import math
    
    # We'll use a simpler local search:
    # - Maintain S as a set
    # - Maintain conflict_count[v] = number of vertices in S that conflict with v
    # - A vertex v can be added if v not in S and conflict_count[v] == 0
    # - A vertex v in S can be removed
    # - After adding/removing, update conflict counts
    
    def local_search(S, current_weight, time_limit):
        S_set = set(S)
        # conflict_count[v] = number of neighbors of v that are in S
        conflict_count = [0] * n
        for v in S:
            idx = v - 1
            # For each neighbor of idx, increment conflict_count
            nb = neighbors[idx]
            b = nb
            while b:
                lsb = b & -b
                j = lsb.bit_length() - 1
                conflict_count[j] += 1
                b ^= lsb
        
        current_weight = float(sum(1.0 / x for x in S))
        
        T = 0.05  # Initial temperature
        T_min = 0.001
        cooling = 0.995
        
        best_local = list(S)
        best_local_weight = current_weight
        
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a vertex
                # Find candidates: not in S, conflict_count == 0
                candidates = []
                for v in range(1, n + 1):
                    if v not in S_set and conflict_count[v - 1] == 0:
                        candidates.append(v)
                if not candidates:
                    continue
                v = random.choice(candidates)
                idx = v - 1
                # Check if adding v is valid (conflict_count should be 0)
                if conflict_count[idx] != 0:
                    continue
                # Add v
                S_set.add(v)
                current_weight += weights[idx]
                # Update conflict counts
                nb = neighbors[idx]
                b = nb
                while b:
                    lsb = b & -b
                    j = lsb.bit_length() - 1
                    conflict_count[j] += 1
                    b ^= lsb
                delta = weights[idx]
                # Accept or reject
                if delta > 0 or random.random() < math.exp(delta / T):
                    if current_weight > best_local_weight:
                        best_local_weight = current_weight
                        best_local = list(S_set)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject: undo
                    S_set.discard(v)
                    current_weight -= weights[idx]
                    nb = neighbors[idx]
                    b = nb
                    while b:
                        lsb = b & -b
                        j = lsb.bit_length() - 1
                        conflict_count[j] -= 1
                        b ^= lsb
                    no_improve += 1
            
            elif move_type < 0.6:
                # Try to remove a vertex
                if not S_set:
                    continue
                v = random.choice(list(S_set))
                idx = v - 1
                # Remove v
                S_set.discard(v)
                current_weight -= weights[idx]
                # Update conflict counts
                nb = neighbors[idx]
                b = nb
                while b:
                    lsb = b & -b
                    j = lsb.bit_length() - 1
                    conflict_count[j] -= 1
                    b ^= lsb
                delta = -weights[idx]
                # Accept or reject
                if delta > 0 or random.random() < math.exp(delta / T):
                    if current_weight > best_local_weight:
                        best_local_weight = current_weight
                        best_local = list(S_set)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject: undo
                    S_set.add(v)
                    current_weight += weights[idx]
                    nb = neighbors[idx]
                    b = nb
                    while b:
                        lsb = b & -b
                        j = lsb.bit_length() - 1
                        conflict_count[j] += 1
                        b ^= lsb
                    no_improve += 1
            
            else:
                # Try a swap: remove one, add another
                if not S_set:
                    continue
                # Pick a vertex to remove
                v_remove = random.choice(list(S_set))
                idx_remove = v_remove - 1
                # Temporarily remove it
                S_set.discard(v_remove)
                current_weight -= weights[idx_remove]
                nb_remove = neighbors[idx_remove]
                b = nb_remove
                while b:
                    lsb = b & -b
                    j = lsb.bit_length() - 1
                    conflict_count[j] -= 1
                    b ^= lsb
                
                # Find candidates to add
                candidates = []
                for v in range(1, n + 1):
                    if v not in S_set and conflict_count[v - 1] == 0:
                        candidates.append(v)
                
                if candidates:
                    v_add = random.choice(candidates)
                    idx_add = v_add - 1
                    # Add it
                    S_set.add(v_add)
                    current_weight += weights[idx_add]
                    nb_add = neighbors[idx_add]
                    b = nb_add
                    while b:
                        lsb = b & -b
                        j = lsb.bit_length() - 1
                        conflict_count[j] += 1
                        b ^= lsb
                    
                    delta = weights[idx_add] - weights[idx_remove]
                    if delta > 0 or random.random() < math.exp(delta / T):
                        if current_weight > best_local_weight:
                            best_local_weight = current_weight
                            best_local = list(S_set)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Reject: undo
                        S_set.discard(v_add)
                        current_weight -= weights[idx_add]
                        nb_add = neighbors[idx_add]
                        b = nb_add
                        while b:
                            lsb = b & -b
                            j = lsb.bit_length() - 1
                            conflict_count[j] -= 1
                            b ^= lsb
                        # Restore v_remove
                        S_set.add(v_remove)
                        current_weight += weights[idx_remove]
                        nb_remove = neighbors[idx_remove]
                        b = nb_remove
                        while b:
                            lsb = b & -b
                            j = lsb.bit_length() - 1
                            conflict_count[j] += 1
                            b ^= lsb
                        no_improve += 1
                else:
                    # No candidates, undo the removal
                    S_set.add(v_remove)
                    current_weight += weights[idx_remove]
                    nb_remove = neighbors[idx_remove]
                    b = nb_remove
                    while b:
                        lsb = b & -b
                        j = lsb.bit_length() - 1
                        conflict_count[j] += 1
                        b ^= lsb
                    no_improve += 1
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            if no_improve > max_no_improve:
                break
        
        return best_local, best_local_weight
    
    # Run multiple restarts with different random seeds
    random.seed(42)
    
    while time.time() - start_time < deadline:
        # Run local search from the current best
        result, result_weight = local_search(best_set, best_weight, deadline - 2.0)
        
        if result_weight > best_weight:
            best_set = result
            best_weight = result_weight
        
        # Try a few random restarts
        for _ in range(3):
            if time.time() - start_time > deadline:
                break
            # Create a random starting point by perturbing the best set
            S = list(best_set)
            # Remove some random elements
            for _ in range(random.randint(0, min(5, len(S)))):
                if S:
                    v = random.choice(S)
                    S.remove(v)
            # Add some random non-conflicting elements
            S_set = set(S)
            for _ in range(random.randint(0, 5)):
                candidates = []
                for v in range(1, n + 1):
                    if v not in S_set:
                        # Check if v conflicts with any in S
                        idx = v - 1
                        conflicts = False
                        for u in S_set:
                            if (1 << (u - 1)) & neighbors[idx]:
                                conflicts = True
                                break
                        if not conflicts:
                            candidates.append(v)
                if candidates:
                    v = random.choice(candidates)
                    S_set.add(v)
                    S = list(S_set)
            
            result, result_weight = local_search(S, sum(1.0 / x for x in S), deadline - 2.0)
            if result_weight > best_weight:
                best_set = result
                best_weight = result_weight
    
    # Verify the result is valid
    S = set(best_set)
    valid = True
    for a in S:
        for b in S:
            if a >= b:
                continue
            c = 2 * b - a
            if c in S:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy
        S = set(greedy_init()[0])
    
    return list(S)

# EVOLVE-BLOCK-END
