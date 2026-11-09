"""Rewritten program using Max-SAT / simulated annealing approach."""
import random
import time

def construct():
    n = 180
    # Precompute forbidden triples
    forbidden = set()
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if b == a:
                continue
            den = b - a
            if den == 0:
                continue
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c < 1 or c > n:
                continue
            if c == a or c == b:
                continue
            triple = frozenset((a, b, c))
            forbidden.add(triple)
            
    triple_list = list(forbidden)
    num_triples = len(triple_list)
    
    # Map element to list of triple indices
    elem_to_triples = [[] for _ in range(n + 1)]
    for idx, tri in enumerate(triple_list):
        for e in tri:
            elem_to_triples[e].append(idx)
            
    # State: x[i] is 1 if i is in S
    # violated[idx] is 1 if triple idx is fully in S
    
    def init_greedy():
        x = [0] * (n + 1)
        for i in range(1, n + 1):
            x[i] = 1
        violated = [1] * num_triples
        
        # Greedy removal
        changed = True
        while changed:
            changed = False
            conflict_count = [0] * (n + 1)
            for idx in range(num_triples):
                if violated[idx]:
                    for e in triple_list[idx]:
                        if x[e]:
                            conflict_count[e] += 1
                            
            max_conf = 0
            max_elem = -1
            for i in range(1, n + 1):
                if x[i] and conflict_count[i] > max_conf:
                    max_conf = conflict_count[i]
                    max_elem = i
                    
            if max_conf == 0:
                break
                
            x[max_elem] = 0
            changed = True
            for idx in elem_to_triples[max_elem]:
                tri = triple_list[idx]
                is_viol = True
                for e in tri:
                    if not x[e]:
                        is_viol = False
                        break
                if is_viol:
                    violated[idx] = 1
                else:
                    violated[idx] = 0
        return x, violated

    x, violated = init_greedy()
    current_energy = 100 * sum(violated) - sum(x[1:])
    best_x = x[:]
    best_energy = current_energy
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    start_time = time.time()
    elements = list(range(1, n + 1))
    
    while time.time() - start_time < 58:
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            candidates = [i for i in elements if not x[i]]
            if not candidates:
                continue
            e = random.choice(candidates)
            
            delta_viol = 0
            for idx in elem_to_triples[e]:
                tri = triple_list[idx]
                other_in = 0
                for o in tri:
                    if o != e and x[o]:
                        other_in += 1
                if other_in == 2:
                    delta_viol += 1
            
            delta_e = 100 * delta_viol - 1
            if delta_e < 0 or random.random() < (2.718281828 ** (-delta_e / T)):
                x[e] = 1
                for idx in elem_to_triples[e]:
                    tri = triple_list[idx]
                    other_in = 0
                    for o in tri:
                        if o != e and x[o]:
                            other_in += 1
                    if other_in == 2:
                        violated[idx] = 1
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x[:]
                    
        elif move_type < 0.8:
            # Remove
            candidates = [i for i in elements if x[i]]
            if not candidates:
                continue
            e = random.choice(candidates)
            
            delta_viol = 0
            for idx in elem_to_triples[e]:
                tri = triple_list[idx]
                other_in = 0
                for o in tri:
                    if o != e and x[o]:
                        other_in += 1
                if other_in == 2:
                    delta_viol -= 1
            
            delta_e = 100 * delta_viol + 1
            if delta_e < 0 or random.random() < (2.718281828 ** (-delta_e / T)):
                x[e] = 0
                for idx in elem_to_triples[e]:
                    tri = triple_list[idx]
                    other_in = 0
                    for o in tri:
                        if o != e and x[o]:
                            other_in += 1
                    if other_in == 2:
                        violated[idx] = 0
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x[:]
        else:
            # Swap
            c_out = [i for i in elements if x[i]]
            c_in = [i for i in elements if not x[i]]
            if not c_out or not c_in:
                continue
            e_out = random.choice(c_out)
            e_in = random.choice(c_in)
            
            # Delta for removing e_out
            delta_viol_out = 0
            for idx in elem_to_triples[e_out]:
                tri = triple_list[idx]
                other_in = 0
                for o in tri:
                    if o != e_out and x[o]:
                        other_in += 1
                if other_in == 2:
                    delta_viol_out -= 1
            
            # Delta for adding e_in (with e_out removed)
            delta_viol_in = 0
            for idx in elem_to_triples[e_in]:
                tri = triple_list[idx]
                other_in = 0
                for o in tri:
                    if o == e_out:
                        continue
                    if x[o]:
                        other_in += 1
                # Note: if e_in and e_out are in the same triple, we handled e_out above.
                # We need to check if the OTHER two (excluding e_in) are in S (with e_out removed).
                # Let's just re-evaluate carefully.
                # Actually, simpler:
                # Count how many of the other 2 elements are in S, EXCLUDING e_out if it's one of them.
                # If e_out is in tri, and e_in is in tri, then the third element is 'other'.
                # If e_out is NOT in tri, then we just check the other 2.
                
                # Let's do it explicitly:
                count = 0
                for o in tri:
                    if o == e_in:
                        continue
                    if o == e_out:
                        # e_out is being removed, so it's not in S
                        continue
                    if x[o]:
                        count += 1
                if count == 2:
                    delta_viol_in += 1
            
            delta_e = 100 * (delta_viol_out + delta_viol_in)
            # Size change is 0
            
            if delta_e < 0 or random.random() < (2.718281828 ** (-delta_e / T)):
                # Update x
                x[e_out] = 0
                x[e_in] = 1
                
                # Update violated for e_out's triples
                for idx in elem_to_triples[e_out]:
                    tri = triple_list[idx]
                    is_viol = True
                    for o in tri:
                        if not x[o]:
                            is_viol = False
                            break
                    if is_viol:
                        violated[idx] = 1
                    else:
                        violated[idx] = 0
                        
                # Update violated for e_in's triples
                for idx in elem_to_triples[e_in]:
                    tri = triple_list[idx]
                    is_viol = True
                    for o in tri:
                        if not x[o]:
                            is_viol = False
                            break
                    if is_viol:
                        violated[idx] = 1
                    else:
                        violated[idx] = 0
                        
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x[:]
                    
        T *= cooling
        if T < T_min:
            T = T_min
            
    result = [i for i in range(1, n + 1) if best_x[i]]
    if not result:
        result = [180]
    return result
