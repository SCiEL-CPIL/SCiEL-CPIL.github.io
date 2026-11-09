import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_VALS = 127 # 2 to 128
    
    def get_products(seq):
        prods = []
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        return prods

    def init_state_and_counts(seq):
        prod_counts = {}
        num_dups = 0
        prods = get_products(seq)
        for p in prods:
            if p in prod_counts:
                if prod_counts[p] == 1:
                    num_dups += 1
                prod_counts[p] += 1
            else:
                prod_counts[p] = 1
        return prod_counts, num_dups

    def calculate_delta_add(state, val, prod_counts):
        # val is the value to add
        # Find insertion point
        ins_pos = 0
        while ins_pos < len(state) and state[ins_pos] < val:
            ins_pos += 1
        
        new_prods = [val]
        p_left = 1
        for i in range(ins_pos - 1, -1, -1):
            p_left *= state[i]
            new_prods.append(p_left * val)
            
        p_right = 1
        for j in range(ins_pos, len(state)):
            p_right *= state[j]
            new_prods.append(p_right * val)
            
        delta_dups = 0
        for p in new_prods:
            count = prod_counts.get(p, 0)
            if count == 1:
                delta_dups += 1
                
        return delta_dups, new_prods

    def calculate_delta_remove(state, idx, prod_counts):
        val = state[idx]
        removed_prods = [val]
        
        p_left = 1
        for i in range(idx - 1, -1, -1):
            p_left *= state[i]
            removed_prods.append(p_left * val)
            
        p_right = 1
        for j in range(idx + 1, len(state)):
            p_right *= state[j]
            removed_prods.append(p_right * val)
            
        delta_dups = 0
        for p in removed_prods:
            count = prod_counts.get(p, 0)
            if count == 2:
                delta_dups -= 1
                
        return delta_dups, removed_prods

    def apply_move(state, val, is_add, prod_counts, new_prods):
        if is_add:
            ins_pos = 0
            while ins_pos < len(state) and state[ins_pos] < val:
                ins_pos += 1
            state.insert(ins_pos, val)
            for p in new_prods:
                if p in prod_counts:
                    prod_counts[p] += 1
                else:
                    prod_counts[p] = 1
        else:
            state.remove(val) # O(K)
            for p in new_prods:
                prod_counts[p] -= 1
                if prod_counts[p] == 0:
                    del prod_counts[p]

    # Greedy Initial
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        tails = [x]
        prod = x
        for a in reversed(greedy_seq):
            prod *= a
            tails.append(prod)
        if not greedy_prods.intersection(tails):
            greedy_seq.append(x)
            greedy_prods.update(tails)
            
    best_seq = greedy_seq[:]
    best_len = len(best_seq)
    
    # Simulated Annealing
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            current_seq = greedy_seq[:]
        else:
            # Random start
            # Try to start with a valid random sequence of size ~ best_len/2
            # Or just empty
            current_seq = []
            # Add some random elements to give it a head start
            for _ in range(10):
                v = random.randint(2, 128)
                if v not in current_seq:
                    current_seq.append(v)
            current_seq.sort()
            
        prod_counts, num_dups = init_state_and_counts(current_seq)
        current_cost = num_dups * 1000 - len(current_seq)
        
        T = 50.0
        T_min = 0.5
        alpha = 0.99
        
        steps = 0
        max_steps = 500000
        
        while steps < max_steps and time.time() < deadline:
            # Random move
            if random.random() < 0.5 and len(current_seq) < 127:
                # Add
                # Pick random value
                val = random.randint(2, 128)
                if val in current_seq:
                    continue
                    
                delta_dups, new_prods = calculate_delta_add(current_seq, val, prod_counts)
                delta_cost = delta_dups * 1000 - 1
                
                accept = False
                if delta_cost < 0:
                    accept = True
                else:
                    if T > T_min:
                        p = math.exp(-delta_cost / T)
                        if random.random() < p:
                            accept = True
                
                if accept:
                    apply_move(current_seq, val, True, prod_counts, new_prods)
                    num_dups += delta_dups
                    current_cost += delta_cost
                    if num_dups == 0 and len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = current_seq[:]
                        
            else:
                # Remove
                if not current_seq:
                    continue
                idx = random.randint(0, len(current_seq) - 1)
                val = current_seq[idx]
                
                delta_dups, removed_prods = calculate_delta_remove(current_seq, idx, prod_counts)
                delta_cost = delta_dups * 1000 + 1 # -(-1)
                
                accept = False
                if delta_cost < 0:
                    accept = True
                else:
                    if T > T_min:
                        p = math.exp(-delta_cost / T)
                        if random.random() < p:
                            accept = True
                
                if accept:
                    apply_move(current_seq, val, False, prod_counts, removed_prods)
                    num_dups += delta_dups
                    current_cost += delta_cost
            
            steps += 1
            T *= alpha
            
    return best_seq
