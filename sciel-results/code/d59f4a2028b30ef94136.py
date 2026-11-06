import time
import math
import random
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit

    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    n_moduli = len(moduli)

    # Precompute GCDs
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)

    # Precompute 1/m as float and Fraction
    inv_m_float = {m: 1.0 / m for m in moduli}
    inv_m_frac = {m: Fraction(1, m) for m in moduli}

    def get_valid_residues(m, state):
        """Get all valid residues for modulus m given current state {m2: r2}"""
        valid_rs = []
        for r in range(m):
            ok = True
            for m2, r2 in state.items():
                if m2 == m:
                    continue
                g = gcd_table.get((m, m2), math.gcd(m, m2))
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                valid_rs.append(r)
        return valid_rs

    def check_valid(m, r, state, exclude_m=None):
        """Check if (m, r) is valid given state"""
        for m2, r2 in state.items():
            if m2 == m:
                continue
            if exclude_m is not None and m2 == exclude_m:
                continue
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if r % g == r2 % g:
                return False
        return True

    def compute_score_float(state):
        return sum(inv_m_float[m] for m in state)

    def compute_score_frac(state):
        total = Fraction(0, 1)
        for m in state:
            total += inv_m_frac[m]
        return total

    def greedy_construct(order=None, randomize=False):
        """Greedy construction: add moduli in given order, picking first valid residue."""
        if order is None:
            order = moduli[:]
            if randomize:
                random.shuffle(order)
        
        state = {}
        for m in order:
            if time.time() > deadline - 5:
                break
            valid_rs = get_valid_residues(m, state)
            if valid_rs:
                if randomize:
                    r = random.choice(valid_rs)
                else:
                    r = valid_rs[0]
                state[m] = r
        return state

    def local_search(state, max_time=10, aggressive=False):
        """
        Local search: try to add moduli that were previously impossible,
        or swap residues to free up space.
        """
        state = dict(state)
        score = compute_score_float(state)
        best_state = dict(state)
        best_score = score
        
        start_time = time.time()
        while time.time() < start_time + max_time and time.time() < deadline - 2:
            # Try adding a random unselected modulus
            unselected = [m for m in moduli if m not in state]
            if unselected:
                m_new = random.choice(unselected)
                valid_rs = get_valid_residues(m_new, state)
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_state = dict(state)
                    new_state[m_new] = r_new
                    new_score = score + inv_m_float[m_new]
                    if new_score > best_score:
                        best_state = new_state
                        best_score = new_score
                        state = new_state
                        score = new_score
                    elif random.random() < 0.1:
                        state = new_state
                        score = new_score
            
            # Try swapping a residue to see if it allows adding another modulus
            if len(state) > 5 and random.random() < 0.3:
                m_var = random.choice(list(state.keys()))
                old_r = state[m_var]
                other_state = {k: v for k, v in state.items() if k != m_var}
                valid_rs = get_valid_residues(m_var, other_state)
                if len(valid_rs) > 1:
                    new_r = random.choice(valid_rs)
                    new_state = dict(state)
                    new_state[m_var] = new_r
                    # Check if this opens up any new moduli
                    for m_add in unselected:
                        if m_add in new_state:
                            continue
                        vrs_add = get_valid_residues(m_add, new_state)
                        if vrs_add:
                            r_add = random.choice(vrs_add)
                            final_state = dict(new_state)
                            final_state[m_add] = r_add
                            final_score = compute_score_float(final_state)
                            if final_score > best_score:
                                best_state = final_state
                                best_score = final_score
                                state = final_state
                                score = final_score
                                break

            if aggressive and time.time() < deadline - 5:
                # More aggressive: try double swaps
                if len(state) >= 2:
                    keys = list(state.keys())
                    m1 = random.choice(keys)
                    m2 = random.choice([k for k in keys if k != m1])
                    other = {k: v for k, v in state.items() if k not in (m1, m2)}
                    vrs1 = get_valid_residues(m1, other)
                    vrs2 = get_valid_residues(m2, other)
                    if vrs1 and vrs2:
                        r1 = random.choice(vrs1)
                        r2 = random.choice(vrs2)
                        new_state = dict(state)
                        new_state[m1] = r1
                        new_state[m2] = r2
                        for m_add in unselected:
                            if m_add in new_state:
                                continue
                            vrs_add = get_valid_residues(m_add, new_state)
                            if vrs_add:
                                r_add = random.choice(vrs_add)
                                final_state = dict(new_state)
                                final_state[m_add] = r_add
                                final_score = compute_score_float(final_state)
                                if final_score > best_score:
                                    best_state = final_state
                                    best_score = final_score
                                    state = final_state
                                    score = final_score
                                    break

        return best_state, best_score

    def simulated_annealing(initial_state, max_time=40):
        """
        Simulated annealing with multiple move types.
        """
        current_state = dict(initial_state)
        current_score = compute_score_float(current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        start_temp = 0.03
        end_temp = 0.00005
        cooling_rate = 0.9995
        
        temp = start_temp
        start_time = time.time()
        
        while time.time() < start_time + max_time and temp > end_temp:
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.30:
                # Add a random unselected modulus, biased toward smaller moduli
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    temp *= cooling_rate
                    continue
                # Weight by 1/m^2 to strongly prefer small moduli
                weights = [inv_m_float[m] ** 2 for m in unselected]
                total_w = sum(weights)
                r_rand = random.random() * total_w
                cum = 0
                m_new = unselected[-1]
                for m, w in zip(unselected, weights):
                    cum += w
                    if r_rand <= cum:
                        m_new = m
                        break
                valid_rs = get_valid_residues(m_new, current_state)
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_state = dict(current_state)
                    new_state[m_new] = r_new
                    new_score = current_score + inv_m_float[m_new]
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current_state = new_state
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = dict(current_state)
                            
            elif move_type < 0.45:
                # Remove a random selected modulus, biased toward larger moduli
                if not current_state:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                # Weight by 1/m to prefer removing larger moduli
                weights = [inv_m_float[m] for m in keys]
                total_w = sum(weights)
                r_rand = random.random() * total_w
                cum = 0
                m_old = keys[-1]
                for m, w in zip(keys, weights):
                    cum += w
                    if r_rand <= cum:
                        m_old = m
                        break
                new_state = dict(current_state)
                del new_state[m_old]
                new_score = current_score - inv_m_float[m_old]
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_state = dict(current_state)
                            
            elif move_type < 0.70:
                # Swap: remove one, add another
                if not current_state:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                weights_rem = [inv_m_float[m] for m in keys]
                total_w_rem = sum(weights_rem)
                r_rand_rem = random.random() * total_w_rem
                cum = 0
                m_old = keys[-1]
                for m, w in zip(keys, weights_rem):
                    cum += w
                    if r_rand_rem <= cum:
                        m_old = m
                        break
                
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    temp *= cooling_rate
                    continue
                weights_add = [inv_m_float[m] ** 2 for m in unselected]
                total_w_add = sum(weights_add)
                r_rand_add = random.random() * total_w_add
                cum = 0
                m_new = unselected[-1]
                for m, w in zip(unselected, weights_add):
                    cum += w
                    if r_rand_add <= cum:
                        m_new = m
                        break
                
                temp_state = dict(current_state)
                del temp_state[m_old]
                valid_rs = get_valid_residues(m_new, temp_state)
                
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_state = temp_state
                    new_state[m_new] = r_new
                    new_score = current_score - inv_m_float[m_old] + inv_m_float[m_new]
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current_state = new_state
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = dict(current_state)
                            
            elif move_type < 0.85:
                # Change residue of a random selected modulus
                if not current_state:
                    temp *= cooling_rate
                    continue
                m_var = random.choice(list(current_state.keys()))
                old_r = current_state[m_var]
                new_r = random.randint(0, m_var - 1)
                
                if new_r != old_r:
                    other_state = {k: v for k, v in current_state.items() if k != m_var}
                    if check_valid(m_var, new_r, other_state):
                        new_state = dict(current_state)
                        new_state[m_var] = new_r
                        # Score unchanged
                        current_state = new_state
                        # No score change, so always accept for exploration
                        
            else:
                # Double swap: remove two, add two
                if len(current_state) < 2:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                m_old1 = random.choice(keys)
                remaining = [k for k in keys if k != m_old1]
                m_old2 = random.choice(remaining)
                
                unselected = [m for m in moduli if m not in current_state]
                if len(unselected) < 2:
                    temp *= cooling_rate
                    continue
                m_new1 = random.choice(unselected)
                remaining_unsel = [m for m in unselected if m != m_new1]
                m_new2 = random.choice(remaining_unsel)
                
                temp_state = dict(current_state)
                del temp_state[m_old1]
                del temp_state[m_old2]
                
                valid_rs1 = get_valid_residues(m_new1, temp_state)
                if not valid_rs1:
                    temp *= cooling_rate
                    continue
                r_new1 = random.choice(valid_rs1)
                temp_state[m_new1] = r_new1
                
                valid_rs2 = get_valid_residues(m_new2, temp_state)
                if not valid_rs2:
                    temp *= cooling_rate
                    continue
                r_new2 = random.choice(valid_rs2)
                temp_state[m_new2] = r_new2
                
                new_score = current_score - inv_m_float[m_old1] - inv_m_float[m_old2] + inv_m_float[m_new1] + inv_m_float[m_new2]
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_state = temp_state
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_state = dict(current_state)
            
            temp *= cooling_rate
            
        return best_state, best_score

    best_solution = {}
    best_score = 0.0

    # Strategy 1: Greedy sorted by 1/m descending (prefer small moduli)
    order_desc = sorted(moduli, key=lambda m: -inv_m_float[m])
    state1 = greedy_construct(order=order_desc)
    sol1, score1 = simulated_annealing(state1, max_time=30)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1

    # Strategy 2: Greedy sorted ascending
    order_asc = sorted(moduli, key=lambda m: inv_m_float[m])
    state2 = greedy_construct(order=order_asc)
    sol2, score2 = simulated_annealing(state2, max_time=20)
    if score2 > best_score:
        best_score = score2
        best_solution = sol2

    # Strategy 3-6: Randomized greedy + SA
    num_restarts = 4
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break
        state = greedy_construct(randomize=True)
        sol, score = simulated_annealing(state, max_time=15)
        if score > best_score:
            best_score = score
            best_solution = sol

    # Final polish on best solution
    if time.time() < deadline - 10 and best_solution:
        sol_final, score_final = simulated_annealing(best_solution, max_time=10)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final

    # Additional local search pass
    if time.time() < deadline - 5 and best_solution:
        sol_ls, score_ls = local_search(best_solution, max_time=5, aggressive=True)
        if score_ls > best_score:
            best_score = score_ls
            best_solution = sol_ls

    if best_solution:
        return [[m, r] for m, r in best_solution.items()]
    
    # Fallback
    state = greedy_construct(randomize=False)
    return [[m, r] for m, r in state.items()]

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Conflict: ({m1},{r1}) and ({m2},{r2})")
    print(f"Valid: {valid}")
