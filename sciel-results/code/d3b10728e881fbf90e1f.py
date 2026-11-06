import time
import math
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    n_moduli = len(moduli)
    
    # Precompute GCDs and inv_m
    gcd_table = {}
    inv_m = {}
    for m1 in moduli:
        inv_m[m1] = 1.0 / m1
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # Precompute for each pair (m1, m2): for each residue r2 of m2,
    # the set of residues r1 of m1 that conflict (r1 % g == r2 % g where g = gcd(m1,m2))
    # Instead, we'll compute on the fly but use a faster method.
    # For given m1, m2, g = gcd(m1,m2), r2: invalid r1 are those where r1 ≡ r2 (mod g)
    # Count of such r1 in [0, m1) is m1 // g
    # We can represent invalid residues as a boolean array or just check r1 % g == r2 % g
    
    def get_valid_residues_fast(m, state):
        """Get all valid residues for modulus m given current state, using fast conflict check."""
        if not state:
            return list(range(m))
        
        # For each selected modulus, mark invalid residues
        invalid = [False] * m
        for m2, r2 in state.items():
            if m2 == m:
                continue
            g = gcd_table[(m, m2)]
            target = r2 % g
            # Mark all r in [0, m) where r % g == target
            for r in range(target, m, g):
                invalid[r] = True
        
        return [r for r in range(m) if not invalid[r]]
    
    def check_single(m, r, state, exclude_m=None):
        """Check if a single residue r for modulus m is valid."""
        for m2, r2 in state.items():
            if m2 == m or m2 == exclude_m:
                continue
            g = gcd_table[(m, m2)]
            if r % g == r2 % g:
                return False
        return True
    
    def simulated_annealing(initial_state, max_time=100):
        """
        Simulated annealing to maximize the score using float arithmetic for speed.
        """
        current_state = dict(initial_state)
        current_score = sum(inv_m[m] for m in current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        # Temperature schedule - start higher to allow exploration
        start_temp = 0.5
        end_temp = 0.0001
        cooling_rate = 0.9998
        
        temp = start_temp
        start_time = time.time()
        
        # Precompute unselected list for faster access
        selected_set = set(current_state.keys())
        
        while time.time() < start_time + max_time and temp > end_temp:
            # Check global deadline
            if time.time() > deadline - 5:
                break
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a random unselected modulus - prefer smaller moduli (higher 1/m)
                unselected = [m for m in moduli if m not in selected_set]
                if not unselected:
                    continue
                # Weight towards smaller moduli
                if random.random() < 0.7:
                    # Pick from the smallest 50% of unselected
                    unselected.sort()
                    half = len(unselected) // 2
                    m_new = random.choice(unselected[:max(1, half)])
                else:
                    m_new = random.choice(unselected)
                
                valid_rs = get_valid_residues_fast(m_new, current_state)
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_score = current_score + inv_m[m_new]
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current_state[m_new] = r_new
                        selected_set.add(m_new)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = dict(current_state)
                        
            elif move_type < 0.60:
                # Remove a random selected modulus - prefer removing larger moduli
                if not current_state:
                    continue
                # Weight towards larger moduli (lower 1/m) for removal
                if random.random() < 0.6:
                    # Pick from the largest 50% of selected
                    selected_list = sorted(current_state.keys())
                    half = len(selected_list) // 2
                    m_old = random.choice(selected_list[max(0, half):])
                else:
                    m_old = random.choice(list(current_state.keys()))
                
                new_score = current_score - inv_m[m_old]
                del current_state[m_old]
                selected_set.discard(m_old)
                current_score = new_score
                # Always accept removal if it leads to better future (but we just track best)
                # Actually, removal always decreases score, so only accept if we want to explore
                # Re-add if we don't want to go down
                if new_score < best_score:
                    # Still keep it, SA allows going down
                    pass
                if current_score > best_score:
                    best_score = current_score
                    best_state = dict(current_state)
                        
            elif move_type < 0.85:
                # Swap: remove one, add another
                if not current_state:
                    continue
                unselected = [m for m in moduli if m not in selected_set]
                if not unselected:
                    continue
                
                # Prefer removing large, adding small
                if random.random() < 0.7:
                    selected_list = sorted(current_state.keys())
                    half = len(selected_list) // 2
                    m_old = random.choice(selected_list[max(0, half):])
                else:
                    m_old = random.choice(list(current_state.keys()))
                
                unselected.sort()
                half_u = len(unselected) // 2
                m_new = random.choice(unselected[:max(1, half_u)])
                
                # Try to find a valid residue for m_new after removing m_old
                temp_state = {k: v for k, v in current_state.items() if k != m_old}
                valid_rs = get_valid_residues_fast(m_new, temp_state)
                
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    delta = inv_m[m_new] - inv_m[m_old]
                    new_score = current_score + delta
                    
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        del current_state[m_old]
                        selected_set.discard(m_old)
                        current_state[m_new] = r_new
                        selected_set.add(m_new)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = dict(current_state)
                            
            else:
                # Change residue of a random selected modulus
                if not current_state:
                    continue
                m_var = random.choice(list(current_state.keys()))
                old_r = current_state[m_var]
                
                # Try a few random new residues
                for _ in range(3):
                    new_r = random.randint(0, m_var - 1)
                    if new_r == old_r:
                        continue
                    other_state = {k: v for k, v in current_state.items() if k != m_var}
                    if check_single(m_var, new_r, other_state):
                        # Residue change doesn't affect score
                        current_state[m_var] = new_r
                        break
            
            # Cool down
            temp *= cooling_rate
            
        return best_state, best_score
    
    def greedy_construct(randomize=False, order=None):
        """
        Greedy construction to create an initial solution.
        """
        if order is None:
            order = moduli[:]
            if randomize:
                random.shuffle(order)
            else:
                # Sort by 1/m descending (prefer small moduli)
                order.sort()
        
        state = {}
        selected_set = set()
        for m in order:
            if time.time() > deadline - 35:
                break
                
            valid_rs = get_valid_residues_fast(m, state)
            if valid_rs:
                if randomize:
                    r = random.choice(valid_rs)
                else:
                    r = valid_rs[0]
                state[m] = r
                selected_set.add(m)
        
        return state
    
    best_solution = {}
    best_score = 0.0
    
    # Strategy:
    # 1. Create a good initial solution using greedy (prefer small moduli)
    # 2. Run simulated annealing from that initial solution
    # 3. Try multiple restarts with different initial solutions
    
    # Run 1: Deterministic greedy + SA
    state1 = greedy_construct(randomize=False)
    sol1, score1 = simulated_annealing(state1, max_time=45)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
        
    # Run 2-5: Randomized greedy + SA
    num_restarts = 4
    
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break
            
        state = greedy_construct(randomize=True)
        sol, score = simulated_annealing(state, max_time=25)
        
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Final: if we have time, run SA on the best solution found
    if time.time() < deadline - 15 and best_solution:
        sol_final, score_final = simulated_annealing(best_solution, max_time=10)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final
    
    # Convert to list of lists
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
