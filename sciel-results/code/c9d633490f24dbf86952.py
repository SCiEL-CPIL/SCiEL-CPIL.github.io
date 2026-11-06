"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # Safety margin for 90s limit
    
    # Candidate moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    # Precompute gcds for speed
    # gcd_matrix[m1-2][m2-2] = gcd(m1, m2)
    # Indices 0..126 correspond to moduli 2..128
    N = 127
    gcd_mat = [[0]*N for _ in range(N)]
    for i in range(N):
        m1 = i + 2
        for j in range(i, N):
            m2 = j + 2
            g = _gcd(m1, m2)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def is_disjoint_fast(r1, idx1, r2, idx2):
        """Fast disjoint check using precomputed gcd."""
        m1 = idx1 + 2
        m2 = idx2 + 2
        g = gcd_mat[idx1][idx2]
        if g == 1:
            return False
        return (r2 - r1) % g != 0

    def get_valid_residues(new_idx, current_state):
        """
        Returns a list of residues r for modulus m = new_idx+2 
        such that (r, m) is disjoint from all in current_state.
        current_state is a dict {modulus: residue}
        """
        m_new = new_idx + 2
        valid = []
        for r in range(m_new):
            ok = True
            for m_exist, r_exist in current_state.items():
                idx_exist = m_exist - 2
                if not is_disjoint_fast(r, new_idx, r_exist, idx_exist):
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid

    # 1. Initial Solution: Greedy construction to get a decent starting point
    # Sort by divisor count descending, then value descending (heuristic from seed)
    def divisor_count(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count

    div_counts = {m: divisor_count(m) for m in all_moduli}
    sorted_moduli = sorted(all_moduli, key=lambda m: (-div_counts[m], -m))
    
    current_state = {} # {modulus: residue}
    for m in sorted_moduli:
        idx = m - 2
        valid_res = get_valid_residues(idx, current_state)
        if valid_res:
            # Pick a random valid residue to start diversity
            current_state[m] = random.choice(valid_res)
    
    best_state = dict(current_state)
    best_size = len(best_state)
    
    # 2. Simulated Annealing
    # State: current_state dict
    # Moves:
    # a) Add a new modulus (if possible)
    # b) Remove a modulus
    # c) Change residue of an existing modulus (to potentially allow others)
    # d) Swap: Remove one, add another (if direct add fails, maybe changing residue of existing helps? 
    #    Actually, changing residue of existing is move (c). 
    #    A combined move: Pick existing X, change its residue, then try to add Y. 
    #    This is expensive. Let's stick to simpler moves first.)
    
    # Let's refine moves:
    # 1. Try to add a random unused modulus. If valid residue exists, accept if size increases or with prob exp(-1/T) if size same (shouldn't happen if we only add if valid).
    #    Actually, adding always increases size. So always accept if possible.
    # 2. Try to remove a random existing modulus. Accept with prob exp(-1/T) (decreases size, so usually reject unless it helps later).
    # 3. Try to change residue of a random existing modulus. 
    #    - If new residue is valid with others, accept. (Size same).
    #    - This might open up space for future additions.
    
    # Better Move Set for SA:
    # 1. Add: Pick unused m. Find valid residues. If any, pick one. New size = old + 1. Always accept.
    # 2. Remove: Pick used m. Remove it. New size = old - 1. Accept with prob exp(-1/T).
    # 3. Residue Change: Pick used m. Pick new random residue r'. 
    #    Check if r' is disjoint from all other existing. 
    #    If yes, accept (size same). This is a neutral move that changes the landscape for future additions.
    #    If no, reject.
    
    # Wait, if we just change residue to a valid one, we don't gain size immediately. 
    # The gain comes from subsequent "Add" moves. 
    # To make SA effective, we need to explore residue spaces that are currently "blocked".
    
    # Let's add a "Perturb" move:
    # 4. Perturb: Pick used m. Change its residue to a RANDOM value (even if invalid). 
    #    This makes the state invalid temporarily? No, the state must always be valid for the "Add" logic to work correctly?
    #    Actually, standard SA requires the state to be a valid configuration? 
    #    If we allow invalid states, the "Add" move becomes complex (need to check if the whole set becomes valid?).
    #    It's safer to keep the state valid at all times.
    
    # So, Move 3 (Residue Change) only accepts if the new residue is valid with the rest.
    # This allows us to "shuffle" the residues of existing moduli to find a configuration 
    # where a specific unused modulus CAN be added.
    
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    # Precompute list of all indices
    all_indices = [m - 2 for m in all_moduli]
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            # Optionally restart with higher T if stuck? 
            # Let's just keep going or reset if stuck for a long time.
            # Simple restart: if no improvement for 1000 steps, T = 1.0
            # We don't track steps easily here, let's just let it cool.
            # If T is very low, we mostly accept improvements (which are only adds) and rare removals.
            # We might get stuck. Let's add a periodic reheat.
            if random.random() < 0.01:
                T = 1.0
                # Optionally randomize current state slightly?
                # Let's just reheat.
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add
            unused = [m for m in all_moduli if m not in current_state]
            if unused:
                m_new = random.choice(unused)
                idx_new = m_new - 2
                valid_res = get_valid_residues(idx_new, current_state)
                if valid_res:
                    r_new = random.choice(valid_res)
                    # Calculate delta: size increases by 1. Always good.
                    current_state[m_new] = r_new
                    if len(current_state) > best_size:
                        best_size = len(current_state)
                        best_state = dict(current_state)
                    # Always accept
                    continue
                else:
                    # Cannot add directly. 
                    # Maybe we should try a "Residue Change" of an existing one to make room?
                    # That's a different move type.
                    pass
                    
        elif move_type < 0.7:
            # Try to Remove
            if current_state:
                m_old = random.choice(list(current_state.keys()))
                r_old = current_state[m_old]
                # Remove
                del current_state[m_old]
                # Delta = -1. Accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    # Accept
                    if len(current_state) > best_size:
                        best_size = len(current_state)
                        best_state = dict(current_state)
                else:
                    # Reject: Undo
                    current_state[m_old] = r_old
                    
        else:
            # Try to Change Residue of an existing modulus
            if current_state:
                m_old = random.choice(list(current_state.keys()))
                idx_old = m_old - 2
                r_old = current_state[m_old]
                m_new_val = m_old
                # Try a random new residue
                r_try = random.randint(0, m_old - 1)
                if r_try == r_old:
                    continue
                
                # Check if r_try is disjoint from all OTHER existing moduli
                ok = True
                for m_exist, r_exist in current_state.items():
                    if m_exist == m_old:
                        continue
                    idx_exist = m_exist - 2
                    if not is_disjoint_fast(r_try, idx_old, r_exist, idx_exist):
                        ok = False
                        break
                
                if ok:
                    # Accept the change (size is same)
                    current_state[m_old] = r_try
                    # Check if this change allows an immediate add? 
                    # We don't do that in this move, but the next "Add" move might benefit.
                    # Update best if size is same? No, best is tracked by size.
                else:
                    # Reject
                    pass
        
        T *= cooling
        
    # Final cleanup: ensure best_state is valid (it should be)
    if not best_state:
        # Fallback
        return [[1, 2]]
    
    # Convert dict to list of [residue, modulus]
    result = [[r, m] for m, r in best_state.items()]
    
    return result
# EVOLVE-BLOCK-END
