import math
import random
import time

def construct():
    moduli = list(range(20, 201))
    
    # Precompute gcds to speed up compatibility checks
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    # We can just use math.gcd on the fly, it's fast enough for ~200 items.
    
    def is_compatible(selected, m, r):
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def get_valid_residues(selected, m):
        valid = []
        for r in range(m):
            if is_compatible(selected, m, r):
                valid.append(r)
        return valid

    def greedy_construct(seed_val):
        random.seed(seed_val)
        selected = []
        remaining = set(moduli)
        
        # Sort by weight descending, but we will pick from remaining dynamically
        # To make it faster, we can just iterate through a sorted list and check if in remaining
        sorted_moduli = sorted(moduli, key=lambda x: -1.0/x)
        
        # We will iterate through sorted_moduli.
        # If a modulus is skipped, we might want to retry it later?
        # For a simple greedy, we just pass through once.
        # But local search will handle adding skipped ones.
        
        skipped = []
        
        for m in sorted_moduli:
            if m not in remaining:
                continue
            valid_res = get_valid_residues(selected, m)
            if valid_res:
                # Pick a random valid residue
                r = random.choice(valid_res)
                selected.append((m, r))
                remaining.remove(m)
            else:
                skipped.append(m)
        
        return selected, skipped

    def local_search(selected, skipped, max_iters=100):
        # Convert to list of tuples for easier manipulation
        # We want to maximize sum(1/m)
        
        # Helper to calculate score
        def score(sel):
            return sum(1.0/m for m, r in sel)
        
        best_score = score(selected)
        best_selected = [tuple(x) for x in selected]
        
        # We will try to add skipped moduli and change residues to enable additions
        # We need to maintain the set of skipped moduli
        skipped_set = set(skipped)
        
        for _ in range(max_iters):
            improved = False
            
            # 1. Try to add any skipped modulus
            # Iterate through skipped in order of weight?
            # Just try all
            for m in sorted(skipped_set, key=lambda x: -1.0/x):
                valid_res = get_valid_residues(best_selected, m)
                if valid_res:
                    # Pick random residue
                    r = random.choice(valid_res)
                    new_selected = best_selected + [(m, r)]
                    new_score = score(new_selected)
                    if new_score > best_score:
                        best_selected = new_selected
                        best_score = new_score
                        skipped_set.remove(m)
                        improved = True
                        break # Restart loop with new state
            
            if improved:
                continue
                
            # 2. Try to change a residue of an existing modulus to unlock a skipped one
            # This is more complex. We change r_i to r_i' and check if any skipped m becomes valid.
            # We only accept the change if it allows adding at least one skipped modulus.
            
            # To avoid O(N^2 * M) complexity, we can be selective.
            # Let's try changing residues for a few random selected moduli.
            
            if best_selected:
                # Pick a random selected modulus to change
                idx = random.randint(0, len(best_selected) - 1)
                m, r_old = best_selected[idx]
                
                # Get all valid residues for m given the OTHER selected moduli
                others = [x for i, x in enumerate(best_selected) if i != idx]
                valid_res = get_valid_residues(others, m)
                
                if len(valid_res) > 1:
                    # Try a different residue
                    # Pick a random new residue different from r_old
                    if r_old in valid_res:
                        valid_res.remove(r_old)
                    
                    if valid_res:
                        r_new = random.choice(valid_res)
                        
                        # Create a temporary new selected list
                        temp_selected = list(best_selected)
                        temp_selected[idx] = (m, r_new)
                        
                        # Check if any skipped modulus can now be added
                        added_any = False
                        for m_skip in sorted(skipped_set, key=lambda x: -1.0/x):
                            if get_valid_residues(temp_selected, m_skip):
                                # We can add m_skip
                                # We should probably add the best one
                                valid_skip_res = get_valid_residues(temp_selected, m_skip)
                                r_skip = random.choice(valid_skip_res)
                                
                                # Apply the change and the addition
                                best_selected = temp_selected + [(m_skip, r_skip)]
                                skipped_set.remove(m_skip)
                                best_score = score(best_selected)
                                improved = True
                                added_any = True
                                break
                        
                        # If we didn't add any, we might still keep the residue change?
                        # No, because it doesn't improve score directly.
                        # But it might help in future iterations?
                        # For simplicity, only keep change if it leads to immediate addition.
                        # This is a "greedy" local search.
                        
            if not improved:
                break
        
        return best_selected

    # Main loop: Run multiple restarts
    best_solution = []
    best_val = 0.0
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    seed = 0
    while time.time() - start_time < time_limit:
        selected, skipped = greedy_construct(seed)
        # Run local search
        improved_sel = local_search(selected, skipped, max_iters=50)
        
        val = sum(1.0/m for m, r in improved_sel)
        if val > best_val:
            best_val = val
            best_solution = improved_sel
        
        seed += 1
        
    return [[m, r] for m, r in best_solution]
