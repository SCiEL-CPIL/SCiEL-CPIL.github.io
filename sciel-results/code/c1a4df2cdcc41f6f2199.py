"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _all_odd_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    base = 51975
    divisors = _all_odd_divisors(base)
    moduli = sorted(divisors)
    num_moduli = len(moduli)
    
    # Precompute step sizes for each modulus
    steps = [base // m for m in moduli]  # number of residues covered by each modulus
    
    best_score = base
    best_classes = None
    
    def greedy_initialize(order):
        """Greedy: for each modulus in given order, pick residue covering most uncovered."""
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli
        
        for idx in order:
            m = moduli[idx]
            best_r = 0
            best_gain = -1
            best_overlap = base
            for r in range(m):
                slice_ = counts[r::m]
                gain = np.count_nonzero(slice_ == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_overlap = np.count_nonzero(slice_ > 0)
                    best_r = r
                elif gain == best_gain:
                    overlap = np.count_nonzero(slice_ > 0)
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_r = r
            residues[idx] = best_r
            counts[best_r::m] += 1
        
        score = np.count_nonzero(counts == 0)
        return residues, counts, score
    
    def compute_delta(counts, m, old_r, new_r):
        """Compute change in uncovered count when changing residue from old_r to new_r for modulus m."""
        # Residues that lose coverage: old_r::m where count was 1
        # Residues that gain coverage: new_r::m where count was 0
        old_slice = counts[old_r::m]
        new_slice = counts[new_r::m]
        
        # Number of residues that become uncovered (were at 1, now at 0)
        lose = np.count_nonzero(old_slice == 1)
        # Number of residues that become covered (were at 0, now at 1)  
        gain = np.count_nonzero(new_slice == 0)
        
        # But if old_r == new_r, no change
        if old_r == new_r:
            return 0
        
        # Net change in uncovered = lose - gain
        # But we need to be careful: a residue might be in both old_r::m and new_r::m?
        # No, since old_r != new_r and both are < m, the sets old_r::m and new_r::m are disjoint.
        return lose - gain
    
    def local_search(residues, counts, score, rng, time_limit):
        """Fast local search with efficient delta computation."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_score = score
        
        temperature = 10.0
        cooling = 0.999
        min_temp = 0.1
        best_local_score = current_score
        best_local_residues = current_residues[:]
        
        start_ls = time.time()
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() - start_ls < time_limit and temperature > min_temp:
            if no_improve_count > max_no_improve:
                # Kick: randomize a few residues to escape local minimum
                for _ in range(5):
                    i = rng.randint(0, num_moduli - 1)
                    m = moduli[i]
                    old_r = current_residues[i]
                    new_r = rng.randint(0, m - 1)
                    if new_r != old_r:
                        current_counts[old_r::m] -= 1
                        current_counts[new_r::m] += 1
                        current_residues[i] = new_r
                # Recalculate score after kick
                current_score = np.count_nonzero(current_counts == 0)
                no_improve_count = 0
                temperature = max(temperature, 5.0)
                continue
            
            # Pick a random modulus index
            i = rng.randint(0, num_moduli - 1)
            m = moduli[i]
            old_r = current_residues[i]
            new_r = rng.randint(0, m - 1)
            
            if new_r == old_r:
                continue
            
            # Compute delta efficiently
            delta = compute_delta(current_counts, m, old_r, new_r)
            
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                # Accept move
                current_counts[old_r::m] -= 1
                current_counts[new_r::m] += 1
                current_residues[i] = new_r
                current_score += delta
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_residues = current_residues[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            temperature *= cooling
        
        return best_local_residues, best_local_score
    
    # Multiple restarts with different strategies
    strategies = [
        lambda rng: list(range(num_moduli)),  # natural order (small to large)
        lambda rng: list(range(num_moduli - 1, -1, -1)),  # reverse (large to small)
        lambda rng: rng.sample(range(num_moduli), num_moduli),  # random
        lambda rng: sorted(range(num_moduli), key=lambda i: moduli[i], reverse=True),  # large to small
        lambda rng: sorted(range(num_moduli), key=lambda i: -steps[i]),  # by coverage size
    ]
    
    rng = random.Random(42)
    
    for restart in range(30):  # more restarts with faster search
        if time.time() > deadline - 10:
            break
        
        # Choose strategy
        strategy_idx = restart % len(strategies)
        order = strategies[strategy_idx](rng)
        
        # Greedy initialization
        residues, counts, score = greedy_initialize(order)
        
        if score < best_score:
            best_score = score
            best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
        
        # Local search with time budget
        if time.time() > deadline - 12:
            break
        
        ls_time = min(8.0, deadline - 12 - time.time())
        if ls_time < 1.0:
            continue
        
        best_residues, best_ls_score = local_search(residues, counts, score, rng, ls_time)
        
        if best_ls_score < best_score:
            best_score = best_ls_score
            best_classes = [[moduli[i], best_residues[i]] for i in range(num_moduli)]
    
    # Final local search on the best solution found
    if best_classes is not None and time.time() < deadline - 3:
        final_residues = [c[1] for c in best_classes]
        final_counts = np.zeros(base, dtype=np.int8)
        for m, r in best_classes:
            final_counts[r::m] += 1
        final_score = np.count_nonzero(final_counts == 0)
        
        remaining_time = deadline - 3 - time.time()
        if remaining_time > 1.0:
            improved_residues, improved_score = local_search(
                final_residues, final_counts, final_score, rng, remaining_time
            )
            if improved_score < best_score:
                best_score = improved_score
                best_classes = [[moduli[i], improved_residues[i]] for i in range(num_moduli)]
    
    return best_classes
# EVOLVE-BLOCK-END
