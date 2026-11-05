"""Optimized odd covering system search for moduli dividing 51975."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (evaluator kills at 150s)
    
    base = 51975
    # 51975 = 3^3 * 5^2 * 7 * 11
    # Get all odd divisors of base greater than 1 using prime factorization
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = set()
    for p1 in range(4):  # 3^p1, p1=0..3
        for p2 in range(3):  # 5^p2, p2=0..2
            for p3 in range(2):  # 7^p3, p3=0..1
                for p4 in range(2):  # 11^p4, p4=0..1
                    d = (3 ** p1) * (5 ** p2) * (7 ** p3) * (11 ** p4)
                    if d > 1:
                        divisors.add(d)
    divisors = sorted(divisors)
    
    # Precompute index maps for fast lookup
    divisor_set = set(divisors)
    
    def compute_gain_for_residue(m, r, counts):
        """Compute gain for a specific residue class (m, r)."""
        return np.count_nonzero(counts[r::m] == 0)
    
    def local_search_polish(classes, counts, deadline):
        """Local search to improve residue assignments with early termination."""
        improved = True
        max_passes = 100
        pass_count = 0
        total_improvements = 0
        
        while improved and pass_count < max_passes and time.time() < deadline - 8:
            improved = False
            pass_count += 1
            pass_improvements = 0
            
            indices = list(range(len(classes)))
            np.random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    return classes, counts, total_improvements
                
                m, old_r = classes[idx]
                
                # Temporarily remove this class
                counts[old_r::m] -= 1
                
                # Compute loss: positions in old_r class now uncovered
                loss_old = np.count_nonzero(counts[old_r::m] == 0)
                
                # Find best new residue
                best_gain = loss_old
                best_r = old_r
                
                # For efficiency, compute gains for all residues
                # For small m, try all; for large m, sample
                if m <= 2000:
                    for r in range(m):
                        if r == old_r:
                            continue
                        gain = np.count_nonzero(counts[r::m] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                else:
                    # Sample a subset of residues for very large m
                    n_sample = min(m, 500)
                    if m <= 500:
                        sample = list(range(m))
                    else:
                        sample = np.random.choice(m, size=n_sample, replace=False)
                        sample = [int(x) for x in sample]
                    for r in sample:
                        if r == old_r:
                            continue
                        gain = np.count_nonzero(counts[r::m] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                
                # Apply the best choice
                if best_r != old_r:
                    counts[best_r::m] += 1
                    classes[idx][1] = best_r
                    improved = True
                    pass_improvements += 1
                else:
                    counts[old_r::m] += 1
            
            total_improvements += pass_improvements
            
            # Early termination if no improvements in this pass
            if pass_improvements == 0:
                break
        
        return classes, counts, total_improvements
    
    # Multiple restarts with different random seeds for diversity
    best_classes = None
    best_uncovered = base + 1
    
    n_restarts = 5  # Number of randomized restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        np.random.seed(restart * 12345 + 67890)
        
        counts = np.zeros(base, dtype=np.int32)
        classes = []
        used_moduli = set()
        remaining_moduli = list(divisors)
        
        # Randomized greedy with adaptive look-ahead
        # At each step, compute marginal gain for each available modulus
        # Use softmax-like probabilistic selection based on gain
        # Adaptive: as we use more moduli, reduce temperature to be more greedy
        
        n_moduli = len(remaining_moduli)
        for step in range(n_moduli):
            if time.time() > deadline - 20:
                break
            
            # Compute gains for each remaining modulus
            # For each modulus, the best residue's gain
            moduli_to_consider = [m for m in remaining_moduli if m not in used_moduli]
            
            if not moduli_to_consider:
                break
            
            # Compute best gain for each modulus
            gains = []
            for m in moduli_to_consider:
                # Find best residue for this modulus
                best_r = 0
                best_gain = -1
                for r in range(m):
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                gains.append((m, best_r, best_gain))
            
            if not gains:
                break
            
            # Adaptive temperature: start high (exploratory), decrease over time
            max_gain = max(g[2] for g in gains)
            if max_gain <= 0:
                break
            
            # Use softmax weighting based on gain
            temperature = max_gain * 0.3 + 1.0  # Adaptive temperature
            
            weights = []
            for m, r, g in gains:
                w = np.exp(g / temperature)
                weights.append(w)
            
            total_w = sum(weights)
            if total_w == 0:
                break
            
            # Probabilistic selection
            probs = [w / total_w for w in weights]
            idx = np.random.choice(len(gains), p=probs)
            
            m, best_r, _ = gains[idx]
            classes.append([m, best_r])
            counts[best_r::m] += 1
            used_moduli.add(m)
            remaining_moduli.remove(m)
        
        # Local search polish
        classes, counts, improvements = local_search_polish(classes, counts, deadline)
        
        # Evaluate
        uncovered = np.count_nonzero(counts == 0)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [list(c) for c in classes]
        
        if best_uncovered == 0:
            break
    
    if best_classes is None:
        # Fallback: return empty
        return []
    
    return best_classes
# EVOLVE-BLOCK-END
