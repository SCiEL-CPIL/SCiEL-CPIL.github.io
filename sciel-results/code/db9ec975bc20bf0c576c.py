import numpy as np
import random
import time

def construct():
    base = 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    n_classes = len(divisors)
    
    def local_search(initial_residues, time_limit=10):
        """
        Perform local search with perturbation to minimize uncovered count.
        """
        residues = initial_residues.copy()
        
        # Initialize counts
        counts = np.zeros(base, dtype=np.int32)
        for i, m in enumerate(divisors):
            r = residues[i]
            counts[r::m] += 1
        
        unc = int(np.sum(counts == 0))
        
        start_time = time.time()
        no_improve_passes = 0
        max_no_improve_passes = 3 # Perturb if stuck for 3 full passes
        
        while (time.time() - start_time) < time_limit:
            improved = False
            
            for i in range(n_classes):
                m = divisors[i]
                r_old = residues[i]
                
                # 1. Remove old residue
                counts[r_old::m] -= 1
                new_uncovered = int(np.sum(counts[r_old::m] == 0))
                unc_temp = unc + new_uncovered
                
                # 2. Calculate gains for all r
                is_zero = (counts == 0)
                # Reshape trick for fast sum
                gains = is_zero.reshape((m, base // m), order='F').sum(axis=1)
                
                best_r = int(np.argmax(gains))
                max_gain = gains[best_r]
                best_unc = unc_temp - max_gain
                
                # 3. Apply best
                if best_r != r_old:
                    counts[best_r::m] += 1
                    residues[i] = best_r
                    unc = best_unc
                    improved = True
                else:
                    counts[r_old::m] += 1
                    # unc unchanged
                    
            if not improved:
                no_improve_passes += 1
                if no_improve_passes >= max_no_improve_passes:
                    # Perturb: change 2-5 random residues
                    k = random.randint(2, 5)
                    indices = random.sample(range(n_classes), k)
                    for idx in indices:
                        m = divisors[idx]
                        old_r = residues[idx]
                        counts[old_r::m] -= 1
                        new_r = random.randint(0, m-1)
                        residues[idx] = new_r
                        counts[new_r::m] += 1
                    # Recalculate unc
                    unc = int(np.sum(counts == 0))
                    no_improve_passes = 0
            else:
                no_improve_passes = 0
                
        return residues, unc

    best_unc = base
    best_res = None
    
    # Use more time, fewer restarts but longer search
    # Total ~110s
    num_restarts = 5
    time_per_restart = 20 # 5 * 20 = 100s
    
    for k in range(num_restarts):
        if k == 0:
            init_res = [0] * n_classes
        else:
            init_res = [random.randint(0, m-1) for m in divisors]
        
        res, unc = local_search(init_res, time_limit=time_per_restart)
        
        if unc < best_unc:
            best_unc = unc
            best_res = res
    
    if best_res is None:
        best_res = [0] * n_classes
        # Fallback evaluation
        counts = np.zeros(base, dtype=np.int32)
        for i, m in enumerate(divisors):
            r = best_res[i]
            counts[r::m] += 1
        best_unc = int(np.sum(counts == 0))

    classes = [[divisors[i], best_res[i]] for i in range(n_classes)]
    return classes
