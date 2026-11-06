def design(problem):
    start_time = time.time()
    time_budget = 50.0
    
    band = problem["band_nm"]
    lambda_0 = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    problem["wavelengths"] = wavelengths
    
    # Generate initial sequences
    initial_seqs = []
    for n in range(1, max_layers + 1, 2): # Odd number of layers often better for AR
        seq_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        seq_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        initial_seqs.append(seq_L)
        initial_seqs.append(seq_H)
        
    # Also include some even
    for n in range(2, max_layers + 1, 2):
        seq_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        seq_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        initial_seqs.append(seq_L)
        initial_seqs.append(seq_H)
        
    best_seq = initial_seqs[0]
    best_scale = 1.0
    best_score = 100.0
    
    for seq in initial_seqs:
        s, score = _optimize_scale_for_seq(seq, problem)
        if score < best_score:
            best_score = score
            best_seq = list(seq)
            best_scale = s
            
    np.random.seed(42)
    
    # Simulated Annealing
    num_restarts = 10
    iters_per_restart = 5000
    T_init = 0.5
    T_final = 0.001
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget - 10:
            break
            
        if restart == 0:
            current_seq = list(best_seq)
            current_scale = best_scale
            current_score = best_score
        else:
            n_layers = np.random.randint(2, max_layers + 1)
            current_seq = ['L' if np.random.rand() < 0.5 else 'H' for _ in range(n_layers)]
            current_scale = np.random.uniform(0.5, 2.0)
            current_score = _evaluate(current_seq, current_scale, problem)
            if current_score < best_score:
                best_score = current_score
                best_seq = list(current_seq)
                best_scale = current_scale
                
        cooling = (T_final / T_init) ** (1.0 / iters_per_restart)
        
        for i in range(iters_per_restart):
            if time.time() - start_time > time_budget - 5:
                break
                
            T = T_init * (cooling ** i)
            
            r = np.random.rand()
            if r < 0.4:
                new_seq = list(current_seq)
                if len(new_seq) > 0:
                    idx = np.random.randint(len(new_seq))
                    new_seq[idx] = 'H' if new_seq[idx] == 'L' else 'L'
                else:
                    new_seq = ['L']
                new_scale = current_scale
            elif r < 0.6:
                if len(current_seq) < max_layers:
                    new_seq = list(current_seq)
                    pos = np.random.randint(len(new_seq) + 1)
                    new_layer = 'L' if np.random.rand() < 0.5 else 'H'
                    new_seq.insert(pos, new_layer)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            elif r < 0.8:
                if len(current_seq) > 1:
                    new_seq = list(current_seq)
                    idx = np.random.randint(len(new_seq))
                    new_seq.pop(idx)
                else:
                    new_seq = list(current_seq)
                new_scale = current_scale
            else:
                new_seq = list(current_seq)
                new_scale = current_scale * (1.0 + 0.2 * np.random.randn())
                new_scale = max(0.3, min(3.0, new_scale))
                
            new_score = _evaluate(new_seq, new_scale, problem)
            
            if new_score < current_score:
                current_seq = new_seq
                current_scale = new_scale
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_seq = list(current_seq)
                    best_scale = current_scale
            else:
                delta = new_score - current_score
                if delta > 0 and T > 0:
                    accept_prob = np.exp(-delta / T)
                    if np.random.rand() < accept_prob:
                        current_seq = new_seq
                        current_scale = new_scale
                        current_score = new_score
                        
    # Final refinement
    best_scale, best_score = _optimize_scale_for_seq(best_seq, problem)
    
    # Try small modifications with full scale optimization
    if time.time() - start_time < time_budget - 3:
        for _ in range(100):
            trial_seq = list(best_seq)
            r = np.random.rand()
            if r < 0.4 and len(trial_seq) < max_layers:
                pos = np.random.randint(len(trial_seq) + 1)
                trial_seq.insert(pos, 'L' if np.random.rand() < 0.5 else 'H')
            elif r < 0.7 and len(trial_seq) > 1:
                idx = np.random.randint(len(trial_seq))
                trial_seq.pop(idx)
            else:
                if len(trial_seq) > 0:
                    idx = np.random.randint(len(trial_seq))
                    trial_seq[idx] = 'H' if trial_seq[idx] == 'L' else 'L'
                    
            s, score = _optimize_scale_for_seq(trial_seq, problem)
            if score < best_score:
                best_score = score
                best_seq = list(trial_seq)
                best_scale = s
                
    materials_labels = best_seq
    thicknesses = []
    for label in best_seq:
        n = mat_dict[label]
        d = best_scale * lambda_0 / (4.0 * n)
        thicknesses.append(d)
        
    return materials_labels, thicknesses
