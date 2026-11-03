    rng = np.random.RandomState(42)
    gen = _init_generator(rng)
    
    for i in range(100): # Max 100 tries
        if time.time() > deadline - 5:
            break
        try:
            init = next(gen)
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            x_opt, score = _local_optimize_slsqp(init_bary.flatten(), deadline)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.reshape(N, 2)
                
        except StopIteration:
            break
        except Exception:
            pass
