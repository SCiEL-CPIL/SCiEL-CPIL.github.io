    def construct():
        start_time = time.time()
        deadline = start_time + 110.0
        
        best_S_int = 0
        best_count = 0
        best_set_list = None # To avoid reconverting
        
        while time.time() < deadline:
            S_int, count = _greedy_initial()
            S_int, count = _local_search(S_int, count, time_limit=5.0)
            
            if count > best_count:
                best_count = count
                best_S_int = S_int
                # Convert to list for return later? Or keep int.
                
            # If time is running out, break
            if time.time() > deadline - 10.0:
                break
                
            # Perturbation: Take best, remove 3 random, re-optimize
            # Only if we have enough time
            if best_count > 0:
                perturbed_S_int = best_S_int
                perturbed_count = best_count
                # Remove 3 random from best
                present = [i for i in range(N_TRIPLES) if (perturbed_S_int & (1 << i))]
                random.shuffle(present)
                for i in present[:3]:
                    perturbed_S_int &= ~(1 << i)
                    perturbed_count -= 1
                    
                perturbed_S_int, perturbed_count = _local_search(perturbed_S_int, perturbed_count, time_limit=5.0)
                if perturbed_count > best_count:
                    best_count = perturbed_count
                    best_S_int = perturbed_S_int

        # Convert back to list of triples
        result = []
        for i in range(N_TRIPLES):
            if best_S_int & (1 << i):
                result.append(list(ALL_TRIPLES[i]))
                
        return result
    