    def local_search(indices):
        current_indices = list(indices)
        
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in current_indices:
            counts[k_to_t_array[idx]] += 1
            
        def is_redundant(idx):
            counts[k_to_t_array[idx]] -= 1
            redundant = np.all(counts > 0)
            counts[k_to_t_array[idx]] += 1
            return redundant

        # Phase 1: Remove redundant
        improved = True
        while improved:
            improved = False
            random.shuffle(current_indices)
            new_indices = []
            for idx in current_indices:
                if is_redundant(idx):
                    counts[k_to_t_array[idx]] -= 1
                    improved = True
                else:
                    new_indices.append(idx)
            current_indices = new_indices
            
        # Phase 2: Swapping
        max_swaps = 300
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            if not current_indices:
                break
                
            pos = random.randint(0, len(current_indices) - 1)
            b = current_indices[pos]
            
            counts[k_to_t_array[b]] -= 1
            
            critical_sets = np.where(counts == 0)[0]
            
            if len(critical_sets) == 0:
                counts[k_to_t_array[b]] += 1
                continue
                
            candidate_set = set()
            for c in critical_sets:
                candidate_set.update(t_to_k_indices[c])
                
            current_set = set(current_indices)
            current_set.discard(b)
            
            best_gain = 0
            best_candidate = -1
            
            for cand in candidate_set:
                if cand in current_set:
                    continue
                
                gain = 0
                cand_t_set = k_to_t_set[cand]
                for c in critical_sets:
                    if c in cand_t_set:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
                    
            if best_candidate != -1 and best_gain > 0:
                current_indices[pos] = best_candidate
                counts[k_to_t_array[best_candidate]] += 1
                
        # Final redundancy
        improved = True
        while improved:
            improved = False
            random.shuffle(current_indices)
            new_indices = []
            for idx in current_indices:
                if is_redundant(idx):
                    counts[k_to_t_array[idx]] -= 1
                    improved = True
                else:
                    new_indices.append(idx)
            current_indices = new_indices
            
        return current_indices
