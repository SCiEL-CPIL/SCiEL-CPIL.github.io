<<<<<<< SEARCH
        def goptimal_polish(labels):
            """
            A more aggressive local search that tries all single flips,
            then all pairs of flips, to find a better local optimum.
            """
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            # First, do standard hill climbing
            for rnd in range(200):
                if time_left() < 2.0:
                    break
                improved = False
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    gains = compute_gains(best_labels)
                    delta = int(gains[v])
                    if delta > 0:
                        best_labels[v] = 1 - best_labels[v]
                        best_cut += delta
                        improved = True
                if not improved:
                    break
            
            # Try pairwise flips
            for rnd in range(10):
                if time_left() < 2.0:
                    break
                improved = False
                # Sample a subset of pairs to try
                pairs = np.random.randint(0, n, size=(min(200, n * n // 2), 2))
                for i, j in pairs:
                    if i == j:
                        continue
                    if time_left() < 1.0:
                        break
                    
                    # Compute gain of flipping both i and j
                    # This is more complex, so we'll do it by actually flipping
                    old_i = best_labels[i]
                    old_j = best_labels[j]
                    best_labels[i] = 1 - old_i
                    best_labels[j] = 1 - old_j
                    new_cut = cut_size(best_labels)
                    delta = new_cut - best_cut
                    if delta > 0:
                        best_cut = new_cut
                        improved = True
                    else:
                        # Revert
                        best_labels[i] = old_i
                        best_labels[j] = old_j
                
                if not improved:
                    break
            
            return best_labels, best_cut
=======
        def goptimal_polish(labels):
            """
            A more aggressive local search that tries all single flips,
            then all pairs of flips, to find a better local optimum.
            """
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            # First, do standard hill climbing with gains computed once per round
            for rnd in range(200):
                if time_left() < 2.0:
                    break
                improved = False
                gains = compute_gains(best_labels)
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    delta = int(gains[v])
                    if delta > 0:
                        best_labels[v] = 1 - best_labels[v]
                        best_cut += delta
                        improved = True
                if not improved:
                    break
            
            # Try pairwise flips
            for rnd in range(10):
                if time_left() < 2.0:
                    break
                improved = False
                pairs = np.random.randint(0, n, size=(min(200, n * n // 2), 2))
                for i, j in pairs:
                    if i == j:
                        continue
                    if time_left() < 1.0:
                        break
                    
                    old_i = best_labels[i]
                    old_j = best_labels[j]
                    best_labels[i] = 1 - old_i
                    best_labels[j] = 1 - old_j
                    new_cut = cut_size(best_labels)
                    delta = new_cut - best_cut
                    if delta > 0:
                        best_cut = new_cut
                        improved = True
                    else:
                        best_labels[i] = old_i
                        best_labels[j] = old_j
                
                if not improved:
                    break
            
            return best_labels, best_cut
>>>>>>> REPLACE
