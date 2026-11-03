<<<<<<< SEARCH
        # If we have a good code, try perturbing it
        if time.time() < deadline - 20 and best_code is not None:
            # Perturb: remove a few random codewords
            k = random.randint(2, 5)
            if len(best_code) > k:
                perturbed = list(best_code)
                remove_indices = random.sample(range(len(perturbed)), k)
                for idx in sorted(remove_indices, reverse=True):
                    perturbed.pop(idx)
                
                # Repair
                counts = np.zeros(total_words, dtype=np.int16)
                for c in perturbed:
                    counts[neighbors_arr[c, :]] += 1
                uncovered = counts == 0
                
                while uncovered.any() and time.time() < deadline - 10:
                    gains = np.sum(uncovered[neighbors_arr], axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                    best_candidates = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_candidates.tolist())
                    perturbed.append(chosen)
                    counts[neighbors_arr[chosen, :]] += 1
                    uncovered = counts == 0
                
                # Polish
                perturbed = reduce_code(perturbed)
                if time.time() < deadline - 10:
                    perturbed = local_search(perturbed, deadline - 5)
                    perturbed = reduce_code(perturbed)
                
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed
=======
        # If we have a good code, try perturbing it
        if time.time() < deadline - 20 and best_code is not None:
            # Perturb: remove a few random codewords
            k = random.randint(1, 4)
            if len(best_code) > k:
                perturbed = list(best_code)
                remove_indices = random.sample(range(len(perturbed)), k)
                for idx in sorted(remove_indices, reverse=True):
                    perturbed.pop(idx)
                
                # Repair
                counts = np.zeros(total_words, dtype=np.int16)
                for c in perturbed:
                    counts[neighbors_arr[c, :]] += 1
                uncovered = (counts == 0).astype(np.int8)
                
                while uncovered.any() and time.time() < deadline - 10:
                    gains = np.dot(neighbors_arr, uncovered)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                    best_candidates = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_candidates.tolist())
                    perturbed.append(chosen)
                    counts[neighbors_arr[chosen, :]] += 1
                    uncovered = (counts == 0).astype(np.int8)
                
                # Polish
                perturbed = reduce_code(perturbed)
                if time.time() < deadline - 10:
                    perturbed = local_search(perturbed, deadline - 5)
                    perturbed = reduce_code(perturbed)
                
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed
>>>>>>> REPLACE

