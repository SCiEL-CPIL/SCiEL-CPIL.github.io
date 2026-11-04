<<<<<<< SEARCH
import numpy as np
import time
import itertools

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 5s margin before 120s kill
    
    if n <= 8 and radius == 1:
        total_words = q ** n
        sphere_size = 1 + n * (q - 1)
        
        # Precompute all neighbor indices for each word
        # neighbors[i] = list of indices within Hamming distance 1 of word i
        neighbors = np.full((total_words, sphere_size), -1, dtype=np.int32)
        for i in range(total_words):
            nbrs = [i]
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                current_val = (i // pow_q) % q
                for new_val in range(q):
                    if new_val != current_val:
                        nbrs.append(i + (new_val - current_val) * pow_q)
            neighbors[i] = nbrs
        
        # Greedy: pick word covering most uncovered
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while time.time() < deadline - 30.0:  # Leave 30s for local search
            gains = (~covered)[neighbors].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            code_indices.append(best_idx)
            covered[neighbors[best_idx]] = True
        
        # Local search: try to remove redundant codewords
        # A codeword is redundant if removing it doesn't uncover any word
        code_set = set(code_indices)
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            # Recompute coverage from current code
            covered_check = np.zeros(total_words, dtype=bool)
            for ci in code_indices:
                covered_check[neighbors[ci]] = True
            
            for idx_in_code, ci in enumerate(code_indices):
                if time.time() > deadline - 10.0:
                    break
                # Check if ci is redundant: is every neighbor of ci covered by another codeword?
                # i.e., for all j in neighbors[ci], is covered_check[j] true after removing ci?
                # Simpler: temporarily remove ci, check if all neighbors still covered
                nbrs_of_ci = neighbors[ci]
                # For each neighbor j of ci, check if j is covered by some other codeword
                # This is expensive. Alternative: just try removing and check full coverage
                # But that's O(len(code) * sphere_size) per removal attempt
                
                # Faster: for each position in nbrs_of_ci, check if covered by another
                # We can precompute for each word, which codewords cover it
                # But that's complex. Let's just do the check:
                redundant = True
                for j in nbrs_of_ci:
                    # Is j covered by any codeword other than ci?
                    # Check all other codewords
                    found = False
                    for other_ci in code_indices:
                        if other_ci == ci:
                            continue
                        if j in neighbors[other_ci]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    code_indices.remove(ci)
                    changed = True
                    break  # Restart the loop since coverage changed
        
        # Convert to words
        result_words = []
        for idx in code_indices:
            w = []
            temp = idx
            for _ in range(n):
                w.append(temp % q)
                temp //= q
            w.reverse()
            result_words.append(w)
        
        return result_words
    else:
=======
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 5s margin before 120s kill
    
    if n <= 8 and radius == 1:
        total_words = q ** n
        sphere_size = 1 + n * (q - 1)
        
        # Precompute all neighbor indices for each word
        # neighbors[i] = list of indices within Hamming distance 1 of word i
        neighbors = np.full((total_words, sphere_size), -1, dtype=np.int32)
        for i in range(total_words):
            nbrs = [i]
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                current_val = (i // pow_q) % q
                for new_val in range(q):
                    if new_val != current_val:
                        nbrs.append(i + (new_val - current_val) * pow_q)
            neighbors[i] = nbrs
        
        # Greedy: pick word covering most uncovered
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while time.time() < deadline - 40.0:  # Leave 40s for local search
            gains = (~covered)[neighbors].sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            code_indices.append(best_idx)
            covered[neighbors[best_idx]] = True
        
        # Local search: try to remove redundant codewords
        # A codeword is redundant if removing it doesn't uncover any word
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            # Recompute coverage from current code
            covered_check = np.zeros(total_words, dtype=bool)
            for ci in code_indices:
                covered_check[neighbors[ci]] = True
            
            for idx_in_code, ci in enumerate(code_indices):
                if time.time() > deadline - 10.0:
                    break
                # Check if ci is redundant: is every neighbor of ci covered by another codeword?
                nbrs_of_ci = neighbors[ci]
                # For each neighbor j of ci, check if j is covered by some other codeword
                redundant = True
                for j in nbrs_of_ci:
                    # Is j covered by any codeword other than ci?
                    found = False
                    for other_ci in code_indices:
                        if other_ci == ci:
                            continue
                        if j in neighbors[other_ci]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    code_indices.remove(ci)
                    changed = True
                    break  # Restart the loop since coverage changed
        
        # Add-then-remove heuristic: try adding a codeword, then removing two to see if we can reduce size
        # This can escape local minima of the simple removal heuristic
        while time.time() < deadline - 5.0:
            # Try random add-then-remove
            if len(code_indices) < 2:
                break
            
            # Pick a random candidate to add
            candidate = random.randint(0, total_words - 1)
            if candidate in code_indices:
                continue
            
            # Temporarily add it
            code_indices.append(candidate)
            
            # Now try to remove two codewords (including possibly the new one)
            # to see if we can end up with fewer total
            removed = 0
            for _ in range(2):
                if time.time() > deadline - 5.0:
                    break
                # Recompute coverage
                covered_check = np.zeros(total_words, dtype=bool)
                for ci in code_indices:
                    covered_check[neighbors[ci]] = True
                
                # Find a redundant codeword to remove
                found_redundant = False
                for idx_in_code, ci in enumerate(code_indices):
                    if time.time() > deadline - 5.0:
                        break
                    nbrs_of_ci = neighbors[ci]
                    redundant = True
                    for j in nbrs_of_ci:
                        # Is j covered by any codeword other than ci?
                        found = False
                        for other_ci in code_indices:
                            if other_ci == ci:
                                continue
                            if j in neighbors[other_ci]:
                                found = True
                                break
                        if not found:
                            redundant = False
                            break
                    if redundant:
                        code_indices.remove(ci)
                        removed += 1
                        found_redundant = True
                        break
                
                if not found_redundant:
                    break
            
            # If we removed 2 but added 1, net -1; if removed 1, net 0; if removed 0, net +1
            # We only keep this if we reduced the size
            if removed < 2:
                # Revert: remove the candidate we added
                if candidate in code_indices:
                    code_indices.remove(candidate)
        
        # Final pass: remove any remaining redundant codewords
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            covered_check = np.zeros(total_words, dtype=bool)
            for ci in code_indices:
                covered_check[neighbors[ci]] = True
            
            for idx_in_code, ci in enumerate(code_indices):
                if time.time() > deadline - 1.0:
                    break
                nbrs_of_ci = neighbors[ci]
                redundant = True
                for j in nbrs_of_ci:
                    found = False
                    for other_ci in code_indices:
                        if other_ci == ci:
                            continue
                        if j in neighbors[other_ci]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    code_indices.remove(ci)
                    changed = True
                    break
        
        # Convert to words
        result_words = []
        for idx in code_indices:
            w = []
            temp = idx
            for _ in range(n):
                w.append(temp % q)
                temp //= q
            w.reverse()
            result_words.append(w)
        
        return result_words
    else:
>>>>>>> REPLACE

