    MAX_WORDS = 1000
    code_matrix = np.zeros((MAX_WORDS, N), dtype=np.uint8)
    code_matrix[0] = all_bits[0]
    current_size = 1
    
    for w in range(1, 1 << N):
        if time.time() > DEADLINE:
            break
        
        bits = all_bits[w]
        
        # Check distance against code_matrix[:current_size]
        dists = np.sum(code_matrix[:current_size] ^ bits, axis=1)
        
        if np.min(dists) >= MIN_DIST:
            words_list.append(w)
            code_matrix[current_size] = bits
            current_size += 1
            
    return words_list
