def construct(n):
    word = [1]
    for m in range(2, n + 1):
        # Extract unique (m-1) permutations
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Create blocks
        blocks = []
        for p in perms:
            blocks.append(list(p) + [m] + list(p))
        
        if not blocks:
            continue
            
        # Greedy TSP to order blocks
        # State: current block index, visited set
        # We just want to concatenate them.
        
        num_blocks = len(blocks)
        if num_blocks == 1:
            new_word = blocks[0]
            word = new_word
            continue
            
        # Precompute overlap matrix? Or just calculate on the fly.
        # For 720 blocks, on the fly is fine.
        
        visited = [False] * num_blocks
        current_block = 0
        visited[0] = True
        new_word = list(blocks[0])
        
        for _ in range(num_blocks - 1):
            best_idx = -1
            best_overlap = -1
            best_start = 0
            
            current_suffix = new_word
            
            for j in range(num_blocks):
                if visited[j]:
                    continue
                
                block = blocks[j]
                # Calculate max overlap
                max_k = min(len(current_suffix), len(block))
                found_overlap = 0
                # Check from largest k down to 0
                # Optimization: only check k up to m-1? 
                # The overlap can be at most m-1? 
                # Block length is 2(m-1)+1 = 2m-1.
                # Overlap can be larger if blocks are similar.
                for k in range(max_k, -1, -1):
                    if k == 0:
                        found_overlap = 0
                        break
                    if current_suffix[-k:] == block[:k]:
                        found_overlap = k
                        break
                
                if found_overlap > best_overlap:
                    best_overlap = found_overlap
                    best_idx = j
            
            if best_idx == -1:
                break
                
            visited[best_idx] = True
            block = blocks[best_idx]
            new_word += block[best_overlap:]
            
        word = new_word
        
    return "".join(map(str, word))
