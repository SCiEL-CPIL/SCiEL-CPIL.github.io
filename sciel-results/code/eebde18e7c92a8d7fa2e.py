<<<<<<< SEARCH
    # Partition 40: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13, rank 108
    
    # Return the best result found
    return best_decomp
=======
    # Partition 40: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13, rank 108
    
    # More 2x2x2 focused partitions
    
    # Partition 41: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13
    
    # Partition 42: n=1+1+2, m=2+2+1, p=2+2+2
    # Already tried as partition 18
    
    # Partition 43: n=2+1+1, m=2+2+1, p=2+2+2
    # 2x2x2 blocks: ni=0 (ns=2): 12 blocks
    # Total: 12*7 + 2*(1*2*2) + 2*(1*2*2) = 84 + 8 + 8 = 100
    result, rank = try_partition([2,1,1], [2,2,1], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 44: n=1+2+1, m=2+2+1, p=2+2+2
    # 2x2x2 blocks: ni=1 (ns=2): 12 blocks
    # Total: 12*7 + 2*(1*2*2) + 2*(1*2*2) = 84 + 8 + 8 = 100
    result, rank = try_partition([1,2,1], [2,2,1], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 45: n=2+2, m=2+2+1, p=1+2+2+1
    # 2x2x2 blocks: pi=1,2: 12 blocks
    # Total: 12*7 + 2*(2*2*1) + 2*(2*2*1) = 84 + 8 + 8 = 100
    result, rank = try_partition([2,2], [2,2,1], [1,2,2,1])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 46: n=2+2, m=1+2+2+1, p=2+2+2
    # 2x2x2 blocks: mi=1,2: 12 blocks
    # Total: 12*7 + 2*(2*1*2) + 2*(2*1*2) = 84 + 8 + 8 = 100
    result, rank = try_partition([2,2], [1,2,2,1], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 47: n=2+2, m=2+1+2+1, p=2+2+2
    # 2x2x2 blocks: mi=0,2: 12 blocks
    # Total: 12*7 + 2*(2*1*2) + 2*(2*1*2) = 84 + 8 + 8 = 100
    result, rank = try_partition([2,2], [2,1,2,1], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 48: n=1+2+1, m=2+2+1, p=2+2+2
    # Already tried as partition 44
    
    # Partition 49: n=2+1+1, m=2+2+1, p=2+2+2
    # Already tried as partition 43
    
    # Partition 50: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13, rank 108
    
    # Return the best result found
    return best_decomp
>>>>>>> REPLACE
