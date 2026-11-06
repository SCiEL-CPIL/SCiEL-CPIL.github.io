import time

def construct():
    start = time.time()
    deadline = start + 85  # 85 second deadline
    
    # Start with Batcher network
    network = batcher_network(16)
    
    # Try to remove redundant comparators
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(len(network)):
            if time.time() > deadline - 5:
                break
            # Try removing network[idx]
            test = network[:idx] + network[idx+1:]
            if is_valid(test):
                network = test
                improved = True
                break
    
    return network
