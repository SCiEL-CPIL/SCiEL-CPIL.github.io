def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Start with Batcher's Odd-Even Merge Sort (63 comparators for n=16)
    network = _batcher(N)
    
    # Precompute all inputs
    inputs = np.arange(TOTAL_INPUTS, dtype=np.int32)
    
    # Verify initial network
    bits = _precompute_bits(inputs)
    if not _verify_network(network, inputs, bits):
        # Should not happen, but return initial if it does
        return network
    
    # Remove redundant comparators
    try:
        network = _remove_redundant(network, inputs, deadline)
    except Exception:
        # If removal process fails, fall back to initial
        pass

    # Final verification
    if _verify_network(network, inputs, bits):
        return network
    else:
        # If final network is invalid (shouldn't happen if _remove_redundant is correct),
        # return the initial valid network
        return _batcher(N)
