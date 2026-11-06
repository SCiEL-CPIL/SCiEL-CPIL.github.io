<<<<<<< SEARCH
def _verify_network(network):
    """Verify network sorts all 2^N Boolean inputs using zero-one principle."""
    n = len(network[0][0]) if network else 0
    # We know N=17, so 2^17 = 131072 inputs
    for mask in range(1 << N):
        bits = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, bits)
        # Check if sorted: all 1s come after all 0s
        seen_one = False
        for b in result:
            if b == 1:
                seen_one = True
            elif seen_one:
                return False
    return True
=======
def _verify_network(network):
    """Verify network sorts all 2^N Boolean inputs using zero-one principle."""
    # We know N=17, so 2^17 = 131072 inputs
    for mask in range(1 << N):
        bits = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, bits)
        # Check if sorted: all 1s come after all 0s
        seen_one = False
        for b in result:
            if b == 1:
                seen_one = True
            elif seen_one:
                return False
    return True
>>>>>>> REPLACE

