<<<<<<< SEARCH
def _try_remove_comparator(network, index, n=N, timeout=0.5):
    """Try to remove comparator at index and verify if network still works."""
    if len(network) <= 1:
        return False, network
    
    new_network = network[:index] + network[index+1:]
    
    if not _is_valid_network(new_network, n):
        return False, network
    
    if _verify_network(new_network, n, timeout):
        return True, new_network
    
    return False, network
=======
def _try_remove_comparator(network, index, n=N):
    """Try to remove comparator at index and verify if network still works."""
    if len(network) <= 1:
        return False, network
    
    new_network = network[:index] + network[index+1:]
    
    if not _is_valid_network(new_network, n):
        return False, network
    
    if _verify_network(new_network, n):
        return True, new_network
    
    return False, network
>>>>>>> REPLACE

