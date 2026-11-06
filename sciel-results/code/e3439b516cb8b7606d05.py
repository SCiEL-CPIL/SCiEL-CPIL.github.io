import sys
import numpy as np
import time
import random

N = 18

def construct():
    """
    Constructs a sorting network for N=18 wires using a local search algorithm.
    """
    
    def odd_even_transposition_network(n):
        """
        Generates the Odd-Even Transposition sorting network.
        """
        comps = []
        for i in range(n):
            for j in range(i % 2, n - 1, 2):
                comps.append((j, j + 1))
        return comps

    # Pre-compute all boolean inputs
    num_inputs = 1 << N
    base_inputs = (np.arange(num_inputs)[:, None] >> np.arange(N)[::-1]) & 1
    
    def is_sorting_network(comparators):
        """
        Checks if the given list of comparators sorts all 2^N boolean inputs.
        """
        state = base_inputs.copy()
        
        for i, j in comparators:
            # Perform compare-exchange
            # min to i, max to j
            temp = state[:, i].copy()
            state[:, i] = np.minimum(temp, state[:, j])
            state[:, j] = np.maximum(temp, state[:, j])
            
        # Check if sorted: non-decreasing
        # state[:, k] <= state[:, k+1]
        if np.all(state[:, :-1] <= state[:, 1:]):
            return True
        else:
            return False

    # Initialize with Odd-Even Transposition
    current_network = odd_even_transposition_network(N)
    
    # Verify initial network
    # if not is_sorting_network(current_network):
    #     raise ValueError("Initial network is not sorting")
    
    start_time = time.time()
    timeout = 55  # Leave 5s buffer
    
    # Local Search: Try to remove comparators
    while time.time() - start_time < timeout:
        # Try a batch of random removals
        # If we find one, we accept it and continue
        # If we don't, we just keep trying (stochastic)
        
        # To avoid getting stuck in a very tight local minimum, 
        # we could also try swapping, but removal is the primary driver.
        
        # Randomly pick indices to try removing
        # We try up to 200 removals per "iteration"
        # If any succeed, we update current_network
        
        for _ in range(200):
            if len(current_network) == 0:
                break
            
            idx = random.randint(0, len(current_network) - 1)
            
            # Create candidate
            # Instead of slicing (which copies), we can use list comprehension or just slice
            # Slicing is O(N) which is small (100-200)
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if is_sorting_network(candidate):
                current_network = candidate
                # Reset the counter or just continue?
                # Just continue to try to remove more.
                # We don't break because we might remove more.
                # But we should probably restart the inner loop logic?
                # No, just continue the for loop.
                
    return current_network

# EVOLVE-BLOCK-END
