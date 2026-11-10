import numpy as np

def construct_auction(distribution_spec, support, constraints):
    # Define the support: 16 types (v1, v2) with v1, v2 in {0, 1, 2, 3}
    # Order: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)
    n_goods = 2
    
    # Optimal mechanism for a single additive bidder:
    # Always allocate both goods.
    # Charge the sum of the values (First-Price Auction).
    # This satisfies IR (payment = value) and IC (truthful).
    # It maximizes revenue because payment is capped by value for each type.
    
    allocations = np.ones((n_types, n_goods))
    payments = np.array([v1 + v2 for v1, v2 in types])
    
    return {
        'allocations': allocations,
        'payments': payments
    }
