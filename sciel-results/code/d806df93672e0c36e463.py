def acquisition(mu, sigma, f_best, cost):
    kappa = 2.0
    # LCB for minimization: lower is better, so we want low mu and high sigma
    # score should be high for good candidates
    # So we want high (mu - kappa*sigma) ? No.
    # We want to minimize f(x).
    # LCB value is mu - kappa*sigma.
    # We want to pick the point with the LOWEST LCB value.
    # But the function returns scores where HIGHER is better.
    # So we should return -LCB / cost? Or (something - LCB)/cost?
    # Let's look at the baseline:
    # EI is positive when mu < f_best.
    # We want high EI.
    # LCB is low when mu is low and sigma is high.
    # So we want to MAXIMIZE (-mu + kappa*sigma) / cost.
    
    return (-mu + kappa * sigma) / cost
