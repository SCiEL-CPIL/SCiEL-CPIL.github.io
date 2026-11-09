<<<<<<< SEARCH
    # 13. Backward path with stronger bottleneck emphasis
    cand_13 = bwd * (1.0 + 1.0 * ratio_norm)
    candidates.append(cand_13)
    
    return candidates
=======
    # 13. Backward path with stronger bottleneck emphasis
    cand_13 = bwd * (1.0 + 1.0 * ratio_norm)
    candidates.append(cand_13)
    
    # 14. Critical Path Density: time / slack where slack = cycle_time - bwd
    #     Tasks with high time relative to their downstream path get high priority
    slack = cycle_time - bwd
    # Avoid division by zero or negative slack
    slack_safe = np.maximum(slack, 1e-6)
    cpd = times / slack_safe
    # Normalize for stability
    cpd_norm = _normalize(cpd)
    candidates.append(cpd)
    
    # 15. CPD combined with successor count
    cand_15 = cpd * (1.0 + 0.5 * succ_count.astype(float))
    candidates.append(cand_15)
    
    # 16. CPD combined with backward path
    cand_16 = cpd_norm * bwd
    candidates.append(cand_16)
    
    # 17. Slack-based: prefer tasks with small slack (tight downstream)
    #     Inverse slack: 1/slack, higher means tighter
    inv_slack = 1.0 / slack_safe
    inv_slack_norm = _normalize(inv_slack)
    cand_17 = times * (1.0 + 2.0 * inv_slack_norm)
    candidates.append(cand_17)
    
    # 18. Composite: blend of CPD, backward, and time
    cand_18 = 0.4 * cpd_norm + 0.35 * p_rank + 0.25 * p_time
    candidates.append(cand_18)
    
    # 19. CPD * critical path
    cand_19 = cpd_norm * crit
    candidates.append(cand_19)
    
    # 20. Bottom level (longest path to end) weighted by time ratio
    #     This is similar to bwd but with explicit time weighting
    cand_20 = bwd * (1.0 + 0.5 * p_time)
    candidates.append(cand_20)
    
    return candidates
>>>>>>> REPLACE
