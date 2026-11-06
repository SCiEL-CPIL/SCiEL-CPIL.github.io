def count_distinct_fast(ca, cb, cc, cd):
    # Inline logic
    distinct = 1
    if cb != ca: distinct += 1
    if cc != ca and cc != cb: distinct += 1
    if cd != ca and cd != cb and cd != cc: distinct += 1
    return distinct
