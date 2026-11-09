   if l_end - l_start <= 1:
       # Check fit
       prev_width = len(layers.get(l_start, []))
       curr_width = len(layers.get(l_end, []))
       if prev_width + curr_width <= budget:
           # Compute
           ops = []
           for v in layers[l_end]:
               ops.append(['compute', v])
           return ops
       else:
           # Must spill
           ops = []
           for v in layers[l_start]:
               ops.append(['store', v])
           for v in layers[l_start]:
               ops.append(['delete', v])
           for v in layers[l_start]:
               ops.append(['load', v])
           for v in layers[l_end]:
               ops.append(['compute', v])
           return ops
   