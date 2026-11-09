   if n >= 13:
       # pos is 13x3
       current_n = 13
   # ...
   while current_n < n:
       # ...
       pos = np.vstack([pos, new_pos.reshape(1, 3)]) # pos is now 14x3
       current_n += 1 # current_n is 14
       # ...
       for step in range(relax_steps):
           # ...
           e, grad = dzugutov_energy_gradient(pos) # pos is 14x3
           forces = -grad # forces is 14x3
           noise = ... (14, 3)
           pos = pos + forces * dt + noise # 14+14+14 -> OK
   