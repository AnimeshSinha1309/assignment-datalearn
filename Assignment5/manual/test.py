roll = 2018113001

state = [1/3, 1/3, 0, 0, 1/3]
color = ['R', 'R', 'G', 'G', 'R']

moves = ['R', 'L', 'L']
obs = ['R', 'G', 'G']

pc_red = [ 0.9, 0.8, 0.85 ]
pc_green = [ 0.85, 0.95, 0.9 ]
x, y = 1 - ((roll % 1000) % 40 + 1) / 100, (roll % 100) % 3

print(roll)
print(x, y)

for step in range(3):
	move_state = [0.0 for i in range(5)]
	for i in range(5):
		move_state[max(i - 1, 0)] += state[i] * (x if moves[step] == 'L' else 1-x)
		move_state[min(i + 1, 4)] += state[i] * (x if moves[step] == 'R' else 1-x)
	
	color_prob = [ (pc_red[y] if color[i] == 'R' else pc_green[y]) for i in range(5) ]
	measure_prob = [ 
		1 - color_prob[i] if obs[step] != color[i] else color_prob[i] 
		for i in range(5) 
	]
	total_prob = [ move_state[i] * measure_prob[i] for i in range(5) ]
	# print(total_prob, move_state, measure_prob)	
	state = [ 
		measure_prob[i] *  move_state[i] / sum(total_prob)
		for i in range(5) 
	]
	for item in state: 
		print(item, end = ' ')
	print()
	
