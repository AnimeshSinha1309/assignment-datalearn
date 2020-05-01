locale = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]

def name_state(state):
	a, t, c = state
	return 'A' + str(a[0] + 2) + str(a[1] + 2) + 'T' + str(t[0] + 1) + str(t[1] + 1) + 'C' + ('1' if c else '0')

def add_tuple(tuple_1: tuple, tuple_2: tuple) -> tuple:
	assert len(tuple_1) == len(tuple_2)
	return tuple([tuple_1[i] + tuple_2[i] for i in range(min(len(tuple_1), len(tuple_2)))])

def sub_tuple(tuple_1: tuple, tuple_2: tuple) -> tuple:
	assert len(tuple_1) == len(tuple_2)
	return tuple([tuple_1[i] - tuple_2[i] for i in range(min(len(tuple_1), len(tuple_2)))])

def manhattan(data: tuple) -> int:
	return sum([abs(el) for el in data])

def bound(data: tuple) -> int:
	return tuple([max(-1, min(1, el)) for el in data])

states = [(loc_agent, loc_target, call) for loc_agent in locale for loc_target in locale for call in (True, False)]
state_names = list(map(name_state, states))
actions = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
action_names = ['stay', 'up', 'down', 'left', 'right']
observations = ['o_here', 'o_up', 'o_down', 'o_left', 'o_right', 'o_away']

with open('gen.pomdp', 'w') as f:
	f.write('discount: 0.5\n')
	f.write('values: reward\n')
	f.write('states: ' + " ".join(state_names) + '\n')
	f.write('actions: ' + " ".join(action_names) + '\n')
	f.write('observations: ' + " ".join(observations) + '\n\n')
	# Making the T (Transition) Matrices
	for action_idx, action in enumerate(actions):
		f.write('T : ' + action_names[action_idx] + '\n')
		for state_u in states:
			for state_v in states:
				agent_u, agent_v = state_u[0], state_v[0]
				target_u, target_v = state_u[1], state_v[1]
				call_u, call_v = state_u[2], state_v[2]
				mul_target = 0.4 + 0.15 * manhattan(target_u) if target_u == target_v \
					else 0.15 if manhattan(sub_tuple(target_u, target_v)) == 1 else 0.0
				mul_agent = 1.0 if action == (0, 0) and agent_u == agent_v else \
					0.98 if bound(add_tuple(agent_u, action)) == agent_v else \
						0.02 if bound(sub_tuple(agent_u, action)) == agent_v else 0.0
				mul_call = (0.8 if call_v else 0.2) if call_u else (0.4 if call_v else 0.6)
				f.write(str(round(mul_agent * mul_call * mul_target, 6)) + ' ')
			f.write('\n')
		f.write('\n')
	f.write('\n')
	# Making the O (Observation) Matrices
	for state_idx, state in enumerate(states):
		f.write('O : * : ' + state_names[state_idx] + '\n')
		agent, target, call = state
		f.write('1.0 ' if agent == target else '0.0 ')
		f.write('1.0 ' if add_tuple(agent, (actions[1])) == target else '0.0 ')
		f.write('1.0 ' if add_tuple(agent, (actions[2])) == target else '0.0 ')
		f.write('1.0 ' if add_tuple(agent, (actions[3])) == target else '0.0 ')
		f.write('1.0 ' if add_tuple(agent, (actions[4])) == target else '0.0 ')
		f.write('1.0 ' if manhattan(sub_tuple(agent, target)) > 1 else '0.0 ')
		f.write('\n\n')
	f.write('\n')
	# Making the R (Rewards) Matrices
	for state_idx, state in enumerate(states):
		f.write('R : * : * : {0} : * {1}\n'.format(state_names[state_idx], 11 if state[0] == state[1] else -1))
	f.write('\n')
