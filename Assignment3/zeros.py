import numpy as np
import client

GENOME = [
    2.9852982378286203e-13, 
    7.802777485506336e-13,
    0.0,
    2.4115061750581433e-13,
    0.0,
    1.2412906650227206e-14,
    0.0,
    2.2380782485499628e-12,
    0.0,
    1.702220628384407e-12,
    6.321486268844409e-13
]

SECRET_KEY = 'EdQPhzkQ1CnpQ9jxCY4AH8eATTHeZm4IwEs2P1jE2xT3p8sCeE'

def get_errors(genes):
    train_error, validation_error = client.get_errors(SECRET_KEY, genes)
    print(genes, train_error, validation_error)
    with open('manual_1.txt', 'a') as f:
        f.write(str(genes) + "," + str(train_error) + ", " + str(validation_error) + "\n")
    return (train_error, validation_error)

current = [1e20, 1e20]
for iter in range(2000):
    upgrade = GENOME.copy()
    upgrade[np.random.randint(11)] = np.clip(
        upgrade[np.random.randint(11)] + (np.random.random() * 1e-11),
        -10, 10)
    errors = get_errors(upgrade)
    if errors[0] + errors[1] < current[0] + current[1]:
        GENOME = upgrade
        current = errors
        print('Accepted')
    