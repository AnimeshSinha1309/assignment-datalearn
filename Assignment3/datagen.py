import numpy as np

OVERFIT_WEIGHTS = np.array([
    0.0,
    0.1240317450077846,
    -6.211941063144333,
    0.04933903144709126,
    0.03810848157715883,
    8.132366097133624e-05,
    -6.018769160916912e-05,
    -1.251585565299179e-07,
    3.484096383229681e-08,
    4.1614924993407104e-11,
    -6.732420176902565e-12
])

DELTA_WEIGHTS = np.array([
    0.0,
    0.0451289378129372,
    -1.311941063144333,
    0.0,
    0.0012312312311133,
    0.0,
    0.0000112139123809,
    0.0000001839183108,
    0.0,
    1.013123123,
    0.0
])

def get_y(x_full: np.ndarray, dataset: str = 'TRAIN') -> float:
    f: float = 0.0 if dataset == 'TRAIN' else np.clip(np.random.normal(loc=1.0, scale=0.01), -8.0, 8.0)
    weights = np.multiply(np.random.normal(loc=1.0, scale=1e-4, size=OVERFIT_WEIGHTS.shape),
                         (OVERFIT_WEIGHTS + DELTA_WEIGHTS * f))
    return np.dot(weights, x_full)


def get_loss(weights, SAMPLES = 10000, dataset: str = 'TRAIN'):
    loss = 0.0
    for x in np.random.random(size=(SAMPLES)) * 1000:
        x_full = np.array([x ** i for i in range(11)])
        loss += (get_y(x_full, dataset) - np.dot(weights, x_full)) ** 2
    return loss / SAMPLES

def get_errors(key, genome):
    if len(key) > 0:
        return (get_loss(genome, dataset='TRAIN'), get_loss(genome, dataset='TEST'))
    else:
        return (99999999999999999999999999999999999999, 99999999999999999999999999999999999999)

if __name__ == "__main__":
    print('OVERFIT ERRORS:', get_errors('test', OVERFIT_WEIGHTS))
    print('BEST ERRORS:', get_errors('test', OVERFIT_WEIGHTS + DELTA_WEIGHTS))