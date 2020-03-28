import client

SECRET_KEY = 'EdQPhzkQ1CnpQ9jxCY4AH8eATTHeZm4IwEs2P1jE2xT3p8sCeE'

def answer(genes):
    train_error, validation_error = client.get_errors(SECRET_KEY, genes)
    print(genes, train_error, validation_error)
    with open('manual_1.txt', 'a') as f:
        f.write(str(genes) + "," + str(train_error) + ", " + str(validation_error) + "\n")

if __name__ == "__main__":
    GENES = [
        0.0, # Relatively useless
        0.0, # Relatively useless
        -5.84706385e+00,
        4.94769298e-02,
        3.80769755e-02,
        7.95740721e-05,
        -6.01705216e-05,
        -1.26196023e-07,
        3.49151483e-08, # 0.01 is the test-train tradeoff point
        4.04442080e-11,
        -6.75167133e-12, # 0.01 is the test-train tradeoff point
    ]
    OVERFIT = [
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
    ]
    answer(genes=GENES)
