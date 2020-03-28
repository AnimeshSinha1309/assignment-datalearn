import client

SECRET_KEY = 'EdQPhzkQ1CnpQ9jxCY4AH8eATTHeZm4IwEs2P1jE2xT3p8sCeE'

ANSWER = [
    0.00000000e+00,
    1.13428508e-01,
    -6.53181838e+00,
    5.07062000e-02,
    3.81279552e-02,
    8.14398119e-05,
    -6.00833892e-05,
    -1.25015245e-07,
    3.49123196e-08,
    4.06034763e-11,
    -6.73078250e-12
]

if __name__ == "__main__":
    submit_status = client.submit(SECRET_KEY, ANSWER)
    assert "submitted" in submit_status
