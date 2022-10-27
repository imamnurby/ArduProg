# ArduProg
A Replication Package for ArduProg

# Environment Setting
Run the following commands:
1. pip install -r requirements.txt
2. docker build dockerfile --tag arduprog:1.0
3. docker run -it arduprog:1.0 /bin/bash
4. git clone https://github.com/imamnurby/ArduProg.git

# Running Training
1. cd `src/<component-you-want-to-train>`
2. python train.py

Optionally, you can change the model and training setting by modifying the dictionary values inside `config.py`

