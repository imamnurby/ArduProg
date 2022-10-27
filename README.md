# ArduProg
A Replication Package for ArduProg

# Environment Setting
Run the following commands:
1. pip install -r requirements.txt
2. docker build dockerfile --tag <image-name>:<image-tag>
3. docker run -it <image-name>:<image-tag> /bin/bash
4. git clone https://github.com/imamnurby/ArduProg.git

# Running Training
1. cd `src/<component-you-want-to-train>`
2. python train.py

Optionally, you can change the model and training setting by modifying the dictionary values inside `config.py`

