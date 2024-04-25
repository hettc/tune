# set up all dependencies for this project

source .env # <-- set huggingface token here

# pip installs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U # upgrade
pip install -r requirements.txt

huggingface-cli login --token $HUGGINGFACE_TOKEN