echo $(which python)  # Ensure the correct Python is being used
pip install -r requirements.txt
pip install -v flash-attn==2.7.4.post1 --no-build-isolation  # This might takes 20 minutes to 1 hour to install, flag -v is for verbose output
pip install transformers==4.49.0 -U
pip install -e .