docker run --gpus all --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  -v $(pwd):/app \
  -v /home/ubuntu/LLAMA_W:/LLAMA_W:ro \
  -it llama-fine-tuning-gpu
