#! /bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to increment version
increment_version() {
  local version=$1
  local delimiterchar=$(echo "$version" | sed 's/[0-9]//g' | cut -c1)
  local parts=($(echo "${version//${delimiterchar}/ }"))
  local last=$((${#parts[@]}-1))
  local parts[last]=$((parts[last]+1))
  echo "${parts[*]}" | sed "s/ /${delimiterchar}/g"
}

# Read the current version
VERSION=$(cat version.txt)

# Increment the version
NEW_VERSION=$(increment_version $VERSION)

# Update version.txt
echo $NEW_VERSION > version.txt

# Commit changes
git add .
git commit -m "Build version $NEW_VERSION"

# Push changes to GitHub
git push origin main


docker build -t llama-fine-tuning-gpu .

echo "Build completed. Starting Docker Version: $NEW_VERSION"

docker run --gpus all --env-file .env \
	  -v ~/.aws:/root/.aws:ro \
	    -v $(pwd):/app \
	      -v ~/LLAMA_W:/LLAMA_W:ro \
	        -it llama-fine-tuning-gpu




