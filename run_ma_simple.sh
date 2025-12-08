#!/bin/bash

# Simple Multi-Agent Runner Script
# Usage: ./run_ma_simple.sh

export DATASET=visualwebarena
export HF_ENDPOINT=https://hf-mirror.com
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export HOMEPAGE="<your_homepage_domain>:4399"

export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"

# Environment variables (modify as needed)
export OPENAI_API_KEY=sk-ba12564bebdb4f129f91944b55147971
export OPENAI_BASE_URL=https://api.deepseek.com/v1

echo "🚀 Starting Multi-Agent Web Arena"
echo "📋 Task: Search for Yao Ming's age and Shaquille O'Neal's age, then calculate the sum"
echo "🌐 URL: https://www.baidu.com"
echo "🧠 Model: deepseek-chat"

# Run the multi-agent script
python run_multi_agent.py \
  --start_url "https://www.baidu.com" \
  --intent "Search yaoming's age and shaquille o'neal's age, tell me the sum of their ages" \
  --max_steps 3 \
  --config_file multi_agent_config_example.json

