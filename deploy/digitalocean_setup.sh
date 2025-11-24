#!/bin/bash
# Setup script for DigitalOcean GPU Droplet

set -e

echo "🚀 Setting up Multimodal AI Agent on DigitalOcean GPU Droplet"
echo "============================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install NVIDIA drivers and CUDA (if not already installed)
echo "🎮 Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-525 nvidia-dkms-525
    echo "⚠️  System reboot required. Run this script again after reboot."
    exit 0
fi

# Verify GPU
echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install NVIDIA Container Toolkit
echo "🔧 Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify Docker GPU access
echo "✓ Testing Docker GPU access..."
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Install Docker Compose
echo "📦 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone or update repository (if applicable)
# git clone <your-repo-url> multimodal-agent
# cd multimodal-agent

# Create .env file from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your Telegram bot token and other settings"
fi

# Create logs directory
mkdir -p logs

# Download model (optional - can be done at runtime)
echo "🤖 Model will be downloaded on first run"

echo ""
echo "============================================================="
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration:"
echo "   nano .env"
echo ""
echo "2. Configure MCP servers in config/mcp_servers.json"
echo ""
echo "3. Build and start the service:"
echo "   docker-compose up -d --build"
echo ""
echo "4. View logs:"
echo "   docker-compose logs -f"
echo ""
echo "5. Check status:"
echo "   docker-compose ps"
echo "============================================================="
