#!/bin/bash
# Create systemd service for the multimodal agent

SERVICE_NAME="multimodal-agent"
WORKING_DIR="$(pwd)"
USER="$(whoami)"

echo "Creating systemd service for Multimodal AI Agent..."

# Create service file
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Multimodal AI Agent with LangGraph and Telegram
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=${USER}
WorkingDirectory=${WORKING_DIR}
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

echo "✓ Service created: ${SERVICE_NAME}"
echo ""
echo "Usage:"
echo "  Start:   sudo systemctl start ${SERVICE_NAME}"
echo "  Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "  Restart: sudo systemctl restart ${SERVICE_NAME}"
echo "  Status:  sudo systemctl status ${SERVICE_NAME}"
echo "  Enable:  sudo systemctl enable ${SERVICE_NAME}  # Start on boot"
echo "  Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
