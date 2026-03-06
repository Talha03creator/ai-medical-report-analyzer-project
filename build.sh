#!/bin/bash
# Force install dependencies to overcome Vercel caching issues
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "Dependencies installed successfully."
