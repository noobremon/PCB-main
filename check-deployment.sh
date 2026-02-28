#!/bin/bash
# Pre-deployment checklist script

echo "🔍 PCB Inspection System - Pre-Deployment Checklist"
echo "=================================================="
echo ""

# Check if git repo is initialized
if [ -d .git ]; then
    echo "✅ Git repository initialized"
else
    echo "❌ Git repository not initialized"
    echo "   Run: git init && git add . && git commit -m 'Initial commit'"
fi

# Check if requirements.txt exists
if [ -f backend/requirements.txt ]; then
    echo "✅ Backend requirements.txt exists"
else
    echo "❌ Backend requirements.txt missing"
fi

# Check if package.json exists
if [ -f frontend/package.json ]; then
    echo "✅ Frontend package.json exists"
else
    echo "❌ Frontend package.json missing"
fi

# Check if render.yaml exists
if [ -f render.yaml ]; then
    echo "✅ render.yaml configuration exists"
else
    echo "❌ render.yaml missing"
fi

# Check if .env files are NOT committed
if git ls-files | grep -q "\.env$"; then
    echo "⚠️  WARNING: .env files are committed (security risk!)"
    echo "   Add .env to .gitignore and remove from git"
else
    echo "✅ .env files not committed (good!)"
fi

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✅ Python version: $PYTHON_VERSION"
else
    echo "⚠️  Python 3 not found"
fi

# Check Node version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✅ Node version: $NODE_VERSION"
else
    echo "⚠️  Node.js not found"
fi

echo ""
echo "📋 Ready to Deploy?"
echo "==================="
echo "1. Create MongoDB Atlas cluster"
echo "2. Push code to GitHub/GitLab"
echo "3. Follow QUICKSTART.md for deployment"
echo ""
echo "See DEPLOYMENT.md for detailed instructions"
