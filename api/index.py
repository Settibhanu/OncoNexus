import sys
import os

# Add the project root to sys.path so we can import from 'backend'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import app

# Vercel needs 'app' to be the Flask instance
if __name__ == "__main__":
    app.run()
