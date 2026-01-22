#!/usr/bin/env python3
"""
AreaSearch - Development Server Launcher

Run this script to start the AreaSearch application.
The web interface will be available at http://localhost:8000
"""

import os
import sys
import webbrowser
from threading import Timer


def open_browser():
    """Open the browser after a short delay."""
    webbrowser.open("http://localhost:8000")


def main():
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Add backend to path
    sys.path.insert(0, os.path.join(script_dir, "backend"))

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██████╗ ███████╗ █████╗ ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗    ║
    ║    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║    ║
    ║    ███████║██████╔╝█████╗  ███████║███████╗█████╗  ███████║██████╔╝██║     ███████║    ║
    ║    ██╔══██║██╔══██╗██╔══╝  ██╔══██║╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║    ║
    ║    ██║  ██║██║  ██║███████╗██║  ██║███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║    ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    ║
    ║                                                               ║
    ║           Missing Persons Search Route Generator              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝

    Starting server...
    """)

    try:
        import uvicorn
        from backend.app import app

        # Open browser after server starts
        Timer(1.5, open_browser).start()

        print("    Server running at: http://localhost:8000")
        print("    Press Ctrl+C to stop the server")
        print("")

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nPlease install dependencies with:")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
