"""
JARVIS Utility Functions
Helper functions for browser detection, window management, etc.
"""

import os
import time

# Windows-specific imports
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("⚠️  pywin32 not available - screenshots will capture all displays. Install with: pip install pywin32")

# Tesseract OCR
try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
    # Configure pytesseract to use the default Windows installation path if not in PATH
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not available - text finding in screenshots will use vision model only")


def get_browser_window_region():
    """Get the region (coordinates) of the browser window for focused screenshot"""
    if not WIN32_AVAILABLE:
        return None
    
    try:
        def enum_handler(hwnd, results):
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            # Look for common browser windows - prioritize windows with actual content
            if any(browser in window_text.lower() or browser in class_name.lower() 
                   for browser in ['chrome', 'firefox', 'edge', 'brave', 'opera', 'safari']):
                if win32gui.IsWindowVisible(hwnd):
                    # Get window rectangle
                    rect = win32gui.GetWindowRect(hwnd)
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    # Only include reasonably sized windows (not minimized)
                    if width > 200 and height > 200:
                        results.append((hwnd, window_text, rect))
        
        windows = []
        win32gui.EnumWindows(enum_handler, windows)
        
        if windows:
            # Get the most recently used browser window
            hwnd, window_text, rect = windows[0]
            # Bring window to foreground
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.5)  # Give window time to come to foreground
            return rect  # Returns (left, top, right, bottom)
    except Exception as e:
        print(f"   Could not find browser window: {e}")
    
    return None


def focus_browser_window():
    """Try to focus/activate the browser window before taking a screenshot (for multi-display setups)"""
    region = get_browser_window_region()
    return region is not None

