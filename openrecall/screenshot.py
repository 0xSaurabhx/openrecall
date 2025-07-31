import os
import time
from typing import List, Tuple
import sys
import tempfile
import threading
import queue
import subprocess

import numpy as np
from PIL import Image

# Try multiple screenshot backends
screenshot_backend = None
try:
    import mss
    screenshot_backend = 'mss'
except ImportError:
    mss = None

try:
    import pyscreenshot as ImageGrab
    if screenshot_backend is None:
        screenshot_backend = 'pyscreenshot'
except ImportError:
    ImageGrab = None

try:
    from PIL import ImageGrab as PILImageGrab
    if screenshot_backend is None:
        screenshot_backend = 'pil'
except ImportError:
    PILImageGrab = None

from openrecall.config import screenshots_path, args
from openrecall.database import insert_entry
from openrecall.nlp import get_embedding
from openrecall.ocr import extract_text_from_image
from openrecall.utils import (
    get_active_app_name,
    get_active_window_title,
    is_user_active,
)


def mean_structured_similarity_index(
    img1: np.ndarray, img2: np.ndarray, L: int = 255
) -> float:
    """Calculates the Mean Structural Similarity Index (MSSIM) between two images.

    Args:
        img1: The first image as a NumPy array (RGB).
        img2: The second image as a NumPy array (RGB).
        L: The dynamic range of the pixel values (default is 255).

    Returns:
        The MSSIM value between the two images (float between -1 and 1).
    """
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        return 0.0  # Return low similarity if shapes don't match
    
    # Convert to grayscale
    gray1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
    gray2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])

    # Parameters for MSSIM calculation
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2

    # Calculate means
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)

    # Calculate variances and covariance
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))

    # Calculate MSSIM
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    # Avoid division by zero
    if abs(denominator) < 1e-10:
        return 1.0 if abs(numerator) < 1e-10 else 0.0

    return numerator / denominator


def take_screenshots_mss() -> List[np.ndarray]:
    """Take screenshots using MSS library (X11 compatible)"""
    screenshots: List[np.ndarray] = []
    
    try:
        print("🔄 MSS: Creating MSS instance...")
        with mss.mss() as sct:
            print("🔄 MSS: Getting monitors...")
            monitor_indices = range(1, len(sct.monitors))
            print(f"🔄 MSS: Found {len(sct.monitors)-1} monitors")
            
            if args.primary_monitor_only:
                monitor_indices = [1]
                print("🔄 MSS: Using primary monitor only")
            
            for i in monitor_indices:
                if i < len(sct.monitors):
                    monitor_info = sct.monitors[i]
                    print(f"🔄 MSS: Capturing monitor {i} - {monitor_info}")
                    try:
                        sct_img = sct.grab(monitor_info)
                        print(f"🔄 MSS: Converting to numpy array for monitor {i}")
                        screenshot = np.array(sct_img)[:, :, [2, 1, 0]]
                        screenshots.append(screenshot)
                        print(f"✅ MSS: Monitor {i} captured: {screenshot.shape}")
                    except Exception as e:
                        print(f"Warning: Failed to capture monitor {i} with MSS: {e}")
                        continue
                else:
                    print(f"Warning: Monitor index {i} out of bounds. Skipping.")
    except Exception as e:
        print(f"MSS screenshot failed: {e}")
        raise
    
    return screenshots


def take_screenshots_pyscreenshot() -> List[np.ndarray]:
    """Take screenshots using command-line tools (Wayland compatible)"""
    import subprocess
    import tempfile
    
    screenshots: List[np.ndarray] = []
    
    # Try different command-line screenshot tools
    tools_to_try = [
        {
            'name': 'gnome-screenshot',
            'cmd': ['gnome-screenshot', '-d', '1', '-f', '{filename}'],
            'timeout': 20
        },
        {
            'name': 'imagemagick',
            'cmd': ['import', '-window', 'root', '{filename}'],
            'timeout': 10
        }
    ]
    
    for tool in tools_to_try:
        try:
            print(f"🔄 Trying command-line tool: {tool['name']}")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_filename = tmp_file.name
            
            # Prepare the command
            cmd = [arg.format(filename=tmp_filename) for arg in tool['cmd']]
            print(f"🔄 Running command: {' '.join(cmd)}")
            
            # Run the command with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=tool['timeout']
            )
            
            if result.returncode == 0:
                # Load the screenshot
                print(f"🔄 Loading screenshot from {tmp_filename}")
                img = Image.open(tmp_filename)
                screenshot = np.array(img)
                if len(screenshot.shape) == 3 and screenshot.shape[2] == 4:  # RGBA
                    screenshot = screenshot[:, :, :3]  # Convert to RGB
                screenshots.append(screenshot)
                print(f"✅ {tool['name']}: Screenshot captured: {screenshot.shape}")
                
                # Clean up
                os.unlink(tmp_filename)
                return screenshots
            else:
                print(f"❌ {tool['name']} failed with code {result.returncode}")
                print(f"   stderr: {result.stderr}")
                # Clean up
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                continue
                
        except subprocess.TimeoutExpired:
            print(f"❌ {tool['name']}: Command timed out after {tool['timeout']} seconds")
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
            continue
        except Exception as e:
            print(f"❌ {tool['name']}: Exception: {e}")
            if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
            continue
    
    # If we get here, all tools failed
    raise RuntimeError("All screenshot tools failed")


def take_screenshots_pil() -> List[np.ndarray]:
    """Take screenshots using PIL ImageGrab"""
    import threading
    import queue
    
    screenshots: List[np.ndarray] = []
    
    def capture_screenshot(result_queue):
        try:
            print("🔄 PIL: Taking screenshot...")
            img = PILImageGrab.grab()
            result_queue.put(('success', img))
        except Exception as e:
            result_queue.put(('error', e))
    
    try:
        # Use threading with timeout
        result_queue = queue.Queue()
        thread = threading.Thread(target=capture_screenshot, args=(result_queue,))
        thread.daemon = True
        thread.start()
        
        # Wait for result with timeout
        try:
            result_type, result = result_queue.get(timeout=10)
            if result_type == 'error':
                raise result
            img = result
        except queue.Empty:
            print("❌ PIL: Screenshot capture timed out after 10 seconds")
            raise TimeoutError("Screenshot capture timed out")
        
        print("🔄 PIL: Converting to numpy array...")
        screenshot = np.array(img)
        if len(screenshot.shape) == 3 and screenshot.shape[2] == 4:  # RGBA
            screenshot = screenshot[:, :, :3]  # Convert to RGB
            print("🔄 PIL: Converted RGBA to RGB")
        screenshots.append(screenshot)
        print(f"✅ PIL: Screenshot captured: {screenshot.shape}")
        
    except Exception as e:
        print(f"❌ PIL ImageGrab failed: {e}")
        raise
    
    return screenshots


def take_screenshots() -> List[np.ndarray]:
    """Take screenshots with fallback to different backends"""
    print("🔍 Detecting display server...")
    
    # Try different backends in order of preference
    backends_to_try = []
    
    # Detect display server and prioritize backends
    display_server = os.environ.get('XDG_SESSION_TYPE', '').lower()
    wayland_display = os.environ.get('WAYLAND_DISPLAY', '')
    x11_display = os.environ.get('DISPLAY', '')
    
    print(f"Display server: {display_server}, Wayland: {wayland_display}, X11: {x11_display}")
    
    if wayland_display and display_server == 'wayland':
        print("🖥️ Using Wayland backends...")
        # Wayland: prefer pyscreenshot or PIL
        if ImageGrab:
            backends_to_try.append(('pyscreenshot', take_screenshots_pyscreenshot))
            print("  - pyscreenshot available")
        if PILImageGrab:
            backends_to_try.append(('pil', take_screenshots_pil))
            print("  - PIL ImageGrab available")
        if mss:
            backends_to_try.append(('mss', take_screenshots_mss))
            print("  - mss available")
    else:
        print("🖥️ Using X11 backends...")
        # X11: prefer MSS, then fallback to others
        if mss:
            backends_to_try.append(('mss', take_screenshots_mss))
            print("  - mss available")
        if ImageGrab:
            backends_to_try.append(('pyscreenshot', take_screenshots_pyscreenshot))
            print("  - pyscreenshot available")
        if PILImageGrab:
            backends_to_try.append(('pil', take_screenshots_pil))
            print("  - PIL ImageGrab available")
    
    last_exception = None
    
    for backend_name, backend_func in backends_to_try:
        try:
            print(f"🔄 Trying {backend_name}...")
            screenshots = backend_func()
            if screenshots:
                print(f"✅ Success with {backend_name}: {len(screenshots)} screenshot(s)")
                return screenshots
        except Exception as e:
            print(f"❌ {backend_name} failed: {e}")
            last_exception = e
            continue
    
    # If all backends failed
    if last_exception:
        print("\n=== Screenshot Troubleshooting ===")
        print("All screenshot backends failed. This might be due to:")
        print("1. Running in a Wayland session without proper X11 compatibility")
        print("2. Missing display server environment variables")
        print("3. Permission issues")
        print("\nEnvironment info:")
        print(f"  XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', 'not set')}")
        print(f"  WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', 'not set')}")
        print(f"  DISPLAY: {os.environ.get('DISPLAY', 'not set')}")
        print("\nTry running with: XDG_SESSION_TYPE=x11 python your_script.py")
        raise last_exception
    else:
        raise RuntimeError("No screenshot backends available")


def save_numpy_array_as_image(screenshot: np.ndarray, filepath: str) -> bool:
    """Save numpy array as image file"""
    try:
        # Ensure the array is in the correct format
        if len(screenshot.shape) != 3 or screenshot.shape[2] != 3:
            print(f"Warning: Unexpected screenshot shape: {screenshot.shape}")
            return False
        
        # Ensure values are in the correct range
        if screenshot.dtype != np.uint8:
            screenshot = np.clip(screenshot, 0, 255).astype(np.uint8)
        
        # Convert numpy array to PIL Image and save as WebP
        img = Image.fromarray(screenshot, 'RGB')
        
        # Change file extension to webp if it's not already
        if not filepath.lower().endswith('.webp'):
            filepath = filepath.rsplit('.', 1)[0] + '.webp'
        
        # Save as WebP with good quality but optimized for size
        img.save(filepath, 'WEBP', quality=85, optimize=True)
        return True
    except Exception as e:
        print(f"Error saving screenshot to {filepath}: {e}")
        return False


def record_screenshots_thread():
    """Record screenshots in a thread with improved error handling"""
    print("🚀 Screenshot thread starting...")
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        print("📸 Taking initial screenshots...")
        last_screenshots = take_screenshots()
        print(f"✓ Initial screenshot capture successful ({len(last_screenshots)} screenshots)")
    except Exception as e:
        print(f"❌ Failed to take initial screenshots: {e}")
        import traceback
        traceback.print_exc()
        return

    consecutive_errors = 0
    max_consecutive_errors = 5
    last_save_time = 0  # Track when we last saved a screenshot
    min_save_interval = 5  # Minimum 5 seconds between saves (reduced from 15 for testing)
    
    while True:
        try:
            # Check user activity with fallback
            try:
                # Use a simpler approach - just assume user is active in Wayland
                # since xprintidle has issues with screen saver extension
                display_server = os.environ.get('XDG_SESSION_TYPE', '').lower()
                if display_server == 'wayland':
                    # Skip user activity check for Wayland - always assume active
                    pass
                else:
                    # Try user activity check for X11
                    if not is_user_active():
                        time.sleep(3)
                        continue
            except Exception as e:
                # If user activity check fails, assume user is active and continue
                pass

            screenshots = take_screenshots()
            print(f"📸 Captured {len(screenshots)} screenshot(s)")

            for i, screenshot in enumerate(screenshots):
                if i >= len(last_screenshots):
                    last_screenshots.append(screenshot)
                    continue
                    
                last_screenshot = last_screenshots[i]

                try:
                    similarity = mean_structured_similarity_index(screenshot, last_screenshot)
                    print(f"🔍 Screenshot {i} similarity: {similarity:.3f}")
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    similarity = 0.0  # Assume different if calculation fails

                if similarity < 0.90:  # Only save if there's significant change (reduced from 0.98 for testing)
                    # Check if enough time has passed since last save
                    current_time = time.time()
                    if current_time - last_save_time < min_save_interval:
                        print(f"- Screenshot {i} - skipping (too soon, {current_time - last_save_time:.1f}s since last save)")
                        continue
                    
                    try:
                        # Save and process screenshot
                        timestamp = int(time.time() * 1000)
                        screenshot_filename = f"{timestamp}.webp"
                        screenshot_path = os.path.join(screenshots_path, screenshot_filename)
                        
                        # Ensure screenshots directory exists
                        os.makedirs(screenshots_path, exist_ok=True)
                        
                        # Save the screenshot as WebP
                        if save_numpy_array_as_image(screenshot, screenshot_path):
                            # Extract text using the file path (not numpy array)
                            try:
                                text = extract_text_from_image(screenshot_path)
                                if text and text.strip():
                                    embedding = get_embedding(text)
                                    app_name = get_active_app_name()
                                    window_title = get_active_window_title()
                                    
                                    insert_entry(
                                        text, timestamp, embedding, app_name, window_title
                                    )
                                    print(f"✓ Processed screenshot {i} - found text: {len(text)} chars")
                                    last_save_time = current_time  # Update last save time
                                else:
                                    print(f"- Screenshot {i} - no text found")
                            except Exception as e:
                                print(f"Error processing text from screenshot {i}: {e}")

                        last_screenshots[i] = screenshot
                        consecutive_errors = 0  # Reset error counter on success

                    except Exception as e:
                        print(f"Error processing screenshot {i}: {e}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"Too many consecutive errors ({consecutive_errors}), stopping")
                            return
                        continue

            # Wait before next capture - increased to significantly reduce CPU usage
            time.sleep(2)  # Reduced from 5 to 2 seconds for testing

        except KeyboardInterrupt:
            print("Screenshot recording stopped by user")
            break
        except Exception as e:
            print(f"Error in screenshot recording loop: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({consecutive_errors}), stopping")
                break
            time.sleep(5)  # Wait before retrying
            continue
    
    print("Screenshot recording thread ended")
