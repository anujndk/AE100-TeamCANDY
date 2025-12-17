from pymavlink import mavutil
from picamera2 import Picamera2
import cv2
import time
import numpy as np
import threading
import os
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================


PIXHAWK_PORT = '/dev/ttyAMA0'
PIXHAWK_BAUD = 921600
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_DELAY = 0.1  # seconds between frames

# Image saving configuration
OUTPUT_FOLDER = 'qr_detections'  # Folder to save detected QR code images


# ============================================================================
# GLOBAL STATE
# ============================================================================


master = None
current_location = {'lat': None, 'lon': None, 'alt': None}
location_lock = threading.Lock()
dataPointCount = 0


# ============================================================================
# IMAGE SAVING UTILITIES
# ============================================================================


def initialize_output_folder():
    """Create output folder if it doesn't exist."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"✓ Created output folder: {OUTPUT_FOLDER}")
    else:
        print(f"✓ Using existing output folder: {OUTPUT_FOLDER}")


def save_qr_image(frame, qr_data, detection_number, gps_coords):
    """
    Save QR detection image with metadata.
    
    Args:
        frame: the image frame to save
        qr_data: the QR code data (content) - can be None if not decoded
        detection_number: sequential number of detection
        gps_coords: tuple of (lat, lon, alt)
    
    Returns:
        filename of saved image
    """
    # Create timestamp for filename (avoiding colons which are invalid in filenames)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]  # Include milliseconds
    
    # Create sanitized QR data for filename
    if qr_data:
        qr_data_safe = qr_data.replace('/', '_').replace('\\', '_').replace(':', '_')[:30]
    else:
        qr_data_safe = "unscanned"
    
    # Create filename
    filename = f"QR_{detection_number:03d}_{timestamp}_{qr_data_safe}.jpg"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    # Save image
    success = cv2.imwrite(filepath, frame)
    
    if success:
        lat, lon, alt = gps_coords
        if qr_data:
            print(f"✓ Image saved: {filename}")
            print(f"  QR Data: {qr_data}")
        else:
            print(f"✓ Image saved (unscanned QR): {filename}")
        print(f"  GPS: ({lat:.8f}, {lon:.8f}, {alt:.2f}m)")
    else:
        print(f"✗ Failed to save image: {filename}")
    
    return filepath if success else None


def create_metadata_file(detection_number, qr_data, gps_coords):
    """
    Create a text file with metadata for each QR detection.
    
    Args:
        detection_number: sequential number of detection
        qr_data: the QR code data (can be None)
        gps_coords: tuple of (lat, lon, alt)
    """
    lat, lon, alt = gps_coords
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    metadata_filename = f"QR_{detection_number:03d}_metadata.txt"
    metadata_filepath = os.path.join(OUTPUT_FOLDER, metadata_filename)
    
    qr_status = qr_data if qr_data else "(not decoded)"
    
    metadata_content = f"""QR Code Detection #{detection_number}
===========================================
Timestamp: {timestamp}
QR Data: {qr_status}

GPS Coordinates:
  Latitude:  {lat:.8f}°
  Longitude: {lon:.8f}°
  Altitude:  {alt:.2f} m

===========================================
"""
    
    try:
        with open(metadata_filepath, 'w') as f:
            f.write(metadata_content)
        print(f"✓ Metadata saved: {metadata_filename}")
    except Exception as e:
        print(f"✗ Failed to save metadata: {e}")


# ============================================================================
# GPS LISTENER THREAD
# ============================================================================


def gps_listener():
    """
    Background thread that continuously receives MAVLink messages
    and updates GPS location whenever GLOBAL_POSITION_INT is received.
    """
    global master, current_location
    
    while True:
        try:
            msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
            if msg:
                with location_lock:
                    current_location['lat'] = msg.lat / 1e7  # Convert from int to degrees
                    current_location['lon'] = msg.lon / 1e7
                    current_location['alt'] = msg.alt / 1000.0  # Convert from mm to meters
            
            time.sleep(0.1)
        except Exception as e:
            print(f"GPS listener error: {e}")
            time.sleep(1)


# ============================================================================
# CAMERA AND QR CODE DETECTION
# ============================================================================


def main():
    global master, current_location, dataPointCount
    
    # Initialize output folder
    initialize_output_folder()
    
    # Connect to Pixhawk via MAVLink
    print("Connecting to Pixhawk on", PIXHAWK_PORT)
    try:
        master = mavutil.mavlink_connection(
            PIXHAWK_PORT,
            baud=PIXHAWK_BAUD,
            timeout=5
        )
        master.wait_heartbeat()
    except Exception as e:
        print(f"Failed to connect to Pixhawk: {e}")
        return
    
    print("Connected to Pixhawk!")
    print(f"Received heartbeat from autopilot")
    
    # Start GPS listener in background thread
    gps_thread = threading.Thread(target=gps_listener, daemon=True)
    gps_thread.start()
    print("GPS listener thread started")
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    
    # Initialize QR code detector
    qr_detector = cv2.QRCodeDetector()
    
    print("QR Code Scanner Ready. Press Ctrl+C to quit.")
    print("-" * 80)
    
    last_qr_bbox = None  # Track previous QR detection to avoid duplicate saves
    
    try:
        while True:
            # Capture frame from camera
            frame = picam2.capture_array()
            
            # Convert from RGB to BGR (OpenCV format)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect QR code (returns bbox even if not fully decoded)
            data, bbox, _ = qr_detector.detectAndDecode(frame_bgr)

            with location_lock:
                lat = current_location['lat']
                lon = current_location['lon']
                alt = current_location['alt']
        
            print(f" GPS Latitude: {lat:.8f}")
            print(f" GPS Longitude: {lon:.8f}")
            print(f" GPS Altitude: {alt:.2f} m")
            print(f"{'='*80}\n")

            # If QR code is detected in frame (bbox is not None)
            if bbox is not None:
                # Convert bbox to integer coordinates
                bbox_int = bbox.astype(int)
                
                # Draw a box around the QR code using polylines
                cv2.polylines(frame_bgr, [bbox_int], True, (0, 255, 0), 3)
                
                # Check if this is a new QR code detection (not the same one as last frame)
                # Convert bbox to tuple for comparison
                bbox_tuple = tuple(map(tuple, bbox_int[0]))
                
                if last_qr_bbox is None or bbox_tuple != last_qr_bbox:
                    # This is a new QR code detection
                    dataPointCount += 1
                    last_qr_bbox = bbox_tuple
                    
                    # Get current GPS location safely
                    with location_lock:
                        lat = current_location['lat']
                        lon = current_location['lon']
                        alt = current_location['alt']
                    
                    # Format output
                    if lat is not None and lon is not None:
                        if data:
                            # QR code was successfully decoded
                            print(f"\n{'='*80}")
                            print(f"QR Code #{dataPointCount} Detected and Decoded!")
                            print(f"  QR Data: {data}")
                            print(f"  GPS Latitude:  {lat:.8f}°")
                            print(f"  GPS Longitude: {lon:.8f}°")
                            print(f"  GPS Altitude:  {alt:.2f} m")
                            print(f"{'='*80}\n")
                        else:
                            # QR code detected but not decoded
                            print(f"\n{'='*80}")
                            print(f"QR Code #{dataPointCount} Detected (not fully scanned)")
                            print(f"  GPS Latitude:  {lat:.8f}°")
                            print(f"  GPS Longitude: {lon:.8f}°")
                            print(f"  GPS Altitude:  {alt:.2f} m")
                            print(f"{'='*80}\n")
                        
                        # Save image and metadata
                        save_qr_image(frame_bgr, data, dataPointCount, (lat, lon, alt))
                        create_metadata_file(dataPointCount, data, (lat, lon, alt))
                        
                        # Put text above the QR code
                        if data:
                            text_line1 = f"QR: {data}"
                        else:
                            text_line1 = f"QR: (not scanned)"
                        
                        text_line2 = f"Lat: {lat:.6f}"
                        text_line3 = f"Lon: {lon:.6f}"
                        text_line4 = f"Alt: {alt:.1f}m"
                        
                        y_offset = bbox_int[0][0][1] - 50
                        cv2.putText(frame_bgr, text_line1, (bbox_int[0][0][0], y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, text_line2, (bbox_int[0][0][0], y_offset + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, text_line3, (bbox_int[0][0][0], y_offset + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, text_line4, (bbox_int[0][0][0], y_offset + 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print(f"QR Code #{dataPointCount} detected")
                        if data:
                            print(f"  QR Data: {data}")
                        print("  WARNING: GPS location not yet available")
                        
                        if data:
                            text_line1 = f"QR: {data}"
                        else:
                            text_line1 = f"QR: (not scanned)"
                        text_line2 = "Waiting for GPS..."
                        
                        y_offset = bbox_int[0][0][1] - 30
                        cv2.putText(frame_bgr, text_line1, (bbox_int[0][0][0], y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, text_line2, (bbox_int[0][0][0], y_offset + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
                        
                        # Still save the image
                        save_qr_image(frame_bgr, data, dataPointCount, (lat, lon, alt))
                        create_metadata_file(dataPointCount, data, (lat, lon, alt))
                else:
                    # Same QR code as last frame, just draw it without saving again
                    text_line1 = "Tracking QR..."
                    y_offset = bbox_int[0][0][1] - 30
                    cv2.putText(frame_bgr, text_line1, (bbox_int[0][0][0], y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            else:
                # No QR code in frame
                last_qr_bbox = None
            
            # Display frame (commented out for headless operation)
            # cv2.imshow("QR Code Scanner", frame_bgr)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(FRAME_DELAY)


    except KeyboardInterrupt:
        print("\nInterrupted by user")


    
    finally:
        print("\nCleaning up...")
        picam2.stop()
        # cv2.destroyAllWindows()
        master.close()
        print("Done!")



if __name__ == "__main__":
    main()
