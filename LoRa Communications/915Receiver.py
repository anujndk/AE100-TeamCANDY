# RFM95W LoRa Receiver - Raspberry Pi 4B
# Compatible with Arduino LoRa receiver
# Receives "WPT:lat1,lon1;lat2,lon2;lat3,lon3;" format packets
# SAVES coordinates to GPS_coords.txt file

# SPDX-License-Identifier: MIT

import board
import busio
import digitalio
import adafruit_rfm9x
import time
import datetime

# Define radio parameters
RADIO_FREQ_MHZ = 915.0  # Must match Arduino frequency

# Define pins connected to the RFM95W on Raspberry Pi 4B
CS = digitalio.DigitalInOut(board.D7)      # GPIO 7 (Pin 26) - Chip Select
RESET = digitalio.DigitalInOut(board.D25)  # GPIO 25 (Pin 22) - Reset

# Initialize SPI bus
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)

# Initialize RFM95W radio
try:
    rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, RADIO_FREQ_MHZ)
    print("✓ RFM95W initialized successfully!")
except RuntimeError as e:
    print(f"✗ Failed to initialize RFM95W: {e}")
    exit(1)

# Configure the radio
rfm9x.tx_power = 20

print("=" * 70)
print("RFM95W LoRa Receiver - Raspberry Pi 4B")
print("Compatible with Arduino LoRa")
print("=" * 70)
print(f"Frequency: {RADIO_FREQ_MHZ} MHz")
print(f"TX Power: {rfm9x.tx_power} dB")
print("\nListening for waypoint packets (WPT:lat1,lon1;lat2,lon2;lat3,lon3;)...")
print("Saving coordinates to GPS_coords.txt")
print("Press Ctrl+C to stop.\n")

# Counter for received packets
packet_count = 0

# GPS log filename
GPS_LOG_FILE = "GPS_coords.txt"

def parse_waypoint_packet(packet_text):
    """
    Parse waypoint packet in format: WPT:lat1,lon1;lat2,lon2;lat3,lon3;
    Returns: list of tuples [(lat1, lon1), (lat2, lon2), (lat3, lon3)] or None if invalid
    """
    if not packet_text.startswith("WPT:"):
        print(f" ⚠ Packet is not a WPT command. Raw: {packet_text}")
        return None
    
    try:
        # Remove "WPT:" prefix (4 characters)
        coords_str = packet_text[0:].strip() 
        
        # Split by semicolon to get individual lat,lon pairs
        pairs = coords_str.split(';')
        
        # Remove empty strings from split (last semicolon creates empty string)
        pairs = [p.strip() for p in pairs if p.strip()]
        
        if len(pairs) != 3:
            print(f" ⚠ Expected 3 coordinate pairs, got {len(pairs)}")
            print(f"     Pairs: {pairs}")
            return None
        
        coordinates = []
        for i, pair in enumerate(pairs):
            parts = pair.split(',')
            if len(parts) != 2:
                print(f" ⚠ Pair {i+1} malformed (expected lat,lon): {pair}")
                return None
            
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                coordinates.append((lat, lon))
            except ValueError as e:
                print(f" ⚠ Could not convert pair {i+1} to float: {pair}")
                return None
        
        return coordinates
    
    except Exception as e:
        print(f" ⚠ Error parsing waypoint packet: {e}")
        return None

def save_gps_coordinates(coordinates):
    """
    Save GPS coordinates to file in format:
    lat1: XXX, long1: XXXX; lat2: XXXX, long2: XXXX; lat3: XXXX, long3: XXXX;
    
    Args:
        coordinates: list of tuples [(lat1, lon1), (lat2, lon2), (lat3, lon3)]
    """
    try:
        # Open file in append mode
        with open(GPS_LOG_FILE, 'a') as f:
            # Format each coordinate pair
            formatted_pairs = []
            for i, (lat, lon) in enumerate(coordinates, 1):
                formatted_pairs.append(f"lat{i}: {lat}, long{i}: {lon}")
            
            # Join with semicolon and space, add trailing semicolon
            output = "; ".join(formatted_pairs) + "; "
            f.write(output)
        
        # Display what was saved
        saved_display = "; ".join([f"lat{i}: {lat:.6f}, long{i}: {lon:.6f}" 
                                   for i, (lat, lon) in enumerate(coordinates, 1)]) + ";"
        print(f" ✓ Saved to {GPS_LOG_FILE}:")
        print(f"     {saved_display}")
        return True
    
    except IOError as e:
        print(f" ✗ Error saving to {GPS_LOG_FILE}: {e}")
        return False

# Main receive loop
try:
    while True:
        # Receive packets with a 0.5 second timeout
        packet = rfm9x.receive(timeout=0.5)
        
        if packet is None:
            # No packet received
            print(".", end="", flush=True)
        else:
            # Received a packet!
            packet_count += 1
            print("\n")
            print("-" * 70)
            print(f"Packet #{packet_count} received!")
            print("-" * 70)
            
            # Print raw bytes
            print(f"Raw bytes: {packet}")
            print(f"Packet size: {len(packet)} bytes")
            
            # Decode to ASCII text
            try:
                packet_text = str(packet, "ascii")
                print(f"Decoded text: {packet_text}")
                
                # Try to parse as waypoint packet
                coordinates = parse_waypoint_packet(packet_text)
                if coordinates:
                    print(f"\nParsed waypoint coordinates:")
                    for i, (lat, lon) in enumerate(coordinates, 1):
                        print(f" Pair {i}: Latitude = {lat:.6f}, Longitude = {lon:.6f}")
                    
                    # Save to GPS file
                    save_gps_coordinates(coordinates)
                
            except UnicodeDecodeError:
                print("Could not decode packet as ASCII")
                print(f"Hex: {packet.hex()}")
            
            # Get signal strength
            rssi = rfm9x.last_rssi
            print(f"\nSignal Strength (RSSI): {rssi} dB")
            
            # Get SNR if available
            try:
                snr = rfm9x.last_snr
                print(f"Signal-to-Noise Ratio (SNR): {snr} dB")
            except AttributeError:
                pass
            
            print("-" * 70)

except KeyboardInterrupt:
    print("\n" + "=" * 70)
    print(f"Receiver stopped. Total packets received: {packet_count}")
    print(f"GPS coordinates saved to {GPS_LOG_FILE}")
    print("=" * 70)
    
    # Display the contents of the GPS file
    try:
        print(f"\nContents of {GPS_LOG_FILE}:")
        print("-" * 70)
        with open(GPS_LOG_FILE, 'r') as f:
            content = f.read()
            print(content)
        print("\n" + "-" * 70)
    except FileNotFoundError:
        print(f"No {GPS_LOG_FILE} created (no packets received)")