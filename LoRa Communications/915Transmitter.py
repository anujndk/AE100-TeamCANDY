# RFM95W LoRa Transmitter - Raspberry Pi 4B
# Compatible with Arduino LoRa receiver
# Sends "WPT; lat,lon,alt;" format packets

# SPDX-License-Identifier: MIT

import board
import busio
import digitalio
import adafruit_rfm9x
import time

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
rfm9x.tx_power = 23

print("=" * 70)
print("RFM95W LoRa Transmitter - Raspberry Pi 4B")
print("Compatible with Arduino LoRa")
print("=" * 70)
print(f"Frequency: {RADIO_FREQ_MHZ} MHz")
print(f"TX Power: {rfm9x.tx_power} dB")
print("\nSending waypoint packets (WPT; lat,lon,alt;)...")
print("Press Ctrl+C to stop.\n")

# Counter for sent packets
packet_count = 0

def send_waypoint(latitude, longitude, altitude):
    """
    Send a waypoint packet in format: WPT; lat,lon,alt;
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        altitude (float): Altitude in meters
    
    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        # Format: WPT; lat,lon,alt;
        message = f"WPT:{latitude:.6f},{longitude:.6f},{altitude:.1f};"
        
        # Convert to bytes
        packet_data = bytes(message, "utf-8")
        
        # Send the packet
        rfm9x.send(packet_data)
        return True
    
    except Exception as e:
        print(f"  ✗ Error sending packet: {e}")
        return False

# Example coordinates (latitude, longitude, altitude in meters)
example_waypoints = [
    (40.712776, -74.005974, 10.5),     # NYC latitude, longitude, 10.5m altitude
    (34.052234, -118.243685, 25.0),    # LA latitude, longitude, 25.0m altitude
    (37.774929, -122.419418, 50.0),    # San Francisco, 50.0m altitude
]

waypoint_index = 0

try:
    while True:
        # Get next waypoint (cycle through example waypoints)
        latitude, longitude, altitude = example_waypoints[waypoint_index % len(example_waypoints)]
        waypoint_index += 1
        
        # Create and display the message
        message = f"WPT; {latitude:.6f},{longitude:.6f},{altitude:.1f};"
        print(f"Sending packet #{packet_count + 1}: {message}")
        
        # Send the waypoint
        if send_waypoint(latitude, longitude, altitude):
            packet_count += 1
            print(f"  ✓ Packet sent successfully!")
        else:
            print(f"  ✗ Failed to send packet")
        
        print(f"Waiting 5 seconds before next transmission...\n")
        time.sleep(5)

except KeyboardInterrupt:
    print("\n" + "=" * 70)
    print(f"Transmitter stopped. Total packets sent: {packet_count}")
    print("=" * 70)