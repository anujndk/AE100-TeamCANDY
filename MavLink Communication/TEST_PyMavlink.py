from pymavlink import mavutil
import time

print("Connecting to vehicle...")
try:
    master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600, timeout=5)
    master.wait_heartbeat()
except Exception as e:
    print(f"Failed to connect: {e}")
    exit(1)

print("Connected!")

# Get heartbeat message
heartbeat = master.messages['HEARTBEAT']
print(f"Autopilot type: {heartbeat.autopilot}")
print(f"Vehicle mode: {heartbeat.custom_mode}")

# Get system status for battery info
# SYS_STATUS is typically streamed by default, so just wait for it
print("\nWaiting for telemetry data...")
sys_status = master.recv_match(type='SYS_STATUS', blocking=True, timeout=5)

if sys_status:
    battery_voltage = sys_status.voltage_battery / 1000.0  # Convert from mV to V
    print(f"Battery voltage: {battery_voltage}V")
else:
    print("Battery voltage: (not available)")

# Wait for GPS data
gps_data = master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=5)

if gps_data:
    satellites = gps_data.satellites_visible
    latitude = gps_data.lat / 1e7  # Convert from int to degrees
    longitude = gps_data.lon / 1e7  # Convert from int to degrees
    altitude = gps_data.alt / 1000.0  # Convert from mm to meters
    
    print(f"GPS satellites: {satellites}")
    print(f"GPS Latitude:  {latitude:.8f}°")
    print(f"GPS Longitude: {longitude:.8f}°")
    print(f"GPS Altitude:  {altitude:.2f} m")
else:
    print("GPS data: (not available)")

# Get altitude from VFR_HUD (ground speed, heading, etc.)
vfr_hud = master.recv_match(type='VFR_HUD', blocking=True, timeout=5)
if vfr_hud:
    print(f"Is armable: {vfr_hud.throttle > 0}")

master.close()
print("\nConnection closed")
