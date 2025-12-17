#!/usr/bin/env python3
"""
Drone GPS Circle Mission Script
Circles between 3 GPS waypoints with wait times at each point
Supports switching to Alt Hold mode for manual control
"""

from pymavlink import mavutil
import time
import sys

# Configuration
WAYPOINTS = [
    {"lat": 37.7749, "lon": -122.4194, "alt": 10},  # Point 1 (relative altitude in meters)
    {"lat": 37.7750, "lon": -122.4195, "alt": 10},  # Point 2
    {"lat": 37.7751, "lon": -122.4196, "alt": 10},  # Point 3
]
WAIT_TIME = 5  # seconds to wait at each waypoint
ACCEPTANCE_RADIUS = 2  # meters - how close to waypoint before considering "reached"
CONNECTION_STRING = '/dev/ttyAMA0'  # Serial port for Raspberry Pi
BAUD_RATE = 57600

class DroneController:
    def __init__(self, connection_string, baud_rate):
        print(f"Connecting to drone on {connection_string}...")
        self.master = mavutil.mavlink_connection(connection_string, baud=baud_rate)
        
        # Wait for heartbeat
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print(f"Heartbeat received (system {self.master.target_system} component {self.master.target_component})")
        
    def set_mode(self, mode):
        """Set flight mode"""
        mode_id = self.master.mode_mapping()[mode]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        
        # Wait for mode change confirmation
        print(f"Setting mode to {mode}...")
        while True:
            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack:
                print(f"Mode change acknowledged: {ack}")
                break
        
        return True
    
    def arm(self):
        """Arm the drone"""
        print("Arming drone...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        
        # Wait for arming confirmation
        self.master.motors_armed_wait()
        print("Drone armed!")
        
    def disarm(self):
        """Disarm the drone"""
        print("Disarming drone...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        
        self.master.motors_disarmed_wait()
        print("Drone disarmed!")
    
    def takeoff(self, altitude):
        """Takeoff to specified altitude"""
        print(f"Taking off to {altitude}m...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        
        # Monitor altitude
        while True:
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
            if msg:
                current_alt = msg.relative_alt / 1000.0  # Convert mm to meters
                print(f"Current altitude: {current_alt:.1f}m")
                if current_alt >= altitude * 0.95:
                    print("Takeoff complete!")
                    break
            time.sleep(0.5)
    
    def goto_position(self, lat, lon, alt):
        """Navigate to GPS position"""
        print(f"Going to position: Lat={lat}, Lon={lon}, Alt={alt}m")
        
        self.master.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_global_int_message(
                10,  # time_boot_ms (not used)
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                int(0b110111111000),  # type_mask (only position enabled)
                int(lat * 1e7),  # lat_int
                int(lon * 1e7),  # lon_int
                alt,  # alt
                0, 0, 0,  # vx, vy, vz
                0, 0, 0,  # afx, afy, afz
                0, 0  # yaw, yaw_rate
            )
        )
    
    def get_distance_to_target(self, target_lat, target_lon):
        """Calculate distance to target position"""
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        if msg:
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            
            # Simple distance calculation (works for short distances)
            from math import sin, cos, sqrt, atan2, radians
            
            R = 6371000  # Earth radius in meters
            lat1, lon1 = radians(current_lat), radians(current_lon)
            lat2, lon2 = radians(target_lat), radians(target_lon)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            return distance
        return None
    
    def wait_for_position(self, lat, lon, acceptance_radius):
        """Wait until drone reaches target position"""
        print(f"Waiting to reach position (within {acceptance_radius}m)...")
        
        while True:
            distance = self.get_distance_to_target(lat, lon)
            if distance is not None:
                print(f"Distance to target: {distance:.2f}m")
                if distance <= acceptance_radius:
                    print("Position reached!")
                    return True
            time.sleep(1)
    
    def check_mode(self):
        """Check current flight mode"""
        msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        if msg:
            mode = mavutil.mode_string_v10(msg)
            return mode
        return None
    
    def monitor_for_mode_change(self, timeout=0.1):
        """Non-blocking check for mode changes (e.g., manual override)"""
        msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
        if msg:
            mode = mavutil.mode_string_v10(msg)
            return mode
        return None

def main():
    try:
        # Initialize drone controller
        drone = DroneController(CONNECTION_STRING, BAUD_RATE)
        
        # Pre-flight checks
        print("\n=== PRE-FLIGHT CHECKLIST ===")
        print("1. Ensure GPS has good fix")
        print("2. Ensure battery is sufficient")
        print("3. Clear airspace around drone")
        print("4. RC transmitter ready for manual override")
        input("\nPress Enter to continue or Ctrl+C to abort...")
        
        # Set to GUIDED mode
        drone.set_mode('GUIDED')
        time.sleep(1)
        
        # Arm and takeoff
        drone.arm()
        time.sleep(2)
        
        drone.takeoff(WAYPOINTS[0]['alt'])
        time.sleep(2)
        
        print("\n=== STARTING CIRCULAR MISSION ===")
        print("Switch to ALT_HOLD on RC transmitter for manual control at any time")
        
        # Main mission loop
        waypoint_index = 0
        loop_count = 0
        
        while True:
            # Check if mode changed (manual override)
            current_mode = drone.check_mode()
            if current_mode and 'ALT_HOLD' in current_mode:
                print("\n!!! ALT_HOLD mode detected - mission paused !!!")
                print("Waiting for return to GUIDED mode...")
                
                while True:
                    current_mode = drone.check_mode()
                    if current_mode and 'GUIDED' in current_mode:
                        print("GUIDED mode restored - resuming mission")
                        break
                    time.sleep(1)
            
            # Get current waypoint
            wp = WAYPOINTS[waypoint_index]
            print(f"\n--- Loop {loop_count + 1}, Waypoint {waypoint_index + 1}/3 ---")
            
            # Navigate to waypoint
            drone.goto_position(wp['lat'], wp['lon'], wp['alt'])
            drone.wait_for_position(wp['lat'], wp['lon'], ACCEPTANCE_RADIUS)
            
            # Wait at waypoint
            print(f"Holding position for {WAIT_TIME} seconds...")
            time.sleep(WAIT_TIME)
            
            # Move to next waypoint
            waypoint_index = (waypoint_index + 1) % len(WAYPOINTS)
            
            # Increment loop counter when completing full circle
            if waypoint_index == 0:
                loop_count += 1
    
    except KeyboardInterrupt:
        print("\n\n!!! Mission interrupted by user !!!")
        print("Switching to ALT_HOLD mode...")
        drone.set_mode('ALT_HOLD')
        time.sleep(1)
        print("Drone in ALT_HOLD - land manually using RC transmitter")
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("Attempting to switch to ALT_HOLD mode...")
        try:
            drone.set_mode('ALT_HOLD')
            print("Drone in ALT_HOLD - land manually using RC transmitter")
        except:
            print("Could not set ALT_HOLD - use RC transmitter immediately!")
        
    finally:
        print("\nScript ended")

if __name__ == "__main__":
    main()
