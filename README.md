# 🚁 AE100 Team CANDY - Operation BEARTRAC

**UCSD Aerospace Engineering AE100: Autonomous QR Code Hunting Quadcopter**

## 📋 Files

| File | Description |
|------|-------------|
| `QR_PostProcessor.py` | Filters noisy drone QR captures → binary fusion image w/ pyzbar + edge/green metrics |
| `LoRa Communications/` | `915Receiver.py` & `915Transmitter.py` - CubeSat coord exchange |
| `AE100__Final_Report.pdf` | Complete technical report: subsystems, testing, reflections |

## 🎯 Mission
Autonomous quadcopter searches 2-acre RFS area for 1m² "Oski" QR codes using CubeSat coords. 11min flights + 10min data cycles w/ battery swaps.

## 🛠️ Stack

Pixhawk 2.4.8 + RPi 4B + Arducam 12MP
3S 5000mAh → 40A ESCs → 1000KV motors (10x4.5 props)
915MHz SiK/LoRa + 2.4GHz RC + WiFi/SSH


## 📈 Key Results
- **Flight:** 13-15min (beat 9min prediction)
- **QR:** 30ft detection, post-processed fusion
- **Comms:** 4x channels validated (LoRa link budget)

**Team CANDY** | UC Berkeley AE100 | Fall 2025
