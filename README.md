# Baby Cry Monitor (Nano 33 BLE + ESP32-C6 MAX30102)

This repository contains two Arduino sketches that work together:

- `research_nano_33_ble/` – Nano 33 BLE Sense client that captures audio, runs on-device ML cry classification (TensorFlow Lite Micro), connects to the ESP32 BLE server for MAX30102 vitals, and publishes results over Wi-Fi/MQTT.
- `esp32_max30102_ble/` – ESP32-C6 BLE server that reads the MAX30102 pulse-oximeter and streams heart-rate/SpO2 to the Nano client.

## Hardware

- Nano 33 BLE Sense (nRF52840)
- Waveshare ESP32-C6-WROOM-1 + MAX30102 sensor
- ESP-01S Wi-Fi module (UART D3 TX → ESP RX, D2 RX → ESP TX, external 3.3V supply with caps, common GND)
- Topic used: `sensors/baby_monitor_data`

## Sketch: `research_nano_33_ble`

- Features: PDM mic capture at 16 kHz, streaming MFCC/GFCC/harmonic features, TensorFlow Lite Micro inference, BLE client to ESP32, MQTT publisher.
- Buffer sizes: `AUDIO_BUFFER_SIZE=4096`, `TENSOR_ARENA_SIZE=60000`.
- Wi-Fi/MQTT: broker `broker.emqx.io`, port `1883`, topic `sensors/baby_monitor_data` (edit in sketch as needed).

### Build steps (Nano 33 BLE)

1. Board: **Arduino Mbed OS Nano Boards → Arduino Nano 33 BLE**.
2. Libraries (Library Manager unless noted): `ArduinoBLE`, `PDM`, `WiFiEspAT`, `ArduinoMqttClient`, `ArduinoJson`, `TensorFlowLite` (Arduino_TensorFlowLite), `CMSIS-DSP` is bundled with the core.
3. Open `research_nano_33_ble.ino`, set Wi-Fi SSID/PASS and flags above.
4. Compile & upload. Use a powered ESP-01S; keep Serial at 115200 for logs.

## Sketch: `esp32_max30102_ble`

- Features: MAX30102 sampling at 100 Hz, validation and confidence scoring, BLE service with HR/SpO2 characteristics, FreeRTOS watchdog.
- Config (top of sketch): I2C pins SDA=6, SCL=7; `SENSOR_SAMPLE_RATE=100`; `BLE_SECURITY_ENABLED` flag; `TASK_STACK_SIZE=4096`.
- UUIDs match the Nano client: service `4fafc201-1fb5-459e-8fcc-c5c9c331914b`, HR char `beb5483e-36e1-4688-b7f5-ea07361b26a8`, SpO2 char `f7d1a7a6-419b-4f2e-e6a8-ea07361b26a9`.

### Build steps (ESP32-C6)

1. Board: **ESP32C6 Dev Module** (Arduino-ESP32 core) and select the correct port.
2. Libraries: `NimBLE-Arduino`, `MAX30105` (SparkFun), `heartRate`, `spo2_algorithm` (bundled with MAX30105 examples), FreeRTOS is built-in.
3. Open `esp32_max30102_ble.ino`, adjust BLE security flag or sample rate if desired.
4. Compile & upload. Sensor is on I2C SDA=6, SCL=7.

## Typical wiring

- MAX30102 → ESP32-C6: SDA→6, SCL→7, 3V3, GND.
- ESP-01S → Nano 33 BLE: D3(TX)→ESP RX, D2(RX)→ESP TX, external 3.3V supply, common ground.

## Running

1. Power ESP32-C6; verify it advertises and streams HR/SpO2.
2. Power Nano 33 BLE; it will connect over BLE, capture audio, run inference (if `ENABLE_INFERENCE=1`), and publish MQTT payloads. With `DEBUG_DUMMY_DATA=true`, it publishes synthetic calm data for pipeline testing.
3. Subscribe to MQTT topic `sensors/baby_monitor_data` to see JSON payloads.

## Troubleshooting

- Audio stalls: watch for `[AUDIO][WARN]` logs; reduce sample rate or reinit PDM as already implemented.
- BLE not connecting: confirm ESP32 is advertising with matching UUIDs; restart both ends and ensure single client.
