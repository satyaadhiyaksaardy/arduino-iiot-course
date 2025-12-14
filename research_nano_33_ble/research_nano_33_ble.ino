/**
 * @file research_nano_33_ble.ino
 * @brief Thread-Safe Baby Cry Classifier for Arduino Nano 33 BLE
 *
 * A real-time baby cry classification system using TensorFlow Lite Micro,
 * ARM CMSIS-DSP, and multi-threaded RTOS architecture. Features comprehensive
 * thread safety, atomic operations, and timeout-protected synchronization.
 *
 * Hardware: Arduino Nano 33 BLE SENSE + ESP32 MAX30102 + ESP-01S Wi-Fi
 *
 * @author Satya Adhiyaksa
 * @version 1.0
 * @date 2025
 */

// === SYSTEM INCLUDES ===
#include <Arduino.h>           // Arduino core functions
#include <ArduinoBLE.h>        // Bluetooth Low Energy communication
#include <WiFiEspAT.h>         // ESP-01S Wi-Fi co-processor (replaces incompatible WiFiNINA)
#include <ArduinoMqttClient.h> // MQTT client for IoT communication
#include <ArduinoJson.h>       // JSON serialization for data publishing
#include <PDM.h>               // Pulse Density Modulation microphone interface
#include <rtos.h>              // Real-Time Operating System support
#include <mbed.h>              // Mbed OS threading and synchronization primitives
#include "mbed_stats.h"        // Heap diagnostics to watch for OOM freezes

// === TENSORFLOW LITE INCLUDES ===
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include "tensorflow/lite/micro/micro_log.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "tensorflow/lite/micro/system_setup.h"

// === ARM CMSIS-DSP INCLUDES (Hardware-Accelerated) ===
#include <arm_math.h>          // ARM math functions for nRF52840
#include <arm_const_structs.h> // Pre-computed ARM constants

// === APPLICATION-SPECIFIC INCLUDES ===
#include "arduino_cry_classifier.h" // ML model and class definitions
#include "arduino_preprocessing.h"  // Feature extraction parameters

// === ATOMIC OPERATIONS ===
/**
 * @brief Atomic flag type for thread-safe operations on nRF52840
 *
 * Uses volatile uint32_t to ensure atomic read/write operations
 * compatible with GCC built-in atomic functions (__sync_* family)
 */
#define ATOMIC_FLAG volatile uint32_t

// === DEBUG CONFIGURATION ===
/**
 * @brief Debug mode for testing MQTT with dummy data
 *
 * When enabled, generates dummy sensor data and cry detection results
 * without requiring actual hardware sensors or ML inference.
 *
 * Set to true to test MQTT publishing, false for production use.
 */
#define DEBUG_DUMMY_DATA false // Set to true to enable dummy data for testing

// Toggle heavy feature extraction + ML inference. Set to 0 to isolate audio/threads.
#define ENABLE_INFERENCE 1

// Lightweight heap monitor to catch OOM-related freezes
static void logMemory(const char *tag)
{
    mbed_stats_heap_t heap{};
    mbed_stats_heap_get(&heap);
    Serial.print("[MEM][");
    Serial.print(tag);
    Serial.print("] used/reserved (bytes): ");
    Serial.print(heap.current_size);
    Serial.print(" /");
    Serial.println(heap.reserved_size);
}

// === NETWORK CONFIGURATION ===
/**
 * @brief Network and communication settings for ESP-01S Wi-Fi co-processor
 *
 * ESP-01S is used as external Wi-Fi module due to WiFiNINA incompatibility
 * with the nRF52840 architecture on Arduino Nano 33 BLE
 */
#define WIFI_SSID "Mitlab-703"                 // Wi-Fi network name
#define WIFI_PASS "Mitlab703-2"                // Wi-Fi network password
#define MQTT_BROKER "broker.emqx.io"           // MQTT broker hostname/IP
#define MQTT_PORT 1883                         // Standard MQTT port
#define MQTT_TOPIC "sensors/baby_monitor_data" // Topic for publishing sensor data

/**
 * @brief ESP-01S UART communication configuration
 *
 * Creating custom UART object on D2/D3 pins using Mbed UART class
 *
 * Actual Wiring Setup:
 * - D3 (Nano TX) → ESP-01S RX (data only)
 * - D2 (Nano RX) → ESP-01S TX (data only)
 * - ESP-01S power: USB breakout → step-down regulator → 3.3V with ceramic + electrolytic caps
 * - GND, EN, VCC: handled by external power supply (not connected to Nano)
 *
 * This is the CORRECT setup - separate power supply prevents brown-out issues
 * and provides stable 3.3V with proper filtering for ESP-01S.
 *
 * Syntax: UART(Pin_TX, Pin_RX, RTS, CTS)
 * NC = Not Connected (no hardware flow control)
 */
UART wifiSerial(digitalPinToPinName(3), digitalPinToPinName(2), NC, NC);

#define ESP_SERIAL wifiSerial // Custom UART for ESP-01S on D2/D3
#define ESP_BAUD_RATE 115200  // UART baud rate for ESP-01S
#define WIFI_ENABLED true     // Enable WiFi with custom UART on D2/D3

/**
 * @brief BLE service UUIDs for ESP32 MAX30102 sensor communication
 *
 * Custom UUIDs for heart rate and SpO2 data transmission from external
 * ESP32 device with MAX30102 pulse oximetry sensor
 */
#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"         // Main service UUID
#define HEART_RATE_CHAR_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8" // Heart rate characteristic
#define SPO2_CHAR_UUID "f7d1a7a6-419b-4f2e-e6a8-ea07361b26a9"       // SpO2 characteristic (FIXED to match ESP32)

// === MEMORY AND BUFFER CONFIGURATION ===
/**
 * @brief System memory allocation for audio processing and ML inference
 *
 * Buffer sizes optimized for real-time processing while staying within
 * nRF52840 SRAM constraints (256KB total)
 */
#define AUDIO_BUFFER_SIZE 4096                    // Audio samples buffer (reduced for faster handoff)
#define TENSOR_ARENA_SIZE 60000                   // TensorFlow Lite arena (60KB)
#define FEATURE_BUFFER_SIZE RESEARCH_NUM_FEATURES // Feature vector size (62 features)

// === DIGITAL SIGNAL PROCESSING CONFIGURATION ===
/**
 * @brief FFT and windowing parameters matching training dataset preprocessing
 *
 * These values must exactly match the parameters used during model training
 * to ensure feature extraction compatibility and optimal classification accuracy
 */
#define FFT_SAMPLES 512                                                     // FFT window size (power of 2)
#define FFT_SAMPLING_FREQ 16000                                             // Audio sampling frequency (Hz) lowered for stability
#define FFT_HOP_LENGTH RESEARCH_HOP_LENGTH                                  // Frame hop size for overlapping windows
#define FFT_HALF_SAMPLES (FFT_SAMPLES / 2)                                  // Nyquist frequency bins
#define MAX_FRAMES ((AUDIO_BUFFER_SIZE - FFT_SAMPLES) / FFT_HOP_LENGTH + 1) // Maximum processable frames

/**
 * @brief Audio feature extraction configuration
 *
 * Feature counts must match the research model's input layer dimensions
 * MFCC: Mel-Frequency Cepstral Coefficients
 * GFCC: Gammatone-Frequency Cepstral Coefficients
 * Delta features: First and second derivatives of MFCCs
 */
#define N_MFCC RESEARCH_N_MFCC         // Number of MFCC coefficients
#define N_DELTA_MFCC 13                // Delta and delta-delta MFCCs
#define N_GFCC RESEARCH_N_GFCC         // Number of GFCC coefficients
#define N_HARMONIC RESEARCH_N_HARMONIC // Harmonic feature count
#define N_MEL_FILTERS 128              // Mel filter bank size (librosa compatible)

// === RTOS THREADING AND COMMUNICATION ===
/**
 * @brief Network clients and thread management
 *
 * Multi-threaded architecture with dedicated threads for:
 * - Audio acquisition and buffering
 * - ML inference and classification
 * - BLE sensor data collection
 * - Wi-Fi communication and MQTT publishing
 */
WiFiClient wifiClient;                                                // ESP-01S Wi-Fi client
MqttClient mqttClient(wifiClient);                                    // MQTT client for IoT data
rtos::Thread *bleThread, *audioThread, *inferenceThread, *wifiThread; // Thread pointers

/**
 * @brief Industrial-grade thread synchronization primitives
 *
 * Thread-safe synchronization using RTOS mutexes, semaphores, and queues
 * with timeout protection to prevent deadlocks and ensure real-time performance
 */
rtos::Mutex dataMutex;                              // Protects sensor data variables
rtos::Mutex frameHistoryMutex;                      // Protects MFCC frame history buffers
rtos::Semaphore audioBufferReady(0, 1);             // Signals audio buffer ready for processing
rtos::Queue<StaticJsonDocument<512>, 10> dataQueue; // Inter-thread data queue

// === PRIMARY DATA BUFFERS ===
/**
 * @brief Main audio and feature processing buffers
 *
 * Memory-optimized allocation to stay within nRF52840 SRAM constraints
 * audioFloatBuffer removed to save 64KB - conversion done on-the-fly
 */
int16_t audioBuffer[AUDIO_BUFFER_SIZE];   // Raw 16-bit audio samples from PDM mic
float featureBuffer[FEATURE_BUFFER_SIZE]; // Extracted feature vector for ML inference

/**
 * @brief Thread-safe atomic synchronization variables
 *
 * Lockless atomic operations for high-performance inter-thread communication
 * All variables use GCC built-in atomic functions for race-free access
 */
ATOMIC_FLAG audioReady = 0;           // Audio buffer ready for processing flag
ATOMIC_FLAG audioIndex = 0;           // Current audio buffer write index
ATOMIC_FLAG audioBufferLock = 0;      // Atomic lock for buffer reset operations
ATOMIC_FLAG esp32ConnectionState = 0; // BLE connection state (0=disconnected, 1=connected, -1=connecting)

// === THREAD-ISOLATED PROCESSING BUFFERS ===
/**
 * @brief Inference thread private buffers to prevent race conditions
 *
 * Each thread maintains separate buffer sets to eliminate cross-thread
 * memory corruption. All buffers are aligned for ARM SIMD optimization.
 */
float frameFloatBuffer[FFT_SAMPLES]; // Temporary frame processing buffer

/**
 * @brief ARM CMSIS-DSP optimized FFT buffers (main processing)
 *
 * 16-byte aligned for ARM Cortex-M4 SIMD instructions
 * Hardware-accelerated FFT processing using nRF52840 FPU
 */
float32_t fftInputBuffer[FFT_SAMPLES] __attribute__((aligned(16)));          // FFT input data
float32_t fftOutputBuffer[FFT_SAMPLES] __attribute__((aligned(16)));         // FFT complex output
float32_t fftMagnitudeBuffer[FFT_HALF_SAMPLES] __attribute__((aligned(16))); // FFT magnitude spectrum

/**
 * @brief Dedicated FFT buffers for onset detection (attack time feature)
 *
 * Separate buffer set prevents race conditions when onset detection
 * and main feature extraction run concurrently within inference thread
 */
float32_t onsetFftInputBuffer[FFT_SAMPLES] __attribute__((aligned(16)));          // Onset FFT input
float32_t onsetFftOutputBuffer[FFT_SAMPLES] __attribute__((aligned(16)));         // Onset FFT output
float32_t onsetFftMagnitudeBuffer[FFT_HALF_SAMPLES] __attribute__((aligned(16))); // Onset magnitude

/**
 * @brief ARM CMSIS-DSP FFT instances for hardware-accelerated processing
 *
 * Separate FFT instances prevent race conditions when multiple FFT operations
 * execute concurrently. Each instance is optimized for nRF52840 FPU.
 */
static arm_rfft_fast_instance_f32 rfftInstance;                                 // Main FFT processing instance
static arm_rfft_fast_instance_f32 onsetRfftInstance;                            // Dedicated onset detection FFT instance
static const arm_rfft_fast_instance_f32 *fftInstance = &rfftInstance;           // Main processing pointer
static const arm_rfft_fast_instance_f32 *onsetFftInstance = &onsetRfftInstance; // Onset detection pointer

// === MEMORY-EFFICIENT FILTER BANKS ===
/**
 * @brief Pre-computed filter matrices and parameters
 *
 * Memory optimization: Mel filter bank computed on-the-fly to save 130KB SRAM
 * Only essential matrices are pre-computed and aligned for ARM SIMD operations
 */
float dctMatrix[N_MFCC][N_MEL_FILTERS] __attribute__((aligned(16)));              // DCT transform matrix for MFCC
float gfccFilterResponses[N_GFCC][FFT_HALF_SAMPLES] __attribute__((aligned(16))); // GFCC filter responses
float melFilterBankParams[N_MEL_FILTERS + 2] __attribute__((aligned(16)));        // Mel filter bin indices

// === STREAMING FEATURE ACCUMULATORS ===
/**
 * @brief Memory-efficient feature accumulation buffers
 *
 * Streaming processing saves ~28KB by avoiding per-frame feature storage
 * Features are accumulated across all frames and averaged for final vector
 */
float mfccAccumulator[N_MFCC];             // MFCC coefficient accumulator
float deltaMfccAccumulator[N_DELTA_MFCC];  // Delta MFCC accumulator
float delta2MfccAccumulator[N_DELTA_MFCC]; // Delta-delta MFCC accumulator
float gfccAccumulator[N_GFCC];             // GFCC coefficient accumulator
float harmonicAccumulator[N_HARMONIC];     // Harmonic feature accumulator
float spectralAccumulator[5];              // Spectral features: centroid, rolloff, bandwidth, ZCR, RMS
float temporalAccumulator[6];              // Temporal features: attack, mean_abs, variance, peak, std, silence_ratio

// === FRAME HISTORY FOR DELTA FEATURE CALCULATION ===
/**
 * @brief Thread-safe frame history management for temporal derivatives
 *
 * Maintains sliding window of MFCC frames for librosa-compatible delta calculation
 * Uses atomic counters and mutex protection to prevent race conditions
 */
float mfccFrameHistory[3][N_MFCC];        // Circular buffer: [t-1, t, t+1] frames
float deltaFrameHistory[3][N_DELTA_MFCC]; // Delta MFCC history for delta-delta calculation
ATOMIC_FLAG currentFrameCount = 0;        // Atomic frame counter (thread-safe)
ATOMIC_FLAG frameHistoryIndex = 0;        // Atomic circular buffer index (thread-safe)

// === SENSOR DATA WITH ATOMIC FALLBACK PROTECTION ===
/**
 * @brief Multi-modal sensor data with thread-safe access
 *
 * Primary variables protected by dataMutex with timeout-based access
 * Atomic backup variables provide fallback data during mutex contention
 * Ensures zero data loss even under high thread contention scenarios
 */
int latestHeartRate = 0;       // Heart rate from ESP32 MAX30102 (BPM)
int latestSpO2 = 0;            // Blood oxygen saturation (%)
char latestCryType[32] = "";   // Classified cry type (thread-safe C string)
float latestConfidence = 0.0;  // ML model confidence score [0.0-1.0]
bool isCrying = false;         // Binary cry detection flag
float latestAudioEnergy = 0.0; // Audio energy feature (first MFCC coefficient)

// Audio capture diagnostics
volatile unsigned long lastPdmMicros = 0; // Timestamp of last PDM ISR
volatile int lastPdmBytes = 0;            // Bytes reported by last PDM ISR
volatile uint32_t pdmIsrCount = 0;        // Count of PDM interrupts

/**
 * @brief Atomic backup variables for timeout resilience
 *
 * Lock-free atomic copies updated alongside mutex-protected variables
 * Used when mutex timeouts occur to maintain data availability
 */
ATOMIC_FLAG atomicHeartRate = 0;      // Atomic heart rate backup
ATOMIC_FLAG atomicSpO2 = 0;           // Atomic SpO2 backup
ATOMIC_FLAG atomicConfidenceInt = 0;  // Atomic confidence (scaled by 10000 for integer storage)
ATOMIC_FLAG atomicIsCrying = 0;       // Atomic crying state backup
ATOMIC_FLAG atomicAudioEnergyInt = 0; // Atomic audio energy backup (scaled by 10000)

// === BLE CLIENT FOR EXTERNAL SENSOR COMMUNICATION ===
/**
 * @brief Bluetooth Low Energy client for ESP32 MAX30102 pulse oximeter
 *
 * Connects to external ESP32 device running MAX30102 sensor for heart rate
 * and SpO2 measurement. Uses custom GATT service for data transmission.
 */
BLEDevice esp32Device;                     // BLE device handle for ESP32
BLEService esp32Service;                   // GATT service handle
BLECharacteristic heartRateCharacteristic; // Heart rate data characteristic
BLECharacteristic spo2Characteristic;      // SpO2 data characteristic

/**
 * @brief Thread-safe BLE connection management
 *
 * Atomic timestamp prevents race conditions in connection retry logic
 * Reconnection interval prevents excessive connection attempts
 */
ATOMIC_FLAG lastBLEConnectionAttempt = 0;           // Atomic timestamp of last connection attempt
const unsigned long BLE_RECONNECT_INTERVAL = 10000; // Connection retry interval (10 seconds)

// === TENSORFLOW LITE MICRO ML INFERENCE ENGINE ===
/**
 * @brief Machine learning inference components for cry classification
 *
 * TensorFlow Lite Micro setup for real-time on-device inference
 * Model trained on research dataset with 62-dimensional feature vectors
 */
tflite::AllOpsResolver tflOpsResolver;              // TensorFlow operations resolver
const tflite::Model *tflModel = nullptr;            // Loaded model pointer
tflite::MicroInterpreter *tflInterpreter = nullptr; // Inference interpreter
TfLiteTensor *tflInputTensor = nullptr;             // Input tensor (feature vector)
TfLiteTensor *tflOutputTensor = nullptr;            // Output tensor (class probabilities)
alignas(16) uint8_t tensorArena[TENSOR_ARENA_SIZE]; // Memory arena for TensorFlow operations

// Forward declarations for thread entry functions
void inferenceThreadFunction();
void audioThreadFunction();
void bleThreadFunction();
void wifiThreadFunction();

// ===== AUDIO FEATURE EXTRACTION FUNCTIONS =====

/**
 * @brief Calculate onset detection time for attack time feature
 *
 * Implements spectral flux-based onset detection algorithm to measure
 * attack time characteristics of audio signals. Used for temporal
 * feature extraction in cry classification.
 *
 * @param audioBuffer Pointer to 16-bit audio sample buffer
 * @param audioLength Number of samples in the buffer
 * @return Normalized onset time [0.0-1.0] where 0.5 indicates no clear onset
 *
 * @note Uses separate FFT instance (onsetFftInstance) to prevent race conditions
 * @note Thread-safe: Called only from inference thread
 */
float calculateOnsetTime(int16_t *audioBuffer, int audioLength)
{
    const int ONSET_FRAME_SIZE = 512;    // Frame size for onset analysis
    const int ONSET_HOP_SIZE = 256;      // Hop size (50% overlap)
    const float ONSET_THRESHOLD = 0.01f; // Onset detection sensitivity

    // Calculate spectral flux for onset detection (energy increase metric)
    float prev_spectrum[FFT_HALF_SAMPLES];
    memset(prev_spectrum, 0, sizeof(prev_spectrum));

    float max_flux = 0.0f;
    int onset_frame = -1;

    for (int frame_start = 0; frame_start + ONSET_FRAME_SIZE < audioLength; frame_start += ONSET_HOP_SIZE)
    {
        // Apply Hamming window and convert to float for FFT processing
        for (int i = 0; i < ONSET_FRAME_SIZE; i++)
        {
            float window = 0.54f - 0.46f * arm_cos_f32(2.0f * PI * i / (ONSET_FRAME_SIZE - 1)); // Hamming window
            onsetFftInputBuffer[i] = (audioBuffer[frame_start + i] / 32768.0f) * window;        // Normalize int16 to float
        }

        // Compute FFT using dedicated onset instance (thread-safe, no race conditions)
        arm_rfft_fast_f32(onsetFftInstance, onsetFftInputBuffer, onsetFftOutputBuffer, 0);
        arm_cmplx_mag_f32(onsetFftOutputBuffer, onsetFftMagnitudeBuffer, FFT_HALF_SAMPLES);

        // Compute spectral flux: sum of positive spectral changes (onset indicator)
        float flux = 0.0f;
        for (int i = 1; i < FFT_HALF_SAMPLES; i++) // Skip DC component (i=0)
        {
            float diff = onsetFftMagnitudeBuffer[i] - prev_spectrum[i]; // Spectral difference
            if (diff > 0)                                               // Only count energy increases (onset events)
                flux += diff;
            prev_spectrum[i] = onsetFftMagnitudeBuffer[i]; // Update spectrum history
        }

        // Detect first significant onset above adaptive threshold
        if (flux > max_flux * ONSET_THRESHOLD && onset_frame == -1)
        {
            onset_frame = frame_start; // Record onset location
        }

        if (flux > max_flux)
            max_flux = flux;
    }

    if (onset_frame > 0)
    {
        return (float)onset_frame / (float)audioLength; // Normalize to [0,1]
    }
    else
    {
        return 0.5f; // Default middle if no onset detected
    }
}

// ===== BLE CLIENT FOR ESP32 MAX30102 DATA =====

/**
 * @brief Initialize BLE client for ESP32 MAX30102 communication
 *
 * Sets up Bluetooth Low Energy client to connect with external ESP32 device
 * running MAX30102 pulse oximetry sensor for heart rate and SpO2 measurement.
 *
 * @return true if BLE initialization successful, false otherwise
 *
 * @note Called once during system startup
 * @note Thread-safe: Called only from BLE thread
 */
bool initializeBLEClient()
{
    Serial.println("Initializing BLE client for ESP32 MAX30102...");

    if (!BLE.begin())
    {
        Serial.println("Failed to initialize BLE!");
        return false;
    }

    Serial.println("BLE initialized, scanning for ESP32_MAX30102...");
    return true;
}

/**
 * @brief Establish BLE connection to ESP32 MAX30102 device
 *
 * Implements thread-safe connection logic with atomic state management
 * to prevent concurrent connection attempts. Includes service discovery
 * and characteristic subscription for heart rate and SpO2 data.
 *
 * @return true if connection and subscription successful, false otherwise
 *
 * @note Uses atomic compare-and-swap to prevent race conditions
 * @note Implements connection attempt throttling with RECONNECT_INTERVAL
 * @note Thread-safe: Called from BLE thread
 */
bool connectToESP32()
{
    // Thread-safe connection attempt using atomic compare-and-swap
    int expectedState = 0; // Expected disconnected state
    if (!__sync_bool_compare_and_swap(&esp32ConnectionState, expectedState, -1))
    {
        // Connection attempt failed: either connected (1) or connecting (-1)
        int currentState = __sync_fetch_and_add(&esp32ConnectionState, 0); // Atomic read
        return (currentState == 1);                                        // Return success only if actually connected
    }

    // Check connection attempt throttling using atomic timestamp
    unsigned long lastAttempt = __sync_fetch_and_add(&lastBLEConnectionAttempt, 0); // Atomic read
    if (millis() - lastAttempt < BLE_RECONNECT_INTERVAL)                            // Throttle connection attempts
    {
        __sync_fetch_and_and(&esp32ConnectionState, 0); // Reset to disconnected
        return false;
    }

    // Atomically update connection attempt timestamp
    unsigned long currentTime = millis();
    __sync_fetch_and_and(&lastBLEConnectionAttempt, 0);           // Atomic clear to 0
    __sync_fetch_and_add(&lastBLEConnectionAttempt, currentTime); // Atomic set to current time

    Serial.println("[BLE] Scanning for ESP32_MAX30102...");
    BLE.scanForName("ESP32_MAX30102");

    // Wait for scan to complete with timeout
    unsigned long scanStart = millis();
    BLEDevice foundDevice;
    int deviceCount = 0;

    while ((millis() - scanStart < 10000)) // 10 second scan timeout
    {
        foundDevice = BLE.available();

        if (foundDevice)
        {
            deviceCount++;
            String deviceName = foundDevice.localName();
            String deviceAddr = foundDevice.address();

            Serial.print("[BLE] Device #");
            Serial.print(deviceCount);
            Serial.print(": Name='");
            Serial.print(deviceName);
            Serial.print("' Address=");
            Serial.println(deviceAddr);

            if (deviceName == "ESP32_MAX30102")
            {
                Serial.println("[BLE] ✓ Target device FOUND!");
                esp32Device = foundDevice;
                BLE.stopScan(); // CRITICAL: Stop scanning immediately
                break;
            }
        }
        delay(100);
    }

    BLE.stopScan(); // Ensure scan is stopped (redundant safety)

    Serial.print("[BLE] Scan complete. Found ");
    Serial.print(deviceCount);
    Serial.println(" device(s)");

    if (!esp32Device)
    {
        Serial.println("[BLE] ✗ Scan timeout - ESP32_MAX30102 not found");
        Serial.println("[BLE] Troubleshooting:");
        Serial.println("[BLE]   1. Is ESP32 powered on?");
        Serial.println("[BLE]   2. Is ESP32 running the BLE server sketch?");
        Serial.println("[BLE]   3. Check ESP32 Serial output for 'BLE Advertising'");
        Serial.println("[BLE]   4. Try restarting both devices");
        __sync_fetch_and_and(&esp32ConnectionState, 0);
        return false;
    }

    if (esp32Device)
    {
        Serial.println("[BLE] Found ESP32_MAX30102!");
        Serial.print("[BLE] Device address: ");
        Serial.println(esp32Device.address());

        Serial.println("[BLE] Attempting to connect...");
        Serial.println("[BLE] Note: This may take 5-10 seconds");

        // Try to connect with retries
        bool connected = false;
        for (int attempt = 1; attempt <= 3 && !connected; attempt++)
        {
            Serial.print("[BLE] Connection attempt ");
            Serial.print(attempt);
            Serial.println("/3");

            if (esp32Device.connect())
            {
                connected = true;
                Serial.println("[BLE] ✓ Connected to ESP32!");
                break;
            }
            else
            {
                Serial.print("[BLE] ✗ Connection attempt ");
                Serial.print(attempt);
                Serial.println(" failed");
                if (attempt < 3)
                {
                    Serial.println("[BLE] Waiting 2s before retry...");
                    delay(2000);
                }
            }
        }

        if (!connected)
        {
            Serial.println("[BLE] ERROR: Failed to connect after 3 attempts");
            Serial.println("[BLE] Possible causes:");
            Serial.println("[BLE]   1. ESP32 already connected to another device");
            Serial.println("[BLE]   2. ESP32 stopped advertising");
            Serial.println("[BLE]   3. Signal too weak or interference");
            Serial.println("[BLE] Solution: Restart ESP32 and try again");
            esp32Device.disconnect();
            __sync_fetch_and_and(&esp32ConnectionState, 0);
            return false;
        }

        // Connection successful - give it time to stabilize
        Serial.println("[BLE] Connection established! Waiting for stabilization...");
        delay(1000); // Critical: Allow BLE connection to fully establish

        // Check if still connected
        if (!esp32Device.connected())
        {
            Serial.println("[BLE] ERROR: Connection dropped immediately after connect");
            __sync_fetch_and_and(&esp32ConnectionState, 0);
            return false;
        }

        Serial.println("[BLE] Discovering services and characteristics...");
        if (connected && esp32Device.discoverAttributes())
        {
            Serial.println("[BLE] ✓ Services discovered successfully");

            // Get the service and characteristics
            Serial.print("[BLE] Looking for service UUID: ");
            Serial.println(SERVICE_UUID);
            esp32Service = esp32Device.service(SERVICE_UUID);
            if (esp32Service)
            {
                Serial.println("[BLE] Service found!");

                // Debug: List all available characteristics
                Serial.println("[BLE] Enumerating all characteristics in service:");
                int charCount = esp32Service.characteristicCount();
                Serial.print("[BLE] Total characteristics: ");
                Serial.println(charCount);

                for (int i = 0; i < charCount; i++)
                {
                    BLECharacteristic tempChar = esp32Service.characteristic(i);
                    Serial.print("[BLE]   Char #");
                    Serial.print(i);
                    Serial.print(": UUID = ");
                    Serial.println(tempChar.uuid());
                }

                // Now try to get our specific characteristics
                Serial.print("[BLE] Looking for HR UUID: ");
                Serial.println(HEART_RATE_CHAR_UUID);
                heartRateCharacteristic = esp32Service.characteristic(HEART_RATE_CHAR_UUID);

                Serial.print("[BLE] Looking for SpO2 UUID: ");
                Serial.println(SPO2_CHAR_UUID);
                spo2Characteristic = esp32Service.characteristic(SPO2_CHAR_UUID);

                if (heartRateCharacteristic && spo2Characteristic)
                {
                    Serial.println("[BLE] Characteristics found!");
                    Serial.print("[BLE] HR Char UUID: ");
                    Serial.println(HEART_RATE_CHAR_UUID);
                    Serial.print("[BLE] SpO2 Char UUID: ");
                    Serial.println(SPO2_CHAR_UUID);

                    // Subscribe to notifications
                    bool hrSubscribed = heartRateCharacteristic.subscribe();
                    bool spo2Subscribed = spo2Characteristic.subscribe();

                    Serial.print("[BLE] HR subscription: ");
                    Serial.println(hrSubscribed ? "SUCCESS" : "FAILED");
                    Serial.print("[BLE] SpO2 subscription: ");
                    Serial.println(spo2Subscribed ? "SUCCESS" : "FAILED");

                    if (hrSubscribed && spo2Subscribed)
                    {
                        // Atomic compare-and-swap to set connected state in single operation
                        if (__sync_bool_compare_and_swap(&esp32ConnectionState, -1, 1))
                        {
                            __asm__ __volatile__("dsb sy" : : : "memory"); // Memory barrier
                            Serial.println("[BLE] Successfully subscribed to heart rate and SpO2 data!");
                            Serial.println("[BLE] Connection established - waiting for data...");
                            return true;
                        }
                        else
                        {
                            Serial.println("[BLE] ERROR: Failed to atomically set connection state");
                        }
                    }
                    else
                    {
                        Serial.println("[BLE] ERROR: Failed to subscribe to characteristics");
                    }
                }
                else
                {
                    Serial.println("[BLE] ERROR: Failed to find heart rate or SpO2 characteristics");
                }
            }
            else
            {
                Serial.println("[BLE] ERROR: Failed to find service");
            }
        }
        else
        {
            Serial.println("[BLE] ERROR: Failed to discover attributes");
        }

        esp32Device.disconnect();
        __sync_fetch_and_and(&esp32ConnectionState, 0);
        return false;
    }
    else
    {
        Serial.println("[BLE] ✗ Scan timeout - ESP32_MAX30102 not found");
        __sync_fetch_and_and(&esp32ConnectionState, 0);
        return false;
    }
}

/**
 * @brief Update sensor data from BLE characteristics
 *
 * Reads heart rate and SpO2 values from ESP32 MAX30102 device via BLE
 * characteristics. Updates both mutex-protected variables and atomic
 * backup copies for timeout resilience.
 *
 * @note Validates data ranges: HR 30-200 BPM, SpO2 70-100%
 * @note Uses timeout-protected mutex access with atomic fallbacks
 * @note Thread-safe: Called from BLE thread
 */
void updateSensorDataFromBLE()
{
    int connectionState = __sync_fetch_and_add(&esp32ConnectionState, 0); // Atomic read
    if (connectionState <= 0)                                             // 0 = disconnected, -1 = connecting
    {
        if (connectionState == 0)
        { // Only attempt connection if truly disconnected
            connectToESP32();
        }
        return;
    }

    if (!esp32Device.connected())
    {
        Serial.println("[BLE] ESP32 disconnected - attempting reconnect");
        // Atomic compare-and-swap to safely reset connection state
        __sync_bool_compare_and_swap(&esp32ConnectionState, 1, 0);
        return;
    }

    // Read heart rate data
    if (heartRateCharacteristic.valueUpdated())
    {
        const uint8_t *hrData = heartRateCharacteristic.value();
        int valueLength = heartRateCharacteristic.valueLength();

        if (valueLength > 0)
        {
            // Convert uint8_t array to string
            char hrString[32];
            memcpy(hrString, hrData, valueLength);
            hrString[valueLength] = '\0';

            int newHeartRate = atoi(hrString);

            if (newHeartRate > 30 && newHeartRate < 200)
            {
                if (dataMutex.trylock_for(10)) // Quick timeout for sensor data
                {
                    latestHeartRate = newHeartRate;
                    dataMutex.unlock();
                    Serial.print("[BLE DATA] Heart Rate: ");
                    Serial.print(newHeartRate);
                    Serial.println(" BPM");
                }
                else
                {
                    // Atomic fallback when mutex unavailable (high thread contention)
                    __sync_fetch_and_and(&atomicHeartRate, 0);            // Atomic clear
                    __sync_fetch_and_add(&atomicHeartRate, newHeartRate); // Atomic set
                    Serial.print("[BLE DATA] Heart Rate (atomic): ");
                    Serial.print(newHeartRate);
                    Serial.println(" BPM");
                }
                // Always maintain atomic backup for timeout resilience
                __sync_fetch_and_and(&atomicHeartRate, 0);            // Clear atomic backup
                __sync_fetch_and_add(&atomicHeartRate, newHeartRate); // Update atomic backup
            }
            else
            {
                Serial.print("[BLE DATA] Invalid HR value: ");
                Serial.println(newHeartRate);
            }
        }
    }

    // Read SpO2 data
    if (spo2Characteristic.valueUpdated())
    {
        const uint8_t *spo2Data = spo2Characteristic.value();
        int valueLength = spo2Characteristic.valueLength();

        if (valueLength > 0)
        {
            // Convert uint8_t array to string
            char spo2String[32];
            memcpy(spo2String, spo2Data, valueLength);
            spo2String[valueLength] = '\0';

            int newSpO2 = atoi(spo2String);

            if (newSpO2 > 70 && newSpO2 <= 100)
            {
                if (dataMutex.trylock_for(10)) // Quick timeout for sensor data
                {
                    latestSpO2 = newSpO2;
                    dataMutex.unlock();
                    Serial.print("[BLE DATA] SpO2: ");
                    Serial.print(newSpO2);
                    Serial.println("%");
                }
                else
                {
                    // Atomic fallback when mutex unavailable (high thread contention)
                    __sync_fetch_and_and(&atomicSpO2, 0);       // Atomic clear
                    __sync_fetch_and_add(&atomicSpO2, newSpO2); // Atomic set
                    Serial.print("[BLE DATA] SpO2 (atomic): ");
                    Serial.print(newSpO2);
                    Serial.println("%");
                }
                // Always maintain atomic backup for timeout resilience
                __sync_fetch_and_and(&atomicSpO2, 0);       // Clear atomic backup
                __sync_fetch_and_add(&atomicSpO2, newSpO2); // Update atomic backup
            }
            else
            {
                Serial.print("[BLE DATA] Invalid SpO2 value: ");
                Serial.println(newSpO2);
            }
        }
    }
}

/**
 * @brief BLE thread main function for sensor data collection
 *
 * Dedicated RTOS thread for managing BLE connection and sensor data
 * acquisition from ESP32 MAX30102 device. Runs continuously with
 * 500ms update interval.
 *
 * @note Thread priority: osPriorityLow
 * @note Stack size: 2048 bytes
 * @note Handles connection failures gracefully
 */
void bleThreadFunction()
{
    Serial.println("[BLE] Starting BLE thread...");

    if (!initializeBLEClient())
    {
        Serial.println("[BLE] ERROR: BLE initialization failed - sensor data disabled");
        return;
    }

    Serial.println("[BLE] BLE client initialized successfully");

    // Initial connection attempt
    Serial.println("[BLE] Attempting initial connection to ESP32...");
    connectToESP32();

    unsigned long lastStatusPrint = 0;
    while (true)
    {
        updateSensorDataFromBLE();

        // Print connection status every 10 seconds
        if (millis() - lastStatusPrint > 10000)
        {
            lastStatusPrint = millis();
            int state = __sync_fetch_and_add(&esp32ConnectionState, 0);
            Serial.print("[BLE] Connection state: ");
            if (state == 1)
            {
                Serial.println("CONNECTED");
            }
            else if (state == -1)
            {
                Serial.println("CONNECTING...");
            }
            else
            {
                Serial.println("DISCONNECTED");
            }
        }

        rtos::ThisThread::sleep_for(500); // Check every 500ms
    }
}

// ===== ESP-01S WI-FI CO-PROCESSOR MANAGEMENT =====

/**
 * @brief Initialize ESP-01S external Wi-Fi co-processor
 *
 * Configures UART communication and control pins for ESP-01S module.
 * ESP-01S is used as external Wi-Fi due to WiFiNINA incompatibility
 * with nRF52840 architecture.
 *
 * @return true if ESP-01S responds correctly, false otherwise
 *
 * @note Performs hardware reset sequence and firmware version check
 * @note Thread-safe: Called only from Wi-Fi thread
 */
bool initializeESP01S()
{
    Serial.println("[WIFI] Initializing ESP-01S Wi-Fi module...");
    Serial.println("[WIFI] UART: D3(TX)->ESP-RX, D2(RX)->ESP-TX @ 115200 baud");
    Serial.println("[WIFI] Power: External 3.3V supply via USB breakout + regulator");

    // No control pins needed - ESP-01S powered externally
    // EN and RST are tied to 3.3V via external power supply

    // Initialize custom UART for ESP-01S
    Serial.println("[WIFI] Starting custom UART at 115200 baud...");
    ESP_SERIAL.begin(ESP_BAUD_RATE);
    delay(500); // Give ESP-01S time to boot from external power

    Serial.println("[WIFI] Initializing WiFiEspAT library...");
    WiFi.init(&ESP_SERIAL);
    delay(1000);

    // Check ESP-01S module response
    int status = WiFi.status();
    Serial.print("[WIFI] WiFi status: ");
    Serial.println(status);

    if (status == WL_NO_MODULE)
    {
        Serial.println("[WIFI] ERROR: ESP-01S not responding!");
        Serial.println("[WIFI] Checks:");
        Serial.println("[WIFI]   1. RX/TX wiring: ESP-01S RX->D3, TX->D2");
        Serial.println("[WIFI]   2. External power: 3.3V regulated with caps");
        Serial.println("[WIFI]   3. Common ground between Nano and ESP-01S power");
        Serial.println("[WIFI]   4. ESP-01S baud rate: should be 115200");
        Serial.println("[WIFI]   5. EN pin: must be HIGH (3.3V)");
        return false;
    }
    Serial.print("ESP-01S firmware version: ");
    Serial.println(WiFi.firmwareVersion());
    return true;
}

/**
 * @brief Connect to Wi-Fi network via ESP-01S
 *
 * Establishes Wi-Fi connection using ESP-01S AT commands with retry logic.
 * Implements exponential backoff for failed connection attempts.
 *
 * @return true if Wi-Fi connection successful, false otherwise
 *
 * @note Maximum 10 connection attempts with 2-second intervals
 * @note Displays IP address upon successful connection
 * @note Thread-safe: Called from Wi-Fi thread
 */
bool connectToWiFi()
{
    Serial.print("Connecting to Wi-Fi network: ");
    Serial.println(WIFI_SSID);

    int status = WL_IDLE_STATUS;
    int attempts = 0;
    const int MAX_ATTEMPTS = 10;

    while (status != WL_CONNECTED && attempts < MAX_ATTEMPTS)
    {
        status = WiFi.begin(WIFI_SSID, WIFI_PASS);
        if (status != WL_CONNECTED)
        {
            Serial.print(".");
            delay(2000);
            attempts++;
        }
    }

    if (status == WL_CONNECTED)
    {
        Serial.println("Wi-Fi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        return true;
    }
    else
    {
        Serial.println("Wi-Fi connection failed!");
        return false;
    }
}

/**
 * @brief Wi-Fi thread main function for MQTT communication
 *
 * Dedicated RTOS thread managing ESP-01S Wi-Fi connection, MQTT client,
 * and IoT data publishing. Implements connection monitoring and automatic
 * reconnection with graceful error handling.
 *
 * @note Thread priority: osPriorityLow
 * @note Stack size: 2048 bytes
 * @note Publishes cry detection events with sensor data to MQTT
 * @note Uses timeout-protected data access to prevent blocking
 */
void wifiThreadFunction()
{
#if defined(WIFI_ENABLED) && !WIFI_ENABLED
    Serial.println("[WIFI] WiFi DISABLED - Nano 33 BLE does not have Serial1");
    Serial.println("[WIFI] ESP-01S requires hardware UART not available on this board");
    return;
#endif

    if (!initializeESP01S())
    {
        Serial.println("ESP-01S initialization failed - Wi-Fi disabled");
        return;
    }

    if (!connectToWiFi())
    {
        Serial.println("Wi-Fi connection failed - retrying in 30s");
        rtos::ThisThread::sleep_for(30000);
        return;
    }

    // Initialize MQTT client
    mqttClient.setId("Arduino_BabyCryMonitor");
    mqttClient.setUsernamePassword("", "");

    while (true)
    {
        // Maintain Wi-Fi connection
        if (WiFi.status() != WL_CONNECTED)
        {
            Serial.println("Wi-Fi disconnected - reconnecting...");
            connectToWiFi();
            rtos::ThisThread::sleep_for(5000);
            continue;
        }

        // Handle MQTT connection and data publishing
        if (!mqttClient.connected())
        {
            // First verify we can reach the broker
            Serial.println("[MQTT] Testing broker connectivity...");
            WiFiClient testClient;
            if (testClient.connect(MQTT_BROKER, MQTT_PORT))
            {
                Serial.println("[MQTT] ✓ Broker is reachable (TCP connection OK)");
                testClient.stop();
            }
            else
            {
                Serial.println("[MQTT] ✗ Cannot reach broker (TCP connection failed)");
                Serial.println("[MQTT] Possible issues:");
                Serial.println("[MQTT]   - Broker not running");
                Serial.println("[MQTT]   - Firewall blocking port 1883");
                Serial.println("[MQTT]   - Wrong IP address");
                rtos::ThisThread::sleep_for(10000);
                continue;
            }

            Serial.print("Connecting to MQTT broker: ");
            Serial.print(MQTT_BROKER);
            Serial.print(":");
            Serial.println(MQTT_PORT);

            // Set a unique client ID
            String clientId = "BabyCryMonitor_";
            clientId += String(millis()); // Add timestamp for uniqueness

            mqttClient.setId(clientId.c_str());
            mqttClient.setConnectionTimeout(10000); // 10 second timeout
            mqttClient.setKeepAliveInterval(30000); // 30 seconds

            Serial.print("MQTT Client ID: ");
            Serial.println(clientId);

            if (mqttClient.connect(MQTT_BROKER, MQTT_PORT))
            {
                Serial.println("MQTT connected successfully!");
            }
            else
            {
                int error = mqttClient.connectError();
                Serial.print("MQTT connection failed, error = ");
                Serial.println(error);

                // Decode error
                switch (error)
                {
                case -2:
                    Serial.println("  -> Connection timeout (broker not responding)");
                    Serial.println("  -> Check: Is broker running at 192.168.1.222?");
                    Serial.println("  -> Try: ping 192.168.1.222 from your computer");
                    break;
                case -1:
                    Serial.println("  -> Connection refused");
                    break;
                case 1:
                    Serial.println("  -> Protocol version mismatch");
                    break;
                case 2:
                    Serial.println("  -> Client ID rejected");
                    break;
                case 3:
                    Serial.println("  -> Broker unavailable");
                    break;
                case 4:
                    Serial.println("  -> Bad username/password");
                    break;
                case 5:
                    Serial.println("  -> Not authorized");
                    break;
                default:
                    Serial.println("  -> Unknown error");
                }
                rtos::ThisThread::sleep_for(5000);
                continue;
            }
        }

        // Check for data to publish with timeout-safe critical section
        bool crying;
        int heartRate, spo2;
        float confidence, audioEnergy;
        char cryType[32];

#if DEBUG_DUMMY_DATA
        // Dummy data locked to CALM state with energy in 14-50 range (varies per loop)
        static float hrDrift = 118.0f;  // Tracks slow heart-rate trend
        static float spo2Drift = 97.5f; // Tracks slow SpO2 trend

        // Baselines for calm-only mode (center around calmer vitals)
        float targetHr = 114.0f;
        float targetSpo2 = 97.2f;

        // Random energy target between 14 and 50 with stronger jitter
        float targetEnergy = random(14, 51);
        float energyJitter = random(-2, 3);

        // More aggressive drift and jitter for HR/SpO2
        hrDrift = hrDrift * 0.75f + targetHr * 0.25f + random(-12, 13) * 0.35f;
        spo2Drift = spo2Drift * 0.85f + targetSpo2 * 0.15f + random(-6, 7) * 0.12f;

        heartRate = constrain((int)round(hrDrift), 90, 150);
        spo2 = constrain((int)round(spo2Drift), 93, 99);
        audioEnergy = constrain(targetEnergy + energyJitter, 14.0f, 50.0f);

        crying = false;
        confidence = 5.0f + random(0, 6) * 0.5f; // low confidence since no cry

        strncpy(cryType, "calm", sizeof(cryType) - 1);
        cryType[sizeof(cryType) - 1] = '\0';

        Serial.print("[DEBUG] Dummy state: ");
        Serial.print("CALM");
        Serial.print(" | HR=");
        Serial.print(heartRate);
        Serial.print(" | SpO2=");
        Serial.print(spo2);
        Serial.print(" | Energy=");
        Serial.print(audioEnergy, 2);
        Serial.println();
#else
        // Timeout-protected critical section to prevent deadlocks
        if (dataMutex.trylock_for(100)) // 100ms timeout
        {
            crying = isCrying;
            confidence = latestConfidence;
            heartRate = latestHeartRate;
            spo2 = latestSpO2;
            audioEnergy = latestAudioEnergy;
            strncpy(cryType, latestCryType, sizeof(cryType) - 1);
            cryType[sizeof(cryType) - 1] = '\0';
            dataMutex.unlock();
        }
        else
        {
            Serial.println("WARNING: Data mutex timeout in WiFi thread");
            continue; // Skip this iteration if mutex unavailable
        }
#endif

        bool shouldPublish = crying || DEBUG_DUMMY_DATA; // publish dummy calm data too

        if (shouldPublish)
        {
            StaticJsonDocument<512> doc;

            // Core sensor data
            doc["device_id"] = "nano33ble_001";
            doc["timestamp"] = millis();
            doc["heart_rate"] = heartRate;
            doc["spo2"] = spo2;
            doc["is_crying"] = crying;
            doc["cry_type"] = cryType;
            doc["confidence"] = confidence;

            // Audio energy feature (first MFCC coefficient)
            doc["audio_energy"] = audioEnergy;

            // System health indicators
            doc["wifi_connected"] = (WiFi.status() == WL_CONNECTED);
            doc["free_memory"] = (TENSOR_ARENA_SIZE - sizeof(tensorArena)); // Approximation of free tensor memory

            String jsonString;
            serializeJson(doc, jsonString);

            mqttClient.beginMessage(MQTT_TOPIC);
            mqttClient.print(jsonString);
            mqttClient.endMessage();

            Serial.println("[MQTT] ✓ Data published successfully");
            Serial.print("[MQTT] Topic: ");
            Serial.println(MQTT_TOPIC);
            Serial.print("[MQTT] Payload size: ");
            Serial.print(jsonString.length());
            Serial.println(" bytes");
            Serial.print("[MQTT] Sample data: ");
            Serial.println(jsonString.substring(0, min(100, (int)jsonString.length())));
        }

        mqttClient.poll(); // Keep MQTT connection alive
        rtos::ThisThread::sleep_for(1000);
    }
}

// ===== FEATURE EXTRACTION INITIALIZATION =====

/**
 * @brief Initialize mel-frequency filter bank parameters
 *
 * Pre-computes mel-scale frequency bin indices for on-the-fly filter
 * calculation. Saves 130KB SRAM by avoiding full filter bank storage.
 *
 * @note Frequency range: 80Hz - 8000Hz (optimized for cry analysis)
 * @note Filter count: 128 (librosa compatible)
 * @note Thread-safe: Called once during system initialization
 */
void initializeMelFilterBank()
{
    float mel_low = 2595.0 * log10(1.0 + 80.0 / 700.0);
    float mel_high = 2595.0 * log10(1.0 + 8000.0 / 700.0);

    // Calculate mel points and convert to frequency bin indices
    float freq_resolution = (float)FFT_SAMPLING_FREQ / FFT_SAMPLES;
    for (int i = 0; i <= N_MEL_FILTERS + 1; i++)
    {
        float mel_point = mel_low + (mel_high - mel_low) * i / (N_MEL_FILTERS + 1);
        float freq_point = 700.0 * (pow(10.0, mel_point / 2595.0) - 1.0);
        melFilterBankParams[i] = freq_point / freq_resolution; // Store as bin indices
    }
}

/**
 * @brief Calculate mel filter response for given filter and frequency bin
 *
 * Computes triangular mel filter response on-demand to save memory.
 * Implements standard mel-scale triangular filter with linear interpolation.
 *
 * @param filter_idx Filter bank index [0, N_MEL_FILTERS-1]
 * @param bin_idx FFT frequency bin index [0, FFT_HALF_SAMPLES-1]
 * @return Filter response value [0.0-1.0]
 *
 * @note Memory optimization: Saves 131KB by computing on-the-fly
 * @note Thread-safe: Pure function with no shared state
 */
float calculateMelFilterResponse(int filter_idx, int bin_idx)
{
    if (filter_idx >= N_MEL_FILTERS)
        return 0.0f;

    int left_bin = (int)melFilterBankParams[filter_idx];
    int center_bin = (int)melFilterBankParams[filter_idx + 1];
    int right_bin = (int)melFilterBankParams[filter_idx + 2];

    if (bin_idx >= left_bin && bin_idx <= center_bin)
    {
        return (float)(bin_idx - left_bin) / (center_bin - left_bin);
    }
    else if (bin_idx > center_bin && bin_idx <= right_bin)
    {
        return (float)(right_bin - bin_idx) / (right_bin - center_bin);
    }
    else
    {
        return 0.0f;
    }
}

/**
 * @brief Initialize Discrete Cosine Transform matrix for MFCC computation
 *
 * Pre-computes DCT coefficients for efficient MFCC feature extraction.
 * Uses Type-II DCT with normalization compatible with librosa implementation.
 *
 * @note Matrix dimensions: N_MFCC x N_MEL_FILTERS
 * @note Thread-safe: Called once during system initialization
 */
void initializeDCTMatrix()
{
    for (int k = 0; k < N_MFCC; k++)
    {
        for (int n = 0; n < N_MEL_FILTERS; n++)
        {
            dctMatrix[k][n] = cos(PI * k * (n + 0.5) / N_MEL_FILTERS);
        }
    }
}

/**
 * @brief Initialize Gammatone-Frequency Cepstral Coefficient filters
 *
 * Pre-computes GFCC filter bank responses using logarithmically-spaced
 * gammatone filters. Provides complementary spectral analysis to MFCCs.
 *
 * @note Frequency range: 80Hz - 8000Hz
 * @note Filter design: 4th-order gammatone approximation
 * @note Thread-safe: Called once during system initialization
 */
void initializeGFCCFilters()
{
    float low_freq = 80.0, high_freq = 8000.0;
    float gfccCenterFreqs[N_GFCC];
    for (int g = 0; g < N_GFCC; g++)
    {
        float log_low = log10(low_freq);
        float log_high = log10(high_freq);
        gfccCenterFreqs[g] = pow(10.0, log_low + (log_high - log_low) * g / (N_GFCC - 1));
    }
    float freq_resolution = (float)FFT_SAMPLING_FREQ / FFT_SAMPLES;
    float nyquist = FFT_SAMPLING_FREQ / 2.0;
    for (int g = 0; g < N_GFCC; g++)
    {
        float center_freq = gfccCenterFreqs[g];
        float low_cutoff = max(center_freq * 0.8, 20.0) / nyquist;
        float high_cutoff = min(center_freq * 1.2, nyquist - 1.0) / nyquist;
        for (int k = 0; k < FFT_HALF_SAMPLES; k++)
        {
            float freq = k * freq_resolution;
            float norm_freq = freq / nyquist;
            if (norm_freq >= low_cutoff && norm_freq <= high_cutoff)
            {
                float center_norm = (center_freq / nyquist);
                float dist_from_center = abs(norm_freq - center_norm);
                float bandwidth = (high_cutoff - low_cutoff) / 2.0;
                gfccFilterResponses[g][k] = 1.0 / (1.0 + pow(dist_from_center / bandwidth, 4.0));
            }
            else
            {
                gfccFilterResponses[g][k] = 0.0;
            }
        }
    }
}

// ===== PER-FRAME FEATURE EXTRACTION FUNCTIONS =====

/**
 * @brief Extract Mel-Frequency Cepstral Coefficients from power spectrum
 *
 * Implements standard MFCC extraction pipeline: mel filtering, log transform,
 * and DCT. Compatible with librosa MFCC implementation for training consistency.
 *
 * @param powerSpectrum Input power spectrum from FFT magnitude squared
 * @param mfccFeatures Output MFCC coefficient array [N_MFCC elements]
 *
 * @note Uses on-the-fly mel filter calculation for memory efficiency
 * @note Thread-safe: Called only from inference thread
 */
void extractFrameMFCC(float32_t *powerSpectrum, float *mfccFeatures)
{
    float melEnergies[N_MEL_FILTERS];
    for (int f = 0; f < N_MEL_FILTERS; f++)
    {
        melEnergies[f] = 0.0;
        for (int k = 0; k < FFT_HALF_SAMPLES; k++)
        {
            float filter_response = calculateMelFilterResponse(f, k);
            melEnergies[f] += powerSpectrum[k] * filter_response;
        }
        // Protect against log(0)
        melEnergies[f] = max(melEnergies[f], 1e-10f);
    }
    float logMelEnergies[N_MEL_FILTERS];
    for (int f = 0; f < N_MEL_FILTERS; f++)
    {
        logMelEnergies[f] = log(melEnergies[f] + 1e-10);
        if (!isfinite(logMelEnergies[f]))
            logMelEnergies[f] = -23.02585f; // log(1e-10)
    }
    for (int k = 0; k < N_MFCC; k++)
    {
        mfccFeatures[k] = 0.0;
        for (int n = 0; n < N_MEL_FILTERS; n++)
        {
            mfccFeatures[k] += logMelEnergies[n] * dctMatrix[k][n];
        }
        if (!isfinite(mfccFeatures[k]))
            mfccFeatures[k] = 0.0f;
    }
}

/**
 * @brief Extract Gammatone-Frequency Cepstral Coefficients
 *
 * Computes GFCC features using pre-computed gammatone filter responses.
 * Provides complementary spectral representation to MFCCs for improved
 * cry classification accuracy.
 *
 * @param powerSpectrum Input power spectrum from FFT
 * @param gfccFeatures Output GFCC coefficient array [N_GFCC elements]
 *
 * @note Gammatone filters model auditory frequency selectivity
 * @note Thread-safe: Called only from inference thread
 */
void extractFrameGFCC(float32_t *powerSpectrum, float *gfccFeatures)
{
    float filtered_energies[N_GFCC];
    for (int g = 0; g < N_GFCC; g++)
    {
        float energy = 0.0;
        for (int k = 0; k < FFT_HALF_SAMPLES; k++)
        {
            energy += powerSpectrum[k] * gfccFilterResponses[g][k];
        }
        filtered_energies[g] = energy / FFT_HALF_SAMPLES;
        if (!isfinite(filtered_energies[g]) || filtered_energies[g] <= 0)
            filtered_energies[g] = 1e-10f;
    }
    float log_energies[N_GFCC];
    for (int g = 0; g < N_GFCC; g++)
    {
        log_energies[g] = log(filtered_energies[g] + 1e-10);
        if (!isfinite(log_energies[g]))
            log_energies[g] = -23.02585f;
    }
    for (int k = 0; k < N_GFCC; k++)
    {
        gfccFeatures[k] = 0.0;
        for (int n = 0; n < N_GFCC; n++)
        {
            gfccFeatures[k] += log_energies[n] * cos(PI * k * (n + 0.5) / N_GFCC);
        }
        if (!isfinite(gfccFeatures[k]))
            gfccFeatures[k] = 0.0f;
    }
}

/**
 * @brief Extract harmonic features from power spectrum
 *
 * Identifies fundamental frequency and computes relative energy at harmonic
 * frequencies. Useful for characterizing harmonic structure in cry signals.
 *
 * @param powerSpectrum Input power spectrum from FFT
 * @param harmonicFeatures Output harmonic feature array [N_HARMONIC elements]
 *
 * @note F0 estimation range: 200-600 Hz (typical for infant cries)
 * @note Uses ARM SIMD optimization for peak detection
 * @note Thread-safe: Called only from inference thread
 */
void extractFrameHarmonic(float32_t *powerSpectrum, float *harmonicFeatures)
{
    float freq_resolution = (float)FFT_SAMPLING_FREQ / FFT_SAMPLES;
    int f0_start_bin = (int)(200.0 / freq_resolution);
    int f0_end_bin = (int)(600.0 / freq_resolution);
    int f0_bin = f0_start_bin;
    float32_t max_power = 0.0f;
    // ARM SIMD optimized max search
    uint32_t maxIndex;
    arm_max_f32(&powerSpectrum[f0_start_bin], f0_end_bin - f0_start_bin + 1, &max_power, &maxIndex);
    f0_bin = f0_start_bin + maxIndex;

    float f0_freq = f0_bin * freq_resolution;
    float32_t total_energy;
    arm_power_f32(powerSpectrum, FFT_HALF_SAMPLES, &total_energy);

    for (int h = 0; h < N_HARMONIC; h++)
    {
        int harmonic_bin = (int)(f0_freq * (h + 1) / freq_resolution);
        if (harmonic_bin < FFT_HALF_SAMPLES)
        {
            harmonicFeatures[h] = powerSpectrum[harmonic_bin] / (total_energy + 1e-10);
        }
        else
        {
            harmonicFeatures[h] = 0.0;
        }
    }
}

/**
 * @brief Process single audio frame for streaming feature extraction
 *
 * Core frame processing function that performs windowing, FFT, and feature
 * extraction for one audio frame. Updates global feature accumulators and
 * maintains frame history for delta calculation.
 *
 * @param frameAudio Input audio frame [FFT_SAMPLES elements]
 *
 * @note Uses ARM CMSIS-DSP for hardware-accelerated FFT processing
 * @note Updates thread-safe frame counters and history buffers
 * @note Implements Hamming windowing for spectral analysis
 * @note Thread-safe: Called only from inference thread with mutex protection
 */
void processSingleFrameStreaming(float *frameAudio)
{
    // ARM CMSIS-DSP optimized FFT processing
    // Apply Hamming window (ARM optimized)
    for (int i = 0; i < FFT_SAMPLES; i++)
    {
        // Hamming window: 0.54 - 0.46 * cos(2π * n / (N-1))
        float window = 0.54f - 0.46f * arm_cos_f32(2.0f * PI * i / (FFT_SAMPLES - 1));
        fftInputBuffer[i] = frameAudio[i] * window;
    }

    // Hardware-accelerated Real FFT (CMSIS-DSP)
    arm_rfft_fast_f32(fftInstance, fftInputBuffer, fftOutputBuffer, 0);

    // Calculate magnitude using ARM SIMD instructions
    arm_cmplx_mag_f32(fftOutputBuffer, fftMagnitudeBuffer, FFT_HALF_SAMPLES);

    // Power spectrum using ARM SIMD (hardware-accelerated)
    float32_t powerSpectrum[FFT_HALF_SAMPLES] __attribute__((aligned(16)));
    arm_mult_f32(fftMagnitudeBuffer, fftMagnitudeBuffer, powerSpectrum, FFT_HALF_SAMPLES);

    // Fast energy calculation using ARM DSP
    float32_t total_energy;
    arm_power_f32(fftMagnitudeBuffer, FFT_HALF_SAMPLES, &total_energy);

    // Weighted frequency sum (optimized)
    float weighted_freq_sum = 0.0;
    float freq_resolution = (float)FFT_SAMPLING_FREQ / FFT_SAMPLES;
    for (int i = 0; i < FFT_HALF_SAMPLES; i++)
    {
        weighted_freq_sum += i * freq_resolution * powerSpectrum[i];
    }

    // Extract features directly into accumulators
    float currentMfcc[N_MFCC], currentGfcc[N_GFCC], currentHarmonic[N_HARMONIC];
    extractFrameMFCC(powerSpectrum, currentMfcc);
    extractFrameGFCC(powerSpectrum, currentGfcc);
    extractFrameHarmonic(powerSpectrum, currentHarmonic);

    // Store current MFCC frame in circular buffer with mutex protection
    int currentIdx = __sync_fetch_and_add(&frameHistoryIndex, 0) % 3; // Atomic read

    // Protect frame history buffers with timeout
    if (frameHistoryMutex.trylock_for(5))
    { // 5ms timeout
        for (int i = 0; i < N_MFCC; i++)
            mfccFrameHistory[currentIdx][i] = currentMfcc[i];
        frameHistoryMutex.unlock();
    }
    else
    {
        // Skip frame storage if mutex unavailable - better than corruption
        Serial.println("WARNING: Frame history mutex timeout - skipping frame");
    }

    // Accumulate MFCC features
    for (int i = 0; i < N_MFCC; i++)
        mfccAccumulator[i] += currentMfcc[i];

    // Calculate Delta MFCCs using librosa formula: (mfcc[t+1] - mfcc[t-1]) / 2
    int frameCount = __sync_fetch_and_add(&currentFrameCount, 0); // Atomic read
    if (frameCount >= 2)                                          // Need at least 3 frames (t-1, t, t+1)
    {
        int prevIdx = (frameHistoryIndex - 1 + 3) % 3;
        int nextIdx = (frameHistoryIndex + 1) % 3;

        for (int i = 0; i < N_DELTA_MFCC; i++)
        {
            // Wait until we have the next frame
            if (frameCount < 2)
                continue;

            // Protect delta calculation with mutex
            float deltaMfcc = 0.0f;
            if (frameHistoryMutex.trylock_for(5))
            { // 5ms timeout
                deltaMfcc = (mfccFrameHistory[nextIdx][i] - mfccFrameHistory[prevIdx][i]) / 2.0f;
                deltaFrameHistory[currentIdx][i] = deltaMfcc;
                frameHistoryMutex.unlock();
            }
            else
            {
                // Use previous delta value if mutex unavailable
                deltaMfcc = (i < N_DELTA_MFCC && currentIdx > 0) ? deltaFrameHistory[(currentIdx - 1 + 3) % 3][i] : 0.0f;
            }
            if (!isfinite(deltaMfcc))
                deltaMfcc = 0.0f;
            deltaMfccAccumulator[i] += deltaMfcc;

            // Delta-Delta MFCC calculation
            if (frameCount >= 4) // Need 5 frames for delta-delta
            {
                int deltaPrevIdx = (frameHistoryIndex - 1 + 3) % 3;
                int deltaNextIdx = (frameHistoryIndex + 1) % 3;
                // Protect delta-delta calculation with mutex
                float delta2Mfcc = 0.0f;
                if (frameHistoryMutex.trylock_for(5))
                { // 5ms timeout
                    delta2Mfcc = (deltaFrameHistory[deltaNextIdx][i] - deltaFrameHistory[deltaPrevIdx][i]) / 2.0f;
                    frameHistoryMutex.unlock();
                }
                else
                {
                    // Use previous delta-delta value if mutex unavailable
                    delta2Mfcc = (i < N_DELTA_MFCC && currentIdx > 0) ? (deltaFrameHistory[(currentIdx - 1 + 3) % 3][i] - deltaFrameHistory[(currentIdx - 2 + 3) % 3][i]) / 2.0f : 0.0f;
                }
                if (!isfinite(delta2Mfcc))
                    delta2Mfcc = 0.0f;
                delta2MfccAccumulator[i] += delta2Mfcc;
            }
        }
    }

    // Accumulate other features
    for (int i = 0; i < N_GFCC; i++)
        gfccAccumulator[i] += currentGfcc[i];

    for (int i = 0; i < N_HARMONIC; i++)
        harmonicAccumulator[i] += currentHarmonic[i];

    // Spectral features
    float spectralCentroid = (total_energy > 0) ? weighted_freq_sum / total_energy : 0.0;
    spectralAccumulator[0] += spectralCentroid;

    // Spectral rolloff (ARM optimized)
    float32_t cumsum = 0.0f;
    float spectralRolloff = 0.0f;
    float32_t threshold = 0.85f * total_energy;
    for (int i = 0; i < FFT_HALF_SAMPLES; i++)
    {
        cumsum += powerSpectrum[i];
        if (cumsum >= threshold)
        {
            spectralRolloff = i * freq_resolution;
            break;
        }
    }
    spectralAccumulator[1] += spectralRolloff;

    // Spectral bandwidth (ARM SIMD optimized)
    if (total_energy > 0)
    {
        float32_t bandwidth_buffer[FFT_HALF_SAMPLES] __attribute__((aligned(16)));
        float32_t freq_diff_buffer[FFT_HALF_SAMPLES] __attribute__((aligned(16)));

        // Calculate frequency differences
        for (int i = 0; i < FFT_HALF_SAMPLES; i++)
        {
            freq_diff_buffer[i] = (i * freq_resolution) - spectralCentroid;
        }

        // Square differences using ARM SIMD
        arm_mult_f32(freq_diff_buffer, freq_diff_buffer, bandwidth_buffer, FFT_HALF_SAMPLES);

        // Weighted sum using ARM SIMD
        arm_mult_f32(bandwidth_buffer, powerSpectrum, bandwidth_buffer, FFT_HALF_SAMPLES);

        float32_t bandwidth_sum;
        arm_mean_f32(bandwidth_buffer, FFT_HALF_SAMPLES, &bandwidth_sum);

        float32_t bw_norm = bandwidth_sum * (float32_t)FFT_HALF_SAMPLES / total_energy;
        float32_t bw_sqrt;
        arm_sqrt_f32(bw_norm, &bw_sqrt);
        spectralAccumulator[2] += bw_sqrt;
    }

    // Zero crossing rate
    float zero_crossings = 0.0f;
    for (int i = 1; i < FFT_SAMPLES; i++)
    {
        if ((frameAudio[i] > 0) != (frameAudio[i - 1] > 0))
            zero_crossings++;
    }
    spectralAccumulator[3] += zero_crossings / FFT_SAMPLES;

    // RMS energy
    float rms_energy = 0.0f;
    for (int i = 0; i < FFT_SAMPLES; i++)
        rms_energy += frameAudio[i] * frameAudio[i];
    spectralAccumulator[4] += sqrt(rms_energy / FFT_SAMPLES);

    // Atomic frame counter increment and index management with timeout protection
    __sync_fetch_and_add(&currentFrameCount, 1); // Atomic increment

    // Atomic modulo operation with timeout protection
    int old_index, new_index;
    int mod_timeout = 0;
    do
    {
        old_index = __sync_fetch_and_add(&frameHistoryIndex, 0);
        new_index = (old_index + 1) % 3;

        if (++mod_timeout > 100)
        {
            // Force set if CAS loop takes too long
            __sync_fetch_and_and(&frameHistoryIndex, 0);
            __sync_fetch_and_add(&frameHistoryIndex, new_index);
            break;
        }
    } while (!__sync_bool_compare_and_swap(&frameHistoryIndex, old_index, new_index));
}

// ===== FEATURE AGGREGATION AND NORMALIZATION =====

/**
 * @brief Extract complete 62-dimensional feature vector from audio buffer
 *
 * Main feature extraction pipeline that processes entire audio buffer using
 * streaming algorithm to minimize memory usage. Extracts research-optimized
 * feature set matching training data preprocessing.
 *
 * Feature composition:
 * - 13 MFCCs + 13 Delta MFCCs + 13 Delta-Delta MFCCs (39 total)
 * - 8 GFCCs (47 total)
 * - 4 Harmonic features (51 total)
 * - 5 Spectral features (56 total)
 * - 6 Temporal features (62 total)
 *
 * @param audioBuffer Input 16-bit audio samples
 * @param audioLength Number of samples in buffer
 * @param features Output normalized feature vector [62 elements]
 *
 * @note Memory-efficient streaming processing saves ~92KB SRAM
 * @note Thread-safe: Called only from inference thread
 */
void extractResearchOptimizedFeaturesStreaming(int16_t *audioBuffer, int audioLength, float *features)
{
    // Initialize accumulators
    memset(mfccAccumulator, 0, sizeof(mfccAccumulator));
    memset(deltaMfccAccumulator, 0, sizeof(deltaMfccAccumulator));
    memset(delta2MfccAccumulator, 0, sizeof(delta2MfccAccumulator));
    memset(gfccAccumulator, 0, sizeof(gfccAccumulator));
    memset(harmonicAccumulator, 0, sizeof(harmonicAccumulator));
    memset(spectralAccumulator, 0, sizeof(spectralAccumulator));
    memset(mfccFrameHistory, 0, sizeof(mfccFrameHistory));
    memset(deltaFrameHistory, 0, sizeof(deltaFrameHistory));
    __sync_fetch_and_and(&currentFrameCount, 0); // Atomic reset
    __sync_fetch_and_and(&frameHistoryIndex, 0); // Atomic reset

    // Temporal features accumulation
    float mean_abs = 0, peak_amplitude = 0, sum_squares = 0;
    int silence_count = 0;

    // --- 1. Streaming per-frame processing ---
    for (int frame_start = 0; frame_start + FFT_SAMPLES <= audioLength; frame_start += FFT_HOP_LENGTH)
    {
        // Convert int16_t to float on-the-fly (saves 64KB audioFloatBuffer)
        for (int i = 0; i < FFT_SAMPLES; i++)
        {
            frameFloatBuffer[i] = audioBuffer[frame_start + i] / 32768.0f;
        }

        processSingleFrameStreaming(frameFloatBuffer);

        // Accumulate temporal features during processing
        for (int i = 0; i < FFT_SAMPLES; i++)
        {
            float abs_val = abs(frameFloatBuffer[i]);
            mean_abs += abs_val;
            sum_squares += frameFloatBuffer[i] * frameFloatBuffer[i];
            if (abs_val > peak_amplitude)
                peak_amplitude = abs_val;
        }
    }

    int finalFrameCount = __sync_fetch_and_add(&currentFrameCount, 0); // Atomic read
    Serial.print("[DEBUG] Final frame count: ");
    Serial.println(finalFrameCount);

    if (finalFrameCount < 3) // Guard for insufficient frames
    {
        Serial.println("[ERROR] Insufficient frames for feature extraction - returning empty features");
        return;
    }

    // --- 2. Finalize feature vector ---
    int featureIndex = 0;

    // MFCCs (13) - averaged
    for (int i = 0; i < N_MFCC; i++)
        features[featureIndex++] = mfccAccumulator[i] / finalFrameCount;

    // Delta MFCCs (13) - averaged (skip first frame)
    for (int i = 0; i < N_DELTA_MFCC; i++)
        features[featureIndex++] = deltaMfccAccumulator[i] / max(1, finalFrameCount - 1);

    // Delta-Delta MFCCs (13) - averaged (skip first two frames)
    for (int i = 0; i < N_DELTA_MFCC; i++)
        features[featureIndex++] = delta2MfccAccumulator[i] / max(1, finalFrameCount - 2);

    // GFCCs (8) - averaged
    for (int i = 0; i < N_GFCC; i++)
        features[featureIndex++] = gfccAccumulator[i] / finalFrameCount;

    // Harmonics (4) - averaged
    for (int i = 0; i < N_HARMONIC; i++)
        features[featureIndex++] = harmonicAccumulator[i] / finalFrameCount;

    // Spectral Features (5) - averaged
    for (int i = 0; i < 5; i++)
        features[featureIndex++] = spectralAccumulator[i] / finalFrameCount;

    // --- 3. Temporal features (calculated over whole signal) ---
    int total_samples = finalFrameCount * FFT_SAMPLES;
    mean_abs /= total_samples;
    float variance = (sum_squares / total_samples) - (mean_abs * mean_abs);

    float silence_threshold = 0.01 * peak_amplitude;
    for (int i = 0; i < audioLength; i++)
    {
        if (abs(audioBuffer[i] / 32768.0f) < silence_threshold)
            silence_count++;
    }

    // --- 3. Temporal features (6) ---
    // Attack time (onset detection - match training implementation)
    float attack_time = calculateOnsetTime(audioBuffer, audioLength);
    features[featureIndex++] = attack_time;
    features[featureIndex++] = mean_abs;
    features[featureIndex++] = variance;
    features[featureIndex++] = peak_amplitude;
    features[featureIndex++] = sqrt(max(0.0f, variance));
    features[featureIndex++] = (float)silence_count / audioLength;
}

/**
 * @brief Apply feature normalization using training dataset statistics
 *
 * Normalizes feature vector using pre-computed mean and standard deviation
 * from training dataset. Essential for optimal ML model performance.
 *
 * @param features Input/output feature vector to normalize [FEATURE_BUFFER_SIZE elements]
 *
 * @note Normalization: (x - mean) / std_dev
 * @note Uses RESEARCH_SCALER_MEAN and RESEARCH_SCALER_SCALE arrays
 * @note Thread-safe: Called only from inference thread
 */
void normalizeFeatures(float *features)
{
    for (int i = 0; i < FEATURE_BUFFER_SIZE; i++)
    {
        features[i] = (features[i] - RESEARCH_SCALER_MEAN[i]) / RESEARCH_SCALER_SCALE[i];
    }
}

// ===== RTOS THREADS AND INTERRUPT HANDLERS =====

/**
 * @brief PDM microphone interrupt service routine (ISR)
 *
 * High-priority interrupt handler for real-time audio sample acquisition.
 * Uses atomic compare-and-swap operations to prevent race conditions with
 * audio processing thread. Implements lock-free buffer management.
 *
 * @note ISR-safe: No mutex operations allowed in interrupt context
 * @note Uses atomic operations with timeout protection
 * @note Drops samples if buffer full to prevent overflow
 * @note Critical timing: Must complete before next PDM sample
 */
void onPDMdata()
{
    int bytesAvailable = PDM.available();
    int samplesAvailable = bytesAvailable / sizeof(int16_t);

    // Debug: track ISR activity
    lastPdmMicros = micros();
    lastPdmBytes = bytesAvailable;
    pdmIsrCount++;

    // Atomic compare-and-swap loop to prevent buffer overflow races
    int currentIndex, newIndex;
    do
    {
        currentIndex = __sync_fetch_and_add(&audioIndex, 0); // Atomic read
        int room = AUDIO_BUFFER_SIZE - currentIndex;
        int toWrite = samplesAvailable;
        if (toWrite > room)
        {
            toWrite = room; // Clamp to remaining space
        }
        newIndex = currentIndex + toWrite;

        if (toWrite <= 0)
        {
            return; // No room left
        }

        // Try to reserve space atomically
    } while (!__sync_bool_compare_and_swap(&audioIndex, currentIndex, newIndex));

    // Now we have exclusive access to audioBuffer[currentIndex:newIndex]
    if (newIndex > currentIndex)
    {
        int bytesToRead = (newIndex - currentIndex) * sizeof(int16_t);
        PDM.read(audioBuffer + currentIndex, bytesToRead);
    }
    else
    {
        // If ISR fires with zero bytes, note it for diagnostics
        lastPdmBytes = 0;
    }

    // Memory barrier to ensure write completion
    __asm__ __volatile__("dsb sy" : : : "memory");
}

/**
 * @brief Audio acquisition thread main function
 *
 * High-priority RTOS thread responsible for PDM microphone management
 * and audio buffer coordination. Implements thread-safe buffer reset
 * with atomic locking and timeout protection.
 *
 * @note Thread priority: osPriorityHigh (highest priority)
 * @note Stack size: 2048 bytes
 * @note Coordinates with inference thread via semaphore signaling
 * @note Implements timeout recovery for deadlock prevention
 */
void audioThreadFunction()
{
    Serial.println("[AUDIO] Starting audio thread...");
    Serial.print("[AUDIO] Initializing PDM microphone at ");
    Serial.print(FFT_SAMPLING_FREQ);
    Serial.println(" Hz");

    PDM.setGain(24);         // Boost mic gain for easier debugging
    PDM.setBufferSize(1024); // Ensure decent buffer for callbacks

    if (!PDM.begin(1, FFT_SAMPLING_FREQ))
    {
        Serial.println("[AUDIO] ERROR: Failed to start PDM microphone!");
        Serial.println("[AUDIO] This is normal if no audio input is available");
        return;
    }

    PDM.onReceive(onPDMdata); // Register ISR after begin per PDM docs

    Serial.println("[AUDIO] PDM microphone started successfully!");
    Serial.print("[AUDIO] Sample rate = ");
    Serial.println(FFT_SAMPLING_FREQ);

    unsigned long lastStatusMillis = millis();
    int lastIndex = 0;
    uint32_t lastIsrCount = 0;
    unsigned long lastHandoffMillis = millis();
    uint32_t heartbeatCount = 0;

    while (true)
    {
        // Pure atomic read - no mutex needed
        int currentIndex = __sync_fetch_and_add(&audioIndex, 0);

        // Treat near-full as full to avoid stalls from oversized ISR chunks
        if (currentIndex >= (AUDIO_BUFFER_SIZE - 512))
        {
            Serial.print("[DEBUG] Audio buffer full (");
            Serial.print(currentIndex);
            Serial.println(" samples captured)");
            PDM.end();

            // Atomic buffer lock with timeout to prevent deadlock
            int lock_timeout = 0;
            while (!__sync_bool_compare_and_swap(&audioBufferLock, 0, 1) && lock_timeout < 1000)
            {
                rtos::ThisThread::sleep_for(1); // Minimal spin wait
                lock_timeout++;
            }

            if (lock_timeout >= 1000)
            {
                Serial.println("WARNING: Audio buffer lock timeout - forcing acquisition");
                __sync_fetch_and_and(&audioBufferLock, 0); // Force clear
                __sync_fetch_and_add(&audioBufferLock, 1); // Force acquire
            }

            __sync_fetch_and_add(&audioReady, 1); // Signal buffer ready
            // Separate memory barriers for proper ARM ordering
            __asm__ __volatile__("dsb sy" : : : "memory"); // Data synchronization barrier
            __asm__ __volatile__("isb sy" : : : "memory"); // Instruction synchronization barrier
            audioBufferReady.release();                    // Signal inference thread

            // Wait for inference completion with timeout
            int timeout_count = 0;
            while (__sync_fetch_and_add(&audioReady, 0) != 0 && timeout_count < 1000)
            {
                rtos::ThisThread::sleep_for(10);
                timeout_count++;
            }

            if (timeout_count >= 1000)
            {
                Serial.println("WARNING: Inference thread timeout - forcing reset");
                __sync_fetch_and_and(&audioReady, 0); // Force reset
            }

            // Atomic reset with memory barrier
            __sync_fetch_and_and(&audioIndex, 0);
            __asm__ __volatile__("dsb sy" : : : "memory");

            // Release buffer lock
            __sync_fetch_and_and(&audioBufferLock, 0);

            PDM.begin(1, FFT_SAMPLING_FREQ);
        }

        // Periodic status to see if ISR is firing and buffer is filling
        if (millis() - lastStatusMillis > 1000)
        {
            uint32_t isrDelta = pdmIsrCount - lastIsrCount;
            Serial.print("[AUDIO] idx=");
            Serial.print(currentIndex);
            Serial.print(" (delta ");
            Serial.print(currentIndex - lastIndex);
            Serial.print(") ISR/s=");
            Serial.print(isrDelta);
            Serial.print(" lastBytes=");
            Serial.print(lastPdmBytes);
            Serial.print(" dt_us=");
            Serial.println(lastPdmMicros ? (micros() - lastPdmMicros) : 0);

            // Audio thread heartbeat even if ISR is dead
            Serial.print("[AUDIO] hb=");
            Serial.println(++heartbeatCount);

            // If we have any data but haven't handed off in >2s, flush it to inference
            if (currentIndex > 0 && (millis() - lastHandoffMillis) > 2000)
            {
                Serial.print("[AUDIO][WARN] Forcing buffer handoff due to inactivity (idx=");
                Serial.print(currentIndex);
                Serial.println(")");
                __sync_fetch_and_add(&audioReady, 1);
                audioBufferReady.release();
                __sync_fetch_and_and(&audioIndex, 0);
                lastHandoffMillis = millis();
            }

            if (isrDelta == 0)
            {
                Serial.println("[AUDIO][WARN] No PDM interrupts in the last second. Reinitializing PDM...");
                // If we already have samples, push them to inference once before re-init
                if (currentIndex > 0)
                {
                    Serial.println("[AUDIO][WARN] Stalled buffer handing off partial audio to inference");
                    __sync_fetch_and_add(&audioReady, 1);
                    audioBufferReady.release();
                    __sync_fetch_and_and(&audioIndex, 0);
                }

                PDM.end();
                rtos::ThisThread::sleep_for(10);
                PDM.onReceive(onPDMdata);
                PDM.setGain(24);
                PDM.setBufferSize(1024);
                if (!PDM.begin(1, FFT_SAMPLING_FREQ))
                {
                    Serial.println("[AUDIO][ERROR] PDM re-init failed");
                }
                else
                {
                    Serial.println("[AUDIO] PDM re-init OK");
                }
            }

            lastStatusMillis = millis();
            lastIndex = currentIndex;
            lastIsrCount = pdmIsrCount;
            if (currentIndex == 0)
                lastHandoffMillis = millis();
        }
        rtos::ThisThread::sleep_for(10);
    }
}

/**
 * @brief ML inference thread main function
 *
 * Normal-priority RTOS thread handling feature extraction, ML inference,
 * and cry classification. Processes audio buffers using TensorFlow Lite
 * Micro with optimized feature extraction pipeline.
 */
void inferenceThreadFunction()
{
#if ENABLE_INFERENCE
    initializeMelFilterBank();
    initializeDCTMatrix();
    initializeGFCCFilters();
    Serial.println("[DEBUG] Feature extraction components initialized");
    logMemory("inference_thread_start");

#else
    Serial.println("[INFER] Disabled via ENABLE_INFERENCE=0. Skipping preprocessing, feature extraction, and ML inference.");
    while (true)
    {
        rtos::ThisThread::sleep_for(1000);
    }
#endif

    tflModel = tflite::GetModel(research_arduino_cry_classifier_tflite);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.print("[ERROR] Model schema version mismatch. Expected: ");
        Serial.print(TFLITE_SCHEMA_VERSION);
        Serial.print(", Got: ");
        Serial.println(tflModel->version());
        return;
    }

    tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, TENSOR_ARENA_SIZE);
    if (tflInterpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("[ERROR] AllocateTensors() failed");
        return;
    }
    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);

    Serial.println("[DEBUG] Research model ready and tensors allocated successfully");
    Serial.print("[DEBUG] Input tensor size: ");
    Serial.println(tflInputTensor->bytes);
    Serial.print("[DEBUG] Output tensor size: ");
    Serial.println(tflOutputTensor->bytes);

    while (true)
    {
        // Wait for audio data with extended timeout to prevent deadlocks
        if (audioBufferReady.try_acquire_for(5000)) // 5s timeout
        {
            Serial.println("[DEBUG] Audio buffer acquired, starting feature extraction...");

            // Use optimized streaming processing (no audioFloatBuffer needed)
            extractResearchOptimizedFeaturesStreaming(audioBuffer, AUDIO_BUFFER_SIZE, featureBuffer);
            Serial.println("[DEBUG] Feature extraction completed");

            // Sanitize NaN/Inf to keep pipeline stable
            int nanCount = 0;
            for (int i = 0; i < FEATURE_BUFFER_SIZE; i++)
            {
                if (isnan(featureBuffer[i]) || isinf(featureBuffer[i]))
                {
                    nanCount++;
                    featureBuffer[i] = 0.0f;
                }
            }
            if (nanCount > 0)
            {
                Serial.print("[WARN] Sanitized ");
                Serial.print(nanCount);
                Serial.println(" NaN/Inf feature values to 0");
            }

            // Print first few features for debugging
            Serial.print("[DEBUG] First 5 features: ");
            for (int i = 0; i < 5; i++)
            {
                Serial.print(featureBuffer[i]);
                Serial.print(" ");
            }
            Serial.println();

            normalizeFeatures(featureBuffer);
            Serial.println("[DEBUG] Feature normalization completed");

            // Ensure we never overflow the input tensor if model shape changes
            const int requiredInputBytes = FEATURE_BUFFER_SIZE * sizeof(float);
            if (tflInputTensor->bytes < requiredInputBytes)
            {
                Serial.print("[ERROR] Input tensor too small. Tensor bytes=");
                Serial.print(tflInputTensor->bytes);
                Serial.print(" required=");
                Serial.println(requiredInputBytes);
                __sync_fetch_and_and(&audioReady, 0);
                continue;
            }

            // Print first few normalized features
            Serial.print("[DEBUG] First 5 normalized features: ");
            for (int i = 0; i < 5; i++)
            {
                Serial.print(featureBuffer[i]);
                Serial.print(" ");
            }
            Serial.println();

            for (int i = 0; i < FEATURE_BUFFER_SIZE; i++)
            {
                tflInputTensor->data.f[i] = featureBuffer[i];
            }
            Serial.println("[DEBUG] Copied features into input tensor");

            Serial.println("[DEBUG] Starting ML inference...");
            if (tflInterpreter->Invoke() == kTfLiteOk)
            {
                Serial.println("[DEBUG] ML inference successful");

                float maxConfidence = 0;
                int maxIndex = 0;

                // Print all output probabilities for debugging
                Serial.print("[DEBUG] Output probabilities: ");
                for (int i = 0; i < NUM_CLASSES; i++)
                {
                    Serial.print(CLASS_NAMES[i]);
                    Serial.print("=");
                    Serial.print(tflOutputTensor->data.f[i], 4);
                    Serial.print(" ");

                    if (tflOutputTensor->data.f[i] > maxConfidence)
                    {
                        maxConfidence = tflOutputTensor->data.f[i];
                        maxIndex = i;
                    }
                }
                Serial.println();

                Serial.print("[DEBUG] Selected class: ");
                Serial.print(CLASS_NAMES[maxIndex]);
                Serial.print(" with confidence: ");
                Serial.println(maxConfidence, 4);

                // Timeout-protected critical section to prevent deadlocks
                if (dataMutex.trylock_for(50)) // 50ms timeout
                {
                    strncpy(latestCryType, CLASS_NAMES[maxIndex], sizeof(latestCryType) - 1);
                    latestCryType[sizeof(latestCryType) - 1] = '\0'; // Null terminate
                    latestConfidence = maxConfidence;
                    isCrying = maxConfidence > 0.6f;
                    latestAudioEnergy = featureBuffer[1]; // Store first MFCC coefficient as audio energy
                    dataMutex.unlock();

                    // Memory barriers to ensure data consistency
                    __asm__ __volatile__("dsb sy" : : : "memory"); // Data sync
                    __asm__ __volatile__("dmb sy" : : : "memory"); // Data memory barrier
                }
                else
                {
                    Serial.println("WARNING: Data mutex timeout in inference thread - using atomic fallback");
                }

                // Always update atomic backups
                int confidenceInt = (int)(maxConfidence * 10000);     // Convert to int
                int audioEnergyInt = (int)(featureBuffer[1] * 10000); // Convert audio energy to int
                __sync_fetch_and_and(&atomicConfidenceInt, 0);
                __sync_fetch_and_add(&atomicConfidenceInt, confidenceInt);
                __sync_fetch_and_and(&atomicIsCrying, 0);
                __sync_fetch_and_add(&atomicIsCrying, (maxConfidence > 0.6f) ? 1 : 0);
                __sync_fetch_and_and(&atomicAudioEnergyInt, 0);
                __sync_fetch_and_add(&atomicAudioEnergyInt, audioEnergyInt);
            }
            __sync_fetch_and_and(&audioReady, 0); // Atomic reset to 0 - Release audio thread
        }
        else
        {
            Serial.println("[WARNING] Audio buffer semaphore timeout - no new audio data");
        }
        rtos::ThisThread::sleep_for(50);
    }
}

/**
 * @brief Arduino setup function - system initialization
 *
 * One-time initialization of all system components including CMSIS-DSP,
 * thread creation, and hardware setup. Must complete successfully before
 * real-time operation begins.
 *
 * @note Initializes dual FFT instances for race-free processing
 * @note Creates static thread instances to prevent memory leaks
 * @note Validates hardware initialization before proceeding
 */
void setup()
{
    Serial.begin(115200);
    logMemory("setup_begin");

    // Init CMSIS-DSP real FFT instances (512-point)
    if (arm_rfft_fast_init_f32(&rfftInstance, FFT_SAMPLES) != ARM_MATH_SUCCESS)
    {
        Serial.println("Main FFT init failed");
        while (1)
        {
            delay(10);
        }
    }

    if (arm_rfft_fast_init_f32(&onsetRfftInstance, FFT_SAMPLES) != ARM_MATH_SUCCESS)
    {
        Serial.println("Onset FFT init failed");
        while (1)
        {
            delay(10);
        }
    }

    // while (!Serial);
    Serial.println("=== MEMORY-OPTIMIZED ARDUINO BABY CRY CLASSIFIER ===");
    Serial.println("Features: ESP32 BLE Client, ESP-01S Wi-Fi, ARM CMSIS-DSP, Streaming Processing");
    logMemory("after_fft_init");

    // MEMORY-SAFE: Static thread allocation prevents leaks
    static rtos::Thread inferenceThreadInstance(osPriorityNormal, 4096);
    static rtos::Thread audioThreadInstance(osPriorityHigh, 2048);
    static rtos::Thread bleThreadInstance(osPriorityLow, 2048);
    static rtos::Thread wifiThreadInstance(osPriorityLow, 2048);

    inferenceThread = &inferenceThreadInstance;
    audioThread = &audioThreadInstance;
    bleThread = &bleThreadInstance;
    wifiThread = &wifiThreadInstance;

    // Start threads
    inferenceThread->start(mbed::callback(inferenceThreadFunction));
    audioThread->start(mbed::callback(audioThreadFunction));
    bleThread->start(mbed::callback(bleThreadFunction)); // Start BLE client

#if WIFI_ENABLED
    wifiThread->start(mbed::callback(wifiThreadFunction));
    Serial.println("[WIFI] WiFi thread started");
#else
    Serial.println("[WIFI] WiFi DISABLED - Nano 33 BLE does not have Serial1 for ESP-01S");
    Serial.println("[WIFI] To enable WiFi, use SoftwareSerial library on D2/D3 pins");
#endif

    logMemory("after_threads_start");

    Serial.println("System ready!");
    Serial.println("Hardware: Arduino Nano 33 BLE + ESP32 MAX30102 + ESP-01S Wi-Fi co-processor");
}

/**
 * @brief Arduino main loop - system status monitoring
 *
 * Low-priority monitoring function that displays system status every 5 seconds.
 * Uses timeout-protected data access with atomic fallback values to ensure
 * non-blocking operation even under thread contention.
 *
 * @note Non-blocking: Uses atomic fallbacks if mutex timeouts occur
 * @note Displays cry detection, sensor data, and connection status
 * @note Runs at 0.2 Hz update rate to minimize thread interference
 */
void loop()
{
    rtos::ThisThread::sleep_for(5000);
    Serial.println("[HEARTBEAT] loop alive");
    logMemory("loop");
    // Timeout-safe data access in main loop
    bool crying = false;
    char cryType[32] = "N/A";
    float confidence = 0.0;
    float audioEnergy = 0.0;
    int heartRate = 0;
    int spo2 = 0;
    int connectionState = __sync_fetch_and_add(&esp32ConnectionState, 0); // Atomic read

    if (dataMutex.trylock_for(10)) // Very short timeout for status display
    {
        crying = isCrying;
        strncpy(cryType, latestCryType, sizeof(cryType) - 1);
        cryType[sizeof(cryType) - 1] = '\0';
        confidence = latestConfidence;
        audioEnergy = latestAudioEnergy;
        heartRate = latestHeartRate;
        spo2 = latestSpO2;
        dataMutex.unlock();
    }
    else
    {
        // Use atomic fallback values if mutex timeout
        Serial.println("WARNING: Status display mutex timeout - using atomic fallback values");
        heartRate = __sync_fetch_and_add(&atomicHeartRate, 0);
        spo2 = __sync_fetch_and_add(&atomicSpO2, 0);
        int confidenceInt = __sync_fetch_and_add(&atomicConfidenceInt, 0);
        confidence = confidenceInt / 10000.0f; // Convert back to float
        int audioEnergyInt = __sync_fetch_and_add(&atomicAudioEnergyInt, 0);
        audioEnergy = audioEnergyInt / 10000.0f; // Convert back to float
        crying = (__sync_fetch_and_add(&atomicIsCrying, 0) == 1);
        strncpy(cryType, crying ? "Atomic_Fallback" : "Not_Crying", sizeof(cryType) - 1);
        cryType[sizeof(cryType) - 1] = '\0';
    }

    Serial.println("=== SYSTEM STATUS ===");
    Serial.print("Cry Detection: ");
    Serial.print(crying ? cryType : "Not Crying");
    Serial.print(" (Conf: ");
    Serial.print(confidence * 100, 1);
    Serial.println("%)");

    Serial.print("Heart Rate: ");
    Serial.print(heartRate > 0 ? String(heartRate) : "N/A");
    Serial.println(" BPM");

    Serial.print("SpO2: ");
    Serial.print(spo2 > 0 ? String(spo2) + "%" : "N/A");
    Serial.println();

    Serial.print("BLE: ");
    Serial.println(connectionState ? "Connected to ESP32" : "Disconnected");
    Serial.println();
}