/**
 * @file esp32_max30102_ble.ino
 * @brief ESP32-C6 MAX30102 BLE Server with FreeRTOS
 *
 * Hardware: Waveshare ESP32-C6-WROOM-1 + MAX30102 sensor
 * @author Satya Adhiyaksa
 * @version 2.0 (ESP32-C6 optimized)
 * @date 2025
 */

#include <NimBLEDevice.h>
#include <Wire.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <freertos/queue.h>
#include <esp_task_wdt.h>
#include <esp_log.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include "heartRate.h"

// === ESP32-C6 HARDWARE CONFIGURATION ===
#define I2C_SDA 6 // Waveshare ESP32-C6 I2C SDA pin
#define I2C_SCL 7 // Waveshare ESP32-C6 I2C SCL pin

// === CONFIGURATION ===
#define MAX_BRIGHTNESS 255
#define SENSOR_SAMPLE_RATE 100     // Hz - optimized for accuracy vs power
#define DATA_VALIDATION_SAMPLES 5  // Consecutive valid samples required
#define BLE_SECURITY_ENABLED false // Enable BLE security and pairing
#define WATCHDOG_TIMEOUT_SEC 30    // Watchdog timeout period
#define TASK_STACK_SIZE 4096       // FreeRTOS task stack size (reduced for single-core)
#define QUEUE_LENGTH 10            // Inter-task queue length
#define SENSOR_ERROR_THRESHOLD 5   // Max consecutive sensor errors before reset

static const char *TAG = "ESP32_MAX30102"; // Logging tag

// === SENSOR AND BLE OBJECTS ===
MAX30105 particleSensor;

NimBLEServer *pServer = nullptr;
NimBLECharacteristic *pHeartRateCharacteristic = nullptr;
NimBLECharacteristic *pSpO2Characteristic = nullptr;

// === THREAD-SAFE STATE VARIABLES ===
volatile bool deviceConnected = false;
volatile bool oldDeviceConnected = false;
volatile bool sensorInitialized = false;
volatile bool systemHealthy = true;

// === FREERTOS SYNCHRONIZATION ===
SemaphoreHandle_t sensorDataMutex = nullptr;
SemaphoreHandle_t bleDataMutex = nullptr;
QueueHandle_t sensorDataQueue = nullptr;
TaskHandle_t sensorTaskHandle = nullptr;
TaskHandle_t bleTaskHandle = nullptr;
TaskHandle_t watchdogTaskHandle = nullptr;

// === SENSOR DATA STRUCTURE ===
typedef struct
{
    int32_t heartRate;
    int32_t spo2;
    bool validHeartRate;
    bool validSpO2;
    uint32_t timestamp;
    float confidence;
    uint32_t sequenceNumber;
} SensorData_t;

#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define HEART_RATE_CHAR_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define SPO2_CHAR_UUID "f7d1a7a6-419b-4f2e-e6a8-ea07361b26a9"

// === ENHANCED SENSOR PROCESSING ===
const byte RATE_SIZE = 8;
byte rateArray[RATE_SIZE];
byte rateSpot = 0;
unsigned long lastBeat = 0;
int beatsPerMinute = 0;
int32_t bufferLength = 50; // Reduced for single-core (0.5s vs 1s)
uint32_t redBuffer[50] __attribute__((aligned(4)));
uint32_t irBuffer[50] __attribute__((aligned(4)));

// === DATA VALIDATION ARRAYS ===
int32_t heartRateHistory[DATA_VALIDATION_SAMPLES];
int32_t spo2History[DATA_VALIDATION_SAMPLES];
int validationIndex = 0;

// === CURRENT SENSOR VALUES ===
volatile int32_t currentHeartRate = 0;
volatile int32_t currentSpO2 = 0;
volatile bool currentValidHR = false;
volatile bool currentValidSpO2 = false;

// === STATISTICS AND MONITORING ===
volatile uint32_t totalSensorReadings = 0;
volatile uint32_t validDataCount = 0;
volatile uint32_t bleTransmissions = 0;
volatile uint32_t sensorErrors = 0;
volatile unsigned long lastHealthCheck = 0;
volatile uint32_t sequenceCounter = 0;

// === BLE SERVER CALLBACKS ===
class ExpertServerCallbacks : public NimBLEServerCallbacks
{
    void onConnect(NimBLEServer *pServer)
    {
        deviceConnected = true;
        ESP_LOGI(TAG, "BLE Client connected");

        // Reset transmission statistics on new connection
        bleTransmissions = 0;
        sequenceCounter = 0;
    }

    void onDisconnect(NimBLEServer *pServer)
    {
        deviceConnected = false;
        ESP_LOGI(TAG, "BLE Client disconnected - Total transmissions: %lu", bleTransmissions);

        // Auto-restart advertising after disconnect
        vTaskDelay(pdMS_TO_TICKS(500));
        pServer->startAdvertising();
        ESP_LOGI(TAG, "BLE Advertising restarted");
    }
};

// === SECURITY CALLBACKS ===
// Note: NimBLE security is configured via device-level methods, not callbacks

// === UTILITY FUNCTIONS ===

/**
 * @brief Validate sensor data using statistical analysis
 * @param heartRate Current heart rate reading
 * @param spo2 Current SpO2 reading
 * @return true if data passes validation checks
 */
bool validateSensorData(int32_t heartRate, int32_t spo2)
{
    // Range validation
    if (heartRate < 30 || heartRate > 200)
        return false;
    if (spo2 < 70 || spo2 > 100)
        return false;

    // Store in validation history
    heartRateHistory[validationIndex] = heartRate;
    spo2History[validationIndex] = spo2;
    validationIndex = (validationIndex + 1) % DATA_VALIDATION_SAMPLES;

    // Statistical validation - check for reasonable variance
    if (validationIndex == 0)
    { // Buffer is full
        int32_t hrSum = 0, spo2Sum = 0;
        for (int i = 0; i < DATA_VALIDATION_SAMPLES; i++)
        {
            hrSum += heartRateHistory[i];
            spo2Sum += spo2History[i];
        }

        float hrMean = hrSum / (float)DATA_VALIDATION_SAMPLES;
        float spo2Mean = spo2Sum / (float)DATA_VALIDATION_SAMPLES;

        // Check if current reading is within reasonable range of recent mean
        if (abs(heartRate - hrMean) > 30)
            return false; // HR variance > 30 BPM
        if (abs(spo2 - spo2Mean) > 10)
            return false; // SpO2 variance > 10%
    }

    return true;
}

/**
 * @brief Calculate confidence score based on signal quality
 * @param redSignal Red LED signal strength
 * @param irSignal IR LED signal strength
 * @return Confidence score (0.0 - 1.0)
 */
float calculateConfidence(uint32_t redSignal, uint32_t irSignal)
{
    // Signal strength analysis
    float redRatio = redSignal / 262144.0f; // Normalize to 0-1
    float irRatio = irSignal / 262144.0f;   // Normalize to 0-1

    // Good signals should be in middle range (not too low, not saturated)
    float redScore = 1.0f - abs(0.5f - redRatio) * 2.0f;
    float irScore = 1.0f - abs(0.5f - irRatio) * 2.0f;

    // Signal ratio should be reasonable
    float ratioScore = 1.0f;
    if (redSignal > 0 && irSignal > 0)
    {
        float ratio = (float)redSignal / (float)irSignal;
        if (ratio < 0.5f || ratio > 2.0f)
            ratioScore = 0.5f;
    }

    return (redScore + irScore + ratioScore) / 3.0f;
}

/**
 * @brief System health monitoring function
 */
void performHealthCheck()
{
    unsigned long now = millis();

    if (now - lastHealthCheck > 10000)
    { // Every 10 seconds
        lastHealthCheck = now;

        float successRate = totalSensorReadings > 0 ? (float)validDataCount / (float)totalSensorReadings * 100.0f : 0.0f;

        ESP_LOGI(TAG, "Health Check - Success Rate: %.1f%%, Errors: %lu, BLE TX: %lu",
                 successRate, sensorErrors, bleTransmissions);

        // Update system health status
        systemHealthy = (successRate > 70.0f) && (sensorErrors < SENSOR_ERROR_THRESHOLD);

        if (!systemHealthy)
        {
            ESP_LOGW(TAG, "System health degraded - considering sensor reset");
        }
    }
}

// === FREERTOS TASKS ===

/**
 * @brief High-priority sensor reading task
 * @param parameter Task parameter (unused)
 */
void sensorTask(void *parameter)
{
    ESP_LOGI(TAG, "Sensor task started on core %d", xPortGetCoreID());

    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1000 / SENSOR_SAMPLE_RATE); // 100 Hz

    int consecutiveErrors = 0;

    while (true)
    {
        totalSensorReadings++;

        // Fill buffer with fresh samples
        bool sensorError = false;
        for (byte i = 0; i < bufferLength; i++)
        {
            int timeout = 0;
            while (particleSensor.available() == false && timeout < 100)
            {
                particleSensor.check();
                vTaskDelay(pdMS_TO_TICKS(1));
                timeout++;
            }

            if (timeout >= 100)
            {
                ESP_LOGW(TAG, "Sensor timeout at sample %d", i);
                sensorError = true;
                sensorErrors++;
                break;
            }

            redBuffer[i] = particleSensor.getRed();
            irBuffer[i] = particleSensor.getIR();
            particleSensor.nextSample();
        }

        if (!sensorError)
        {
            // Reset consecutive error count on successful read
            consecutiveErrors = 0;

            // Run Maxim algorithm for SpO2 and HR
            int32_t spo2, heartRate;
            int8_t validSPO2, validHeartRate;

            maxim_heart_rate_and_oxygen_saturation(
                irBuffer, bufferLength, redBuffer,
                &spo2, &validSPO2, &heartRate, &validHeartRate);

            // Calculate confidence based on signal quality
            uint32_t avgRed = 0, avgIR = 0;
            for (int i = 0; i < bufferLength; i++)
            {
                avgRed += redBuffer[i];
                avgIR += irBuffer[i];
            }
            avgRed /= bufferLength;
            avgIR /= bufferLength;

            float confidence = calculateConfidence(avgRed, avgIR);

            // Validate data
            bool dataValid = (validHeartRate && validSPO2 &&
                              validateSensorData(heartRate, spo2) &&
                              confidence > 0.3f);

            if (dataValid)
            {
                validDataCount++;

                // Thread-safe update of current values
                if (xSemaphoreTake(sensorDataMutex, pdMS_TO_TICKS(10)) == pdTRUE)
                {
                    currentHeartRate = heartRate;
                    currentSpO2 = spo2;
                    currentValidHR = true;
                    currentValidSpO2 = true;

                    // Create sensor data packet for queue
                    SensorData_t sensorData = {
                        .heartRate = heartRate,
                        .spo2 = spo2,
                        .validHeartRate = true,
                        .validSpO2 = true,
                        .timestamp = millis(),
                        .confidence = confidence,
                        .sequenceNumber = ++sequenceCounter};

                    xSemaphoreGive(sensorDataMutex);

                    // Send to BLE task via queue (non-blocking)
                    xQueueSend(sensorDataQueue, &sensorData, 0);

                    ESP_LOGD(TAG, "Valid data: HR=%ld, SpO2=%ld, Conf=%.2f",
                             heartRate, spo2, confidence);
                }
                else
                {
                    ESP_LOGW(TAG, "Failed to acquire sensor mutex");
                }
            }
            else
            {
                ESP_LOGD(TAG, "Invalid data: HR=%s, SpO2=%s, Conf=%.2f",
                         validHeartRate ? "OK" : "BAD",
                         validSPO2 ? "OK" : "BAD",
                         confidence);
            }
        }
        else
        {
            consecutiveErrors++;
            if (consecutiveErrors >= SENSOR_ERROR_THRESHOLD)
            {
                ESP_LOGE(TAG, "Too many consecutive sensor errors - attempting reset");

                // Attempt sensor reset
                Wire.begin();
                if (particleSensor.begin())
                {
                    particleSensor.setup();
                    particleSensor.setPulseAmplitudeRed(0x0A);
                    particleSensor.setPulseAmplitudeGreen(0);
                    consecutiveErrors = 0;
                    ESP_LOGI(TAG, "Sensor reset successful");
                }
                else
                {
                    ESP_LOGE(TAG, "Sensor reset failed - will retry");
                }
            }
        }

        // Feed watchdog
        esp_task_wdt_reset();

        // Maintain precise timing
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

/**
 * @brief BLE communication task
 * @param parameter Task parameter (unused)
 */
void bleTask(void *parameter)
{
    ESP_LOGI(TAG, "BLE task started on core %d", xPortGetCoreID());

    SensorData_t receivedData;

    while (true)
    {
        // Wait for sensor data from queue
        if (xQueueReceive(sensorDataQueue, &receivedData, pdMS_TO_TICKS(1000)) == pdTRUE)
        {

            if (deviceConnected && pHeartRateCharacteristic && pSpO2Characteristic)
            {
                // Thread-safe BLE transmission
                if (xSemaphoreTake(bleDataMutex, pdMS_TO_TICKS(50)) == pdTRUE)
                {

                    String hrValue = String(receivedData.heartRate);
                    String spo2Value = String(receivedData.spo2);

                    try
                    {
                        // Set values and notify connected clients
                        pHeartRateCharacteristic->setValue(hrValue.c_str());
                        pHeartRateCharacteristic->notify();

                        pSpO2Characteristic->setValue(spo2Value.c_str());
                        pSpO2Characteristic->notify();

                        bleTransmissions++;

                        ESP_LOGD(TAG, "BLE TX #%lu: HR=%s, SpO2=%s, Seq=%lu",
                                 bleTransmissions, hrValue.c_str(), spo2Value.c_str(),
                                 receivedData.sequenceNumber);
                    }
                    catch (...)
                    {
                        ESP_LOGE(TAG, "BLE transmission error");
                    }

                    xSemaphoreGive(bleDataMutex);
                }
                else
                {
                    ESP_LOGW(TAG, "Failed to acquire BLE mutex");
                }
            }
        }

        // Handle BLE connection state changes
        if (!deviceConnected && oldDeviceConnected)
        {
            vTaskDelay(pdMS_TO_TICKS(500));
            if (pServer)
            {
                pServer->startAdvertising();
                ESP_LOGI(TAG, "Restarted BLE advertising");
            }
            oldDeviceConnected = deviceConnected;
        }

        if (deviceConnected && !oldDeviceConnected)
        {
            oldDeviceConnected = deviceConnected;
        }

        // Feed watchdog
        esp_task_wdt_reset();

        vTaskDelay(pdMS_TO_TICKS(50)); // 20 Hz update rate
    }
}

/**
 * @brief System watchdog and health monitoring task
 * @param parameter Task parameter (unused)
 */
void watchdogTask(void *parameter)
{
    ESP_LOGI(TAG, "Watchdog task started on core %d", xPortGetCoreID());

    while (true)
    {
        performHealthCheck();

        // Monitor task health
        TaskStatus_t sensorTaskStatus, bleTaskStatus;
        vTaskGetInfo(sensorTaskHandle, &sensorTaskStatus, pdTRUE, eInvalid);
        vTaskGetInfo(bleTaskHandle, &bleTaskStatus, pdTRUE, eInvalid);

        ESP_LOGD(TAG, "Task States - Sensor: %d, BLE: %d, Free Heap: %lu bytes",
                 sensorTaskStatus.eCurrentState, bleTaskStatus.eCurrentState,
                 esp_get_free_heap_size());

        // Feed watchdog
        esp_task_wdt_reset();

        vTaskDelay(pdMS_TO_TICKS(5000)); // 5 second intervals
    }
}

// === SETUP FUNCTION ===
void setup()
{
    Serial.begin(115200);
    while (!Serial)
    {
        delay(10);
    } // Wait for USB CDC on ESP32-C6

    ESP_LOGI(TAG, "=== EXPERT ESP32-C6 MAX30102 BLE SERVER ===");
    ESP_LOGI(TAG, "Version 2.0 - ESP32-C6 Single-Core Optimized");
    ESP_LOGI(TAG, "Free Heap: %lu bytes", esp_get_free_heap_size());

    // === HARDWARE INITIALIZATION ===
    Wire.begin(I2C_SDA, I2C_SCL);
    ESP_LOGI(TAG, "I2C initialized: SDA=GPIO%d, SCL=GPIO%d", I2C_SDA, I2C_SCL);

    if (!particleSensor.begin(Wire))
    {
        ESP_LOGE(TAG, "MAX30105 sensor not found! Check wiring/power.");
        while (1)
        {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    ESP_LOGI(TAG, "MAX30105 sensor initialized successfully");
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
    sensorInitialized = true;

    // === FREERTOS SYNCHRONIZATION INITIALIZATION ===
    sensorDataMutex = xSemaphoreCreateMutex();
    bleDataMutex = xSemaphoreCreateMutex();
    sensorDataQueue = xQueueCreate(QUEUE_LENGTH, sizeof(SensorData_t));

    if (!sensorDataMutex || !bleDataMutex || !sensorDataQueue)
    {
        ESP_LOGE(TAG, "Failed to create FreeRTOS synchronization objects!");
        while (1)
        {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    ESP_LOGI(TAG, "FreeRTOS synchronization objects created");

    // === BLE INITIALIZATION ===
    NimBLEDevice::init("ESP32_MAX30102");

    // Create BLE server first
    pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new ExpertServerCallbacks());

    // Configure BLE security (NimBLE style)
    if (BLE_SECURITY_ENABLED)
    {
        NimBLEDevice::setSecurityAuth(true, true, true);
        NimBLEDevice::setSecurityPasskey(123456);
        NimBLEDevice::setSecurityIOCap(BLE_HS_IO_NO_INPUT_OUTPUT);
        ESP_LOGI(TAG, "BLE Security enabled");
    }

    NimBLEService *pService = pServer->createService(SERVICE_UUID);

    // Create characteristics with enhanced properties (NimBLE flags)
    pHeartRateCharacteristic = pService->createCharacteristic(
        HEART_RATE_CHAR_UUID,
        NIMBLE_PROPERTY::READ |
            NIMBLE_PROPERTY::NOTIFY |
            NIMBLE_PROPERTY::INDICATE);

    pSpO2Characteristic = pService->createCharacteristic(
        SPO2_CHAR_UUID,
        NIMBLE_PROPERTY::READ |
            NIMBLE_PROPERTY::NOTIFY |
            NIMBLE_PROPERTY::INDICATE);

    // NimBLE automatically adds 2902 descriptor for NOTIFY/INDICATE

    pService->start();

    // Configure advertising (NimBLE style)
    NimBLEAdvertising *pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setMinInterval(32); // 20ms (32 * 0.625ms)
    pAdvertising->setMaxInterval(64); // 40ms (64 * 0.625ms)
    NimBLEDevice::startAdvertising();

    ESP_LOGI(TAG, "BLE Server initialized and advertising started");

    // === WATCHDOG INITIALIZATION ===
    esp_task_wdt_config_t wdt_config = {
        .timeout_ms = WATCHDOG_TIMEOUT_SEC * 1000,
        .idle_core_mask = (1 << portNUM_PROCESSORS) - 1,
        .trigger_panic = true};
    esp_task_wdt_init(&wdt_config);

    // === FREERTOS TASK CREATION (Single-Core Optimized) ===
    ESP_LOGI(TAG, "Creating tasks for single-core ESP32-C6...");

    xTaskCreate(
        sensorTask,
        "SensorTask",
        TASK_STACK_SIZE,
        nullptr,
        configMAX_PRIORITIES - 1, // Highest priority
        &sensorTaskHandle);

    xTaskCreate(
        bleTask,
        "BLETask",
        TASK_STACK_SIZE,
        nullptr,
        configMAX_PRIORITIES - 2, // High priority
        &bleTaskHandle);

    xTaskCreate(
        watchdogTask,
        "WatchdogTask",
        TASK_STACK_SIZE / 2,
        nullptr,
        tskIDLE_PRIORITY + 1, // Low priority
        &watchdogTaskHandle);

    // Add watchdog for critical tasks only (safer on ESP32-C6)
    esp_task_wdt_add(sensorTaskHandle);
    esp_task_wdt_add(bleTaskHandle);
    esp_task_wdt_add(watchdogTaskHandle);
    // NOTE: NOT adding loop task to avoid TWDT panics on ESP32-C6

    ESP_LOGI(TAG, "FreeRTOS tasks created and pinned to cores");
    ESP_LOGI(TAG, "System initialization complete - Place finger on sensor");
    ESP_LOGI(TAG, "Free Heap after init: %lu bytes", esp_get_free_heap_size());
}

// === MAIN LOOP ===
void loop()
{
    // In expert implementation, main loop only handles system monitoring
    // All sensor reading and BLE communication is handled by FreeRTOS tasks

    static unsigned long lastStatusReport = 0;
    unsigned long now = millis();

    // Display system status every 30 seconds
    if (now - lastStatusReport > 30000)
    {
        lastStatusReport = now;

        // Read current values safely
        int32_t currentHR = 0, currentSpo2Val = 0;
        bool validData = false;

        if (xSemaphoreTake(sensorDataMutex, pdMS_TO_TICKS(10)) == pdTRUE)
        {
            currentHR = currentHeartRate;
            currentSpo2Val = currentSpO2;
            validData = currentValidHR && currentValidSpO2;
            xSemaphoreGive(sensorDataMutex);
        }

        ESP_LOGI(TAG, "=== SYSTEM STATUS ===");
        ESP_LOGI(TAG, "Current Data: HR=%ld BPM, SpO2=%ld%% (Valid: %s)",
                 currentHR, currentSpo2Val, validData ? "YES" : "NO");
        ESP_LOGI(TAG, "BLE Connected: %s, Transmissions: %lu",
                 deviceConnected ? "YES" : "NO", bleTransmissions);
        ESP_LOGI(TAG, "System Health: %s, Free Heap: %lu bytes",
                 systemHealthy ? "GOOD" : "DEGRADED", esp_get_free_heap_size());
        ESP_LOGI(TAG, "Statistics - Total: %lu, Valid: %lu, Errors: %lu",
                 totalSensorReadings, validDataCount, sensorErrors);
    }

    // Feed main loop watchdog
    esp_task_wdt_reset();

    // Minimal delay for main loop - tasks handle the real work
    vTaskDelay(pdMS_TO_TICKS(1000));
}