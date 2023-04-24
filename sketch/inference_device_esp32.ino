/*

Created 21/03/2023
By Rodrigo Costa

*/

#include <ADS1015_WE.h>
#include <Wire.h>

#include <WiFi.h>
#include <PubSubClient.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "model.h" "For your model created by TFL"

/*ADS1015 defines*/
#define I2C_ADDRESS 0x48
#define ALERTPIN 16  //gpio_16 - RX2

/*Others defines */
#define NUMSAMPLES 214
#define NOTEBOOK_CHARGER 1
#define THERMAL_BRUSH_V1 2
#define THERMAL_BRUSH_V2 3
#define THERMAL_BRUSH_V3 4
#define DISCONNECTED 5
#define CELLPHONE_CHARGER 6
#define FAN_V3 7
#define FAN_V2 8
#define FAN_V1 9

/* MQTT defines */
#define TOPIC_SUBSCRIBE_CURRENTSENSOR "curr_sensor"
#define PLENGTH 1024
#define ID_MQTT "esp32_pub"

/*ADS1015 Library params */
ADS1015_WE adc = ADS1015_WE(I2C_ADDRESS);
constexpr int READY_PIN = ALERTPIN;
volatile bool conv_ready = false;
void IRAM_ATTR convReadyAlert() {
  conv_ready = true;
}

/* WiFi Library params */
WiFiClient wifiClient;
const char* SSID = "";           // your network SSID (name)
const char* PASSWORD = "";  // your network password

/* MQTT Library params */
PubSubClient MQTT(wifiClient);
const char* BROKER_MQTT = "";  // ip address broker mqtt
uint16_t BROKER_PORT = 1883;               // broker port

/* TensorFlow Lite params */
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
// Create an area of memory to use for input, output, and intermediate arrays.
const int kTensorArenaSize = 107 * 1024;
__attribute__((aligned(16))) uint8_t* tensor_arena;
}

char data[8] = " ";
char buffer_samples[1200] = " ";
unsigned long last_time = millis();
uint8_t count_samples = 0;

void initTFLite(void);
void initADS1015(void);
void initWiFi(void);
void initMQTT(void);
void connectToBrokerMQTT(void);
void connectToWiFi(WiFiEvent_t event, WiFiEventInfo_t info);
void addressWiFi(WiFiEvent_t event, WiFiEventInfo_t info);
void reconnectToWiFi(WiFiEvent_t event, WiFiEventInfo_t info);
void checkMQTTConnection(void);
uint8_t outInference(TfLiteTensor* tensor);
void bufferInsertion(int16_t data);

void setup() {

  strcpy(buffer_samples, "h|");
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);
  if (!adc.init()) {
    Serial.println(F("ADS1015 disconnected!"));
  }
  initADS1015();  
  initWiFi();
  WiFi.onEvent(connectToWiFi, WiFiEvent_t::ARDUINO_EVENT_WIFI_STA_CONNECTED);
  WiFi.onEvent(addressWiFi, WiFiEvent_t::ARDUINO_EVENT_WIFI_STA_GOT_IP);
  WiFi.onEvent(reconnectToWiFi, WiFiEvent_t::ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
  WiFi.begin(SSID, PASSWORD);
  initMQTT();
  connectToBrokerMQTT();
  initTFLite();
}

void loop() {
  checkMQTTConnection();
  if (conv_ready) {

    int16_t raw_data = adc.getRawResult();    //getting current values
    input->data.f[count_samples] = raw_data;  //put values into input tensor to make inference

    bufferInsertion(raw_data);
    count_samples++;

    if (count_samples >= NUMSAMPLES) {     
           
      TfLiteStatus invoke_status = interpreter->Invoke();  // make inference
      if (invoke_status != kTfLiteOk) {
        Serial.println(F("Invoke failed!"));
        while (1)
          ;
        return;
      }
      
      uint8_t result = outputInferenceValue(output);
      switch (result) {
        case NOTEBOOK_CHARGER:
          bufferInsertion(NOTEBOOK_CHARGER);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;

        case THERMAL_BRUSH_V1:
          bufferInsertion(THERMAL_BRUSH_V1);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;

        case THERMAL_BRUSH_V2:
          bufferInsertion(THERMAL_BRUSH_V2);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;

        case THERMAL_BRUSH_V3:
          bufferInsertion(THERMAL_BRUSH_V3);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;

        case DISCONNECTED:          
          Serial.println(F("Disconnected!"));          
          break;

        case CELLPHONE_CHARGER:
          bufferInsertion(CELLPHONE_CHARGER);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;

        case FAN_V3:
          bufferInsertion(FAN_V3);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;

        case FAN_V2:
          bufferInsertion(FAN_V2);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;

        case FAN_V1:
          bufferInsertion(FAN_V1);
          MQTT.publish(TOPIC_SUBSCRIBE_CURRENTSENSOR, buffer_samples, PLENGTH);
          break;
        default:
          Serial.println(F("Unknown output!"));
      }
      //printf("Inference output: %d\n", result);
      //Serial.println(buffer_samples);
      strcpy(buffer_samples, "h|");
      count_samples = 0;
      delay(600);           
  }
  conv_ready = false;
}
MQTT.loop();
}

uint8_t outputInferenceValue(TfLiteTensor* tensor) {
  uint8_t index = 0;
  float maxVal = tensor->data.f[0];
  for (uint8_t k = 0; k < 10; k++) {
    if (tensor->data.f[k] > maxVal) {
      maxVal = tensor->data.f[k];
      index = k;
    }
  }
  return index;
}

void bufferInsertion(int16_t raw_data) {
  snprintf(data, sizeof(data), "%d", raw_data);
  strcat(buffer_samples, data);
  strcat(buffer_samples, " ");
}

void initADS1015(void) {

  Serial.println(F("Setup ADS1015 params"));
  Serial.println();

  /* Set the voltage range of the ADC to adjust the gain:
  * Please note that you must not apply more than VDD + 0.3V to the input pins!
  *
  * ADS1015_RANGE_6144  ->  2/3x gain +/- 6.144V  1 bit = 3mV 
  * ADS1015_RANGE_4096  ->  1x gain   +/- 4.096V  1 bit = 2mV 
  * ADS1015_RANGE_2048  ->  2x gain   +/- 2.048V  1 bit = 1mV
  * ADS1015_RANGE_1024  ->  4x gain   +/- 1.024V  1 bit = 0.5mV
  * ADS1015_RANGE_0512  ->  8x gain   +/- 0.512V  1 bit = 0.25mV
  * ADS1015_RANGE_0256  ->  16x gain  +/- 0.256V  1 bit = 0.125mV
  */
  adc.setVoltageRange_mV(ADS1015_RANGE_2048);  //comment line/change parameter to change range

  /* Set the inputs to be compared
  *
  * ADS1015_COMP_0_1    ->  compares 0 with 1 (default)
  * ADS1015_COMP_0_3    ->  compares 0 with 3
  * ADS1015_COMP_1_3    ->  compares 1 with 3
  * ADS1015_COMP_2_3    ->  compares 2 with 3
  * ADS1015_COMP_0_GND  ->  compares 0 with GND
  * ADS1015_COMP_1_GND  ->  compares 1 with GND
  * ADS1015_COMP_2_GND  ->  compares 2 with GND
  * ADS1015_COMP_3_GND  ->  compares 3 with GND
  */
  adc.setCompareChannels(ADS1015_COMP_0_1);

  /* Set number of conversions after which the alert pin will be active
  * - or you can disable the alert
  *
  *  ADS1015_ASSERT_AFTER_1  -> after 1 conversion
  *  ADS1015_ASSERT_AFTER_2  -> after 2 conversions
  *  ADS1015_ASSERT_AFTER_4  -> after 4 conversions
  *  ADS1015_DISABLE_ALERT   -> disable comparator // alert pin (default)
  */
  adc.setAlertPinMode(ADS1015_ASSERT_AFTER_1);

  /* Set the conversion rate in SPS (samples per second)
  * Options should be self-explaining:
  *
  * ADS1015_128_SPS
  * ADS1015_250_SPS
  * ADS1015_490_SPS
  * ADS1015_920_SPS
  * ADS1015_1600_SPS (default)
  * ADS1015_2400_SPS
  * ADS1015_3300_SPS
  * ADS1015_3300_SPS
  */
  adc.setConvRate(ADS1015_3300_SPS);

  /* Set continuous or single shot mode:
  *
  * ADS1015_CONTINUOUS  ->  continuous mode
  * ADS1015_SINGLE     ->  single shot mode (default)
  */
  adc.setMeasureMode(ADS1015_CONTINUOUS);

  /* With this function the alert pin will be active, when a conversion is ready.
  */
  adc.setAlertPinToConversionReady();

  attachInterrupt(digitalPinToInterrupt(READY_PIN), convReadyAlert, FALLING);
}

void initTFLite(void) {

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  //There is 520 KB of available SRAM on the ESP32, where 320 KB used to DRAM (DATA) and 200 KB to IRAM (INSTRUCTIONS)
  //However, due to a techinical limitation, the maximum statically allocated DRAM usage is 160 KB.
  //The remaining 160 KB (for a total of 320 KB of DRAM) can only be allocated ar runtime as Heap
  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      tflModel->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  static tflite::MicroMutableOpResolver<12> micro_op_resolver;
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddTanh();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddDequantize();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    tflModel, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println(F("AllocateTensors() failed"));
    return;
  }

  size_t used_bytes = interpreter->arena_used_bytes();
  TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void initWiFi(void) {
  Serial.println(F("------ Wi-Fi Connection ------"));
  Serial.print(F("Attempting to connect to AP SSID: "));
  Serial.println(SSID);
  Serial.println(F("Wait..."));
}

void initMQTT(void) {
  MQTT.setBufferSize(PLENGTH);
  MQTT.setServer(BROKER_MQTT, BROKER_PORT);  
}

void connectToBrokerMQTT(void) {

  while (!MQTT.connected()) {
    Serial.print(F("Attempting MQTT connection to the Broker address: "));
    Serial.println(BROKER_MQTT);
    if (MQTT.connect(ID_MQTT)) {
      Serial.println(F("You're connected to Broker MQTT!"));
    } else {
      Serial.println(F("Failed to connect to Broker."));
      Serial.println(F("Try again in 3 seconds"));
      delay(3000);
    }
  }
}

void checkMQTTConnection(void) {
  if (!MQTT.connected())
    connectToBrokerMQTT();  
}

void connectToWiFi(WiFiEvent_t event, WiFiEventInfo_t info) {
  Serial.println(F("Connected to AP successfully!"));
}

void addressWiFi(WiFiEvent_t event, WiFiEventInfo_t info) {
  Serial.println(F("WiFi connected"));
  Serial.print(F("IP address: "));
  Serial.println(WiFi.localIP());
}

void reconnectToWiFi(WiFiEvent_t event, WiFiEventInfo_t info) {
  Serial.println(F("Disconnected from WiFi access point"));
  Serial.print(F("WiFi lost connection. Reason: "));
  Serial.println(info.wifi_sta_disconnected.reason);
  Serial.println(F("Trying to Reconnect"));
  WiFi.begin(SSID, PASSWORD);
}
