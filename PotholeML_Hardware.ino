#define echoPin 9
#define trigPin 10
#define buzzPin 2

long duration;
int distance;
void setup(){
  Serial.begin(9600);
  pinMode(trigPin,OUTPUT);
  pinMode(echoPin,INPUT);
  pinMode(buzzPin, OUTPUT);
 }
void loop(){
  digitalWrite(trigPin,LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin,HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin,LOW);
  
 duration=pulseIn(echoPin,HIGH);
  distance=(duration*0.034/2);
  Serial.print("Distance : ");
  Serial.print(distance);
  Serial.println(" cm ");
  delay(1000);

  if(distance > 10)
  {
  Serial.println("Pothole Detected");
  digitalWrite(buzzPin,HIGH);
  }
  else
  {
  Serial.println("Normal Road");
  digitalWrite(buzzPin,LOW);
  }
  delay(1000);
               
}