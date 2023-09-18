<h1 align="center">Drone AI vision control</h1>

<p align="center">
    <a href="https://youtu.be/aaa">
        <img src="https://user-images.githubusercontent.com/fff/ddd.jpg" alt="count-objects-crossing-line">
    </a>
</p>

## :book: Introduction
This demo is the proof of concept of drones and AI vision combination. If we can enforce the drone to take off and landing by gestures we can also do mach more complicated tasks. The only limitation is our imagination :)

## :hammer: Tech setup
This demo is prepared to use with **DJI** drones which supports MobileSDK **4.16.4** (V5 drones are not supported - like Mini 3 series). In my case it is DJI Mini 2.
You can find [drone application](https://github.com/pazdzioch87/remote_guard_drone) on my another repository which is fork from original DJI sample. This application has three main tasks: 
- send RTMP stream
- receive SignalR masseges 
- and execute commands to drone
**For more details look on "Diagram_Drone_Complex.png"**

> **Note**
For simplicity: you can also run this Demo without any drone - by using **OBS Studio** as a RTMP stream provider. You can also preview the commands being sent (normally to drone) in ControlBroker debug console.
**For more details look on "Diagram_OBS_Simple.png"**



## 💻 Install (Docker & Ngrok account required)
Installation instruction assumes that docker engine and docker compose plugin are installed on your machine.
**1.** Run ControlBroker service to handle SignalR communication
```bash
# from the main repo directory - call following command:
docker compose up controlapi
```
**2.** Start up Nginx with RTMP module to handle RTMP stream (from Drone or from OBS)
```bash
# from the main repo directory - call following command:
docker compose up nginx
```
**3.** Run Ngrok to publish services endpoint to the internet (required only for drone verion - complex). Please read what Ngrok is and please be aware of publishing your machine port to the internet.
```bash
# this step let us to avoid router and OS firewall which could bloks services port - preventing correct work
# before we run ngrok we had to place our personal auth token to the docker-compose.yml
# righ here NGROK_AUTHTOKEN: YOUR_NGROK_AUTH_TOKEN_HERE

# run ngrok
docker compose up ngrok
```
**4.** Set up [drone application](https://github.com/pazdzioch87/remote_guard_drone) or forward your PC camera by OBS Studio to Nginx service.
Depending on your local IP addres it could be something like that: rtmp://192.168.1.XX:1935/app/stream
Port 1935 and "app" is obligatory. It is due to the nginx configuration. Stream key ("stream" in my case) could be set as you wish.

**5.** Now when we have video stream provided we can run computer vision processing.
```bash
# go to AIVisionProcessing directory and run terminal
# install dependencies
pip install -r requirements.txt
# run video processing
python run.py --source rtmp://192.168.1.18:1935/app/stream --weights drone_gestures.pt --show-preview --process-connection
```
## :memo: License
**AGPL-3.0 License**
