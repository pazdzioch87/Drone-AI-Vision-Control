<h1 align="center">Drone AI vision control</h1>

<p align="center">
    <a href="https://youtu.be/aaa">
        <img src="https://user-images.githubusercontent.com/fff/ddd.jpg" alt="count-objects-crossing-line">
    </a>
</p>

## :book: Introduction
This demo is the proof of concept of drones and AI vision combination. If we can enforce the drone to take off and landing by gestures we can also do mach more complicated tasks. The only limitation is our imagination :)

## :hammer: Tech setup
This demo is prepared to use with **DJI** drones which supports MobileSDK **4.16.4** (V5 drones are not supported). In my case it is DJI Mini 2.
You can find [drone application](https://github.com/pazdzioch87/remote_guard_drone) on my another repository which is fork from original DJI sample. This application has three main tasks: 
- send RTMP stream
- receive SignalR masseges 
- and execute commands to drone
**For more details look on "Diagram_Drone_Complex.png"**

> **Note**
For simplicity: you can also run this Demo without any drone - by using **OBS Studio** as a RTMP stream provider. You can also preview the commands being (normally to drone) sent in ControlBroker debug console.
**For more details look on "Diagram_OBS_Simple.png"**



## 💻 Install

```bash
# create python virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## 📸 Execute

```bash
python3 -m run
```