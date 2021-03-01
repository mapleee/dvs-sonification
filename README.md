# Event Based Camera Sonification
Use the events to do sonification

## Quick Start 
### Prepare environment
1. There is already one event file in data/event/insight.   
Other example event txt files can be downloaded in the links below.   
Unzip it and put all the event txt files in data/event/insight
https://pan.baidu.com/s/1VK1yF2E0GSSuSAy60OvEtQ  
password: 3zhm

2. Download the given sf2 sound font. Unzip it and put the whole file folder into data/sf2 (need to create) 
http://schristiancollins.com/soundfonts/GeneralUser_GS_1.442-MuseScore.zip

3. Install ffmpeg  
https://ffmpeg.org/download.html

4. Install requirements

```bash
pip install -r requirements.txt 
```

### Have a quick test
Use the events provided in data/event. Then just type

```bash
export PYTHONPATH=.
python main.py
```

Then, you can find the generated video in data/output/insight/label
