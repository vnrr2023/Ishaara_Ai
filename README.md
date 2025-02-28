
# Ishaara Ai : The Indian Sign Language Translator For the Indian Deaf And Mute

Over 63 million people in India, or about 6.3% of the population, are hard of hearing. Though they do not regard their hearing loss as a disability, but rather a different way of life, their social interactions can be significantly limited.

To prevent isolation within the often exclusive deaf community, we, the students of [MH Saboo Siddik College of Engineering](https://www.mhssce.ac.in/), affiliated with the [University of Mumbai’s](https://mu.ac.in/) Computer Engineering Department, took the challenge to bridge this communication gap using machine learning and computer vision techniques.

Our project focuses on creating a robust, scalable system to predict, interpret, and translate Indian Sign Language (ISL) in real-time, without requiring specialized hardware. ISL combines actions, facial expressions, and body language, differing from other sign languages like American Sign Language (ASL), which may use single-hand gestures; ISL usually uses both hands. This complexity presents challenges in developing an accurate machine learning model for ISL interpretation.

For our third-year Mini Project, we began our research with a visit to the [Bombay Institution For Deaf & Mutes](https://www.justdial.com/Mumbai/Bombay-Institute-For-The-Deaf-Mutes-Opposite-Mazgaon-Court-Mazgaon/022P860275_BZDET). We initially gathered information about commonly used actions and words through the institution's teachers.

## View Ishaara Ai website (Live Now 🔥)
[![image](https://img.shields.io/badge/IshaaraAi_v1-red?style=flat)](https://ishaara.netlify.app/)
[![image](https://img.shields.io/badge/IshaaraAi_v2-yellow?style=flat)](https://ishaara.vercel.app/)

Table Of Contents
- [Tech Stack and Libraries](https://github.com/vnrr2023/ishaara_ml/tree/main?tab=readme-ov-file#tech-stack)
- [Object Detection based IshaaraAi](https://github.com/vnrr2023/ishaara_ml/tree/main?tab=readme-ov-file#object-detection-based-ishaara-model)
- [Gru Based Sequence2Sequence IshaaraAi](https://github.com/vnrr2023/ishaara_ml/tree/main?tab=readme-ov-file#gru-based-sequence-to-sequence-model-using-mediapipe-version-2)
- [ConvLSTM Based IshaaraAi](https://github.com/vnrr2023/ishaara_ml/tree/main#convlstm-based)
- [Ishaara Net using 3d Convolution.](https://github.com/vnrr2023/ishaara_ml/tree/main#conv3d-based-video-classification-approach)
- [Future Scope and practices](https://github.com/vnrr2023/ishaara_ml/tree/main#future-scope-and-other-practices)
- [Contributors](https://github.com/vnrr2023/ishaara_ml/tree/main#-contributors)
- [System Architecture.](https://github.com/vnrr2023/ishaara_ml/tree/main#architecture-of-entire-system)



## Tech Stack

![My Skills](https://skillicons.dev/icons?i=python,javascript,react,tailwind,netlify&perline=10)
## Libraries and ML Frameworks
![My Skills](https://skillicons.dev/icons?i=tensorflow,pytorch,sklearn,&perline=10)
 
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) 
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

### Object Detection Based Ishaara Model
* Collected Images with static actions.
* Labelled them using Roboflow and LabelImg
* Trained model on cloud using roboflow.
* Run it locally without any external installation.
   - Click [here](https://github.com/vnrr2023/ishaara_ml/blob/main/RCNN%2Broboflow/sampleApp%20(1).zip) to get the ```Zip``` Folder.
   - click the ```download``` icon.
   - Extract the folder and open ```isl-actions-v4```.
   - Run the ``` index.html``` file and start Signing.
* Metrics
  ![metrics](https://github.com/vnrr2023/ishaara_ml/blob/main/RCNN+roboflow/metrics/metrics.png?raw=true)

###  GRU Based Sequence To Sequence Model using Mediapipe 

<p align="center">
  <img src="https://github.com/vnrr2023/ishaara_ml/blob/main/GIFS/rnn.gif?raw=true" width="200px" height="200px" />
</p>

- Extracted landmarks using Mediapipe from videos collected from school students.
- Works with 99% testing accuracy with 6 actions namely 
  + blank (no action)
  + hello
  + how are you
  + sorry
  + welcome
  + thank you
- Also consists of ready model and depth detection model to automate the users process of signing.
- For sentence creation ChatGPT was used.
- For trying Out 
  + Intall python 3.11

```bash
  pip install -r requirements.txt
  git clone https://github.com/vnrr2023/ishaara_ml/tree/main
  cd "GRU+Mediapipe"
  cd "Version_2"
  py main.py

```
### ConvLSTM Based 
<p align="center">
  <img src="https://github.com/vnrr2023/ishaara_ml/blob/main/GIFS/convlstm.gif?raw=true" width="200px" height="200px" />
</p>

- This is a combination of Convolution layers and Lstm cells through which we tried ``` Video Classification for actions ```.
- Collected Videos and Trained a model for 8hrs on CPU.
- The [models](https://github.com/vnrr2023/ishaara_ml/tree/main/ConvLSTM/models) folder contains all the ConvLSTM models.
- The [main.py](https://github.com/vnrr2023/ishaara_ml/blob/main/ConvLSTM/main.py) contains file through u can do real time predictions.
- The [train.py](https://github.com/vnrr2023/ishaara_ml/blob/main/ConvLSTM/train.py) contains the script to train the model. (First you need to run the ```model_builder.py``` script).
- The [model_builder.py](https://github.com/vnrr2023/ishaara_ml/blob/main/ConvLSTM/model_builder.py) script when run creates a model and stores it as ``` raw_model.h5``` which is then used for training. (This is done to increase the efficiency and system wont go in deadlock statte)
* Metrics <br/> 
  ![metrics](https://github.com/vnrr2023/ishaara_ml/blob/main/ConvLSTM/Stats/stats.png?raw=true)

### Conv3D Based Video Classification Approach.

<p align="center">
  <img src="https://github.com/vnrr2023/ishaara_ml/blob/main/GIFS/1%20(1).gif?raw=true" width="200px" height="200px" />
</p>

- Sadly we could not develop this model as we had less powerfull resources.
- It creates a 3Gb model and needs 32gb ram + a powerfull gpu && cpu.
- U can train it  (Mail chouhanvivek207@gmail.com for data).

## Future Scope and other practices.
- Develop an object detection model on 100 words and 26 alphabets by collection large volume of data.
- A specialized online meet extension through which ishaara mode can be turned on and deaf and mute people can easily interact without any human intervention.
- A quick Comm model so that users can use it for quick communication like eg ``` where is abc marg ? how to go there ? ```
- Educational based website where students can practice ISL using AI.

## Challenges and lessons
- Deep learning models to train from scratch is difficult.  They need High volume of data and in proper format.
- More powerfull resources (hardware) are required like GPU to parallely process the data which are infact Matrices.


## 🔗 Contributors
<p align="center">
  <img src="https://github.com/vnrr2023/ishaara_ml/blob/main/assets/IMG-20240429-WA0044.jpg?raw=true" width="550" />
</p>


* Vivek Chouhan (extreme right) <br/>
  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/vivek-chouhan/)

* Nishikant Raut (extreme left) <br/>
  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nishidev/)

* Rehan Sayyed (black shirt-middle) <br/>
  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rehan42/)

* Rohit Deshmukh (purple shirt- middle) <br/>
   [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rohit-deshmukh-/)


## Silent Contributors

We thank the entire staff of [Bombay Institution For Deaf & Mutes](https://www.justdial.com/Mumbai/Bombay-Institute-For-The-Deaf-Mutes-Opposite-Mazgaon-Court-Mazgaon/022P860275_BZDET) for their valuable cooperation throughout the project and specially the children.
<br/> They selflessly helped us collect data and gave us valuable information about ISL.

<p align="center">
  <img src="https://github.com/vnrr2023/ishaara_ml/blob/main/assets/IMG-20231020-WA0004.jpg?raw=true" width="500" />
</p>




## Architecture Of Entire system
![architecture](https://github.com/vnrr2023/ishaara_ml/blob/main/RCNN+roboflow/architecture%20.gif?raw=true)

More icons
https://badges.pages.dev/
