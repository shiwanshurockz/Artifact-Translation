# Signboard-Translation-from-Vernacular-Languages
##### [Click here for detailed description of Project](https://ai4bharat.org/articles/sign-board)

## Problem Statement
The goal of this project is to develop an App which translates the text written on a signboard to another language as desired by the user. The user will just point at the signboard using the camera of his/her phone and the App should then translate the text written on the signboard. We will first design a system which works for names (such as road names, city names, organisation names shop names etc.) which typically contain 1-2 words and are rarely longer than 4-5 words. We will cater to five languages in the first phase (i.e., the app can read and translate text from and to one of these five languages). In the next phase we will increase the number of languages to 15. Finally, in the last phase of the project, we will support translation of longer texts written on signboards, such as “Please do not throw garbage here”.

## Why this is relevant in the Indian context
India has 22 constitutionally recognised languages written in 13 different scripts. An average traveller, on a business or pleasure trip, often gets confused by the various signboards written in an unfamiliar language in a new region. This often spoils the experience of visiting a new place and the traveler goes back with not-so-fond memories. This often leads to language tussles wherein people of region A may feel that people of region B are not considerate enough to display signboards in their language and vice versa. In reality, this is simply a logistics problem. It is just impossible to have every signboard in  every city/town/village across the country written in 22 different languages. The real estate available on the signboard may allow the text to be written in 2-3 languages only. Hence there are bound to be many languages which will be left out. The resulting unfortunate, unpleasant and unproductive bitterness can easily be avoided by building better Apps. We believe that the App developed as a part of this project will allow people from different regions to read and understand text written in native languages as they are traveling across the country.

## Dataset Used
1. [Text Detection (Detecting bounding boxes containing text in the images)](https://drive.google.com/open?id=1Z6Qxr-q-F54iYB2G1AyoDymBh64f5REZ)
(428 real images with annotations)

2. [Text Recognition (Getting the text from the detected crop)](https://drive.google.com/open?id=1C0-mc0WAIdssS5KJwOjghaWaqiImZZUr)
(1740 cropped word images from real pictures with annotations)

3. [Combined Synthetic Train Set for 1 & 2:](https://drive.google.com/open?id=1E5kI8CLoC-XffqQMTWwSpBIPp1Wb2tne)
(consist of approx ~ 100,000 images with annotations)

4. [Transliteration (Transliterating Indic text to English)](http://workshop.colips.org/news2018/dataset.html)

5. [Translation ](http://www.cfilt.iitb.ac.in/iitb_parallel/)

## Pipeline for Sign Board translation :
![No of models implemented](https://github.com/shiwanshurockz/Signboard-Translation-from-Vernacular-Languages/blob/master/Images/img1.jpg "No of models implemented")

![Deep Learning Trends for Scene-Text Detection and Recognition](https://github.com/shiwanshurockz/Signboard-Translation-from-Vernacular-Languages/blob/master/Images/img2.jpg "Deep Learning Trends for Scene-Text Detection and Recognition")

## Final Result

Sucessfully build this complete project where the user front end is written as an Android application, user upload the image from its Android device containing the text which is not understood by the user.
When user click the upload button, image is uploaded to the server for further proccessing where all the above mentioned model is implemented. The server process the image then return back the same image which contains transliterated text in required language by the user.

![Gif of execution](https://github.com/shiwanshurockz/Signboard-Translation-from-Vernacular-Languages/blob/master/Images/gif.gif) { width: 200px; }