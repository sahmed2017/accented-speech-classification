# Accented Speech Classification through Audio Features Collaboration

The recognition of accented speech still remains a dominant  problem  in  automatic  speech  recognition  (ASR)  systems. Statistical  analysis  has  shown  that  accent  is  one  of  the  key factors in speaker variability that affects the performance of ASR.  The problem of classifying accented speech is approached through audio segmentation and the collaboration of several audio features, MFCC  (Mel-frequency  cepstral  coefficients), and the average of RMS  (Root-mean-square), ZCR (Zero Crossing Rate), Chromagram, Spectral Centroid, Spectral Bandwidth, and Spectral Rolloff. The  results  show  that  the  collaboration  of  the  specified audio  features,  when  unscaled,  led  to  either  a  neutral  or negative effect on the classifiers’ accuracy. Frequency-domain  features,  in  particular,  were  found  to  mitigate  the scores  of  all  the  classifiers,  except  Random  Forest  (RF). These could be recognized through graphing the training and testing loss of the ANN in that they all shared an initial high value of loss that sharply decreased and gradually declined. However, scaling the features resulted in significant increases  in  accuracy  for  Support  Vector  Machines  (SVM), k-Nearest Neighbors (kNN), and the Artificial Neural Network  (ANN).  Random  Forest  (RF)  remained  consistent throughout. 


Keywords:  accented  speech  classification;  feature  scaling; audio features collaboration; audio segmentation. 

Within  the  the  scope  of  this  work,  audio  features  are divided  into  a  main  feature,  the  MFCC,  which  yields enough accuracy on its own, and assist features, the remainder  of  the  features,  which  are  paired  with  the  main feature. Forty audio files each of Arabic, English, 
Mandarin,  Spanish,  and  Korean  were  selected  from  the Speech Accent Archive  in  which  participants  were  asked to  read  a  set  passage.  Each  audio  file was  then  segmented  into  three  second  clips,  with  some shorter  segments  in  order  to  contain  the  remainder  of  the audio.

Setup

1. Audio Segmentation

<img width="361" alt="Screen Shot 2022-12-15 at 8 38 00 PM" src="https://user-images.githubusercontent.com/118930981/208009256-d9760776-607f-418c-bfab-83660f27756f.png">

<img width="461" alt="Screen Shot 2022-12-15 at 8 38 40 PM" src="https://user-images.githubusercontent.com/118930981/208009352-0cc7f2a1-037f-483c-a9fa-992b502e8cdb.png">

2. Feature Scaling
3. 7-Fold Cross Validation
4. ANN Model -- The same ANN model was not used. For scaled features, an ANN of three dense and two dropout layers was implemented. For the unscaled features, the same ANN was used without the dropout layers. This was done in order to optimize the results for both categories.


Results

<img width="974" alt="Screen Shot 2022-12-15 at 8 40 55 PM" src="https://user-images.githubusercontent.com/118930981/208009643-921cceb4-1629-45f2-a6b0-9a9c77dab7b9.png">
