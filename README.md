# REDBot
The aim of this project is to build a movie recommendation system by filtering emotions using multi-label mood classification of the text input to the system by the user

## Demo
![](demo.gif)

## Approach
- Text input to the system is used for Multi-label classification using RoBERTa which outputs the probabilities of all the 5 classes
- emotion2genre matrix has been created using data from a survey conducted
- Emotion vector is converted to vector of genres by matrix multiplication with emotion2genre matrix
- Movies are recommended using cosine similarity of the genre vector with other genre vectors in the movies dataframe
- The entire pipeline is deployed using Flask

## Dataset
- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset by the Google research team has be used
  - All text with "Neutral" and "Disgust" emotion have been removed from the dataset
  - The emotions have been reduced to 5 from 28 using the [ekmann mapping](https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json)
  - The emotions have been one hot encoded to convert it into appropriate form for Multi-label Classification
- [IMDB](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset) dataset has been used for the recommendation system.
  - Movies with 8+ IMDB rating are retained
  - Movies with Hindi and English languages are retained
  - Movies pertaining to India, USA and UK are retained
 
 ## Team Members
 - [Debarshi Chanda](https://github.com/DebarshiChanda)
 - [Manan Kakkar](https://github.com/manankakkar13)
