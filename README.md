# Spam Classifier

A spam classifier that classifies user input as spam/not spam

To run :

Clone repository and run deploy.py. Application will be served at http://localhost:8000/

## Prerequisites

You must have the following packages installed :

* Scikit Learn
* Pandas
* Flask

## Project Structure

This project contains 3 major parts :

1. deploy.py : This contains the code that classifies input as spam/not spam using the Naive Bayes algorithm ( with 72% accuracy)
2. templates : This folder contains the HTML template that allows users to input data and detect spam
3. ham_spam.csv : The csv file use to train the model
